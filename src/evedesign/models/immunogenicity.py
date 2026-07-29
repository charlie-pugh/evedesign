import subprocess
from collections import OrderedDict
from os import PathLike, path
from tempfile import TemporaryDirectory
from typing import Sequence, Literal, Any, Self
import numpy as np
import pandas as pd

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback, DeviceType
from evedesign.utils import status_start, status_done, status_progress

try:
    from aiki_hla.inference import score_dataframe
    from aiki_hla.scan.extract import extract_peptides, unique_peptide_strings
    from aiki_hla.scan.alleles import resolve_allele_panel
    from aiki_hla.scan.aggregate import aggregate_hotspots
    from aiki_hla.viability_gate import score_gate_batch, compose
    from aiki_hla.data.sequences import list_alleles
    AIKI_HLA_AVAILABLE = True
except ImportError:
    AIKI_HLA_AVAILABLE = False


class LRU(OrderedDict):
    """
    Helper class for maintaining our own LRU dictionary,
    which allows to add computations in a batched fashion
    """
    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)          # mark as recently used
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class MixMHC2Pred(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around MixMHC2Pred
    """
    available = True
    name: str = "MixMHC2Pred"
    citations: list[str] = ["doi.org/10.1038/s41587-019-0289-6", "10.1016/j.immuni.2023.03.009"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = False  # TODO: could be set to True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        alleles: dict[str, float],
        binary: PathLike,
        peptide_lengths: Sequence[int] = (15,),
        truncate_rank: float | None = 10.0,
        prediction_cache_size: int = 10 ** 8,
    ):
        """
        # TODO: docs
        """
        self.alleles = {
            allele.replace("HLA-", "").replace("*", "_").replace(":", "_"): weight
            for allele, weight in alleles.items()
        }
        self.peptide_lengths = peptide_lengths
        self.binary = binary
        self.truncate_rank = truncate_rank
        self._lru_cache = LRU(
            maxsize=prediction_cache_size
        )
        self._system = None

    @staticmethod
    def extract_peptides(
        seq: str,
        lengths: Sequence[int],
        n_flank_length: int | None = None,
        c_flank_length: int | None = None,
    ):
        peps = []
        for length in lengths:
            for i in range(len(seq) - length + 1):
                pep = seq[i:i + length]

                if n_flank_length is not None:
                    n_flank = seq[max(0, i - n_flank_length):i]
                else:
                    n_flank = None

                if c_flank_length is not None:
                    c_flank = seq[i + length:i + length + c_flank_length]
                else:
                    c_flank = None

                peps.append(
                    (pep, i, n_flank, c_flank)
                )

        return peps

    def _run_mixmhc2pred(
        self,
        peptides_with_context: set[tuple[str, str]],
    ):
        with TemporaryDirectory() as tmpdir:
            input_file = path.join(tmpdir, "input.txt")
            output_file = path.join(tmpdir, "output.txt")

            with open(input_file, "w") as f:
                for (pep, pep_ctx) in peptides_with_context:
                    f.write(f"{pep}\t{pep_ctx}\n")

            cmd = [
                self.binary, "--input", input_file, "--output", output_file, "-a"
            ] + list(self.alleles)

            subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                cwd=None, shell=False, env=None,
            )

            return pd.read_csv(output_file, sep="\t", comment="#")

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        # for now only handle single-protein systems but can trivially extend to multiple proteins
        for entity_idx, entity in enumerate(system):
            if entity.type != "protein":
                return False, "Can only handle protein entities"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score immunogenicity for full sequence (higher = more immunogenic)
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        status_start(status_callback, "Starting scoring")

        # first collect all peptides for this run
        unique_peps = {}

        # also collect which peptides are not yet in cache
        new_peps = set()
        reusing = 0

        for instance_idx, instance in enumerate(instances):
            for entity_idx, entity in enumerate(instance):
                # remove deletions and insertions from sequence
                seq = "".join(entity.normalized_rep())

                # extract all peptides for current instance
                peps = self.extract_peptides(
                    seq=seq,
                    lengths=self.peptide_lengths,
                    n_flank_length=3,
                    c_flank_length=3,
                )

                # iterate peptides, compute context and store in global map
                for pep_seq, seq_idx, ctx_n, ctx_c in peps:
                    ctx = "".join([
                        ctx_n.ljust(3, "-"), pep_seq[:3], pep_seq[-3:], ctx_c.rjust(3, "-")
                    ])

                    key = (pep_seq, ctx)

                    if key not in unique_peps:
                        unique_peps[key] = []

                    unique_peps[key].append(
                        (instance_idx, entity_idx, seq_idx)
                    )

                    if key not in self._lru_cache:
                        new_peps.add(key)
                    else:
                        reusing += 1

        # predict peptides we haven't seen before and stored in cache
        if len(new_peps) > 0:
            new_preds = self._run_mixmhc2pred(
                new_peps
            ).set_index(
                ["Peptide", "Context"]
            ).to_dict("index")
        else:
            new_preds = {}

        print("NEW PEPS:", len(new_peps), "REUSING:", reusing)   # TODO: remove

        # deduplicate binding cores
        core_map = {}

        for key, hit_list in unique_peps.items():
            try:
                stats = new_preds[key]
            except KeyError:
                stats = self._lru_cache[key]

            for allele, allele_weight in self.alleles.items():
                rank = stats["%Rank_" + allele]
                core_hit = stats["CoreP1_" + allele]

                # skip cores if truncation is enabled and rank not high enough
                if self.truncate_rank is not None and rank > self.truncate_rank:
                    continue

                # assign cores to instances/entities
                for instance_idx, entity_idx, pos_idx in hit_list:
                    if instance_idx not in core_map:
                        core_map[instance_idx] = {}

                    # 1-based index for returned binding cores, add pos from hit list and
                    # first_index of respective entity
                    pos_remapped = self._system[entity_idx].first_index + pos_idx + core_hit - 1
                    instance_pos = (entity_idx, pos_remapped)

                    if instance_pos not in core_map[instance_idx]:
                        core_map[instance_idx][instance_pos] = {}

                    if rank < core_map[instance_idx][instance_pos].get(allele, 999):
                        core_map[instance_idx][instance_pos][allele] = rank


        # update cache with latest predictions (only at end to avoid losing entries)
        for k, v in new_preds.items():
            self._lru_cache[k] = v

        # assign scores to instances, this creates shallow copy after which we
        # can also attach binding cores as metadata
        scores = np.zeros((len(instances)))
        for instance_idx, pos_to_cores in core_map.items():
            instance_sum = 0
            for pos, cores in pos_to_cores.items():
                # normalize ranks to 0-1 range, apply allele weights
                pos_transformed = -np.log10(
                    [self.alleles[core] * rank / 100.0 for core, rank in cores.items()]
                )

                instance_sum += pos_transformed.sum()

            scores[instance_idx] = instance_sum

        # first assign scores, this creates a shallow copy of instances
        scored_instances = assign_scores_to_instances(
            instances, np.asarray(scores, dtype=float)
        )

        # also attach epitope hits to instances
        for instance_idx, pos_to_cores in core_map.items():
            if scored_instances[instance_idx].metadata is None:
                scored_instances[instance_idx].metadata = {}
            scored_instances[instance_idx].metadata["t_cell_epitopes"] = pos_to_cores

        return scored_instances


class AikiHLA(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around aiki-hla MHCI/II predictor
    """
    available = AIKI_HLA_AVAILABLE
    name: str = "aiki-hla"
    citations: list[str] = ["doi.org/10.64898/2026.06.18.733075"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = False  # TODO: could be set to True
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        model_dir_path: str | None = None,
        mhc_class: Literal["I", "II", "both"] = "II",
        allele_mode: Literal["iedb27", "supertype", "custom"] = "supertype",
        custom_alleles: list[str] | None= None,
        binder_mode: Literal["percentile", "absolute"] = "percentile",
        binder_percentile: float = 2.0,
        binder_threshold: float = 0.5,
        prediction_cache_size: int = 10**6,
        device: DeviceType = "cpu",
    ):
        """
        Create new AIKI-HLA wrapper

        Parameters
        ----------
        model_dir_path
            Path to directory with model checkpoints
        mhc_class
            MHC class to model (I, II, or both)
        allele_mode
            Select precurated allele profile or custom profile
        custom_alleles
            List of custom alleles (requires allele_mode="custom")
        binder_mode
            Mode of selecting binders based on rank or absolute probability
        binder_percentile
            If binder_mode = "percentile", use this cutoff
            (default: 2.0 is netMHCpan default for weak binder)
        binder_threshold
            If binder_mode = "absolute", use this absolute probability cutoff (default = 0.5)
        prediction_cache_size
            Size of LRU cache to reuse peptide/allele predictions without recomputing them
        device
            Device for pytorch model
        """
        if not self.available:
            raise ImportError("aiki-hla package could not be imported. Is it installed already?")

        self.model_dir_path = model_dir_path
        self.mhc_class = mhc_class
        self.allele_mode = allele_mode
        self.device = device
        self.binder_mode = binder_mode
        self.binder_percentile = binder_percentile
        self.binder_threshold = binder_threshold

        if custom_alleles is not None:
            available_alleles = list_alleles()
            for allele in custom_alleles:
                if allele not in available_alleles:
                    raise ValueError(
                        f"Allele '{allele}' not in available alleles: {available_alleles}"
                    )

                if self.mhc_class == "I" and allele.startswith("HLA-D"):
                    raise ValueError(
                        f"Not a valid class I allele: {allele}"
                    )

                if self.mhc_class == "II" and not allele.startswith("HLA-D"):
                    raise ValueError(
                        f"Not a valid class II allele: {allele}"
                    )

        self.allele_panel = resolve_allele_panel(
            mode=self.allele_mode, mhc_class=self.mhc_class, custom_alleles=custom_alleles
        )

        self._lru_cache = LRU(
            maxsize=prediction_cache_size
        )
        self._system = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        # for now only handle single-protein systems but can trivially extend to multiple proteins
        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _predict_peptides(self, peptide_allele_pairs: list[dict[str, str]]) -> list[dict[str, float]]:
        not_available = [
            pair for pair in peptide_allele_pairs
            if (pair["peptide"], pair["allele"]) not in self._lru_cache
        ]
        print(len(not_available), len(peptide_allele_pairs), len(self._lru_cache))  # TODO: Remove

        if len(not_available) > 0:
            # build dataframe of missing predictions
            df = score_dataframe(
                pd.DataFrame(not_available),
                checkpoint_dir=self.model_dir_path,
                device=self.device,
            )

            gates = score_gate_batch(
                peptides=df["peptide"].astype(str).str.upper().tolist(),
                alleles=df["allele"].astype(str).tolist(),
            )

            df["p_gate"] = [g.p_gate for g in gates]
            df["gate_quality"] = [g.quality for g in gates]

            p_aiki_col = next(
                (c for c in ("binding_prob", "score", "prob") if c in df.columns),
                None,
            )
            df["p_composite"] = [
                compose(float(pa), float(pg))
                for pa, pg in zip(df[p_aiki_col].values, df["p_gate"].values)
            ]

            df = df.rename(columns={p_aiki_col: "p_aiki"})

            # turn into same dictionary format as LRU cache
            new_preds = df.set_index(["peptide", "allele"]).to_dict("index")
        else:
            new_preds = {}

        all_preds = [
            (
                {
                    **pair,
                    **self._lru_cache[(pair["peptide"], pair["allele"])]
                }
                if (pair["peptide"], pair["allele"]) in self._lru_cache
                else {
                    **pair,
                    **new_preds[(pair["peptide"], pair["allele"])]
                }
            )
            for pair in peptide_allele_pairs
        ]

        # update LRU cache last (to avoid losing predictions before we reuse them)
        for k, v in new_preds.items():
            self._lru_cache[k] = v

        return all_preds

    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score immunogenicity for full sequence (higher = more immunogenic)
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Starting scoring")

        scores = []
        for instance_idx, instance in enumerate(instances):
            # remove deletions and insertions from sequence
            seq = "".join(instance[0].normalized_rep())

            peps = extract_peptides(
                sequence=seq,
                mhc_class=self.mhc_class
            )

            unique_peps = unique_peptide_strings(peps)

            combinations = [
                {"peptide": pep, "allele": allele} for pep in unique_peps for allele in self.allele_panel.alleles
            ]

            scored_df = pd.DataFrame(
                self._predict_peptides(combinations)
            ).rename(
                columns={
                    "p_aiki": "binding_prob",
                }
            )

            # compute aggregated immunogenicity risk
            agg = aggregate_hotspots(
                protein_length=len(seq),
                peptides=peps,
                scored_df=scored_df,
                panel=self.allele_panel,
                binder_mode=self.binder_mode,
                binder_percentile=self.binder_percentile,
                binder_threshold=self.binder_threshold,
            )

            scores.append(agg.aggregate_risk)
            status_progress(status_callback, instance_idx / len(instances) * 100)

        status_done(status_callback, "Finished scoring")

        return assign_scores_to_instances(
            instances, np.asarray(scores, dtype=float)
        )
