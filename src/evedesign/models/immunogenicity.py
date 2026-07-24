from collections import OrderedDict
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
