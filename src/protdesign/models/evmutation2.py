"""
Wrapper class around EVmutation2/picasso model
"""
from os import PathLike
from typing import Self, Tuple, Sequence, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
from loguru import logger

from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList, Mutant
from protdesign.constants import MASK
from protdesign.sequence import valid_protein_sequence
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.types import DeviceType, StatusCallback, BatchSize

try:
    from picasso_model import model, features, parsers
    import torch
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class EVmutation2(BaseModel, Scorer, Generator):
    """
    Wrapper class around EVmutation2/picasso model
    """
    available = IMPORT_AVAILABLE
    name: str = "EVmutation2"

    requires_heavy_build: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    requires_target: bool = True
    requires_seqs: bool = True
    requires_msa: bool = True
    requires_3d: bool = False
    requires_fixed_length: bool = True
    handles_deletions: bool = True

    def __init__(
        self,
        model_file_path: str | PathLike,
        encoder_num_samples: int = 1,
        encoder_num_recycling_steps: int = 4,
        encoder_max_num_msa: int | None = 2048,
        decoder_batch_size: BatchSize = 64,
        decoder_num_full_samples: int = 16,
        decoder_num_mutant_samples: int = 16,
        decoder_share_order_across_encodings: bool = True,
        keep_model: bool = False,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate new EVcouplings2 model

        TODO: support min_p sampling and sample_gaps

        Parameters
        ----------
        model_file_path
            Path to Lightning checkpoint
        encoder_num_samples
            Number of encoder samples to draw (at least 1), can improve model performance
        encoder_num_recycling_steps
            Recycling steps to run when computing encoding
        encoder_max_num_msa
            Number of sequences to sample from MSA when computing encoding
        decoder_batch_size
            Maximum number of sequences to decode concurrently
        decoder_num_full_samples
            Number of sampled decoding orders when computing full sequence scores
        decoder_num_mutant_samples
            Number of sampled decoding orders when computing mutant scores
        decoder_share_order_across_encodings
            Reuse decoding order across multiple encodings (if more than 1 used)
        keep_model
            If True, keep model parameters asssociated to instance after build step
            to avoid reloading when scoring/generating. If serializing model, set to
            False to avoid storing model parameters repeatedly.
        device
            Device to use for computations
        """
        super().__init__()
        self.model_file_path = model_file_path
        self.keep_model = keep_model
        self.device = device

        # modelled system
        self.system = None

        # lazy-load model when needed
        self.model = None

        # model parameters for encoding and decoding during inference
        self.encoder_num_samples = encoder_num_samples
        self.encoder_num_recycling_steps = encoder_num_recycling_steps
        self.encoder_max_num_msa = encoder_max_num_msa
        self.decoder_batch_size = decoder_batch_size
        self.decoder_num_full_samples = decoder_num_full_samples
        self.decoder_num_mutant_samples = decoder_num_mutant_samples
        self.decoder_share_order_across_encodings = decoder_share_order_across_encodings

        if self.encoder_num_samples < 1 or self.decoder_num_full_samples < 1 or self.decoder_num_mutant_samples < 1:
            raise ValueError(
                "encoder_num_samples, decoder_num_single_samples and decoder_num_single_samples must all be > 0"
            )

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.decoder_batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")

        # encodings created when calling build() method
        self.encoding = None
        self.pos_mask = None

    @property
    def ready(self):
        return self.system is not None and self.encoding is not None

    @classmethod
    def can_model(cls, system: System) -> Tuple[bool, str]:
        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]
        if target.sequences is None or len(target.sequences.seqs) == 0:
            return False, "Must provide sequences for model inference"

        if not target.sequences.aligned:
            return False, "Provided sequences must be aligned"

        # this should be ensured by construction of system but check again to be safe
        if not valid_protein_sequence(
            target.rep, allow_mask=True, allow_gap=False, allow_ambiguous=True
        ):
            return False, "Input sequence may only contain AA symbols or mask"

        # TODO: more checks on alignment: does length match target rep;
        #  and is alignment compatible with a3m format

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        # TODO: implement meaningful requirements depending on target size instead of made up values
        return RequiredResources(
            min_gpu_cores=1,
            min_gpu_memory_per_core=16000,
            min_cpu_cores=1,
            min_cpu_memory_per_core=16000,
            max_batch_size=512,
            time=1,
        )

    def _load_model(self):
        # avoid reloading if already loaded
        if self.model is not None:
            return

        self.model = model.Model.load_from_checkpoint(
            self.model_file_path, map_location=torch.device(self.device)
        )

        # switch to evaluation mode
        self.model.eval()

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self._release_cache()

    def build(
        self,
        system: System,
        status_callback: StatusCallback | None = None
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system)

        # store system with this instance
        self.system = system
        target = self.system[0]

        # load MSA
        # TODO: clunky hack - reassemble sequences back into one string and pass into parser;
        #  should really update parser to receive sequences and headers
        msa_a3m = target.sequences.to_a3m()
        a3m_lines = "".join(
            f">{seq.id_}\n{seq.seq}\n" for seq in msa_a3m.seqs
        )

        msa = parsers.parse_a3m(a3m_lines)

        # ideally would move this check over to can_model() but checking can
        # then become more resource-intensive
        if len(msa.sequences[0]) != len(target.rep):
            raise ValueError(
                "Length of MSA does not map to length of target representation"
            )

        # featurize and batch; add structure features here eventually as well when
        # that part of EVmutation2 model is finished
        d = features.extract_msa_feature_data(msa)
        f = features.prepare_msa_features(*d)
        input_features = features.batch_features(
            [f], device=self.device
        )

        # also store position mask for prediction time
        self.pos_mask = input_features.pos_mask.cpu()

        # context for loading (and possibly destroying model parameters)
        with model_param_context(self._load_model, self._delete_model, self.keep_model):
            with torch.no_grad():
                s, p = [], []
                # create requested number of encoder samples (single and pair representation)
                for i in range(self.encoder_num_samples):
                    cur_s, cur_p = self.model.encoder(
                        input_features,
                        num_recycling_steps=self.encoder_num_recycling_steps,
                        max_num_msa=self.encoder_max_num_msa,
                    )
                    s.append(cur_s)
                    p.append(cur_p)

                # concatenate into one tensor each for single and pair representation
                s = torch.cat(s, dim=0)
                p = torch.cat(p, dim=0)

                # store encodings, make sure these are moved to CPU for good serialization behaviour
                self.encoding = (
                    s.cpu(), p.cpu()
                )

        # TODO: automatically estimate robust maximum possible batch size for decoder if set to "auto"
        #  depending on available resources, and update self.decoder_batch_size

        # return self to allow method chaining
        return self

    def positions(
        self
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()

        # implementation here is very simple: we model all positions of exactly one target
        # protein sequence; none of the positions along the sequence are excluded so
        # we can simply enumerate starting from first_index
        target = self.system[0]
        return [
            (0, idx) for idx, _ in enumerate(target.rep, start=target.first_index)
        ]

    @contextmanager
    def _prepare(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper to move all necessary information to target device
        when calling inference methods

        Returns
        -------
        Tuple of single representations, pair representations, and position_mask on
        target device
        """
        # move representations and position mask to device
        try:
            (s, p) = self.encoding
            s = s.to(self.device)
            p = p.to(self.device)
            pos_mask = self.pos_mask.to(self.device)
            assert s.shape[0] == p.shape[0], "Number of single and pair representations does not agree"

            yield s, p, pos_mask
        finally:
            del s
            del p
            del pos_mask
            self._release_cache()

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        """
        TODO: support min_p sampling and sample_gaps parameters eventually
        """
        self.ready_or_raise()

        # verify validity of entity selection, even if not used since
        # at this point method can only handle single entity
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError("Can only design single entity (entities = [0] | None)")
        else:
            # not used for now
            entities = [0]

        target = self.system[0]

        # extract fixed pos for single chain
        if fixed_pos is not None:
            if len(fixed_pos) != 1 or list(fixed_pos)[0] != 0:
                raise ValueError(
                    "Only accepting position mapping for entity 0"
                )

            fixed_pos = set(fixed_pos[0])
            # verify if all positions are valid
            self.valid_positions(fixed_pos, entity=0, raise_invalid=True)
        else:
            fixed_pos = set()

        if len(fixed_pos) == len(target.rep):
            raise ValueError("All positions fixed, need to sample at least one position")

        # mark which positions to design (with mask symbol)
        base_seq = [
            symbol if pos in fixed_pos else MASK
            for pos, symbol in enumerate(
                target.rep, start=target.first_index
            )
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model),
            self._prepare() as (s, p, pos_mask)
        ):
            # sampling function expects number of designs to be a multiple of batch_size,
            # so adjust accordingly
            if rem := num_designs % self.decoder_batch_size != 0:
                num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            else:
                num_designs_adj = num_designs

            logger.warning(
                "Sampling using a preliminary inefficient O(N^3) implementation which needs to be "
                "improved to O(N^2) for production use"
            )

            # note: method has @torch.inference_mode() so no_grad not necessary here
            # TODO: update sampling method to update generation status dynamically with callback
            designs, _ = self.model.decoder.sample_inefficient(
                single=s,
                pairwise=p,
                pos_mask=pos_mask,
                seq=base_seq,
                batch_size=self.decoder_batch_size,
                num_samples=num_designs_adj,
                temperature=temperature,
                # min_p=None,  # TODO: implement
                # sample_gaps=None,  # TODO: implement
            )

        # score the designs relative to entity sequence (ideally, user supplied WT sequence, but user can
        # always rescore the designs later if needed)

        # prepend reference sequence, and create instances
        ref_and_designs = [target.rep] + list(designs.seq)
        instances = [
            SystemInstance(
                EntityInstance(rep=rep)
            ) for rep in ref_and_designs
        ]

        # score and attach to instances (normalize by reference score)
        scores = self.score(instances)
        ref_score = scores[0]

        instances_with_score = [
            SystemInstance(
                EntityInstance(rep=seq),
                score=score - ref_score
            ) for seq, score in zip(ref_and_designs, scores)
        ]

        # return designs, remove reference in first position again
        return instances_with_score[1:]

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # validate sequences
        _ = (
            self.system.valid_instance(
                instance, fixed_length=True, validate_reps=True, raise_invalid=True
            ) for instance in instances
        )

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model),
            self._prepare() as (s, p, pos_mask)
        ):
            scores = self.model.decoder.score_full_probability(
                [instance[0].rep for instance in instances],
                single=s,
                pairwise=p,
                pos_mask=pos_mask,
                batch_size=self.decoder_batch_size,
                num_samples=self.decoder_num_full_samples,
                share_decoding_order_across_encodings=self.decoder_share_order_across_encodings,
            )

        # average the logits across encoder and decoder samples,
        # and make sure aggregated dataframe it is sorted by sequence index
        scores_agg = scores.groupby(
            level="seq_idx"
        ).mean().sort_index()

        # return as numpy vector
        return scores_agg["score"].values

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        self.ready_or_raise()

        # check instance against molecular system, requiring fixed length of sequence
        # as was used for entity specification as we have a fixed-length model
        self.system.valid_instance(
            instance, fixed_length=True, validate_reps=True, raise_invalid=True,
        )

        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # extract single target entity from system, nd get sequence from instance
        # (we safely can access this as we have verified instance against system)
        target = self.system[0]
        instance_seq = instance[0].rep

        # validate positions
        if positions is not None:
            positions = self.valid_positions(positions, raise_invalid=True)

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model),
            self._prepare() as (s, p, pos_mask)
        ):
            # get number of encoder samples (single/pair representations)
            num_encodings = s.shape[0]

            # iterate through encoder samples; we can average these as these are log-odds scores, i.e.
            # different decoding orders will already have cancelled out. ultimately, this functionality should
            # probably go inside the score_single_mutants() method in picasso...
            effects = {}
            for idx_enc in range(num_encodings):
                # note: method has @torch.inference_mode() so no_grad not necessary here
                effects[idx_enc] = self.model.decoder.score_single_mutants(
                    seq=instance_seq,
                    first_index=target.first_index,
                    single=s[[idx_enc]],
                    pairwise=p[[idx_enc]],
                    pos_mask=pos_mask,
                    position_subset=positions,
                    num_samples=self.decoder_num_mutant_samples,
                    batch_size=self.decoder_batch_size,
                )

            # assemble multiple scores and average logits
            effects = pd.concat(
                effects, axis=0, names=["encoder_sample"]
            ).groupby(
                level=["pos", "wt_aa"]
            ).mean()

        # TODO: rename index level
        # TODO: assign entity index to table
        # TODO: remove symbols except AAs, gap and mask (also drop "wt" column)
        # TODO: define good return type (custom dataframe?)
        # TODO: transform results into proper format (axes annotation, position/AA ordering, right alphabet)
        return effects

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # check instance against molecular system, requiring fixed length of sequence
        # as was used for entity specification as we have a fixed-length model
        self.system.valid_instance(
            instance, fixed_length=True, validate_reps=True, raise_invalid=True,
        )

        # verify if mutants are valid relative to system and instance
        self.system.valid_mutants(
            instance, mutants, allow_gap=False, raise_invalid=True
        )

        # extract single target entity from system, and get sequence from instance
        # (we safely can access this as we have verified instance against system)
        target = self.system[0]
        instance_seq = instance[0].rep

        # transform mutants into format expected by EVmutation2
        mutants_transformed = [
            [
                (subs.pos, subs.ref, subs.to) for subs in mutant
            ] for mutant in mutants
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model),
            self._prepare() as (s, p, pos_mask)
        ):
            # get number of encoder samples (single/pair representations)
            num_encodings = s.shape[0]

            # iterate through encoder samples; we can average these as these are log-odds scores, i.e.
            # different decoding orders will already have cancelled out. ultimately, this functionality should
            # probably go inside the score_single_mutants() method in picasso...
            effects = {}
            for idx_enc in range(num_encodings):
                # note: method has @torch.inference_mode() so no_grad not necessary here
                effects[idx_enc] = self.model.decoder.score_mutants(
                    seq=instance_seq,
                    mutants=mutants_transformed,
                    first_index=target.first_index,
                    single=s[[idx_enc]],
                    pairwise=p[[idx_enc]],
                    pos_mask=pos_mask,
                    num_samples=self.decoder_num_mutant_samples,
                    batch_size=self.decoder_batch_size,
                )

            # assemble multiple scores and average logits
            effects = pd.concat(
                effects, axis=0, names=["encoder_sample"]
            ).groupby(
                level="mutant"
            ).mean()
            # TODO: make sure sorting order is not messed up

        # TODO: cannot assume that output list is necessarily ordered
        # TODO: update return type, in signature and also in abstract class
        return effects

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        self.ready_or_raise()

        # validate instance sequences
        [
            self.system.valid_instance(
                instance, fixed_length=True, validate_reps=True, raise_invalid=True
            ) for instance in instances
        ]

        # validate entity specification (only handle single entity for now)
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        target = self.system[0]

        # validate positions
        self.valid_positions(positions, entity=0, raise_invalid=True)

        # extract sequences
        seqs = [
            instance[0].rep for instance in instances
        ]

        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model),
            self._prepare() as (s, p, pos_mask)
        ):
            scores = self.model.decoder.score_conditional(
                seqs=seqs,
                positions=positions,
                first_index=target.first_index,
                single=s,
                pairwise=p,
                pos_mask=pos_mask,
                batch_size=self.decoder_batch_size,
                num_samples=self.decoder_num_mutant_samples,
                share_decoding_order_across_encodings=self.decoder_share_order_across_encodings,
            )

        # average encoder and decoder samples
        scores_agg = scores.groupby(
            level=["seq_idx", "pos"]
        ).mean().sort_index()

        # TODO: assign entity index to table
        # TODO: remove symbols except AAs, gap and mask
        # TODO: define good return type (custom dataframe?)
        return scores_agg

