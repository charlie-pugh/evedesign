"""
Wrapper class around EVmutation2/picasso model
"""
from os import PathLike
from typing import Self, Tuple, Sequence, List
from contextlib import contextmanager

import numpy as np

from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import EntityOrEntityList, SystemInstance
from protdesign.constants import MASK
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
        decoder_num_single_samples: int = 16,
        keep_model: bool = False,
        device: DeviceType = "cpu",
    ):
        # TODO: document parameters
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
        self.decoder_num_single_samples = decoder_num_single_samples

        if self.encoder_num_samples < 1 or self.decoder_num_single_samples < 1 or self.decoder_num_single_samples < 1:
            raise ValueError(
                "encoder_num_samples, decoder_num_single_samples and decoder_num_single_samples must all be > 0"
            )

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        # encodings created when calling build() method
        self.encoding = None
        self.pos_mask = None

    @property
    def ready(self):
        return self.encoding is not None

    @classmethod
    def can_model(cls, system: EntityOrEntityList) -> Tuple[bool, str]:
        system = ensure_sequence(system)
        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        if system[0].sequences is None or len(system[0].sequences.seqs) == 0:
            return False, "Must provide sequences for model inference"

        if not system[0].sequences.aligned:
            return False, "Provided sequences must be aligned"

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: EntityOrEntityList,
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

    def _delete_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def build(
        self,
        system: EntityOrEntityList,
        status_callback: StatusCallback | None = None
    ) -> Self:
        # verify if we can model the system
        can_model, can_model_msg = self.can_model(system)
        if not can_model:
            raise ValueError(can_model_msg)

        self.system = ensure_sequence(system)
        target = self.system[0]

        # load MSA
        # TODO: clunky hack - reassemble sequences back into one string and pass into parser;
        #  should really update parser to receive sequences and headers
        msa_a3m = target.sequences.to_a3m()
        a3m_lines = "".join(
            f">{seq.id_}\n{seq.seq}\n" for seq in msa_a3m.seqs
        )

        msa = parsers.parse_a3m(a3m_lines)
        # TODO: might want to move this check over to can_model
        if len(msa.sequences[0]) != len(target.rep):
            raise ValueError(
                "Length of MSA does not map to length of target representation"
            )

        # featurize and batch
        # TODO: add structure features here eventually as well
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

        # return self to allow method chaining
        return self

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

            yield s, p, pos_mask
        finally:
            del s
            del p
            del pos_mask

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: Sequence[Sequence[int]] | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        # TODO: support min_p sampling and sample_gaps via model parameters
        # TODO: implement auto-estimation of batch size?
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError("Can only design single entity (entities = [0] | None)")
        else:
            entities = [0]

        # extract fixed pos for single chain
        if fixed_pos is not None:
            if len(fixed_pos) != len(entities):
                raise ValueError(
                    "There must be one list of fixed positions (possibly empty) for ech designed entity"
                )
            fixed_pos = set(fixed_pos[0])
        else:
            fixed_pos = set()

        # mark which positions to design (with mask symbol)
        base_seq = [
            symbol if pos in fixed_pos else MASK
            for pos, symbol in enumerate(
                self.system[0].rep, start=self.system[0].first_index
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

            # note: method has @torch.inference_mode() so no_grad not necessary here
            designs = self.model.decoder.sample_inefficient(
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
            print(designs)

        # TODO: score the designs

        instances = [
            SystemInstance(reps=[row.seq]) for _, row in designs[0].iterrows()
        ]
        return instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        raise NotImplementedError()

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        raise NotImplementedError()

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        raise NotImplementedError()

    def positions(self):
        raise NotImplementedError()
