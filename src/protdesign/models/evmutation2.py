"""
Wrapper class around EVmutation2/picasso model
"""
from os import PathLike
from typing import Self, Tuple

from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import EntityOrEntityList, SystemInstance
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.types import DeviceType, StatusCallback

try:
    from picasso_model import model, features, parsers
    import torch
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class EVmutation2(BaseModel, Scorer, Generator):
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
        keep_model: bool = False,
        device: DeviceType = "cpu",
    ):
        super().__init__()
        self.model_file_path = model_file_path
        self.keep_model = keep_model
        self.device = device

        # lazy-load model when needed
        self.model = None

        # TODO: store encoder params
        #   num_encoder_samples=1,
        #   num_recycling_steps=4,
        #   max_num_msa=2048,
        #   decoder - num_samples
        # TODO: store decoder params

        # encodings created when calling build() method
        self.encoding = None

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

        # TODO: verify alignment length matches target sequence length (sequence match itself
        #   not stringly needed)

        # TODO target_seq = msa.sequences[0]

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

        target = system[0]

        # load MSA
        # TODO: clunky hack - reassemble sequences back into one string and pass into parser;
        #  should really update parser to receive sequences and headers
        msa_a3m = target.sequences.to_a3m()
        a3m_lines = "".join(
            f">{seq.id_}\n{seq.seq}\n" for seq in msa_a3m.seqs
        )

        msa = parsers.parse_a3m(a3m_lines)

        # featurize and batch
        # TODO: add structure features eventually too
        d = features.extract_msa_feature_data(msa)
        f = features.prepare_msa_features(*d)
        input_features = features.batch_features(
            [f], device=self.device
        )

        # context for loading (and possibly destroying model parameters)
        with model_param_context(self._load_model, self._delete_model, self.keep_model):
            print("in context:", self.model is not None)   # TODO: remove
            with torch.no_grad():
                s, p = self.model.encoder(
                    input_features,
                    # **encoder_kwargs,  # TODO: forward these
                )
                # TODO: implement multiple samples

                self.encoding = (
                    s.cpu().numpy(), p.cpu().numpy()
                )
        print("after context:", self.model is not None)   # TODO: remove

        return self

    def score(self):
        return 123.

    def score_conditional(self):
        return 124.

    def single_mutation_scan(self):
        return None

    def positions(self):
        return []

    def generate(self) -> None:
        return "LALALA"
