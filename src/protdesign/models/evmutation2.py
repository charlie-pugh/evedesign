"""
Wrapper class around EVmutation2/picasso model
"""
from os import PathLike
from typing import Self, Tuple

from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import EntityOrEntityList
from protdesign.utils import ensure_sequence, DeviceType, StatusCallback

try:
    import picasso_model
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
    handles_insertions: bool = False
    handles_deletions: bool = True

    def __init__(
        self,
        model_file_path: str | PathLike,
        keep_model_loaded: bool = False,
        device: DeviceType = "cpu",
    ):
        super().__init__()
        self.model_file_path = model_file_path
        self.keep_model_loaded = keep_model_loaded
        self.device = device

        # lazy-load model when needed
        self.model = None

        # TODO: store encoder params
        # TODO: store decoder params

        # encodings created when calling build() method
        self.encoding = None

    @classmethod
    def can_model(cls, system: EntityOrEntityList) -> Tuple[bool, str]:
        system = ensure_sequence(system)
        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        if system[0].sequences is None:
            return False, "Must provide sequences for model inference"

        # TODO: check that sequences are aligned

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: EntityOrEntityList,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        # TODO: implement meaningful requirements instead of defaults
        return RequiredResources(
            min_gpu_cores=1,
            min_gpu_memory_per_core=16000,
            min_cpu_cores=1,
            min_cpu_memory_per_core=16000,
            time=1,
        )

    def _load_model(self):
        device = "cpu"  # TODO: how to best set this?
        # TODO: use load_from_checkpoint map_location argument instead of .to()?
        m = picasso_model.model.Model.load_from_checkpoint(
            self.model_file_path
        ).to(device)

        # switch to evaluation mode
        m.eval()

        return m

    def build(
        self,
        system: EntityOrEntityList,
        status_callback: StatusCallback | None = None
    ) -> Self:
        print("building...")
        # TODO: verify if we can actually model the system
        print(self.can_model(system))

        # TODO: keep model or not depending on setting
        return self

    def score(self):
        return 123.

    def score_single(self):
        return 124.

    def generate(self) -> None:
        return "LALALA"
