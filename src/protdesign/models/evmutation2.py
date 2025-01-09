"""
Wrapper class around EVmutation2/picasso model
"""
from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import EntityOrEntityList, PROTEIN
from protdesign.utils import ensure_sequence
from typing import Protocol, Self, Tuple

try:
    import picasso_model
    import_available = True
except ImportError:
    import_available = False


class EVmutation2(BaseModel, Scorer, Generator):
    available = import_available

    def __init__(
        self,
        model_file_path: str,
    ):
        # TODO: call super constructor?
        # TODO: where to specify device?
        """
        """
        self.model_file_path = model_file_path

        # lazy-load model when needed
        self.model = None

        # TODO: device?
        # TODO: encoder params
        # TODO: decoder params

        # encodings created when calling build() method
        self.encoding = None

    @classmethod
    def can_model(cls, system: EntityOrEntityList) -> Tuple[bool, str]:
        system = ensure_sequence(system)
        if len(system) != 1 or system[0].entity_type != PROTEIN:
            return False, "Can only handle single-component protein system"

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

    def build(self, system: EntityOrEntityList) -> Self:
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
