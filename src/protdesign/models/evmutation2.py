"""
Wrapper class around EVmutation2 model
"""
from protdesign.model_base import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import EntityOrEntitySequence, PROTEIN
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
        checkpoint_path: str,
    ):
        # TODO: call super constructor?
        # TODO: where to specify device?
        """
        m = model.Model.load_from_checkpoint(
            model_path
        ).to(device)

        # switch to evaluation mode
        m.eval()
        """
        pass

    @classmethod
    def can_model(cls, system: EntityOrEntitySequence) -> Tuple[bool, str]:
        system = ensure_sequence(system)
        if len(system) != 1 or system[0].entity_type != PROTEIN:
            return False, "Can only handle single-component protein system"

        return True, ""

    @classmethod
    def required_resources(cls, system: EntityOrEntitySequence) -> RequiredResources:
        return "hallo"

    def build(self) -> Self:
        print("building...")
        return self

    def score(self):
        return 123.

    def score_single(self):
        return 124.

    def generate(self) -> None:
        return "LALALA"
