from typing import Any, Self, Sequence
import numpy as np

from evedesign.model import BaseModel, Transformer
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import BioPolymers, StatusCallback


class OneHot(BaseModel, Transformer):
    """
    Model wrapper that transforms biopolymer sequences into per-residue one-hot embeddings.

    Each biopolymer entity is encoded against its own type's alphabet:
      - protein: 20 canonical amino acids + gap (21 columns)
      - dna/rna: canonical nucleotides (4 columns)

    The resulting embedding is a 2D array of shape [len(rep), len(alphabet)] stored on
    EntityInstance.embedding. Class order matches Entity.alphabet() for the entity type.

    Any symbol in a representation that is not part of the corresponding entity's alphabet raises a ValueError.

    Note: gaps are encodable for proteins -> (hence handles_deletions is True)
    """
    name: str = "OneHot"
    citations: list[str] = []

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(self):
        self._system = None
        # per-entity-index alphabet and symbol to column lookup--constructed in build()
        self._alphabets: dict[int, list[str]] = {}
        self._symbol_to_index: dict[int, dict[str, int]] = {}

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @staticmethod
    def _entity_alphabet(entity: Entity) -> list[str]:
        """
        proteins include the gap symbol (21 symbols),
        nucleotide entities use the 4 canonicals.
        """
        return entity.alphabet(
            include_gap=(entity.type == "protein"), include_inserts=False
        )

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not take data"

        if len(system) == 0:
            return False, "System must contain at least one entity"

        if not any(entity.type in BioPolymers for entity in system):
            return False, "System must contain at least one biopolymer entity to encode"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        # assign alphabet and index mapper
        self._alphabets = {
            entity_idx: self._entity_alphabet(entity)
            for entity_idx, entity in enumerate(system)
            if entity.type in BioPolymers
        }
        self._symbol_to_index = {
            entity_idx: {symbol: col for col, symbol in enumerate(alphabet)}
            for entity_idx, alphabet in self._alphabets.items()
        }

        return self

    def _one_hot(self, entity_idx: int, rep: np.ndarray) -> np.ndarray:
        """
        One-hot encode a entity representation into a [len(rep), len(alphabet)] one-hot array.
        """
        symbol_to_index = self._symbol_to_index[entity_idx]
        encoding = np.zeros((len(rep), len(symbol_to_index)), dtype=np.float32)

        for pos, symbol in enumerate(rep):
            symbol = str(symbol)
            col = symbol_to_index.get(symbol)
            if col is None:
                raise ValueError(
                    f"Symbol {symbol!r} at position {pos} of entity {entity_idx} is not part of its "
                    f"one-hot alphabet {self._alphabets[entity_idx]}"
                )
            encoding[pos, col] = 1.0

        return encoding

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Transform system instances by adding one-hot embeddings for their biopolymer entities.

        If entity is None, all biopolymer entities in the system are encoded; otherwise only the
        selected entity is encoded (must be a biopolymer).
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # determine entities to encode
        if entity is not None:
            if not 0 <= entity < len(self.system):
                raise ValueError(f"Invalid entity index: {entity}")
            if self.system[entity].type not in BioPolymers:
                raise ValueError(
                    f"Entity {entity} is of type {self.system[entity].type!r}, can only one-hot biopolymers"
                )
            target_entities = [entity]
        else:
            target_entities = sorted(self._alphabets)

        transformed_instances = []
        for inst_idx, instance in enumerate(instances):
            # shallow copy to avoid mutating input
            new_instance = instance.copy()

            for entity_idx in target_entities:
                entity_instance = new_instance[entity_idx]
                if entity_instance.rep is None:
                    raise ValueError(
                        f"Entity {entity_idx} of instance {inst_idx} has no rep to encode"
                    )
                entity_instance.embedding = self._one_hot(entity_idx, entity_instance.rep)

            transformed_instances.append(new_instance)

            # if people are curious
            if status_callback is not None:
                progress = ((inst_idx + 1) / len(instances)) * 500
                status_callback(
                    "running", progress, f"One-hot encoded instance {inst_idx + 1}/{len(instances)}"
                )

        return transformed_instances
