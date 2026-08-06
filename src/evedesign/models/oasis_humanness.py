"""OASis-compatible antibody humanness scoring via :mod:`promb`."""

from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any, Self

from evedesign.model import BaseModel, MutationScorer, Scorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

try:
    from promb import init_db  # type: ignore[import-untyped]

    IMPORT_AVAILABLE = True
except ImportError:
    init_db = None
    IMPORT_AVAILABLE = False


Aggregation = str | Callable[[Sequence[float]], float]
AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": fmean,
    "min": min,
    "max": max,
}


class OASisHumanness(BaseModel, Scorer, MutationScorer):
    """Score proteins by the fraction of 9-mers found in human OAS.

    Scores range from 0.0 to 1.0, with higher values indicating greater
    humanness. This uses promb's fixed peptide-content definition, not BioPhi's
    configurable prevalence thresholds.
    """

    available = IMPORT_AVAILABLE
    name: str = "OASisHumanness"
    citations: list[str] = ["doi:10.1080/19420862.2021.2020203"]

    requires_target: bool = True
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_cpu_parallel: bool = False
    supports_gpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        aggregation: Aggregation = "mean",
        entities: Sequence[int] | None = None,
    ):
        if not self.available:
            raise ImportError(
                "promb could not be imported. Install evedesign with the "
                "'promb' optional dependency."
            )

        if isinstance(aggregation, str):
            valid_aggregation = aggregation in AGGREGATIONS
        else:
            valid_aggregation = callable(aggregation)
        if not valid_aggregation:
            raise ValueError("aggregation must be 'mean', 'min', 'max', or a callable")

        self.aggregation = aggregation
        self.entities = None if entities is None else tuple(entities)
        self._system: System | None = None
        self._selected_entities: tuple[int, ...] | None = None
        self._db: Any = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if not system:
            return False, "System must contain at least one protein entity"

        for entity in system:
            if entity.type != "protein":
                return False, "Can only handle protein entities"
            if not entity.defined_sequence():
                return False, "All entities must have defined rep sequences"
            rep = entity.rep
            if rep is None or len(rep) < 9:
                return False, "Protein sequences must contain at least 9 residues"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        selected_entities = (
            tuple(range(len(system)))
            if self.entities is None
            else self.entities
        )
        if not selected_entities:
            raise ValueError("entities must select at least one entity")
        if len(set(selected_entities)) != len(selected_entities):
            raise ValueError("entities must not contain duplicate indices")
        if any(
            not isinstance(entity, int) or not 0 <= entity < len(system)
            for entity in selected_entities
        ):
            raise ValueError("entities must contain valid system entity indices")

        self._system = system
        self._selected_entities = selected_entities
        return self

    def _load_db(self) -> None:
        if self._db is None:
            self._db = init_db("human-oas", verbose=False)

    def _aggregate(self, chain_scores: Sequence[float]) -> float:
        if callable(self.aggregation):
            return float(self.aggregation(chain_scores))

        return float(AGGREGATIONS[self.aggregation](chain_scores))

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        self.ready_or_raise()
        self._validate_instances(instances)

        if not instances:
            return []

        assert self._selected_entities is not None
        sequences = []
        for instance in instances:
            instance_sequences = []
            for entity in self._selected_entities:
                rep = instance[entity].rep
                if rep is None or len(rep) < 9:
                    raise ValueError("Protein sequences must contain at least 9 residues")
                instance_sequences.append("".join(rep))
            sequences.append(instance_sequences)

        status_start(status_callback, "Scoring human peptide content")

        self._load_db()

        scores = []
        for instance_sequences in sequences:
            chain_scores = [
                self._db.compute_peptide_content(sequence)
                for sequence in instance_sequences
            ]
            scores.append(self._aggregate(chain_scores))

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, scores)
