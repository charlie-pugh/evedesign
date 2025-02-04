"""
Restraining generated sequence distance to reference sequences
"""
from typing import Tuple, Self, List, Sequence

import numpy as np
import pandas as pd
from protdesign.model import BaseModel, Scorer, RequiredResources
from protdesign.entity import System, SystemInstance, Mutant
from protdesign.types import StatusCallback
from protdesign.utils import str_to_np_char_view
from pygments.lexer import include

EntityToReferenceSeqs = dict[int, list[str]]


class LinearSeqDistRestraint(BaseModel, Scorer):
    """
    Linear distance restraint between generated sequences and a set of reference sequences.
    For simplicity, assumes all compared sequences (i.e. on a per-entity basis) have the same
    length and are aligned.

    # TODO: not yet optimized for performance (when using large sequence sets, bring in numba)
    # TODO: constructor param for number of CPUs to use (when parallelizing)?

    Note on sign convention:
    Scoring methods return distance (or delta of distance) to reference sequences; i.e. a positive
    weight on this restraint during sampling will enforce sequences to become more dissimilar
    to the reference sequence(s); a negative weight will enforce designs to become more similar.
    """
    available = True
    name: str = "LinearSeqDistRestraint"

    requires_heavy_build: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True

    requires_seqs: bool = False
    requires_msa: bool = False
    requires_3d: bool = False

    def __init__(
        self,
        exclude_gaps_from_distance: bool = True,
    ):
        """
        Create new linear sequence distance restrain

        Parameters
        ----------
        exclude_gaps_from_distance
            If True, do not count positions where either of two compared sequences
            has a gap symbol
        """
        super().__init__()
        self._system = None

        # will hold mapped and verified reference sequences for comparison
        self._ref_seqs = None

        self.exclude_gaps_from_distance = exclude_gaps_from_distance

    @property
    def ready(self):
        return self.system is not None and self._ref_seqs is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(
        cls,
        system: System,
        data: EntityToReferenceSeqs,
    ) -> Tuple[bool, str]:
        # core requirements: we need at least one restrained biopolymer sequence,
        # and length of all sequences per entity must agree with reference sequence

        # determine valid sequence entities that could be restrained
        valid_entities_to_len = {
            entity_idx: len(entity.rep) for entity_idx, entity in enumerate(system) if entity.defined_sequence()
        }

        if len(data) == 0:
            return False, "Must specify at least one entity with reference sequences"

        # iterate through all specified reference sequences on each of the entities and verify
        for entity_idx, ref_seqs in data.items():
            if entity_idx not in valid_entities_to_len:
                return False, (
                    f"Restraint specified on entity {entity_idx} but valid "
                    f"entities with defined representation are {list(valid_entities_to_len.keys())}"
                )

            cur_entity_length = valid_entities_to_len[entity_idx]
            invalid = [seq for seq in ref_seqs if len(seq) != cur_entity_length]
            if len(invalid) > 0:
                return False, f"Reference sequence(s) do not have correct length of {cur_entity_length}: {invalid}"

        # TODO: check if all given sequences are valid or simply match on character level for more flexibility?

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
        data: EntityToReferenceSeqs,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        # TODO need to implement taking into account size of sequences to compare to
        # TODO: also depends on number of instances? or batch?
        raise NotImplementedError()

    def build(
        self,
        system: System,
        data: EntityToReferenceSeqs,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # store system with this instance
        self._system = system

        # store reference sequences for comparison (already checked validity via can_model() above)
        self._ref_seqs = {
            entity_idx: str_to_np_char_view(
                entity_ref_seqs
            ) for entity_idx, entity_ref_seqs in data.items()
        }

        return self

    def positions(
        self
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()

        # restraint is able to model all positions in biopolymer sequence
        # (even if not constrained, we return all of them and score as neutral if mutated)
        return [
            (entity_idx, pos)
            for entity_idx, entity in enumerate(self._system)
            if entity.defined_sequence()
            for pos, _ in enumerate(entity.rep, start=entity.first_index)
        ]

    def _validate_instances(
        self,
        instances: Sequence[SystemInstance],
    ) -> None:
        # validate instance sequences; must all have the same length
        [
            self.system.valid_instance(
                instance, fixed_length=True, validate_reps=True, raise_invalid=True
            ) for instance in instances
        ]

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # validate instance sequences with specific requirements for this class
        self._validate_instances(instances)

        # for accumulating distances across all entities and instances
        dists = np.zeros(len(instances), dtype="int")

        # loop through target entities
        for entity_idx, cur_ref_seqs in self._ref_seqs.items():
            # extract sequences for current entity from instances as numpy array
            # (do not use np.array(list) as this is way slower)
            x = str_to_np_char_view(
                [inst[entity_idx].rep for inst in instances]
            )

            # iterate through references one by one;
            # TODO: optimize with numba or scipy cdist if large reference sequence sets
            #  (e.g. comparing to MSA) become relevant
            for ref in cur_ref_seqs[:]:
                # silence type warnings by wrapping in array()
                diff = np.array(ref != x)

                if self.exclude_gaps_from_distance:
                    diff = diff & (ref != "-") & (x != "-")

                dists += diff.sum(axis=1)

        assert len(dists) == len(instances)
        return dists

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        self.ready_or_raise()

        if not len(instances) == len(entities) == len(positions):
            raise ValueError("Sequences for instances, entities and positions must all have same length")

        # validate instance sequences with specific requirements for this class
        self._validate_instances(instances)

        # validate entities / positions
        self.valid_positions(
            positions=positions, entities=entities, raise_invalid=True
        )

        # only allow entities of same type to be scored at same time for now
        unique_entities = sorted(set(entities))
        entity_types = {
            self._system[entity_idx].type_ for entity_idx in unique_entities
        }
        if len(entity_types) != 1:
            raise ValueError("For now, can only score entities of one type")

        # get alphabet for this one entity type
        alphabet = self._system[unique_entities[0]].alphabet(include_gap=True)

        # compute distances entity by entity for simplicity
        res = pd.DataFrame({
            "instance": np.arange(len(instances)),
            "entity": entities,
            "pos": positions,
        })

        # accumulate current symbol at each position and whether it is affected by restraint
        # TODO: need to loop through different reference sequences
        # TODO: compare to all possible other symbols

        print(alphabet)  # TODO: Remove

        # TODO: sufficient to compute delta here for all mutants to each site (reference symbol = 0,
        #  do other symbols make sequence more similar or dissimilar to reference?)

        # TODO: need to return zero scores for non-restrained positions
        # TODO: assert length of dataframe

        assert len(res) == len(instances)
        return res

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        raise NotImplementedError()
