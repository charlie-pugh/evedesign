"""
Restraining generated sequence distance to reference sequences
"""
from typing import Tuple, Self, List, Sequence

import numpy as np
import pandas as pd
from protdesign.model import BaseModel, Scorer, RequiredResources
from protdesign.entity import System, SystemInstance, Mutant
from protdesign.types import StatusCallback

EntityToReferenceSeqs = dict[int, list[str]]


class LinearSeqDistRestraint(BaseModel, Scorer):
    """
    Linear distance restraint between generated sequences and a set of reference sequences.
    For simplicity, assumes all compared sequences (i.e. on a per-entity basis) have the same
    length and are aligned.

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
        # core requirements: we need at least one constrained biopolymer sequence,
        # and length of all sequences per entity must agree with reference sequence

        # determine valid seq entities
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
                return False, f"Reference sequence do not have correct length of {cur_entity_length}"

        # TODO: check if all are valid biopolymer sequences

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

        # store reference sequences for comparison
        self._ref_seqs = data
        # TODO: map the data

        print("building", data)  # TODO: remove
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

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        # TODO: verify all instances have the same length
        raise NotImplementedError()

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        # TODO: verify all instances have the same length
        # TODO: sufficient to compute delta here for all mutants to each site (reference symbol = 0,
        #  do other symbols make sequence more similar or dissimilar to reference?)
        # TODO: need to return zero scores for non-restrained positions
        raise NotImplementedError()

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        # TODO: wrap around score_conditional
        raise NotImplementedError()

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        raise NotImplementedError()
