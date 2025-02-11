"""
Restraining generated sequence distance to reference sequences
"""
from typing import Tuple, Self, List, Sequence

import numpy as np
import pandas as pd

from protdesign.model import BaseModel, Scorer, RequiredResources, ConditionalMutationScorer
from protdesign.entity import System, SystemInstance
from protdesign.types import StatusCallback
from protdesign.utils import str_to_np_char_view, map_array

EntityToReferenceSeqs = dict[int, list[str]]


class LinearSeqDistRestraint(BaseModel, Scorer, ConditionalMutationScorer):
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
    handles_insertions: bool = False

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
        self._system = None

        # will hold mapped and verified reference sequences for comparison
        self._ref_seqs = None
        self._alphabets = None
        self._alphabet_mappings = None
        self._ref_seqs_mapped = None

        self.exclude_gaps_from_distance = exclude_gaps_from_distance

    @property
    def ready(self):
        return (
            self.system is not None and
            self._ref_seqs is not None and
            self._alphabets is not None and
            self._alphabet_mappings is not None and
            self._ref_seqs_mapped is not None
        )

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

        # store alphabets for each entity
        self._alphabets = {
            entity_idx: entity.alphabet(include_gap=True)
            for entity_idx, entity in enumerate(self._system)
            if entity.defined_sequence()
        }

        self._alphabet_mappings = {
            entity_idx: {symbol: idx for idx, symbol in enumerate(alphabet)}
            for entity_idx, alphabet in self._alphabets.items()
        }

        # map to numerical indices
        try:
            self._ref_seqs_mapped = {
                entity_idx: map_array(
                    entity_ref_seqs,
                    {symbol: idx for idx, symbol in enumerate(self._alphabets[entity_idx])}
                )
                for entity_idx, entity_ref_seqs in self._ref_seqs.items()
            }
        except KeyError as e:
            raise ValueError("Invalid symbol in reference sequences") from e

        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
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
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=True,
                raise_invalid=True
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
        # TODO: refactor this to a reusable method, and also in Gibbs sampler
        unique_entities = sorted(set(entities))
        entity_types = {
            self._system[entity_idx].type_ for entity_idx in unique_entities
        }
        if len(entity_types) != 1:
            raise ValueError("For now, can only score entities of one type")

        # get alphabet for this one entity type
        alphabet = self._alphabets[unique_entities[0]]

        # initialize table of instance/entity/pos triplets and add current
        # instance symbol for later comparison to restraint sequences
        entity_to_first_index = {
            entity_idx: entity.first_index for entity_idx, entity in enumerate(self._system)
        }

        # prepare empty scoring matrix
        res = pd.DataFrame({
            "instance": np.arange(len(instances)),
            "entity": entities,
            "pos": positions,
        }).set_index(
            ["instance", "entity", "pos"]
        ).reindex(
            alphabet, axis=1, fill_value=0.0
        )

        # determine instance symbol for each row, this allows to reuse the scores for single_mutation_scan()
        inst_symbol = np.array([
            instance[entity_idx].rep[
                pos - entity_to_first_index[entity_idx]
                ]
            for (instance, entity_idx, pos) in zip(instances, entities, positions)
        ])
        inst_symbol_idx = map_array(
            inst_symbol, self._alphabet_mappings[unique_entities[0]]
        )

        # compare sequences entity by entity and accumulate updated subgroup dataframes
        groups = res.groupby("entity", sort=False)

        for entity_idx, all_row_idx in groups.indices.items():
            entity_idx = int(entity_idx)  # noqa

            # keep neutral scores to positions in entities that are restrained
            if entity_idx not in self._ref_seqs:
                continue

            # map requested position for each instance
            cur_positions = (
                res.iloc[all_row_idx].index.get_level_values("pos").values - entity_to_first_index[entity_idx]
            )

            # compare to all reference sequences for current entity
            # (use version mapped to indices for direct fancy indexing into numpy array)
            cur_ref_seqs = self._ref_seqs_mapped[entity_idx]

            # iterate through individual reference sequences
            # TODO: may need to make this more efficient for larger sets of restraint sequences
            #  (e.g. comparing against entire MSA)
            for i in range(len(cur_ref_seqs)):
                # extract symbols at different positions in this reference sequence
                ref_symbols = cur_ref_seqs[i, cur_positions]

                # update in place
                res.values[all_row_idx, ref_symbols] += 1

                # treat gap special case
                if self.exclude_gaps_from_distance:
                    # TODO: implement
                    pass

        # retrieve value for instance symbol across all rows, then subtract from full matrix to normalize
        inst_symbol_val = res.values[np.arange(len(res)), inst_symbol_idx]
        res.values[:, :] -= inst_symbol_val[:, None]

        assert len(res) == len(instances)
        return res

    """
    # TODO: implement MutationScorer interface if it would be useful in practice 
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
    """
