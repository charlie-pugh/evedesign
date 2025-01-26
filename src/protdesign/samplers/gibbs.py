"""
Sequence generation with Gibbs sampling.

Implementation assumes fixed length of sequences (no inserts, deletions can be sampled if part of alphabet).
"""
from random import choices, choice
from typing import Sequence, Literal, Callable, Tuple
from loguru import logger
import numpy as np
from protdesign.constants import VALID_AA_OR_GAP_SORTED, VALID_AA_SORTED
from protdesign.model import Generator, BaseModelAndScorer
from protdesign.entity import SystemInstance, EntityPosList, EntityInstance, Entity
from protdesign.types import StatusCallback, EntityType
from protdesign.utils import status_progress, ensure_sequence

ScanOrder = Literal[
    "random_without_replacement", "random_with_replacement", "sequential"
]
InitStrategy = Literal["random", "system"]

# maps from initial temperature, current step and total number of steps to current temperature for step
TemperatureSchedule = Callable[[float, int, int], float]


class GibbsSampler(Generator):
    """
    Gibbs sampling from linear combination of Scorers

    # TODO: energy sign convention

    Notes and design choices:
    1. This sampler does not parallelize individual chains, as this does not play nicely with
     parallelized GPU-based computations (better to batch individual steps), and as this precludes
     interactions between the different chains right away (e.g. library diversity constraints)
     At this point, parallelization / device choice is entirely up to individual scorers so
     each scorer can optimize individually for its bottlenecks (e.g. number of GPUs, available CPUs,
     memory, ...), and to keep the sampler implementation as lean as possible.

     TODO: may decouple/parallelize GPU-based and CPU-based computations with multiprocessing,
      so the CPU-based computations happen in parallel to heavy GPU-based computations

    2. Current implementation assumes for simplicity that all chains have the same length;
     this requirement could be relaxed eventually to each chain can have its own length (however fixed
     along chain!)

    3. Current implementation can only sample entities of same type (protein or nucleotide entities only,
     but not combinations of types, e.g. design protein and nucleotide entities simultaneously)
    """
    def __init__(
        self,
        scorers: Sequence[BaseModelAndScorer],
        weights: Sequence[float] | None = None,
        num_sweeps: int = 1000,
        init_strategy: InitStrategy = "random",
        scan_order: ScanOrder = "random_without_replacement",
        temperature_schedule: TemperatureSchedule | None = None,
        require_strict_pos : bool = True,
    ):
        """
        Create new Gibbs sampler

        Parameters
        ----------
        scorers
            Scores to combine into joint score for optimization
        weights
            Weight each scorer will be multiplied with (weight_1 * score_1 + ... + weight_n * score_n).
            If specified, needs to have same length as scorers parameter. If None, all weights will be set to 1.0.
            Use negative weights to invert the semantics of a scorer (e.g. to design against or in favor
            of the occurrence of a certain sequence motif)
        num_sweeps
            Number of Gibbs sweeps across entire system (number of total sampling steps will be
            number of sites x number of sweeps)
        init_strategy
            Create starting samples by randomly sampling designed positions from available alphabet ("random"),
            or use the representations associated with the entities in the system ("system"). For fixed positions,
            will always use representation from system.
        scan_order
            Strategy to determine scan order for each chain. Will either sample randomly (with or without replacement)
            from intersection of positions available from all scorers, or iterate through these sequentially
            as specified by scorers.
        temperature_schedule
            Function mapping from starting temperature, current step and num_steps to temperature for the
            current step. Set to None for constant temperature (specified in generate() function call).
        require_strict_pos
            If True, verify that all scorers model the same set of positions in the system or raise
            a ValueError
        """
        # must have at least one scorer
        if len(scorers) == 0:
            raise ValueError("Must provide at least one scorer")

        # assume all weights to be 1.0 if weights is None, otherwise check number of weights matches scorers
        if weights is None:
            weights = np.ones(len(scorers))

        if len(scorers) != len(weights):
            raise ValueError("Number of scorers must match number of weights")

        # verify all scorers and store available positions
        for i, scorer in enumerate(scorers):
            if not scorer.ready:
                raise ValueError(
                    f"Scorer {i} is not yet ready, call build() first before passing into sampler"
                )

            if scorer.system is None:
                raise ValueError(
                    f"Scorer {i} does not have an associated system"
                )

            if scorer.system != scorers[0].system:
                raise ValueError(
                    f"Scorer {i} system is not equal to first system (all systems must be identical across scorers)"
                )

        # make a copy of system for easier access
        self._system = scorers[0].system

        self.scorers = scorers
        self.weights = weights
        self.num_sweeps = num_sweeps
        self.temperature_schedule = temperature_schedule
        self.scan_order = scan_order
        self.init_strategy = init_strategy

        # store available positions for each scorer
        self.scorer_to_pos = {
            idx: scorer.positions() for idx, scorer in enumerate(self.scorers)
        }

        # determine shared positions by intersection, will only be able to design those
        self.shared_pos = set.intersection(
            *(set(v) for v in self.scorer_to_pos.values())
        )

        # require at least one position to model
        if len(self.shared_pos) == 0:
            raise ValueError(
                "No shared positions between scorers, will not be able to sample from system. " +
                f"Positions per scorer: {self.scorer_to_pos}"
            )

        # if strict position requirement is enabled, check that all scorers model the same positions
        if require_strict_pos:
            all_pos = set.union(
                *(set(v) for v in self.scorer_to_pos.values())
            )

            if len(all_pos) != len(self.shared_pos):
                raise ValueError(
                    "Inconsistent position lists between scorers"
                )

    def _design_params(
        self,
        entities: Sequence[int] | None,
        fixed_pos: EntityPosList | None,
        deletions: bool,
    ) -> Tuple[list[int], EntityType, list[str], list[Tuple[int, int]]]:
        """
        Helper method to verify specified entities and fixed positions, and compute
        list of positions in entities that are used for design

        Parameters
        ----------
        entities
            Cf generate() method documentation
        fixed_pos
            Cf generate() method documentation
        deletions
            Cf generate() method documentation

        Returns
        -------
        Entity type and list of (entity_idx, position_in_entity) tuples that
        are selected for design
        """
        entities_to_type = {
            idx: entity.type_ for idx, entity in enumerate(self._system)
        }

        # determine and verify entities and positions to design
        if entities is not None:
            entities = ensure_sequence(entities)
            if (set(entities) & set(entities_to_type)) != set(entities):
                raise ValueError(
                    f"Invalid entity selection {entities}, available entities are {list(entities_to_type)}"
                )
        else:
            # otherwise, use all entities
            entities = sorted(entities_to_type)

        # ensure all designed entities are of same type to simplify sampling step
        # (can eventually relax this requirement)
        designed_types = {
            entities_to_type[entity] for entity in entities
        }
        if len(designed_types) != 1:
            raise ValueError("All designed entities must be of same type")

        designed_type = list(designed_types)[0]

        # verify all designed entities have an existing representation (so we can assume length)
        for entity in entities:
            if self._system[entity].rep is None or len(self._system[entity].rep) == 0:
                raise ValueError(
                    "All designed entities must have a specified representation with nonzero length"
                )

        # verify fixed position specification
        if fixed_pos is not None:
            # verify fixed positions
            if set(fixed_pos) & set(entities) != set(fixed_pos):
                raise ValueError(
                    "Entities specified in fixed_pos must be included in entities to design"
                )

            # turn into flat tuple representation
            fixed_pos = set([
                (entity_idx, pos) for entity_idx, pos_list in fixed_pos.items() for pos in pos_list
            ])

            # verify fixed positions are all available in joint model used for scoring
            invalid_fixed_pos = fixed_pos - self.shared_pos

            if len(invalid_fixed_pos) > 0:
                raise ValueError(f"Invalid fixed positions not available for sampling detected: {invalid_fixed_pos}")
        else:
            fixed_pos = set()

        # remove fixed positions from all available positions, we need at least one to design
        design_pos = sorted(
            entity_pos for entity_pos in self.shared_pos if entity_pos not in fixed_pos
        )

        if len(design_pos) == 0:
            raise ValueError("No positions left to design after removing fixed positions")

        # set up alphabet based on designed_type
        if designed_type == "protein":
            alphabet = VALID_AA_OR_GAP_SORTED if deletions else VALID_AA_SORTED
        else:
            raise NotImplementedError(
                f"Entity type '{designed_type}' not yet supported"
            )

        return entities, designed_type, alphabet, design_pos

    def _init_samples(
        self,
        num_designs: int,
        entities: list[int],
        alphabet: list[str],
        pos_to_design: list[Tuple[int, int]],
    ) -> dict[int, np.ndarray]:
        """
        Initialize samples based on system and random sampling

        Parameters
        ----------
        num_designs
            Number of initialized samples to build
        entities
            Indices of designed entities
        alphabet
            Characters to initialize samples from
        pos_to_design
            Variable positions that should be initialized

        Returns
        -------
        Mapping from entity index to samples
        """
        rng = np.random.default_rng()

        alphabet_set = set(alphabet)
        samples = {}
        for entity_idx in entities:
            entity = self._system[entity_idx]
            # randomize full sequence across all chains
            seq_len = len(entity.rep)
            x = rng.choice(
                alphabet, size=(num_designs, seq_len), replace=True
            )

            # set fixed positions based on system representation
            for pos, symbol in enumerate(entity.rep, start=entity.first_index):
                if symbol not in alphabet_set:
                    raise ValueError(
                        "Fixed position in system representation is not part of alphabet" +
                        f" (entity: {entity_idx}, pos: {pos}, symbol: {symbol}, valid alphabet: {alphabet})"
                    )

                # set to fixed symbol
                if (entity_idx, pos) not in pos_to_design:
                    x[:, pos - entity.first_index] = symbol

            samples[entity_idx] = x

        return samples

    def _scan_order(
        self,
        num_designs: int,
        pos_to_design: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Determine/sample scan order for current sweep

        Parameters
        ----------
        num_designs
            Cf. generate()
        pos_to_design
            List of positions that are sampled (determines length of sweep)

        Returns
        -------
        Matrix with sweep indices along columns (each chain has its own row)
        """
        # number of positions to design defines length of one sweep
        num_pos = len(pos_to_design)
        rng = np.random.default_rng()

        # turn into numpy array of tuples and repeat once per chain
        pos_array = np.array(pos_to_design, dtype="int, int")
        sequential_order = np.tile(
            pos_array, num_designs
        ).reshape(
            num_designs, num_pos
        )

        if self.scan_order == "sequential":
            # keep order as is
            order = sequential_order
        elif self.scan_order == "random_with_replacement":
            # TODO
            order = None
        elif self.scan_order == "random_without_replacement":
            # shuffle independently per chain
            order = rng.permuted(sequential_order, axis=1)
        else:
            raise ValueError(f"Invalid scan order {self.scan_order}")

        # TODO: profile this
        print(order.shape)
        # print("ORDER", order)  # TODO: remove
        print(order[0])
        print(order[2])
        return order

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        # verify/update entity selection and extract positions to design
        entities, designed_type, alphabet, pos_to_design = self._design_params(
            entities, fixed_pos, deletions
        )

        # initialize samples for all designed chains, we represent these as numpy arrays
        # internally since we assume entity representations that all have the same length;
        # we will assemble these into strings for passing into individual scorers
        # TODO: or use joint matrix, is this easier to update in loop below
        samples = self._init_samples(
            num_designs, entities, alphabet, pos_to_design
        )

        # iterate through sweeps (sweep = one full scan of all designed positions)
        for sweep in range(self.num_sweeps):
            # update status (fraction of sweeps completed)
            status_progress(status_callback, sweep / self.num_sweeps)

            # determine scan order for all of the chains for current sweep (random or sequential);
            # only determine order for current sweep to not allocate a huge matrix that sits around unused
            order = self._scan_order(num_designs, pos_to_design)

            # compute temperature for current sweep based if schedule is defined, otherwise
            # use same temperature
            # TODO: also include step in current sweep in callback
            if self.temperature_schedule is not None:
                step_temperature = self.temperature_schedule(
                    temperature, sweep, self.num_sweeps
                )
            else:
                step_temperature = temperature

            print("sweep", sweep, "... T =", step_temperature)  # TODO: remove
            break

            # apply scorers to current instances
            # TODO: need to create current instances from matrices (later on update)...

            # TODO: how to best parallelize? Decouple GPU computation? Parallelize inside constraints?
            # TODO: handle gap via parameter during sampling

            # TODO: replace nan with -inf
            # TODO: is there a np.log_softmax?
            # TODO: sign convention for scores

            # choose update and apply
            # TODO
            #  conditional_probs = np.exp(-U / self.T)
            #  conditional_probs = conditional_probs / np.nansum(conditional_probs)
            #  s[k] = np.random.choice(np.arange(1, 21), p=conditional_probs)

            # TODO: log times per scorer if verbose

            if sweep > 10:
                break  # TODO: remove

        # # Calculate the conditional probabilities
        # conditional_probs = np.exp(-U / self.T)
        #

        # # And normalize them
        # conditional_probs = conditional_probs / np.nansum(conditional_probs)
        #
        # if type(self.pos_constraint) == type(None):
        #
        #     # Without a position constrant, just choose from 20 amino acids
        #     s[k] = np.random.choice(np.arange(1, 21), p=conditional_probs)
        #
        # else:
        #     # Or choose from the allowed amino acids at that position
        #     s[k] = np.random.choice(np.arange(1, 21)[self.pos_constraint[k]], p=conditional_probs)

        # TODO: think about suitable parallelization strategies... can we parallelize GPU and CPU calculation?
        # TODO: note that GPU will typically be the bottleneck that all samples need to pass through
        #  (except for CPU-only calculations)

        # select alphabet for initialization
        # if designed_type == "protein":
        #     alphabet = VALID_AA_OR_GAP if deletions else VALID_AA
        # elif designed_type == "dna":
        #     alphabet = VALID_DNA_OR_GAP if deletions else VALID_DNA
        # elif designed_type == "rna":
        #     alphabet = VALID_RNA_OR_GAP if deletions else VALID_RNA
        # else:
        #     raise NotImplementedError(
        #         f"Entity type '{designed_type}' not supported"
        #     )
        # alphabet = list(alphabet)

        # def _init_rep(entity_idx: int, entity: Entity) -> str:
        #     # create random sequence of same length as entity rep
        #     # (may not use all of it depending on which positions are designed and which are fixed)
        #     random_rep = choices(alphabet, k=len(entity.rep))
        #
        #     # TODO: conditional code if all positions are designed, none, or mixed?
        #     # TODO: this incurs major runtime cost
        #     # s = [
        #     #     (random_symbol if (entity_idx, pos) in pos_to_design else system_symbol)
        #     #     for pos, (system_symbol, random_symbol)
        #     #     in enumerate(
        #     #         zip(entity.rep, random_rep), start=entity.first_index
        #     #     )
        #     # ]
        #
        #     # s = [
        #     #     (choice(alphabet) if (entity_idx, pos) in pos_to_design else system_symbol)
        #     #     for pos, system_symbol
        #     #     in enumerate(
        #     #         entity.rep, start=entity.first_index
        #     #     )
        #     # ]
        #     return "".join(random_rep)
        #
        # samples = [
        #     SystemInstance([
        #         EntityInstance(
        #             _init_rep(entity_idx, entity)
        #         ) for entity_idx, entity in enumerate(self._system)
        #     ]) for _ in range(num_designs)
        # ]
        #
        # return samples
        # return []

        # TODO: attach results to instances
        # TODO: create final instances (with chain in metadata) and return
        return samples
