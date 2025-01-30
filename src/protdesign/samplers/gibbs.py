"""
Sequence generation with Gibbs sampling.

Implementation assumes fixed length of sequences (no inserts, deletions can be sampled if part of alphabet).
"""
from typing import Sequence, Literal, Callable, Tuple
import numpy as np
import pandas as pd
import torch
from protdesign.constants import VALID_AA_OR_GAP_SORTED, VALID_AA_SORTED, GAP
from protdesign.model import Generator, BaseModelAndScorer
from protdesign.entity import SystemInstance, EntityPosList, EntityInstance
from protdesign.types import StatusCallback, EntityType
from protdesign.utils import status_progress, ensure_sequence

ScanOrder = Literal[
    "random", "sequential"
]

InitStrategy = Literal[
    "random", "system"
]

# maps from initial temperature, current sweep and total number of sweeps to current temperature for sweep
TemperatureSchedule = Callable[
    [
        float,  # initial temperature (via generate() parameter)
        int,  # current sweep
        int,  # total number of sweeps
        int,  # current step
        int,  # total number of steps per weep
    ],
    float   # current temperature
]

_ENTITY = "entity"
_POS = "pos"


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
        scan_order: ScanOrder = "random",
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

        if scan_order not in ["random", "sequential"]:
            raise ValueError("Invalid scan order")

        if init_strategy not in ["random", "system"]:
            raise ValueError("invalid initialization strategy")

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
    ) -> Tuple[np.ndarray, dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
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

        # prepare auxiliary mappings around array
        # (e.g. designed entities [1,3] -> [0, 1] in design matrix)
        entity_to_array_idx = {
            entity_idx: array_idx for array_idx, entity_idx in enumerate(entities)
        }

        # mapping from entity index to length of each entity (use to slice
        # design matrix) - both designed and fixed positions
        entity_to_len = {
            entity_idx: len(self._system[entity_idx].rep) for entity_idx in entities
        }

        # array-based maps for fancy indexing (populated further down)
        entity_to_array_idx_linear = np.zeros((max(entities) + 1), dtype="int")
        entity_to_first_index_linear = np.zeros((max(entities) + 1), dtype="int")

        # initialize empty design matrix for number to num_designs x designed_entity x max_num_positions
        # (longest designed entity determines size of array in last dimension)
        samples = np.empty(
            (num_designs, len(entities), max(entity_to_len.values())),
            dtype="<U1"
        )

        alphabet_set = set(alphabet)
        for array_idx, entity_idx in enumerate(entities):
            entity = self._system[entity_idx]

            # initialize array-based mappings
            entity_to_array_idx_linear[entity_idx] = array_idx
            entity_to_first_index_linear[entity_idx] = entity.first_index

            # randomize full sequence across all chains
            seq_len = len(entity.rep)

            # initialize relevant slice of array for each entity across all chains/samples
            samples[
                :, array_idx, :seq_len
            ] = rng.choice(
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
                    samples[:, array_idx, pos - entity.first_index] = symbol

        return samples, entity_to_array_idx, entity_to_len, entity_to_array_idx_linear, entity_to_first_index_linear

    @classmethod
    def _init_scan_order(
        cls,
        num_designs: int,
        pos_to_design: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Initialize sequential scan order

        Parameters
        ----------
        num_designs
            Cf. generate()
        pos_to_design
            List of positions that are sampled (determines length of sweep)

        Returns
        -------
        2D array with sweep indices along columns (each chain has its own row)
        """
        # number of positions to design defines length of one sweep
        num_pos = len(pos_to_design)

        # turn into numpy array of tuples and repeat once per chain
        pos_array = np.array(
            pos_to_design, dtype=[(_ENTITY, "int"), (_POS, "int")]
        )

        sequential_order = np.tile(
            pos_array, num_designs
        ).reshape(
            num_designs, num_pos
        )

        return sequential_order

    @classmethod
    def _verify_and_update_scores(
        cls,
        scores: pd.DataFrame,
        deletions: bool,
        scorer_idx: int,
        alphabet: Sequence[str],
        num_designs: int,
    ):
        assert len(scores) == num_designs, "Invalid length of scoring dataframe"

        if deletions:
            if GAP not in scores.columns:
                raise ValueError(
                    f"Scorer {scorer_idx} did not provide values for gap, but deletions=True"
                )
        else:
            if GAP in scores.columns:
                scores = scores.drop([GAP], axis=1)

        returned_alphabet = "".join(scores.columns)
        expected_alphabet = "".join(alphabet)

        assert (
            returned_alphabet == expected_alphabet
        ), f"Invalid alphabet returned by scorer {scorer_idx} ({returned_alphabet} vs {expected_alphabet}"

        return scores

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

        # auxiliary variables for fancy indexing into design array
        alphabet_array = np.array(alphabet)
        design_idx_all = np.arange(num_designs)

        # initialize samples for all designed chains, we represent these as numpy arrays
        # internally since we assume entity representations that all have the same length;
        # we will assemble these into strings for passing into individual scorers
        (
            samples, entity_to_array_idx, entity_to_len,
            entity_to_array_idx_linear, entity_to_first_index_linear
        ) = self._init_samples(
            num_designs, entities, alphabet, pos_to_design
        )

        # initialize full instances to pass to scorers;
        # numpy view speeds up instance creation by ~10x for large sample sets
        # compared to iteration and string joining
        # TODO: move to its own method to reuse below
        # TODO: make this a build_or_update method?
        samples_joined = {
            entity_idx: samples[
                :, array_idx, :entity_to_len[entity_idx]
            ].view(
                f"<U{entity_to_len[entity_idx]}"
            )[:, 0]
            for entity_idx, array_idx in entity_to_array_idx.items()
        }

        # build molecular system instance list; will update in-place with
        # new sequences after each Gibbs step
        instances = [
            SystemInstance([
                EntityInstance(
                    rep=(
                        samples_joined[entity_idx][design_idx]
                        if entity_idx in entities
                        else self._system[entity_idx].rep
                    )
                ) for entity_idx, entity in enumerate(self._system)
            ]) for design_idx in range(num_designs)
        ]

        # initialize sequential scan order for all positions to be designed (will be reshuffled
        # per sweep in case random scan order is chosen)
        order = self._init_scan_order(num_designs, pos_to_design)
        rng = np.random.default_rng()

        # iterate through sweeps (sweep = one full scan of all designed positions)
        for sweep in range(self.num_sweeps):
            # update status (fraction of sweeps completed)
            status_progress(status_callback, sweep / self.num_sweeps)

            # number of steps per sweep is the number of positions we want to design
            num_steps = len(pos_to_design)

            # permute the current sweep scan order if using random order
            # (we always sample without replacement for now for simplicity);
            # note that rng.shuffle is not applicable here since all chains
            # would be shuffled in same way
            if self.scan_order == "random":
                order = rng.permuted(order, axis=1)

            assert (
                order.shape[0] == num_designs and order.shape[1] == num_steps
            ), "Scan order array has wrong shape"

            # iterate through all steps for current sweep
            for step in range(num_steps):
                # determine temperature for current sweep/step if we have a temperature schedule in place
                if self.temperature_schedule is not None:
                    step_temp = self.temperature_schedule(
                        temperature, sweep, self.num_sweeps, step, num_steps
                    )
                else:
                    step_temp = temperature

                # extract entity and position to sample for each chain in current step as flat arrays
                step_ent = order[_ENTITY][:, step]
                step_pos = order[_POS][:, step]
                assert len(step_ent) == len(step_pos) == num_designs

                # apply all scorers to current instances and compute weighted sum of scores;
                # we could decouple GPU and CPU-based computations here with multiprocessing
                # eventually to increase speed
                agg_scores = None
                for scorer_idx, (scorer, weight) in enumerate(zip(self.scorers, self.weights)):
                    # compute weighted score
                    s = scorer.score_conditional(
                        instances, step_ent, step_pos
                    ) * weight

                    # verify conditional score dataframe, and remove gap if present but not sampling deletions
                    s = self._verify_and_update_scores(
                        s, deletions=deletions, scorer_idx=scorer_idx, alphabet=alphabet, num_designs=num_designs
                    )

                    if agg_scores is None:
                        agg_scores = s
                    else:
                        agg_scores = agg_scores.add(s, axis=0)

                    assert (
                        len(agg_scores) == num_designs
                    ), f"Invalid length of aggregated scoring matrix after scorer {scorer_idx}"

                # Gibbs step

                # replace any missing values to exclude from sampling, and scale by temperature for current step;
                # Note we are using an inverted scale here (e.g. not -E/T but E/T where higher E means "better");
                # we go through pytorch here to use the parallelized multinomial implementation which is much
                # more suitable here
                scores_scaled = torch.from_numpy(
                    agg_scores.replace(np.nan, np.NINF).values
                ) / step_temp

                p = scores_scaled.softmax(dim=-1)

                sampled_token_idx = torch.multinomial(
                    p, num_samples=1
                ).flatten().numpy()

                sampled_tokens = alphabet_array[sampled_token_idx]

                # update sample matrix and instances for next step
                assert len(design_idx_all) == len(step_ent) == len(step_pos) == len(sampled_tokens)

                print(step_ent, step_pos, sampled_tokens)  # TODO: remove
                # print("first index", entity_to_array_idx_linear[step_ent])  # TODO: remove
                print("first index", step_pos - entity_to_first_index_linear[step_ent])  # TODO: remove

                # TODO: remove
                for _i, (_ent, _pos) in enumerate(zip(step_ent, step_pos)):
                    print(_i, _ent, _pos, "->", samples[_i, _ent, _pos - self._system[0].first_index])
                samples_before = samples.copy()
                # TODO: remove

                samples[
                    design_idx_all,
                    entity_to_array_idx_linear[step_ent],
                    step_pos - entity_to_first_index_linear[step_ent],
                ] = sampled_tokens

                # TODO: remove
                _diff = (samples_before != samples).sum()
                print("DIFF COUNT", _diff)
                for _i, (_ent, _pos) in enumerate(zip(step_ent, step_pos)):
                    print(_i, _ent, _pos, "->", samples[_i, _ent, _pos - self._system[0].first_index])
                # TODO: remove

                # TODO: verify index went through correctly
                # TODO: update entities based on new samples array state

                # TODO: also record and eventually attach chain information to metadata
                break  # TODO: remove

            if sweep >= 0:  # TODO: remove
                print("BREAK", sweep)
                break  # TODO: remove

        return instances
