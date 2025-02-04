from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol, List, Self, Tuple, Sequence, Any
import numpy as np
import pandas as pd
from protdesign.entity import System, SystemInstance, EntityPosList, Mutant
from protdesign.types import StatusCallback


class Scorer(Protocol):
    """
    Interface implemented by classes that can score
    (e.g. density/log likelihood) for existing designs/sequences
    (scalar value per design/sequence)

    Please refer to comments on each function on the excepted semantics and format
    of the returned scores.
    """
    @abstractmethod
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        """
        Score different realizations of the modelled system (e.g. different sequences
        generated from a model)

        Note:
        1. Scores returned by this function should be raw logits comparable between all instances
         scored in the same call. Scores between multiple calls do not have to be comparable (user
         is responsible for including a reference instance for normalization in these cases)

        Parameters
        ----------
        instances
            Designs to score with model
        status_callback
            Callback function to track computation status

        Returns
        -------
        Vector of scores (one per instance, in same order as instances input parameter)
        """
        pass

    @abstractmethod
    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Compute scores for all substitutions in a single position
        across a batch of sequences (single position can differ between instances), e.g.
        for Gibbs sampling-based generation of multiple designs in parallel.

        Note:
        1. This function allows to exploit the fact that often single mutations for
         one position can be computed more efficiently than arbitrary full sequences
         (e.g. in Potts model hamiltonian). If no customized implementation is available,
         this method should still wrap around score() for applications like Gibbs sampling.

        2. Logits are not relative to any particular sequence (e.g. "wildtype"), but
         meant to be interpreted relative to each other (i.e. should be treated as raw logits)
         across possible symbols *per* sampled instance/entity/position combination

        TODO: how handle different types of alphabets sampled at the same time? Or require that all entities
         must have same type/alphabet (e.g. protein)

        TODO: if we encounter at least one relevant case of a method that is able to
          compute P(x_i | x_\i) but not P(x_1, ..., x_n), break this method out into a
          separate interface "ConditionalScorer" for use with the Gibbs sampler

        Parameters
        ----------
        instances
            Target instances/sequences for which scores should be calculated. Must
            have same length as entities and positions.
        entities
            List of entity indexes which selects exactly one entity per instance for scoring.
            Must have same length as instances and positions.
        positions
            List of positions which selects exactly one position per instance/entity pair.
            Must have same length as instances and entities.
        status_callback
            Callback function to track computation status

        Returns
        -------
        Dataframe with raw logit scores (seq x aa); row index is over instance index/entity index/position triplets;
        columns index over different symbols (amino acids etc.). Guaranteed to have same length as instance,
        entities and positions. Rows must be in the same order as input instance/entity/position triplets.
        Columns must be in same order as constants.VALID_AA_OR_GAP_SORTED, missing predictions must be
        encoded by np.nan
        """
        pass

    @abstractmethod
    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Compute all single substitutions to one particular instance (aka "single mutation scan")
        batching across different positions. This is different to score_conditional() which
        batches substitutions to exactly one single position across many different instances.

        Note:
        1. Mutation logits should be *relative* to the given instance (like a log-odds ratio),
         so that self-substitutions are assigned are score of 0, beneficial substitutions are score > 0,
         and damaging substitutions  a score < 0. This differs from score_conditional, where there is
         no notion of a "wildtype" sequence to compute relative scores to.

        2. The implementation of this function can draw on score(), score_conditional(), score_mutants()
         or any method-specific implementations as needed to provide the most efficient/accurate way
         to single mutant effect calculation

        Parameters
        ----------
        instance
            Target system instance specification to mutate
        entity
            Index of entity for which mutation scan should be computed. Defaults to first entity.
        positions
            Subset of positions to score. If None, scores for all positions will be computed.
        status_callback
            Callback function to track computation status

        Returns
        -------
        Dataframe with log-odds scores (seq x aa) relative to instance; rows index over
        entity/position/ref triplets, columns index over different symbols (amino acids etc.).
        Columns must be in same order as constants.VALID_AA_OR_GAP_SORTED,
        missing predictions must be coded by np.nan.
        """
        pass

    @abstractmethod
    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        """
        Compute logit scores for a list of mutations to a specified system instance
        (can be any single or higher-order mutants); this method is to allow specialized, more efficient
        or accurate implementations of mutant calculations than computing the full score of the WT and
        mutant sequence. In case no such specialization is possible or needed for a method, it can simply
        call out to the score() function.

        Note:
        1. Mutation logits should be *relative* to the given instance (like a log-odds score),
         so that self-substitutions are assigned are score of 0, beneficial substitutions are score > 0,
         and damaging substitutions  a score < 0. This differs from score_conditional, where there is
         no notion of a "wildtype" sequence to compute relative scores to.

        2. Implementations of this method may either compute mutant and reference scores for substraction
         with the score() method or draw on any specialized implementations of single and higher-order mutation
         scoring that are more accurate / efficient.

        Parameters
        ----------
        instance
            Target system instance specification to mutate
        mutants
            List of mutations of any order to compute
        status_callback
            Callback function to track computation status

        Returns
        -------
        Vector of scores, guaranteed to be in the same order as mutants list
        """
        pass


class Generator(Protocol):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)

    TODO: add parameters to bias or select/avoid amino acids (global or position-specific)
    TODO: add flags to allow/disallow indels (also need to specify min/max length range)
    TODO: add parameters for sampling strategy where available (e.g. min-p, top-k, etc.)
    """
    @abstractmethod
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        """
        Sample new sequences from generative model

        Note: Implementation should raise ValueError if any of the specified design options are not supported

        Note: Method must always return at least num_designs elements in the output list,
        but may also return more designs than requested e.g. if beneficial due to batch size

        Parameters
        ----------
        num_designs
            Number of designs to generate
        entities
            Indices of entities in system that should be designed during generation (others will be kept fixed).
            If None, will attempt to design all entities.
        fixed_pos
            Mapping from entity index to positions that should be fixed during design. Any entity referenced
            in the mapping must be also included in the entities' parameter. Numbering of fixed positions must match
            sequence numbering of system entity representation (with corresponding value of first_index,
            by default 1; i.e. one-based indexing of positions!)
        temperature
            Sampling temperature (higher values generate more diversity)
        deletions
            If True, allow the model to sample deletions relative to the entities representation
        status_callback
            Callback function to track computation status

        Returns
        -------
        Designed instances (sequences/structures) of system (guaranteed to contain at least num_design instances)
        """
        pass


class Embedder(Protocol):
    """
    Interface implemented by methods than can compute per-position embeddings
    (designs/sequences, vector per token)

    TODO: add interface for combined scoring and embedding (don't compute twice, as embeddings will be
        computed whenever density is computed

    TODO: add separate interface for protein-level embedding (rather than positional embeddings, which can be pooled)

    TODO: check if beneficial to add specialized methods for single-mutant embeddings?

    TODO: all instances must have same length

    TODO: make method more flexible so can we compute embeddings across all entities?
    """
    @abstractmethod
    def embed(
        self,
        instances: Sequence[SystemInstance],
        entity: int,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
        """
        Transform system instances to embeddings

        Parameters
        ----------
        instances
            List of system instances to be transformed
        entity:
            The index of the entity to embed
        status_callback
            Callback function to track computation status

        Returns
        -------
        Embeddings for given instances (instance x positions x feature dimension);
        actual embedding features are in last dimension of tensor
        """
        pass


@dataclass
class RequiredResources:
    """
    All memory resources in megabytes, times in minutes
    """
    min_gpu_cores: int | None
    min_gpu_memory_per_core: int | None

    min_cpu_cores: int | None
    min_cpu_memory_per_core: int | None

    max_batch_size: int | None

    time: int | None


class BaseModel(ABC):
    """
    Core abstract definition of a protein design model
    """
    @property
    @abstractmethod
    # plain-text name of method
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    # whether model has long-running build step (e.g. EVE VAE)
    def requires_heavy_build(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *must* be run on GPU
    def requires_gpu(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be run on GPU (implies this is an advantage, otherwise set this to False)
    def supports_gpu(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be parallelized on CPU (implies this is an advantage, otherwise set this to False)
    def supports_cpu_parallel(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model *can* be parallelized on GPU (implies this is an advantage, otherwise set this to False)
    def supports_gpu_parallel(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs a specified target sequence
    def requires_target(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs unaligned sequences as input
    def requires_seqs(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs aligned sequences as input
    def requires_msa(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model needs 3D structures as input
    def requires_3d(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model requires fixed-length sequences
    # (implies insertions cannot be modeled)
    def requires_fixed_length(self) -> bool:
        pass

    @property
    @abstractmethod
    # whether model is able to model deletions (may be possible for models
    # with required fixed length depending on alphabet)
    def handles_deletions(self) -> bool:
        pass

    @property
    @abstractmethod
    # indicates if model was built and is ready for scoring/generation
    def ready(self) -> str:
        pass

    @property
    @abstractmethod
    # must return system modelled by the current instance (after build), or None otherwise
    def system(self) -> System | None:
        pass

    def ready_or_raise(self) -> None:
        """
        Verifies if model is ready for predictions by checking ready property,
        or raises a ValueError otherwise
        """
        if not self.ready:
            raise ValueError("Must call build() first to use model")

    @classmethod
    @abstractmethod
    def can_model(
        cls,
        system: System,
        data: Any,
    ) -> Tuple[bool, str]:
        """
        Check if the model is able to perform computations on the specified
        molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)

        Returns
        -------
        bool
            True if model is able to handle the system, False otherwise
        str
            Message specifying why model is not able to handle the system
        """
        pass

    @classmethod
    def can_model_or_raise(
        cls,
        system: System,
        data: Any,
    ) -> None:
        """
        Check if the model is able to perform computations on the specified
        molecular system via can_model(), raise a ValueError otherwise

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)

        Returns
        -------
        bool
            True if model is able to handle the system, False otherwise
        str
            Message specifying why model is not able to handle the system
        """
        can_model, can_model_msg = cls.can_model(system, data)
        if not can_model:
            raise ValueError(can_model_msg)

    @classmethod
    @abstractmethod
    def required_resources(
        cls,
        system: System,
        data: Any,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        """
        Estimate the required resources to perform computations on molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (cf. documentation for build() method)
        use_gpu
            Set to True if you want to estimate resources making use of GPU
            (only for models supporting GPU-based computations)
        build
            Set as True to estimate resources for model building. Set as False to
            estimate resources for inference (scoring / sampling).

        Returns
        -------
        RequiredResources
            CPU/GPU/RAM requirements for running computations on molecular system
        """
        pass

    @abstractmethod
    def build(
        self,
        system: System,
        data: Any,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Prepare model for calculations on a given molecular system (e.g. scoring or sampling).
        Conditional approaches will typically perform computations here whereas unconditional approaches
        may simply do nothing other than return self.
        In the case of inference-only conditional models, implementations of this method will be very
        light (e.g. compute an encoding), whereas for other conditional models this method may be
        compute-heavy (e.g. EVE VAE models trained on a family-specific MSA)

        Notes re implementation:
        1) Should always verify if the system can
        be modelled using self.can_model() or raise a ValueError instead

        2) Should always assign system to self.system

        3) Should always return self to allow method chaining

        4) Should pay careful attention whether any external model parameters
        (e.g. PyTorch model) are stored inside the class to avoid potential problems and inflated
        memory usage if instances of the class are serialized; use the available context managers
        to handle this behavior reliably

        Parameters
        ----------
        system
            Molecular system to be modelled
        data
            Arbitrary additional data specific to model that is not a descriptive property of system itself
            (could be labelled data points, external sequences to compare to, etc.)
        status_callback
            Callback function to receive progress updates

        Returns
        -------
        self
            Reference to the instance for method chaining
        """
        pass

    @abstractmethod
    def positions(
        self
    ) -> List[Tuple[int, int]]:
        """
        Return list of all available modelled positions per entity that can be potentially mutated

        Note: positions that are not modelled (e.g. lowercase letters in EVmutation) should not
        be returned by this method

        Note: returned positions should be ordered in ascending order
        by i) entity index, ii) position index in entity

        Returns
        -------
        List of position lists (outer list indexes over entities, inner list contains all positions)
        """
        pass

    def valid_positions(
        self,
        positions: Sequence[int],
        entities: int | Sequence[int] = 0,
        raise_invalid: bool = False,
    ) -> List[tuple[int, int]]:
        """
        Helper method to verify if a list of positions for a given entity in system is valid (via positions())

        Parameters
        ----------
        positions
            List of unique positions to check
        entities
            List of entities corresponding to each position (if sequence);
            or can be fixed to one entity which will be applied to all positions (if int)
        raise_invalid
            If invalid position contained in input list, raise a ValueError

        Returns
        -------
        List of valid position tuples
        """
        # available_positions = set(
        #     pos for (entity_idx, pos) in self.positions() if entity_idx == entity
        # )

        if isinstance(entities, int):
            given_pos = [
                (entities, pos) for pos in positions
            ]
        else:
            if len(positions) != len(entities):
                raise ValueError("Length of entities and positions must agree")

            given_pos = [
                (entity, pos) for entity, pos in zip(entities, positions)
            ]

        available_pos = set(self.positions())

        valid_pos = [
            entity_pos for entity_pos in given_pos if entity_pos in available_pos
        ]

        if raise_invalid and len(valid_pos) != len(positions):
            raise ValueError(
                f"Invalid positions given, valid options are {available_pos}"
                f" but given are {given_pos}"
            )

        return valid_pos


class BaseModelAndScorer(BaseModel, Scorer, ABC):
    """
    Auxiliary class for typing pa

    TODO: If we find that all Scorers need to be a BaseModel, better to have Scorer inherit from BaseModel;
     we already know from the Gibbs sampler that not all Generators will be a BaseModel
    """
    pass
