from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, List, Self, Tuple, Sequence
import numpy as np
from protdesign.entity import System, SystemInstance, EntityPosList
from protdesign.types import StatusCallback


class Scorer(Protocol):
    """
    Interface implemented by classes that can score
    (e.g. density/log likelihood) for existing designs/sequences
    (scalar value per design/sequence)

    All methods of this interface are expected to return raw logits that can be compared
    relatively within the returned array of scores (but not necessarily between different
    calls to the function, where normalization e.g. to the target sequence
    should be employed)

    # TODO: add specialized method to score higher-order mutations (e.g. doubles)?
    # TODO: add batch size to params? or infer in build() method?
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
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """
        Compute scores for all substitutions in a single position
        across a batch of sequences (single position can differ between instances), e.g.
        for Gibbs sampling-based generation of multiple designs in parallel.

        This function allows to exploit the fact that often single mutations for
        one position can be computed more efficiently than arbitrary full sequences
        (e.g. in Potts model hamiltonian). If no customized implementation is available,
        this method should still wrap around score() for applications like Gibbs sampling.

        Note that logits are not relative to any particular sequence (e.g. "wildtype"), but
        meant to be interpreted relative to each other (i.e. should be treated as raw logits)
        *per* sampled instance/entity/position combination

        TODO: how handle different types of alphabets sampled at the same time?

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
        Matrix of logit scores (seq x aa); first dimension indexes along different instances,
        second dimension indexes over different states
        """
        pass

    @abstractmethod
    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """
        Compute all single substitutions to one particular instance (aka "single mutation scan")
        batching across different positions. This is markedly different to score_single_pos() which
        batches substitutions to exactly one single position across many different instances.

        Note that mutation logits should be *relative* to the given instance, so that self-substitutions
        are assigned are score of 0. This differs from score_conditional, where there is no notion of a
        "wildtype" sequence to compute relative scores to.

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
        Matrix of logit scores (pos x aa); first dimension indexes over positions and second
        dimension indexes over possible substitutions.
        """
        pass

    @abstractmethod
    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: List[str],
        status_callback: StatusCallback | None = None
    ) -> None:
        """
        Compute logit scores for a list of mutations to a specified system instance
        (can be any single or higher-order mutants); this method is to allow specialized, more efficient
        implementations of mutant calculations than computing the full score of the WT and mutant sequence.
        In case no such specialization is possible or needed for a method, it can simply call out to the
        score() function.

        # TODO: mutant format specification, how to best handle mutants across different entities?

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
        # TODO: define
        """
        pass


class Generator(Protocol):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)

    # TODO: add parameters to bias or select/avoid amino acids (global or position-specific)
    # TODO: add parameter to allow indels (also need to specify min/max length range)
    # TODO: add parameters for sampling strategy where available (e.g. min-p, top-k, etc.)
    """
    @abstractmethod
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
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
            in the mapping must be also included in the entities parameter. Numbering of fixed positions must match
            sequence numbering of system entity representation (with corresponding value of first_index,
            by default 1; i.e. one-based indexing of positions!)
        temperature
            Sampling temperature (higher values generate more diversity)
        status_callback
            Callback function to track computation status

        Returns
        -------
        Designed instances (sequences/structures) of system
        """
        pass


class Embedder(Protocol):
    """
    Interface implemented by methods than can compute embeddings
    (designs/sequences, vector per token)

    # TODO: add method for combined scoring and embedding (don't compute twice, separate interface)?
    # TODO: add specialized methods for single-mutant embeddings?
    # TODO: pooling / protein-level embedding?
    # TODO: all instances must have same length unless pooling
    # TODO: can we compute embeddings across all entities?
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
    ) -> Tuple[bool, str]:
        """
        Check if the model is able to perform computations on the specified
        molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled

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
    ) -> None:
        """
        Check if the model is able to perform computations on the specified
        molecular system via can_model(), raise a ValueError otherwise

        Parameters
        ----------
        system
            Molecular system to be modelled

        Returns
        -------
        bool
            True if model is able to handle the system, False otherwise
        str
            Message specifying why model is not able to handle the system
        """
        can_model, can_model_msg = cls.can_model(system)
        if not can_model:
            raise ValueError(can_model_msg)

    @classmethod
    @abstractmethod
    def required_resources(
        cls,
        system: System,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        """
        Estimate the required resources to perform computations on molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled
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

        2) Sould always assign system to self.system

        3) Should always return self to allow method chaining

        4) Should pay careful attention whether any external model parameters
        (e.g. PyTorch model) are stored inside the class to avoid potential problems and inflated
        memory usage if instances of the class are serialized; use the available context managers
        to handle this behavior reliably

        # TODO: add parameter for labelled examples for supervised setting or keep this base class zero shot-only?
        # TODO: add parameter "limit" to restrict system scoring to a certain region?

        Parameters
        ----------
        system
            Molecular system to be modelled
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
        entity: int = 0,
        raise_invalid: bool = False,
    ) -> List[int]:
        """
        Helper method to verify if a list of positions for a given entity in system is valid (via positions())

        Parameters
        ----------
        positions
            List of unique positions to check
        entity
            Index of entity in system to check positions in
        raise_invalid
            If invalid position contained in input list, raise a ValueError

        Returns
        -------
        List of valid positions
        """
        available_positions = set(
            pos for (entity_idx, pos) in self.positions() if entity_idx == entity
        )

        valid_positions = [
            pos for pos in positions if pos in available_positions
        ]

        if raise_invalid and len(valid_positions) != len(positions):
            raise ValueError(
                f"Invalid positions for entity {entity}, valid options are {', '.join(map(str, available_positions))}"
                f" but given are {', '.join(map(str, positions))}"
            )

        return valid_positions