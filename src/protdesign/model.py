from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, List, Self, Tuple, Sequence
import numpy as np
from numpy.typing import ArrayLike
from protdesign.entity import EntityOrEntityList, SystemInstance
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
    def positions(
        self
    ) -> List[List[int]]:
        """
        Return list of all available positions per entity that can be mutated using score_single()

        Returns
        -------
        List of position lists (outer list indexes over entities, inner list contains all positions)
        """
        pass

    @abstractmethod
    def score_single_pos(
        self,
        instances: Sequence[SystemInstance],
        entities: ArrayLike,
        positions: ArrayLike,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """
        Compute scores for all substitutions in a single position
        across a batch of sequences (single position can differ between instances), e.g.
        for Gibbs sampling multiple designs in parallel.

        This function allows to exploit the fact that often single mutations for
        one position can be computed more efficiently than arbitrary full sequences
        (e.g. in Potts model hamiltonian). If no customized implementation is available,
        this method should still wrap around score() for applications like Gibbs sampling.

        # TODO:
        #  Break this method out into its own Protocol? Some methods may be able to compute
        #  P(x_i | x_\i) but not P(x_1, ..., x_n) - for Gibbs, we only need the former!

        Parameters
        ----------
        instances
            Target instances/sequences for which scores should be calculated
        entities
            List of entity indexes for which single mutant should be computed
        positions
            Position in instance/entity combination for which mutant scores
            should be computed
        status_callback
            Callback function to track computation status

        Returns
        -------
        Matrix of logit scores (seq x aa); first dimension indexes along different instances,
        second dimension indexes over different states
        """
        pass

    def score_landscape(
        self,
        instance: SystemInstance,
        entity: int,
        positions: ArrayLike | None = None,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        """
        Compute all single substitutions to given instance (aka "single mutation scan/DMS")
        batching across all (or some subset thereof) positions. This is markedly different to score_single()
        which batches substitutions to single position across many different target sequences.


        Parameters
        ----------
        instance
            Target system instance specification to mutate
        entity
            Index of entity for which mutation scan should be computed
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


class Generator(Protocol):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)

    # TODO: add batch size to params? or infer in build() method?
    # TODO: add parameters to bias or select/avoid amino acids (global or position-specific)
    """
    @abstractmethod
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: Sequence[int] | None = None,  # TODO: specify as list of lists
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        # TODO: document parameters
        # TODO: what entities to design/fix
        pass


class Embedder(Protocol):
    """
    Interface implemented by methods than can compute embeddings
    (designs/sequences, vector per token)

    # TODO: add method for combined scoring and embedding (don't compute twice, separate interface)?
    # TODO: add specialized methods for single-mutant embeddings?
    # TODO: pooling / protein-level embedding?
    # TODO: add batch size to params? or infer in build() method?
    """
    @abstractmethod
    def embed(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int, int], np.dtype[float]]:
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

    @classmethod
    @abstractmethod
    def can_model(
        cls,
        system: EntityOrEntityList
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
    @abstractmethod
    def required_resources(
        cls,
        system: EntityOrEntityList,
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
        system: EntityOrEntityList,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Prepare model for calculations on a given molecular system (e.g. scoring or sampling).
        In the case of inference-only approaches, implementations of this method will be very light
        (e.g. do nothing, or compute an encoding), whereas for others this method may be compute-heavy
        (e.g. EVE VAE models trained on a family-specific MSA)

        Note: implementations of this method should always verify if the system can
        be modelled or raise a ValueError instead

        Note: Implementations of this method should always return self to allow method chaining

        Note: implementations should pay careful attention whether any external model parameters
        (e.g. PyTorch model) are stored inside the class to avoid potential problems and inflated
        memory usage if instances of the class are serialized

        # TODO: add parameter for labelled examples for supervised setting

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
