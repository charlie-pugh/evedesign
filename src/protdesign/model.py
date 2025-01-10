from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, Tuple
from protdesign.entity import EntityOrEntityList
from protdesign.utils import StatusCallback


class Scorer(Protocol):
    """
    Interface implemented by classes that can score
    (e.g. density/log likelihood) for existing designs/sequences
    (scalar value per design/sequence)
    """
    @abstractmethod
    def score(self) -> None:
        # TODO: add actual return types
        # TODO: mutants or just absolute sequences?
        pass

    @abstractmethod
    def score_single(self) -> None:
        # TODO: break this method out into its own Protocol? Some methods may be able to compute
        #  P(x_i | x_\i) but not P(x_1, ..., x_n) - for Gibbs, we only need the former!

        # TODO: this should score one position across many different WT sequences
        #  (batch across sequences)

        # TODO: add actual return types
        # TODO: should this also support deletions/insertions?
        pass

    # TODO: add status callback argument
    # TODO: another method to score all singles for a given sequence (batch across positions)
    # TODO: method to return all sites of interest for Gibbs sampler?
    # TODO: device specification


class Generator(Protocol):
    """
    Interface implemented by classes that can generate new samples
    (e.g. generative models or samplers on top of scoring models)
    """
    @abstractmethod
    def generate(self) -> None:
        # TODO: add actual return types
        # TODO: parameters? number of designs, flexible positions, etc.
        # TODO: device specification
        # TODO: add status callback argument
        pass


class Embedder(Protocol):
    """
    Interface implemented by methods than can compute embeddings
    (designs/sequences, vector per token)

    # TODO: add more efficient method to score and embed?
    # TODO: add methods for single-mutant embeddings
    # TODO: pooling / protein-level embedding
    """
    @abstractmethod
    def embed(self) -> None:
        # TODO: add return types
        # TODO: device specification
        # TODO: add status callback argument
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
    def can_model(cls, system: EntityOrEntityList) -> Tuple[bool, str]:
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
        # TODO: rename this method to _build and create an implementation for build() that
        #   calls can_model and sets model.build = True?

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
