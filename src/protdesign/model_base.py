from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, Tuple
from protdesign.entity import EntityOrEntitySequence


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
        # TODO: add actual return types
        pass

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
        pass


class Embedder(Protocol):
    """
    Interface implemented by methods than can compute embeddings
    (designs/sequences, vector per token)

    # TODO: add more efficient method to score and embed?
    # TODO: pooling / protein-level embedding
    """
    @abstractmethod
    def embed(self) -> None:
        # TODO: add return types
        # TODO: device specification
        pass


@dataclass
class RequiredResources:
    # TODO: seperate between training and inference?
    # TODO: training time
    # TODO: inference time
    gpu_required: bool
    gpu_cores: int | None
    gpu_ram_per_core: int | None
    cpu_cores: int | None
    cpu_ram_per_core: int | None


class BaseModel(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def can_model(cls, system: EntityOrEntitySequence) -> Tuple[bool, str]:
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
    def required_resources(cls, system: EntityOrEntitySequence) -> RequiredResources:
        """
        Estimate the required resources to perform computations on molecular system

        Parameters
        ----------
        system
            Molecular system to be modelled

        Returns
        -------
        RequiredResources
            CPU/GPU/RAM requirements for running computations on molecular system
        """
        pass

    @abstractmethod
    def build(self, system: EntityOrEntitySequence) -> Self:
        """
        Prepare model for calculations on a given molecular system (e.g. scoring or sampling).
        In the case of inference-only approaches, implementations of this method will be very light
        (e.g. compute an encoding), whereas for others this method may be compute-heavy (e.g.
        VAE models trained on a specific MSA)

        Note: implementations of this method should always verify if the system can
        be modelled or raise a ValueError instead

        Note: Implementations of this method should always return self to allow method chaining

        Note: implementations should pay careful attention whether any external model parameters
        (e.g. PyTorch model) are stored inside the class to avoid potential problems and inflated
        memory usage if instances of the class are serialized

        Parameters
        ----------
        system
            Molecular system to be modelled

        Returns
        -------
        self
            Reference to the instance for method chaining
        """
        # TODO: add extra parameters for supplying MSA, structures, etc.
        pass

