"""
BoltzGen: wraps BoltzGen protein design into the
evedesign Generator interface.

Uses only the diffusion/generation step of BoltzGen
(no inverse folding, no refolding, no filtering).
"""

import os
from typing import Any, Self, Sequence

try:
    import boltzgen  # noqa: F401
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

DEFAULT_CACHE_DIR: str = os.path.expanduser("~/.cache/boltzgen")

from evedesign.model import BaseModel, Generator
from evedesign.system import System, SystemInstance
from evedesign.types import DeviceType, StatusCallback


class BoltzGenGenerator(BaseModel, Generator):
    available = IMPORT_AVAILABLE
    name: str = "BoltzGen"
    citations: list[str] = []

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        protocol: str = "protein-anything",
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        device: DeviceType = "cuda",
        sampling_steps: int = 200,
        placeholder_residue: str = "G",
        keep_model_after_build: bool = False,
    ):
        if not self.available:
            raise ValueError(
                "boltzgen package could not be imported. "
                "Install with: pip install boltzgen "
                "(requires CUDA-12)"
            )

        self.protocol = protocol
        self.cache_dir = cache_dir
        self.device = device
        self.sampling_steps = sampling_steps
        self.placeholder_residue = placeholder_residue
        self.keep_model_after_build = keep_model_after_build
        # keep parameters loaded once loaded for prediction
        # purposes to avoid reloading over and over
        self.keep_model_after_pred = True

        self._system = None
        self.model = None

    @property
    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(
        cls,
        system: System,
        data: Any = None
    ) -> tuple[bool, str]:
        raise NotImplementedError("can_model stub")

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        raise NotImplementedError("build stub")

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        raise NotImplementedError("generate stub")
