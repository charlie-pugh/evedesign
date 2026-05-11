"""
BoltzGen: wraps BoltzGen protein design into the
evedesign Generator interface.

Uses only the diffusion/generation step of BoltzGen
(no inverse folding, no refolding, no filtering).
"""

try:
    import boltzgen  # noqa: F401
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

from evedesign.model import BaseModel, Generator


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
