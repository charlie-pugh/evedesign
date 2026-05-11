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
    pass
