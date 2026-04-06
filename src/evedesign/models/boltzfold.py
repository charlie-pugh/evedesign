"""
BoltzFold: wraps Boltz-2 structure prediction into the
evedesign Transformer interface.

NOTE: Template conditioning via Entity.structures is not
yet implemented. Structures present on entities will be
ignored with a warning.
"""

try:
    from boltz.main import process_inputs, BoltzProcessedInput  # noqa
    from boltz.model.models.boltz2 import Boltz2  # noqa
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # noqa
    from boltz.data.types import Manifest  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False
