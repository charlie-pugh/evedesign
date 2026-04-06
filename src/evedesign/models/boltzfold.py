"""
BoltzFold: wraps Boltz-2 structure prediction into the
evedesign Transformer interface.

NOTE: Template conditioning via Entity.structures is not
yet implemented. Structures present on entities will be
ignored with a warning.
"""

from os import PathLike
from typing import Literal

try:
    from boltz.main import process_inputs  # noqa
    from boltz.model.models.boltz2 import Boltz2  # noqa
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # noqa
    from boltz.data.types import Manifest  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

from evedesign.model import BaseModel, Transformer, Scorer
from evedesign.types import DeviceType, BatchSize


class BoltzFoldTransformer(BaseModel, Transformer, Scorer):
    """
    Wraps Boltz-2 into the evedesign Transformer interface.
    Folds protein sequences into 3D structures.
    Confidence scores are returned as a side effect of transform().
    """
    available = IMPORT_AVAILABLE
    name: str = "BoltzFold"
    citations: list[str] = []

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        model_dir_path: str | PathLike | None = None,
        batch_size: BatchSize = 1,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        use_msa_server: bool = False,
        use_msa: bool = True,
        score_attribute: Literal[
            "iptm", "ptm", "confidence_score", "complex_plddt"
        ] = "iptm",
    ):
        if not self.available:
            raise ValueError(
                "boltz package could not be imported. Is it installed already?"
            )

        self.model_dir_path = model_dir_path
        self.batch_size = batch_size
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = True
        self.device = device
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.recycling_steps = recycling_steps
        self.use_msa_server = use_msa_server
        self.use_msa = use_msa
        self.score_attribute = score_attribute

        self._system = None
        self.model = None
