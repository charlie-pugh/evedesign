"""
BoltzGen Generator: wraps BoltzGen diffusion-based de novo
protein structure design into the evedesign Generator interface.

Generates new protein backbones conditioned on a target
structure. Returns SystemInstance objects with structures
populated but placeholder sequences. Use LigandMPNN downstream
for sequence design and BoltzFoldTransformer for refolding.

NOTE: Requires the boltzgen package (pip install evedesign[boltzdesign]).
The boltzgen package pins cuequivariance_* dependencies that do
not install on macOS or CPU-only Linux.
"""
import shutil

from loguru import logger

try:
    # boltzgen is CLI-only — we only check it is on PATH
    IMPORT_AVAILABLE = shutil.which("boltzgen") is not None
except ImportError:
    IMPORT_AVAILABLE = False

from evedesign.model import BaseModel, Generator
from evedesign.system import System
from evedesign.types import DeviceType


# Default checkpoint references (HuggingFace)
DEFAULT_DESIGN_CHECKPOINTS = [
    "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
    "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
]
DEFAULT_INVERSE_FOLD_CHECKPOINT = (
    "huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt"
)
DEFAULT_FOLDING_CHECKPOINT = (
    "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt"
)

PROTOCOLS = [
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
    "antibody-anything",
    "protein-redesign",
]


class BoltzGenGenerator(BaseModel, Generator):
    """
    Wraps BoltzGen diffusion-based de novo structure
    design into the evedesign Generator interface.

    Generates de novo protein backbones conditioned
    on a target structure. Returns SystemInstance
    objects with structures populated.

    For sequence design after backbone generation use
    LigandMPNN. For refolding use BoltzFoldTransformer.
    """
    available = IMPORT_AVAILABLE
    name: str = "BoltzGen"
    citations: list[str] = ["doi.org/10.1101/2025.11.20.689494"]

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = True
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = [
        "structures",
        "secondary_structure",
        "interactions",
        "atom_bonds",
        "copies",
        "min_length",
        "max_length",
    ]

    def __init__(
        self,
        protocol: str = "protein-anything",
        device: DeviceType = "cuda",
        num_devices: int | None = None,
        num_workers: int = 1,
        diffusion_batch_size: int | None = None,
        design_checkpoints: list[str] | None = None,
        inverse_fold_checkpoint: str | None = None,
        folding_checkpoint: str | None = None,
        skip_inverse_folding: bool = False,
        inverse_fold_num_sequences: int = 1,
        step_scale: str | None = None,
        noise_scale: str | None = None,
        budget: int = 30,
        alpha: float | None = None,
    ):
        if not self.available:
            logger.warning(
                "boltzgen CLI not found on PATH. "
                "Install via: pip install boltzgen "
                "(generate() will fail until boltzgen is installed)"
            )

        if protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol '{protocol}', "
                f"valid options: {PROTOCOLS}"
            )

        self.protocol = protocol
        self.device = device
        self.num_devices = num_devices
        self.num_workers = num_workers
        self.diffusion_batch_size = diffusion_batch_size
        self.design_checkpoints = (
            design_checkpoints
            or DEFAULT_DESIGN_CHECKPOINTS
        )
        self.inverse_fold_checkpoint = (
            inverse_fold_checkpoint
            or DEFAULT_INVERSE_FOLD_CHECKPOINT
        )
        self.folding_checkpoint = (
            folding_checkpoint
            or DEFAULT_FOLDING_CHECKPOINT
        )
        self.skip_inverse_folding = skip_inverse_folding
        self.inverse_fold_num_sequences = (
            inverse_fold_num_sequences
        )
        self.step_scale = step_scale
        self.noise_scale = noise_scale
        self.budget = budget
        self.alpha = alpha

        self._system = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system
