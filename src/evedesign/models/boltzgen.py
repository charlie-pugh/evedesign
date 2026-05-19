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


try:
    # boltzgen is CLI-only — we only check it is on PATH
    IMPORT_AVAILABLE = shutil.which("boltzgen") is not None
except ImportError:
    IMPORT_AVAILABLE = False


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
