"""
BoltzGen Generator: wraps BoltzGen diffusion-based de novo
protein structure design into the evedesign Generator interface.

NOTE: Requires the boltzgen package (pip install evedesign[boltzgen]).
A CUDA GPU is mandatory: the boltzgen CLI calls torch.cuda.get_device_capability()
unconditionally, so there is no CPU path.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Self, Sequence

from loguru import logger

try:
    # boltzgen is CLI-only — we only check it is on PATH
    IMPORT_AVAILABLE = shutil.which("boltzgen") is not None
except ImportError:
    IMPORT_AVAILABLE = False

from evedesign.model import BaseModel, Generator
from evedesign.models.boltz.convert_design import (
    _is_design_entity,
    parse_design_output,
    system_to_boltzgen_yaml,
)
from evedesign.system import System, SystemInstance
from evedesign.types import DeviceType, EntityPosList, StatusCallback


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

    Generates de novo protein backbones, optionally
    conditioned on a target. Returned SystemInstance
    objects have both structures and designed sequences
    populated.
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
        keep_tmp_dir: bool = False,
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
        self.keep_tmp_dir = keep_tmp_dir

        self._system = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(
        cls,
        system: System,
        data: Any = None,
    ) -> tuple[bool, str]:
        """
        Check if the system is suitable for BoltzGen
        de novo design.

        Requires at least one designable entity
        (rep=None or min_length/max_length specified).
        """
        if data is not None:
            return False, (
                "BoltzGen does not accept a data "
                "parameter (must be None)"
            )

        has_design = False
        for i, entity in enumerate(system):
            if entity.type not in ("protein", "ligand"):
                return False, (
                    f"Entity {i} has type "
                    f"'{entity.type}'. Only protein "
                    "and ligand entities are currently "
                    "supported."
                )
            if _is_design_entity(entity):
                has_design = True

        if not has_design:
            return False, (
                "No designable entities found. At least "
                "one entity must have rep=None or "
                "min_length/max_length set to be designed."
            )

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """Validate system and register for generation."""
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _build_cli_command(
        self,
        yaml_path: Path,
        output_dir: Path,
        num_designs: int,
    ) -> list[str]:
        """
        Build the boltzgen CLI command from the
        generator's configuration and per-call args.

        Returns the command as a list of strings
        suitable for subprocess.run.
        """
        cmd = [
            "boltzgen", "run",
            str(yaml_path),
            "--output", str(output_dir),
            "--protocol", self.protocol,
            "--num_designs", str(num_designs),
            "--num_workers", str(self.num_workers),
            "--inverse_fold_num_sequences",
            str(self.inverse_fold_num_sequences),
            "--budget", str(self.budget),
        ]
        cmd += [
            "--design_checkpoints"
        ] + self.design_checkpoints
        cmd += [
            "--inverse_fold_checkpoint",
            self.inverse_fold_checkpoint,
        ]
        cmd += [
            "--folding_checkpoint",
            self.folding_checkpoint,
        ]

        if self.num_devices is not None:
            cmd += ["--devices", str(self.num_devices)]
        if self.diffusion_batch_size is not None:
            cmd += [
                "--diffusion_batch_size",
                str(self.diffusion_batch_size),
            ]
        if self.step_scale is not None:
            cmd += ["--step_scale", self.step_scale]
        if self.noise_scale is not None:
            cmd += ["--noise_scale", self.noise_scale]
        if self.alpha is not None:
            cmd += ["--alpha", str(self.alpha)]
        if self.skip_inverse_folding:
            cmd += ["--skip_inverse_folding"]

        return cmd

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Generate N de novo designs via BoltzGen.

        Writes a BoltzGen design YAML from self._system,
        shells out to the boltzgen CLI, then parses
        the output CIFs into SystemInstance objects.

        BoltzGen determines designable entities from
        the System spec itself (entities with rep=None
        or min_length/max_length). The entities and
        fixed_pos parameters are accepted for interface
        compatibility but currently logged as warnings.
        """
        self.ready_or_raise()

        if entities is not None:
            logger.warning(
                "BoltzGen determines designable entities "
                "from the System specification. The "
                "'entities' parameter is ignored — entity "
                "selection is controlled via Entity.rep / "
                "Entity.min_length / Entity.max_length."
            )

        if fixed_pos is not None:
            logger.warning(
                "BoltzGen does not yet support fixed_pos "
                "via the evedesign interface — ignored."
            )

        # Map temperature to step_scale if user didn't
        # set it explicitly. BoltzGen recommends
        # step_scale ~ 1.5 * temperature for sampling.
        effective_step_scale = self.step_scale
        if effective_step_scale is None and temperature != 1.0:
            effective_step_scale = str(
                round(1.5 * temperature, 2)
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="boltzgen_"))
        try:
            # 1. Write the YAML design spec
            yaml_path = tmp_dir / "design_spec.yaml"
            system_to_boltzgen_yaml(self._system, yaml_path)
            logger.info(
                f"BoltzGen YAML written to {yaml_path}"
            )

            # 2. Build the CLI command and invoke
            output_dir = tmp_dir / "output"
            output_dir.mkdir()

            # Temporarily apply effective_step_scale to
            # the instance so _build_cli_command picks
            # it up
            original_step_scale = self.step_scale
            if effective_step_scale is not None:
                self.step_scale = effective_step_scale
            cmd = self._build_cli_command(
                yaml_path, output_dir, num_designs
            )
            self.step_scale = original_step_scale

            logger.info(
                f"Running BoltzGen: {' '.join(cmd)}"
            )
            if status_callback is not None:
                status_callback(
                    "running",
                    0.0,
                    "Starting BoltzGen pipeline",
                )

            env = {**os.environ}
            if self.device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                logger.error(
                    f"BoltzGen stderr:\n{result.stderr}"
                )
                raise RuntimeError(
                    f"BoltzGen pipeline failed "
                    f"(exit code {result.returncode}).\n"
                    f"stderr: {result.stderr[-2000:]}"
                )

            logger.info(
                "BoltzGen pipeline completed successfully"
            )
            if status_callback is not None:
                status_callback(
                    "running",
                    0.8,
                    "Parsing BoltzGen outputs",
                )

            # 3. Parse outputs into list[SystemInstance]
            instances = parse_design_output(
                output_dir=output_dir,
                system=self._system,
            )

            if status_callback is not None:
                status_callback(
                    "done",
                    1.0,
                    f"{len(instances)} designs parsed",
                )

            return instances

        finally:
            if not self.keep_tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                logger.info(
                    f"keep_tmp_dir=True — outputs preserved "
                    f"at {tmp_dir}"
                )
