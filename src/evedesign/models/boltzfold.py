"""
evedesign/models/boltzfold.py

Boltz-2 model wrapper: Transformer (folding) + Scorer (confidence).

Folds sequences into 3D structures using the Boltz-2 diffusion model.
Returns per-instance Structure objects and confidence metrics (pLDDT, pTM, ipTM).

Integration strategy: in-process via `boltz` Python package.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Self, Sequence

import torch
from loguru import logger

from evedesign.model import BaseModel, Scorer, Transformer
from evedesign.system import System, SystemInstance
from evedesign.utils import model_param_context
from evedesign.types import DeviceType, StatusCallback, BatchSize

from evedesign.models.boltz.convert import (
    system_instance_to_yaml,
    prediction_to_instance,
)

try:
    from boltz.main import (
        Boltz2DiffusionParams,
        PairformerArgsV2,
        MSAModuleArgs,
        BoltzSteeringParams,
        download_boltz2,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class BoltzFoldTransformer(BaseModel, Transformer, Scorer):
    """
    Wrapper class around Boltz-2 structure prediction / diffusion model.
    """

    available = IMPORT_AVAILABLE
    name: str = "BoltzFold"
    citations: list[str] = []

    # Core properties
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
        # Boltz-2 folding parameters
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        use_msa_server: bool = False,
    ):
        if not self.available:
            raise ValueError(
                "boltz package could not be imported. "
                "Install with: pip install boltz"
            )

        self.model_dir_path = Path(
            model_dir_path
        ) if model_dir_path is not None else None
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = True
        self.device = device
        self.batch_size = batch_size

        # Boltz-2 specific parameters
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.recycling_steps = recycling_steps
        self.use_msa_server = use_msa_server

        # Framework state (set during build)
        self._system: System | None = None
        self.model = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        """
        Check if this model can handle the given system.

        Currently supports: protein entities only (multi-chain OK).
        DNA/RNA/ligand support planned for future versions.
        """
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) == 0:
            return False, "System must have at least one entity"

        for i, entity in enumerate(system):
            if entity.type != "protein":
                return False, (
                    f"Entity {i} has type '{entity.type}'. "
                    f"Currently only protein entities are supported. "
                    f"DNA/RNA/ligand support coming soon."
                )
            if not entity.defined_sequence():
                return False, f"Entity {i} must have a defined sequence"

        return True, ""

    def _load_model(self) -> None:
        """TODO: CHANGE TO BOLTZFOLD framework
        Load Boltz-2 model weights onto device.
        """
        if self.model is not None:
            return

        # Download model weights if not cached
        cache = Path("~/.boltz").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)
        checkpoint = cache / "boltz2_conf.ckpt"

        # Model parameters (Boltz-2 defaults from main.py)
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5

        predict_args = {
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "diffusion_samples": self.diffusion_samples,
            "max_parallel_samples": None,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        self.model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels="cuda" in str(self.device),
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )),
            steering_args=asdict(BoltzSteeringParams()),
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Boltz-2 model loaded from {checkpoint}")

    def _release_cache(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self) -> None:
        self.model = None
        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        """
        self.can_model_or_raise(system, data)
        self._system = system
        return self
    
    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Fold sequences into 3D structures.

        Parameters
        ----------
        entity
            If None, fold all entities as a complex.
            If specified, fold only that entity in isolation and
            merge the result back into the full instance.
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # Determine which system/instances to fold
        if entity is not None:
            if entity < 0 or entity >= len(self._system):
                raise ValueError(
                    f"Entity index {entity} out of range "
                    f"(system has {len(self._system)} entities)"
                )
            fold_system = System([self._system[entity]])
            fold_instances = [
                SystemInstance([inst[entity]]) for inst in instances
            ]
        else:
            fold_system = self._system
            fold_instances = instances

        # Resolve cache directory (same as _load_model)
        cache = Path("~/.boltz").expanduser()
        mol_dir = cache / "mols"
        ccd_path = cache / "ccd.pkl"

        # Create temp working directory
        work_dir = Path(tempfile.mkdtemp(prefix="boltzfold_"))
        yaml_dir = work_dir / "inputs"
        yaml_dir.mkdir()

        try:
            # Step 1: Write YAML files for each instance
            yaml_paths = []
            record_ids = []
            for i, instance in enumerate(fold_instances):
                record_id = f"fold_{i}"
                yaml_path = yaml_dir / f"{record_id}.yaml"
                system_instance_to_yaml(
                    fold_system,
                    instance,
                    yaml_path,
                    use_msa_server=self.use_msa_server,
                )
                yaml_paths.append(yaml_path)
                record_ids.append(record_id)

            # Step 2: Process inputs (YAML → tokenized structures + MSA)
            # Note: process_inputs writes manifest to disk but doesn't return it
            process_inputs(
                data=yaml_paths,
                out_dir=work_dir,
                ccd_path=ccd_path,
                mol_dir=mol_dir,
                use_msa_server=self.use_msa_server,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
                boltz2=True,
            )
            processed_dir = work_dir / "processed"
            manifest = Manifest.load(processed_dir / "manifest.json")

            if not manifest.records:
                logger.warning("No records to process after process_inputs")
                return [inst.copy() for inst in instances]

            # Step 3: Create data module and get dataloader
            data_module = Boltz2InferenceDataModule(
                manifest=manifest,
                target_dir=processed_dir / "structures",
                msa_dir=processed_dir / "msa",
                mol_dir=mol_dir,
                num_workers=0,
                constraints_dir=(
                    (processed_dir / "constraints")
                    if (processed_dir / "constraints").exists()
                    else None
                ),
                template_dir=(
                    (processed_dir / "templates")
                    if (processed_dir / "templates").exists()
                    else None
                ),
                extra_mols_dir=(
                    (processed_dir / "mols")
                    if (processed_dir / "mols").exists()
                    else None
                ),
            )
            data_module.setup(stage="predict")
            dataloader = data_module.predict_dataloader()

            # Map record_id → instance index for reassembly
            record_id_to_idx = {rid: i for i, rid in enumerate(record_ids)}
            fold_results = [None] * len(fold_instances)
            structures_dir = processed_dir / "structures"

            with model_param_context(
                self._load_model,
                self._delete_model,
                self.keep_model_after_pred,
            ):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        # Move tensors to device
                        batch_device = {
                            k: v.to(self.device)
                            if isinstance(v, torch.Tensor)
                            else v
                            for k, v in batch.items()
                        }

                        # Run prediction
                        pred_dict = self.model.predict_step(
                            batch_device, batch_idx=batch_idx
                        )

                        if pred_dict.get("exception", False):
                            records = batch["record"]
                            for record in records:
                                idx = record_id_to_idx.get(record.id)
                                if idx is not None:
                                    logger.warning(
                                        f"Prediction failed for instance {idx}"
                                    )
                                    fold_results[idx] = fold_instances[idx].copy()
                            continue

                        # Convert tensors directly to SystemInstance
                        record = batch["record"][0]
                        idx = record_id_to_idx.get(record.id)
                        if idx is not None:
                            fold_results[idx] = prediction_to_instance(
                                pred_dict=pred_dict,
                                batch=batch,
                                structures_dir=structures_dir,
                                system=fold_system,
                                instance=fold_instances[idx],
                            )

            # Fill any missing results
            for i in range(len(fold_results)):
                if fold_results[i] is None:
                    logger.warning(
                        f"No prediction for instance {i}, returning copy"
                    )
                    fold_results[i] = fold_instances[i].copy()

        finally:
            # Clean up temp directory
            shutil.rmtree(work_dir, ignore_errors=True)

        # Merge back into full instances if entity was specified
        if entity is not None:
            results = []
            for orig_inst, fold_inst in zip(instances, fold_results):
                entity_instances = list(orig_inst)
                entity_instances[entity] = fold_inst[0]
                results.append(SystemInstance(
                    entity_instances,
                    score=fold_inst.score,
                    confidence=fold_inst.confidence,
                    metadata=fold_inst.metadata,
                ))
            return results

        return fold_results

    def score(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        raise NotImplementedError

    def _instance_has_fold(self, instance: SystemInstance) -> bool:
        """
        Check if an instance already has folded structures.

        An instance is considered "folded" if at least one EntityInstance
        has a non-None .models field (which stores the Structure objects).
        """
        return any(ei.models is not None for ei in instance)