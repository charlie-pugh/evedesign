"""
Structure <> BoltzFold format conversion
"""


from __future__ import annotations

import string
from pathlib import Path

import re

import numpy as np
import yaml
import biotite.structure as struc

from evedesign.structure import Structure
from evedesign.system import System, SystemInstance, EntityInstance


def _get_chain_ids(n: int) -> list[str]:
    """
    Generate chain IDs for n entities.

    Uses A, B, C, ... Z, then AA, AB, etc.
    Matches the pattern used by LigandMPNN wrapper.
    """
    ids = list(string.ascii_uppercase)
    if n <= len(ids):
        return ids[:n]
    # Extend with two-letter IDs if needed
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            ids.append(a + b)
            if len(ids) >= n:
                return ids[:n]
    raise ValueError(f"Too many entities: {n}")

def system_instance_to_yaml(
    system: System,
    instance: SystemInstance,
    output_path: Path,
    use_msa_server: bool = False,
) -> Path:
    """
    Write a Boltz-2 YAML input file from a System and SystemInstance.

    Parameters
    ----------
    system
        The System defining entity types, IDs, and constraints.
    instance
        The SystemInstance containing sequences (in EntityInstance.rep).
    output_path
        Path to write the YAML file.
    use_msa_server
        If False, sets msa: empty for single-sequence mode.
        If True, omits msa field so Boltz-2 generates MSAs.

    Returns
    -------
    Path
        The written YAML file path.
    """
    chain_ids = _get_chain_ids(len(system))
    sequences = []

    for entity_idx, (_, entity_instance) in enumerate(
        zip(system, instance)
    ):
        # Get sequence string from numpy array of single chars
        seq = "".join(entity_instance.rep)

        entry = {
            "id": chain_ids[entity_idx],
            "sequence": seq,
        }

        # MSA: single-sequence mode unless server is enabled
        if not use_msa_server:
            entry["msa"] = "empty"

        # Wrap in entity type
        # For now only protein is supported (enforced by can_model)
        sequences.append({"protein": entry})

    boltz_yaml = {
        "version": 1,
        "sequences": sequences,
    }

    # Write YAML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(boltz_yaml, f, default_flow_style=False, sort_keys=False)

    return output_path


def _boltz_structure_to_atom_array(
    boltz_structure,
    coords: np.ndarray,
) -> struc.AtomArray:
    """
    Build an AtomArray from a Boltz StructureV2 and predicted coordinates.

    Parameters
    ----------
    boltz_structure
        Boltz StructureV2 with atom/residue/chain metadata.
    coords
        Predicted coordinates, shape (n_atoms, 3), already unpadded.
    """
    from boltz.data import const

    atoms = boltz_structure.atoms
    residues = boltz_structure.residues
    chains = boltz_structure.chains

    n_atoms = len(atoms)
    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = coords

    chain_ids = np.empty(n_atoms, dtype="<U5")
    res_ids = np.zeros(n_atoms, dtype=int)
    res_names = np.empty(n_atoms, dtype="<U5")
    atom_names = np.empty(n_atoms, dtype="U4")
    elements = np.empty(n_atoms, dtype="U2")
    ins_codes = np.full(n_atoms, "", dtype="U1")

    for chain in chains:
        chain_name = str(chain["name"])
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]

        for residue in residues[res_start:res_end]:
            res_name = str(residue["name"])
            residue_index = residue["res_idx"] + 1  # 1-based
            atom_start = residue["atom_idx"]
            atom_end = atom_start + residue["atom_num"]

            for i in range(atom_start, atom_end):
                if not atoms[i]["is_present"]:
                    continue

                atom_name = str(atoms[i]["name"])
                chain_ids[i] = chain_name
                res_ids[i] = residue_index
                res_names[i] = res_name
                atom_names[i] = atom_name

                # Element inference (same logic as boltz mmcif writer)
                atom_key = re.sub(r"\d", "", atom_name)
                if atom_key in const.ambiguous_atoms:
                    amb = const.ambiguous_atoms[atom_key]
                    if isinstance(amb, str):
                        element = amb
                    elif res_name in amb:
                        element = amb[res_name]
                    else:
                        element = amb["*"]
                else:
                    element = atom_key[0]
                elements[i] = element.upper()

    atom_array.chain_id = chain_ids
    atom_array.res_id = res_ids
    atom_array.res_name = res_names
    atom_array.atom_name = atom_names
    atom_array.element = elements
    atom_array.ins_code = ins_codes

    # Filter to present atoms only
    present = atoms["is_present"]
    if not present.all():
        atom_array = atom_array[present]

    return atom_array


def prediction_to_instance(
    pred_dict: dict,
    batch: dict,
    structures_dir,
    system: System,
    instance: SystemInstance,
    model_idx: int = 0,
) -> SystemInstance:
    """
    Convert Boltz-2 prediction tensors directly into a SystemInstance.

    Skips the disk round-trip (CIF write → read) by building biotite
    AtomArrays directly from predicted coordinates and structure metadata.

    Parameters
    ----------
    pred_dict
        Output from Boltz2.predict_step().
    batch
        The batch dict from the dataloader (contains record, masks).
    structures_dir
        Path to processed/structures/ directory with .npz files.
    system
        The System defining entity types.
    instance
        The input SystemInstance (sequences preserved in output).
    model_idx
        Which diffusion sample to use (0 = best ranked).
    """
    from boltz.data.types import StructureV2
    import torch

    records = batch["record"]
    if len(records) != 1:
        raise ValueError(
            f"prediction_to_instance expects batch_size=1, got {len(records)} records"
        )
    record = records[0]
    pad_mask = pred_dict["masks"]

    # Use ranking if available
    if "confidence_score" in pred_dict:
        argsort = torch.argsort(
            pred_dict["confidence_score"], descending=True
        )
        ranked_idx = argsort[model_idx].item()
    else:
        ranked_idx = model_idx

    # Extract coordinates
    coords = pred_dict["coords"]
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    model_coords = coords[ranked_idx]
    coords_unpad = model_coords[pad_mask.bool()].cpu().numpy()

    # Load the processed structure NPZ for atom metadata
    structure_path = Path(structures_dir) / f"{record.id}.npz"
    boltz_structure = StructureV2.load(structure_path)
    boltz_structure = boltz_structure.remove_invalid_chains()

    # Build biotite AtomArray directly
    atom_array = _boltz_structure_to_atom_array(boltz_structure, coords_unpad)
    full_structure = Structure(atom_array)

    # Split into per-entity chain structures
    chain_ids = _get_chain_ids(len(system))
    entity_structures = {}
    for chain_id in chain_ids:
        if chain_id in full_structure.chains():
            entity_structures[chain_id] = full_structure.get_chain(chain_id)

    # Extract confidence from tensors directly
    confidence_dict = {}
    for key in [
        "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
    ]:
        if key in pred_dict:
            confidence_dict[key] = pred_dict[key][ranked_idx].item()

    if "pair_chains_iptm" in pred_dict:
        confidence_dict["chains_ptm"] = {
            idx: pred_dict["pair_chains_iptm"][idx][idx][ranked_idx].item()
            for idx in pred_dict["pair_chains_iptm"]
        }
        confidence_dict["pair_chains_iptm"] = {
            idx1: {
                idx2: pred_dict["pair_chains_iptm"][idx1][idx2][ranked_idx].item()
                for idx2 in pred_dict["pair_chains_iptm"][idx1]
            }
            for idx1 in pred_dict["pair_chains_iptm"]
        }

    complex_plddt = confidence_dict.get("complex_plddt", 0.0)

    plddt_array = None
    if "plddt" in pred_dict:
        plddt_array = pred_dict["plddt"][ranked_idx].cpu().numpy()

    # Build new SystemInstance
    new_entity_instances = []
    for entity_idx, (entity, entity_instance) in enumerate(
        zip(system, instance)
    ):
        chain_id = chain_ids[entity_idx]
        models = (
            {chain_id: entity_structures[chain_id]}
            if chain_id in entity_structures
            else None
        )
        new_entity_instances.append(
            EntityInstance(
                rep=entity_instance.rep,
                embedding=entity_instance.embedding,
                models=models,
            )
        )

    return SystemInstance(
        new_entity_instances,
        score=complex_plddt,
        confidence=complex_plddt,
        metadata={
            "boltz_confidence": confidence_dict,
            "plddt": plddt_array,
        },
    )


