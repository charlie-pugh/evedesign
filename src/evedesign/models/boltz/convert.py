"""
Translation layer between evedesign data structures and Boltz-2 inputs/outputs.

This module is the only place that knows about Boltz-2 internals.
boltzfold.py calls these functions without importing boltz directly.

NOTE: Template conditioning (Entity.structures → YAML templates)
is not yet implemented. Structures are ignored with a warning.
"""

from pathlib import Path
from typing import Literal

import yaml
import json
from loguru import logger

from evedesign.system import Entity, EntityInstance, System, SystemInstance, StructureChainMap, Structure
from evedesign.structure import StructureFile

# 1. evedesign Entity --> to Boltz-2 YAML

# Chain ID generation
# Boltz-2 uses chain IDs to identify entities in the system. We need to generate unique chain IDs for each entity instance. 
# The convention is to use uppercase letters (A-Z) for the first 26 entities, then double letters (AA, AB, AC, ...) for additional entities. 
# This function generates the appropriate chain ID based on the entity's position in the system.
def _get_chain_id(chain_num: int) -> str:
    """Generate chain ID: A-Z, then AA, AB, AC, ..."""
    if chain_num < 26:
        return chr(65 + chain_num)
    else:
        first = chr(65 + (chain_num - 26) // 26)
        second = chr(65 + (chain_num - 26) % 26)
        return first + second


def _get_chain_ids(system: System) -> list[str]:
    """
    Generate one chain ID per chain in the system.

    IDs follow the sequence A-Z, then AA, AB, AC, ...
    Each entity consumes max(entity.copies, 1) IDs.
    """
    total = sum(
        max(e.copies, 1) if e.copies is not None else 1
        for e in system
    )
    return [_get_chain_id(i) for i in range(total)]


def _chain_to_entity_map(system: System) -> dict[str, int]:
    """Maps each Boltz-2 chain ID back to its evedesign entity index."""
    chain_ids = _get_chain_ids(system)
    result: dict[str, int] = {}
    pointer = 0
    for entity_idx, entity in enumerate(system):
        copies = entity.copies if entity.copies is not None else 1
        for chain_id in chain_ids[pointer:pointer + copies]:
            result[chain_id] = entity_idx
        pointer += copies
    return result


# A3M writer
# a3m format is a simple extension of FASTA that allows for insertions in the MSA.
def _write_a3m(
    entity: Entity,
    entity_instance: EntityInstance,
    output_path: Path,
) -> Path:
    """Write an A3M file with the query sequence followed by MSA hits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Query sequence first
        header = entity.id or "query"
        f.write(f">{header}\n{''.join(entity_instance.rep)}\n")

        # MSA hits
        for seq in entity.sequences.seqs:
            f.write(f">{seq.id_ or 'seq'}\n{seq.seq}\n")

    return output_path


def _write_csv(
    entity: Entity,
    entity_instance: EntityInstance,
    output_path: Path,
) -> Path:
    """
    Write a Boltz-2 compatible CSV MSA file preserving
    pairing keys from Sequence.key.

    Format matches what Boltz-2's own server path produces:
    - Header: key,sequence
    - Query sequence first with key=0
    - Paired sequences with integer key (from "pair-{i}")
    - Unpaired sequences with key=-1

    This format is required for paired MSAs in multi-chain
    complexes. Boltz-2's CSV parser uses the key column as
    taxonomy_id to match paired rows across chains.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["key,sequence"]
    # Query sequence is always the first row
    rows.append(f"0,{''.join(entity_instance.rep)}")
    for seq in entity.sequences.seqs:
        if seq.key is not None and seq.key.startswith("pair-"):
            key = seq.key[len("pair-"):]
        else:
            key = "-1"
        rows.append(f"{key},{seq.seq}")
    output_path.write_text("\n".join(rows) + "\n")
    return output_path


def _resolve_msa_field(
    entity: Entity,
    entity_instance: EntityInstance,
    chain_id: str,
    yaml_path: Path,
    use_msa: bool,
) -> str | None:
    """
    Decide the msa field value for a single entity in the Boltz-2 YAML.

    Returns an absolute path string when a local MSA file was written
    (CSV when pairing keys are present, A3M otherwise), or "empty" when
    no MSA is available.
    """
    # Warn about unsupported template conditioning
    if entity.structures is not None and len(entity.structures) > 0:
        logger.warning(
            f"Entity '{entity.id}': structures are present but "
            f"template conditioning is not yet implemented — ignoring."
        )

    if (
        use_msa
        and entity.sequences is not None
        and len(entity.sequences.seqs) > 0
    ):
        # Use CSV format when pairing keys are present
        # (multi-chain complexes with paired MSA).
        # Use A3M for single-sequence or unpaired MSA.
        has_pairing = any(
            s.key is not None and s.key.startswith("pair-")
            for s in entity.sequences.seqs
        )
        if has_pairing:
            msa_path = _write_csv(
                entity, entity_instance,
                yaml_path.parent / "msa" / f"{chain_id}.csv",
            )
        else:
            msa_path = _write_a3m(
                entity, entity_instance,
                yaml_path.parent / "msa" / f"{chain_id}.a3m",
            )
        return str(msa_path.resolve())

    return "empty"


def system_instance_to_yaml(
    system: System,
    instance: SystemInstance,
    output_path: Path,
    use_msa: bool = True,
) -> Path:
    """
    Convert an evedesign System + SystemInstance into a Boltz-2 input YAML.

    Parameters
    ----------
    system
        The evedesign System (defines entities, copies, MSA, structures).
    instance
        A specific SystemInstance whose sequences will be written.
    output_path
        Where to write the YAML file.
    use_msa
        If True and MSA data is available on the entity, write a local
        MSA file (CSV when pairing keys are present, A3M otherwise) and
        reference it. If False, set msa to "empty".

    Returns
    -------
    Path to the written YAML file.

    
    Homo-oligomer mapping:
        evedesign represents a homodimer as a single Entity with copies=2.
        There is one EntityInstance per entity regardless of copy count.
        Boltz-2 expects a list of chain IDs for homo-oligomers:
            copies=1  →  id: "A"        (scalar, monomer)
            copies=2  →  id: ["A","B"]  (list, homodimer)
        Boltz-2 then creates separate chains sharing the same entity_id
        and incrementing sym_id, so it knows they are symmetry-related.
    """
    chain_ids = _get_chain_ids(system)
    pointer = 0
    sequences = []

    for entity, entity_instance in zip(system, instance):
        copies = entity.copies if entity.copies is not None else 1
        first_chain = chain_ids[pointer]
        id_field = first_chain if copies == 1 else chain_ids[pointer:pointer + copies]

        seq = "".join(entity_instance.rep)

        msa = _resolve_msa_field(
            entity, entity_instance, first_chain, output_path,
            use_msa=use_msa,
        )

        entry: dict = {"id": id_field, "sequence": seq}
        if msa is not None:
            entry["msa"] = msa

        sequences.append({"protein": entry})
        pointer += copies

    data = {"version": 1, "sequences": sequences}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return output_path


def prediction_to_instance(
    record_id: str,
    predictions_dir: Path,
    system: System,
    instance: SystemInstance,
    chain_to_entity: dict[str, int],
    score_attribute: Literal[
        "iptm", "ptm", "confidence_score", "complex_plddt"
    ] = "confidence_score",
    confidence_attribute: Literal[
        "iptm", "ptm", "confidence_score", "complex_plddt"
    ] = "complex_plddt",
) -> SystemInstance:
    """
    Parse BoltzWriter output files for one record into a
    SystemInstance with populated structures and scores.

    Only the best-ranked model (model_0, highest confidence)
    is parsed and stored in EntityInstance.models.

    NOTE: Boltz-2 always numbers residues from 1 internally.
    Output residue numbering is remapped to match each
    entity's first_index before populating EntityInstance.models.

    NOTE: Support for returning all diffusion samples is not
    yet implemented. When added, the CIF parsing logic should
    be extracted into a separate _parse_cif_to_chain_map()
    helper and iterated over all model_{i}.cif files.
    """
    # Locate output files for this record
    record_dir = predictions_dir / record_id
    if not record_dir.exists():
        raise ValueError(
            f"No prediction output found for record "
            f"'{record_id}' in {predictions_dir}"
        )

    cif_files = sorted(record_dir.glob("*.cif"))
    json_files = sorted(record_dir.glob("confidence_*.json"))

    if not cif_files:
        raise ValueError(
            f"No .cif files found for record "
            f"'{record_id}' in {record_dir}"
        )

    # Load all per-rank confidence JSONs
    all_confidence: dict[str, dict] = {}
    for rank_idx, json_path in enumerate(json_files):
        rank_key = f"model_{rank_idx}"
        all_confidence[rank_key] = json.loads(
            json_path.read_text()
        )

    # The score and confidence on SystemInstance are
    # taken from model_0 (best ranked by Boltz-2)
    best_confidence = all_confidence.get("model_0", {})

    # SystemInstance.score holds the score_attribute
    # value (e.g. "confidence_score") of the best-ranked
    # diffusion sample (model_0). Per-sample scores for
    # all ranks are stored in metadata["scores"].
    score = best_confidence.get(score_attribute, None)
    if score is None and best_confidence:
        raise ValueError(
            f"'{score_attribute}' not found in Boltz-2 "
            f"confidence output. "
            f"Available keys: {list(best_confidence.keys())}"
        )

    # SystemInstance.confidence holds the
    # confidence_attribute value (e.g. "complex_plddt")
    # of the best-ranked diffusion sample (model_0).
    confidence_val = best_confidence.get(confidence_attribute, None)
    if confidence_val is None and best_confidence:
        raise ValueError(
            f"'{confidence_attribute}' not found in Boltz-2 "
            f"confidence output. "
            f"Available keys: {list(best_confidence.keys())}"
        )

    entity_models: dict[int, StructureChainMap] = {}

    for rank_idx, cif_path in enumerate(cif_files):
        rank_key = f"model_{rank_idx}"
        entity_chains: dict[int, list[Structure]] = {}

        sf = StructureFile(str(cif_path), format="cif")
        full_structure = sf.get_model()

        for chain_id in full_structure.chains():
            if chain_id not in chain_to_entity:
                logger.warning(
                    f"Chain '{chain_id}' not in "
                    f"chain_to_entity mapping — skipping"
                )
                continue
            entity_idx = chain_to_entity[chain_id]
            entity = system[entity_idx]

            chain_structure = full_structure.get_chain(chain_id)

            n = len(entity.rep)
            mapping = {
                i: i + entity.first_index - 1
                for i in range(1, n + 1)
            }
            remapped = chain_structure.remap(mapping)

            if entity_idx not in entity_chains:
                entity_chains[entity_idx] = []
            entity_chains[entity_idx].append(remapped)

        for entity_idx, chains in entity_chains.items():
            if entity_idx not in entity_models:
                entity_models[entity_idx] = {}
            if len(chains) == 1:
                entity_models[entity_idx][rank_key] = chains[0]
            else:
                entity_models[entity_idx][rank_key] = chains

    # Build output EntityInstance objects (shallow copy, add structures)
    new_entity_instances = []
    for i, entity_instance in enumerate(instance):
        new_ei = EntityInstance(
            rep=entity_instance.rep.copy(),
            embedding=entity_instance.embedding,
            models=entity_models.get(i, None),
        )
        new_entity_instances.append(new_ei)

    metadata = {
        "scores": all_confidence,
    }

    return SystemInstance(
        new_entity_instances,
        score=score,
        confidence=confidence_val,
        metadata=metadata,
    )
