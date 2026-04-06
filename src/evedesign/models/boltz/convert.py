"""
Translation layer between evedesign data structures and Boltz-2 inputs/outputs.

This module is the only place that knows about Boltz-2 internals.
boltzfold.py calls these functions without importing boltz directly.

NOTE: Template conditioning (Entity.structures → YAML templates)
is not yet implemented. Structures are ignored with a warning.
"""

from pathlib import Path

import yaml
from loguru import logger

from evedesign.system import Entity, EntityInstance, System, SystemInstance

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


def _resolve_msa_field(
    entity: Entity,
    entity_instance: EntityInstance,
    chain_id: str,
    yaml_path: Path,
    use_msa: bool,
    use_msa_server: bool,
) -> str | None:
    """
    Decide the msa field value for a single entity in the Boltz-2 YAML.

    Returns None when the field should be omitted (server mode),
    an absolute path string when a local A3M was written,
    or "empty" when no MSA is available.
    """
    # Warn about unsupported template conditioning
    if entity.structures is not None and len(entity.structures) > 0:
        logger.warning(
            f"Entity '{entity.id}': structures are present but "
            f"template conditioning is not yet implemented — ignoring."
        )

    # Case 1: let Boltz-2's own MSA server handle it
    if use_msa_server:
        return None

    # Case 2: we have local MSA data — write it as A3M
    if (
        use_msa
        and entity.sequences is not None
        and len(entity.sequences.seqs) > 0
    ):
        a3m_path = _write_a3m(
            entity, entity_instance,
            yaml_path.parent / "msa" / f"{chain_id}.a3m",
        )
        return str(a3m_path.resolve())

    # Case 3: no MSA available or not requested
    return "empty"


def system_instance_to_yaml(
    system: System,
    instance: SystemInstance,
    output_path: Path,
    use_msa: bool = True,
    use_msa_server: bool = False,
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
        If True and MSA data is available on the entity, write an .a3m
        file and reference it. If False, set msa to "empty".
    use_msa_server
        If True, omit the msa field entirely so Boltz-2 will query its
        own MSA server at runtime.

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
            use_msa=use_msa, use_msa_server=use_msa_server,
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
