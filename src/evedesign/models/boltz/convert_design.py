"""
Translation layer between evedesign data structures
and BoltzGen inputs/outputs.

This module converts evedesign System/SystemInstance
objects into BoltzGen design specification YAMLs and
parses BoltzGen's generated structures back into
SystemInstance objects.

NOTE: This module is the only place that knows about
BoltzGen file conventions. boltzgen.py calls these
functions without importing boltzgen directly.
"""

from evedesign.models.boltz.chains import (
    _chain_to_entity_map,
    _get_chain_id,
    _get_chain_ids,
)
from evedesign.system import (
    Entity,
)


# ─── Entity classification ────────────────────────────


def _is_design_entity(entity: Entity) -> bool:
    """
    Determine whether an entity should be designed
    by BoltzGen.

    Returns True when:
    - rep is None (no sequence specified), OR
    - min_length / max_length are specified (length
      range for design)
    """
    if entity.rep is None:
        return True
    if (
        getattr(entity, "min_length", None) is not None
        or getattr(entity, "max_length", None) is not None
    ):
        return True
    return False


def _entity_to_sequence_spec(entity: Entity) -> str:
    """
    Convert entity length info into BoltzGen sequence
    spec string.

    BoltzGen YAML accepts three formats for the
    sequence field of a designable entity:
    - "80..140" for variable length (range)
    - "80" for fixed length (no specific sequence)
    - Actual sequence string (for fixed entities,
      handled elsewhere)

    Resolution order:
    1. min_length and max_length both set → "min..max"
    2. min_length only → "min"
    3. max_length only → "max"
    4. rep is set → len(rep) (fixed length matching rep)
    5. Fallback → "60..120"
    """
    if (
        entity.min_length is not None
        and entity.max_length is not None
    ):
        return f"{entity.min_length}..{entity.max_length}"
    if entity.min_length is not None:
        return str(entity.min_length)
    if entity.max_length is not None:
        return str(entity.max_length)
    if entity.rep is not None:
        return str(len(entity.rep))
    return "60..120"


# ─── YAML entity emitters ─────────────────────────


def _emit_design_entity(
    entity: Entity,
    chain_ids: list[str],
    pointer: int,
) -> tuple[dict, int]:
    """
    Emit a YAML entity dict for a designable entity.

    For proteins, emits:
        {"protein": {"id": "A", "sequence": "60..100", ...}}

    For homo-oligomers (copies > 1), id becomes a list:
        {"protein": {"id": ["A", "B"], "sequence": ...}}

    For ligands, emits:
        {"ligand": {"id": "A", "smiles": "..."}} or
        {"ligand": {"id": "A", "ccd": "..."}}

    Returns the YAML dict and the updated chain ID pointer.
    """
    copies = (
        entity.copies if entity.copies is not None
        else 1
    )
    id_field = (
        chain_ids[pointer]
        if copies == 1
        else chain_ids[pointer:pointer + copies]
    )

    # Ligand entity
    if entity.type == "ligand":
        if (
            entity.ligand_rep_type == "smiles"
            and entity.rep is not None
        ):
            entry = {
                "ligand": {
                    "id": id_field,
                    "smiles": "".join(entity.rep)
                }
            }
        elif (
            entity.ligand_rep_type == "ccd"
            and entity.rep is not None
        ):
            entry = {
                "ligand": {
                    "id": id_field,
                    "ccd": "".join(entity.rep)
                }
            }
        else:
            # Designable ligand with no rep — default to UNK
            entry = {
                "ligand": {"id": id_field, "ccd": "UNK"}
            }
        return entry, pointer + copies

    # Protein / DNA / RNA — use type if valid, else protein
    seq_spec = _entity_to_sequence_spec(entity)
    entity_type = (
        entity.type
        if entity.type in ("protein", "dna", "rna")
        else "protein"
    )

    entry_inner: dict = {
        "id": id_field,
        "sequence": seq_spec,
    }

    entry = {entity_type: entry_inner}
    return entry, pointer + copies
