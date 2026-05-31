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
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from evedesign.models.boltz.chains import (
    _chain_to_entity_map,
    _get_chain_id,
    _get_chain_ids,
)
from evedesign.system import (
    Entity,
    EntityInstance,
    System,
    SystemInstance,
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
    5. Fallback → "80..140" (matches BoltzGen's vanilla
       binder default; emits a warning when triggered)
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
    logger.warning(
        "Designable entity has no min_length, "
        "max_length, or rep — defaulting to "
        "BoltzGen's vanilla binder range '80..140'. "
        "Set min_length/max_length on the Entity "
        "to suppress this warning and control the "
        "design length range explicitly."
    )
    return "80..140"


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


def _emit_context_entity(
    entity: Entity,
    entity_idx: int,
    chain_ids: list[str],
    pointer: int,
    tmp_dir: Path,
) -> tuple[dict, int]:
    """
    Emit a YAML entity dict for a context entity
    (fixed structure target).

    Writes the entity's structure to a CIF file in
    tmp_dir/structures/ and references it from the YAML.

    Falls back to a protein sequence-only entry if
    no structure is available.

    Returns the YAML dict and the updated chain ID
    pointer.
    """
    copies = (
        entity.copies if entity.copies is not None
        else 1
    )

    # Fallback: no structure attached → emit as
    # sequence-only protein entry
    if (
        entity.structures is None
        or len(entity.structures) == 0
    ):
        id_field = (
            chain_ids[pointer]
            if copies == 1
            else chain_ids[pointer:pointer + copies]
        )
        seq = (
            "".join(entity.rep)
            if entity.rep is not None
            else "A" * 10
        )
        entry = {
            "protein": {"id": id_field, "sequence": seq}
        }
        return entry, pointer + copies

    # Write the structure to a CIF in tmp_dir/structures/
    cif_path = (
        tmp_dir / "structures" / f"entity_{entity_idx}.cif"
    )
    cif_path.parent.mkdir(parents=True, exist_ok=True)

    first_key = next(iter(entity.structures))
    model = entity.structures[first_key]
    if isinstance(model, list):
        model = model[0]
    model.to_file(str(cif_path), format="cif")

    # Build the file entry — currently includes only
    # the chain for entity_idx (no binding-site or
    # interaction constraints yet — those will be
    # added in a later prompt)
    chain_id = chain_ids[pointer]
    file_entry: dict = {
        "path": str(cif_path.resolve()),
        "include": [{"chain": {"id": chain_id}}],
    }

    entry = {"file": file_entry}
    return entry, pointer + copies


# ─── Top-level System → YAML ─────────────────────


def system_to_boltzgen_yaml(
    system: System,
    output_path: Path,
) -> Path:
    """
    Convert an evedesign System into a BoltzGen design
    specification YAML file.

    Each entity in the system is classified as either:
    - Designable (rep=None or min_length/max_length set)
      → emitted as a sequence/length spec via
      _emit_design_entity
    - Context (has a fixed structure attached)
      → emitted as a file: reference via
      _emit_context_entity (the structure CIF is
      written to output_path.parent / structures/)

    Returns the output_path for chaining.
    """
    tmp_dir = output_path.parent
    chain_ids = _get_chain_ids(system)
    pointer = 0

    entities_list: list[dict] = []

    n_design = sum(
        1 for e in system if _is_design_entity(e)
    )
    n_context = len(system) - n_design
    logger.info(
        f"System has {len(system)} entities: "
        f"{n_design} designable, {n_context} context"
    )

    for entity_idx, entity in enumerate(system):
        if _is_design_entity(entity):
            entry, pointer = _emit_design_entity(
                entity, chain_ids, pointer,
            )
        else:
            entry, pointer = _emit_context_entity(
                entity,
                entity_idx,
                chain_ids,
                pointer,
                tmp_dir,
            )
        entities_list.append(entry)

    yaml_data: dict = {"entities": entities_list}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    return output_path


# ─── Output parsing ─────────────────────────────────


def _parse_single_design(
    cif_path: Path,
    system: System,
    chain_to_entity: dict[str, int],
    metrics_row: dict | None = None,
) -> SystemInstance:
    """
    Parse a single BoltzGen output CIF into a
    SystemInstance.

    Each chain in the CIF is matched against
    chain_to_entity to find which evedesign entity
    it belongs to. Sequence and structure are extracted
    for designed entities; original entity.rep is kept
    for context entities that have no chain in the CIF.

    metrics_row (optional) — a dict from the BoltzGen
    metrics CSV. Used to populate instance.metadata
    and instance.score (using iptm, falling back to
    ptm, falling back to complex_plddt).
    """
    from evedesign.structure import Structure, StructureFile

    sf = StructureFile(str(cif_path), format="cif")
    full_structure = sf.get_model()

    # Group chains by entity index based on chain_to_entity
    entity_chains: dict[int, list[Structure]] = {}
    for chain_id in full_structure.chains():
        if chain_id not in chain_to_entity:
            logger.warning(
                f"Chain '{chain_id}' in {cif_path.name} "
                f"not in chain_to_entity mapping — skipping"
            )
            continue
        entity_idx = chain_to_entity[chain_id]
        chain_structure = full_structure.get_chain(chain_id)
        entity_chains.setdefault(entity_idx, []).append(
            chain_structure
        )

    # Build EntityInstance for each entity
    entity_instances = []
    for entity_idx, entity in enumerate(system):
        chains = entity_chains.get(entity_idx)

        if chains is None or len(chains) == 0:
            # Entity missing from CIF — preserve its
            # original rep
            rep = (
                entity.rep.copy()
                if entity.rep is not None
                else None
            )
            entity_instances.append(EntityInstance(rep=rep))
            continue

        # Extract sequence from the primary chain
        primary_chain = chains[0]
        res_df = primary_chain.res_df()
        if "res_name_oneletter" in res_df.columns:
            seq_array = np.array(
                list(res_df["res_name_oneletter"].fillna("X")),
                dtype="U1",
            )
        else:
            seq_array = np.array(
                ["X"] * len(res_df), dtype="U1"
            )

        # Build the models dict — single chain or
        # homo-oligomer list
        if len(chains) == 1:
            models = {"model_0": chains[0]}
        else:
            models = {"model_0": chains}

        entity_instances.append(
            EntityInstance(rep=seq_array, models=models)
        )

    instance = SystemInstance(entity_instances)

    # Attach metadata
    design_id = cif_path.stem  # e.g. "design_spec_0"
    instance.metadata = {"boltzgen_design_id": design_id}

    if metrics_row is not None:
        instance.metadata["boltzgen_metrics"] = metrics_row
        # Use iptm as primary score, fall back through
        # ptm and complex_plddt
        score_val = (
            metrics_row.get("iptm")
            or metrics_row.get("ptm")
            or metrics_row.get("complex_plddt")
        )
        if score_val is not None:
            try:
                instance.score = float(score_val)
            except (ValueError, TypeError):
                pass

    return instance


def parse_design_output(
    output_dir: Path,
    system: System,
    chain_to_entity: dict[str, int],
    metrics_csv_name: str = "aggregate_metrics_analyze.csv",
) -> list[SystemInstance]:
    """
    Parse a BoltzGen output directory into a list of
    SystemInstance objects.

    output_dir must be the directory passed to
    boltzgen via --output (the parent containing
    intermediate_designs/ etc.).

    Each design CIF in intermediate_designs/ is parsed
    into one SystemInstance via _parse_single_design.
    Per-design metrics from aggregate_metrics_analyze.csv
    are attached to each instance's metadata when
    available.

    Returns an empty list with a warning if no designs
    are found.
    """
    designs_dir = output_dir / "intermediate_designs"

    if not designs_dir.exists():
        logger.warning(
            f"BoltzGen designs directory not found: "
            f"{designs_dir}"
        )
        return []

    # Find design CIFs — pattern: design_spec_<idx>.cif
    # at the top level of intermediate_designs/
    cif_files = sorted(
        p for p in designs_dir.glob("design_spec_*.cif")
        if p.is_file()
    )

    if not cif_files:
        logger.warning(
            f"No design CIFs found in {designs_dir}"
        )
        return []

    # Load per-design metrics from CSV if present
    metrics_by_id: dict[str, dict] = {}
    metrics_path = designs_dir / metrics_csv_name
    if metrics_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_path)
            if "id" in df.columns:
                for _, row in df.iterrows():
                    metrics_by_id[str(row["id"])] = (
                        row.to_dict()
                    )
                logger.info(
                    f"Loaded metrics for "
                    f"{len(metrics_by_id)} designs "
                    f"from {metrics_csv_name}"
                )
            else:
                logger.warning(
                    f"{metrics_csv_name} has no 'id' "
                    f"column — metrics not attached"
                )
        except Exception as e:
            logger.warning(
                f"Could not load metrics CSV "
                f"{metrics_path}: {e}"
            )
    else:
        logger.info(
            f"No metrics CSV at {metrics_path} — "
            "designs will have no metrics in metadata"
        )

    # Parse each CIF into a SystemInstance
    instances: list[SystemInstance] = []
    for cif_path in cif_files:
        design_id = cif_path.stem  # "design_spec_0"
        metrics_row = metrics_by_id.get(design_id)
        try:
            instance = _parse_single_design(
                cif_path=cif_path,
                system=system,
                chain_to_entity=chain_to_entity,
                metrics_row=metrics_row,
            )
            instances.append(instance)
        except Exception as e:
            logger.warning(
                f"Failed to parse {cif_path.name}: {e}"
            )

    logger.info(
        f"Parsed {len(instances)} BoltzGen designs "
        f"from {designs_dir}"
    )
    return instances
