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
import copy
import re
from itertools import groupby
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from evedesign.models.boltz.chains import (
    _chain_to_entity_map,
    _get_chain_ids,
)
from evedesign.structure import Structure, StructureFile
from evedesign.system import (
    Entity,
    EntityInstance,
    System,
    SystemInstance,
)
from evedesign.types import EntityPosList

# INPUT: evedesign System -> BoltzGen design YAML


# 1a. Sequence spec (letters fixed, numbers designed)


def _entity_to_sequence_spec(
    entity: Entity,
    fixed_pos: Sequence[int] | None = None,
) -> str:
    """
    Convert entity length info into BoltzGen sequence
    spec string.

    In a BoltzGen spec, letters are held fixed and numbers
    are designed, chosen per residue rather than per chain
    (res_design_mask, schema.py:530-547). An entity counts as
    designed when its spec contains any digit
    (schema.py:879). So "60..80" designs a whole chain, while
    "3EFG4" designs 3 residues, keeps EFG, then designs 4.

    BoltzGen YAML accepts three formats for the
    sequence field of a designable entity:
    - "80..140" for variable length (range)
    - "80" for fixed length (no specific sequence)
    - Actual sequence string (for fixed entities,
      handled elsewhere)

    Resolution order:
    1. fixed_pos given -> interleaved motif spec built from
       entity.rep (requires rep; length fixed at len(rep))
    2. min_length and max_length both set -> "min..max"
    3. min_length only -> "min"
    4. max_length only -> "max"
    5. rep is set -> len(rep) (fixed length matching rep)
    6. Fallback -> "80..140" (matches BoltzGen's vanilla
       binder default; emits a warning when triggered)

    Parameters
    ----------
    fixed_pos : Sequence[int], optional
        1-based positions to hold fixed (motif scaffolding).
        Requires entity.rep, since the fixed residues are
        taken from it.
    """
    if fixed_pos:
        if entity.rep is None:
            raise ValueError(
                f"Entity '{entity.id}': fixed_pos needs rep, which "
                "supplies the fixed residues."
            )
        rep = list(entity.rep)
        length = len(rep)
        keep = {int(p) for p in fixed_pos}

        out_of_range = sorted(p for p in keep if not 1 <= p <= length)
        if out_of_range:
            raise ValueError(
                f"Entity '{entity.id}': fixed_pos {out_of_range} outside "
                f"1..{length}."
            )

        # Kept residues emit their letter, designed runs emit their length
        spec: list[str] = []
        for kept, group in groupby(
            range(1, length + 1), key=lambda pos: pos in keep
        ):
            run = list(group)
            if kept:
                spec.extend(str(rep[pos - 1]) for pos in run)
            else:
                spec.append(str(len(run)))
        return "".join(spec)

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
        "max_length, or rep, defaulting to "
        "BoltzGen's vanilla binder range '80..140'. "
        "Set min_length/max_length on the Entity "
        "to suppress this warning and control the "
        "design length range explicitly."
    )
    return "80..140"

# 1b. Per-entity conditioning blocks

# Each helper mirrors one block of BoltzGen's spec parser in
# boltzgen/data/parse/schema.py; line refs are for boltzgen 0.3.2.


# evedesign H/E/C (helix/sheet/coil) -> BoltzGen's range keys
_SS_TO_BOLTZGEN = {"H": "helix", "E": "sheet", "C": "loop"}


def _positions_to_range_spec(positions: Sequence[int]) -> str:
    """
    Collapse 1-based positions into BoltzGen range syntax,
    e.g. [5, 6, 7, 13] -> "5..7,13".

    Consumed by parse_range (schema.py:646-680), which reads
    single values and ranges as 1-indexed and end-inclusive.
    """
    ordered = sorted({int(p) for p in positions})
    if not ordered:
        return ""

    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = prev = pos
    runs.append((start, prev))

    return ",".join(
        str(a) if a == b else f"{a}..{b}" for a, b in runs
    )


def _spec_length(entity: Entity) -> int | None:
    """
    Residue count for per-residue specs, or None if unknown.

    schema.py:1127-1132 pads a short per-residue string with
    UNSPECIFIED but raises when it is longer than the sampled
    chain, so anchor on the shortest length the entity allows.
    """
    if entity.rep is not None:
        return len(entity.rep)
    if entity.min_length is not None:
        return entity.min_length
    if entity.max_length is not None:
        return entity.max_length
    return None


def _secondary_structure_spec(entity: Entity) -> dict[str, str] | None:
    """
    Secondary structure as {"helix": "10..30"}, or None.

    The range form BoltzGen's examples use, e.g.
    "secondary_structure: {sheet: 1,3..11}" beside a
    variable-length sequence. Ranges carry no length, so
    positions past min_length stay expressible, unlike the
    per-residue string form. Parsed at schema.py:1134-1140;
    only valid on designed residues (data.py:1957).

    An item with pos=None covers the whole chain, which
    BoltzGen spells "all".
    """
    ss = getattr(entity, "secondary_structure", None)
    if not ss:
        return None

    by_key: dict[str, list[int]] = {}
    for item in ss:
        if item.type not in _SS_TO_BOLTZGEN:
            raise ValueError(
                f"Entity '{entity.id}': bad secondary structure type "
                f"{item.type!r}, expected one of {sorted(_SS_TO_BOLTZGEN)}."
            )
        key = _SS_TO_BOLTZGEN[item.type]
        if item.pos is None:
            return {key: "all"}
        by_key.setdefault(key, []).append(int(item.pos))

    return {
        k: _positions_to_range_spec(v) for k, v in by_key.items()
    }


def _binding_types_spec(entity: Entity) -> dict[str, str] | None:
    """
    binding_types mapping from Entity.interactions, or None.

    avoid=False -> "binding", avoid=True -> "not_binding";
    pos=None covers the whole entity. Keys consumed at
    schema.py:1094-1100.
    """
    interactions = getattr(entity, "interactions", None)
    if not interactions:
        return None

    length = _spec_length(entity)
    binding: list[int] = []
    not_binding: list[int] = []

    for item in interactions:
        if item.partner_ids:
            raise ValueError(
                f"Entity '{entity.id}': Interaction '{item.id}' sets "
                "partner_ids, which binding_types cannot express."
            )
        target = not_binding if item.avoid else binding
        if item.pos is None:
            if length is None:
                raise ValueError(
                    f"Entity '{entity.id}': Interaction '{item.id}' "
                    "covers the whole entity but its length is unknown."
                )
            target.extend(range(1, length + 1))
            continue
        target.extend(int(p) for p in item.pos)

    overlap = sorted(set(binding) & set(not_binding))
    if overlap:
        raise ValueError(
            f"Entity '{entity.id}': positions {overlap} marked both "
            "binding and not_binding."
        )

    spec: dict[str, str] = {}
    if binding:
        spec["binding"] = _positions_to_range_spec(binding)
    if not_binding:
        spec["not_binding"] = _positions_to_range_spec(not_binding)
    return spec or None


def _attach_conditioning(entity: Entity, inner: dict) -> None:
    """
    Attach the optional conditioning blocks to an inline
    entity dict, in place.

    BoltzGen decides which combinations are legal: secondary
    structure is rejected on non-designed residues and binding
    types on designed ones (data.py:1953-1960). File entities
    take list-of-chain forms, handled by the caller.
    """
    ss_spec = _secondary_structure_spec(entity)
    if ss_spec is not None:
        inner["secondary_structure"] = ss_spec

    binding_spec = _binding_types_spec(entity)
    if binding_spec is not None:
        inner["binding_types"] = binding_spec

    if entity.cyclic:
        inner["cyclic"] = True


# 1c. Entity emitters: the two input cases

def _emit_design_entity(
    entity: Entity,
    chain_ids: list[str],
    pointer: int,
    fixed_pos: Sequence[int] | None = None,
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

    Conditioning blocks from section 1c are attached when the
    entity specifies them.

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
            # Designable ligand with no rep, default to UNK
            entry = {
                "ligand": {"id": id_field, "ccd": "UNK"}
            }
        return entry, pointer + copies

    # Ligands returned above, so only protein is left: can_model
    # (boltzgen.py) admits protein and ligand entities only.
    if entity.type != "protein":
        raise ValueError(
            f"Entity '{entity.id}': type {entity.type!r} unsupported; "
            "BoltzGen models protein and ligand entities."
        )

    entry_inner: dict = {
        "id": id_field,
        "sequence": _entity_to_sequence_spec(entity, fixed_pos=fixed_pos),
    }
    _attach_conditioning(entity, entry_inner)

    entry = {entity.type: entry_inner}
    return entry, pointer + copies


def _emit_context_entity(
    entity: Entity,
    entity_idx: int,
    chain_ids: list[str],
    pointer: int,
    tmp_dir: Path,
) -> tuple[dict, int]:
    """
    Emit a YAML entry for a fixed (context) entity: a file:
    entry when structures are attached (PDB written to
    tmp_dir/structures/), else an inline protein: entry with
    the literal rep. Conditioning attaches as a dict inline
    but as a list of chain blocks on file entries
    (schema.py:2187-2191).

    Returns the entry and the advanced chain ID pointer.
    """
    copies = (
        entity.copies if entity.copies is not None
        else 1
    )

    # Fallback: no structure attached -> emit as
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
        if entity.rep is None:
            raise ValueError(
                f"Entity '{entity.id}': nothing to emit as fixed, "
                "needs rep or structures."
            )
        inner: dict = {"id": id_field, "sequence": "".join(entity.rep)}
        _attach_conditioning(entity, inner)
        entry = {"protein": inner}
        return entry, pointer + copies

    # PDB, not CIF: BoltzGen branches on the suffix
    # (schema.py:1920) and its mmCIF reader needs
    # _entity_poly_seq (mmcif.py:980)
    structure_path = (
        tmp_dir / "structures" / f"entity_{entity_idx}.pdb"
    )
    structure_path.parent.mkdir(parents=True, exist_ok=True)

    first_key = next(iter(entity.structures))
    model = entity.structures[first_key]
    if isinstance(model, list):
        model = model[0]
    model.to_file(str(structure_path), format="pdb")

    # Build the file entry for the chain at entity_idx.
    chain_id = chain_ids[pointer]
    file_entry: dict = {
        "path": str(structure_path.resolve()),
        "include": [{"chain": {"id": chain_id}}],
    }

    binding_spec = _binding_types_spec(entity)
    if binding_spec is not None:
        file_entry["binding_types"] = [
            {"chain": {"id": chain_id, **binding_spec}}
        ]

    entry = {"file": file_entry}
    return entry, pointer + copies

def _entity_to_boltzgen_yaml(
    entity: Entity,
    entity_idx: int,
    chain_ids: list[str],
    pointer: int,
    tmp_dir: Path,
    designed: bool,
    fixed_pos: Sequence[int] | None = None,
) -> tuple[dict, int]:
    """
    Emit one entity's YAML entry, designed or fixed.

    Ligands always take the ligand: form. fixed_pos yields a spec
    with digits, which BoltzGen counts as designed (schema.py:879).
    """
    if entity.type == "ligand" or designed or fixed_pos:
        if entity.structures and entity.type != "ligand":
            raise ValueError(
                f"Entity '{entity.id}' has structures, so it cannot be "
                "designed and used as a target at once. Exclude it from "
                "entities to keep it fixed (in-place redesign is not "
                "implemented)."
            )
        return _emit_design_entity(
            entity, chain_ids, pointer, fixed_pos=fixed_pos
        )
    return _emit_context_entity(
        entity, entity_idx, chain_ids, pointer, tmp_dir
    )


# 1d. System level


def _atom_bond_constraints(
    system: System,
    chain_ids: list[str],
) -> list[dict]:
    """
    Collect Entity.atom_bonds into BoltzGen's top-level
    constraints block:

        constraints:
          - bond:
              atom1: [<chain>, <res>, <atom>]
              atom2: [<chain>, <res>, <atom>]

    Read at schema.py:1703-1710, which only forms covalent
    connections, so other BondType values are rejected
    rather than silently dropped.
    """
    entity_id_to_chain: dict[str, str] = {}
    pointer = 0
    for entity in system:
        copies = entity.copies if entity.copies is not None else 1
        if entity.id is not None:
            entity_id_to_chain[entity.id] = chain_ids[pointer]
        pointer += copies

    constraints: list[dict] = []
    pointer = 0
    for entity in system:
        copies = entity.copies if entity.copies is not None else 1
        source_chain = chain_ids[pointer]
        pointer += copies

        for bond in getattr(entity, "atom_bonds", None) or []:
            if bond.type != "covalent":
                raise ValueError(
                    f"Entity '{entity.id}': AtomBond type {bond.type!r} "
                    "unsupported; BoltzGen models covalent bonds only."
                )
            target_chain = entity_id_to_chain.get(bond.target_entity_id)
            if target_chain is None:
                raise ValueError(
                    f"Entity '{entity.id}': AtomBond targets unknown "
                    f"entity id {bond.target_entity_id!r}."
                )
            if bond.source_pos is None or bond.target_pos is None:
                raise ValueError(
                    f"Entity '{entity.id}': AtomBond needs explicit "
                    "source_pos and target_pos."
                )
            constraints.append({
                "bond": {
                    "atom1": [
                        source_chain,
                        int(bond.source_pos),
                        bond.source_atom,
                    ],
                    "atom2": [
                        target_chain,
                        int(bond.target_pos),
                        bond.target_atom,
                    ],
                }
            })
    return constraints


def system_to_boltzgen_yaml(
    system: System,
    output_path: Path,
    fixed_pos: EntityPosList | None = None,
    entities: Sequence[int] | None = None,
) -> Path:
    """
    Convert an evedesign System into a BoltzGen design
    specification YAML.

    entities selects which entity indices are designed, the rest
    are held fixed; defaults to all, per Generator.generate.
    Designed entities become sequence/length specs, fixed ones a
    file: entry or literal sequence (writing a PDB to
    output_path.parent / structures/). Entity.atom_bonds become
    the top-level constraints block.

    fixed_pos maps entity index -> 1-based positions held fixed
    within that entity (motif scaffolding), taken from its rep.

    Returns output_path for chaining.
    """
    design = set(
        range(len(system)) if entities is None else entities
    )
    tmp_dir = output_path.parent
    chain_ids = _get_chain_ids(system)
    pointer = 0

    entities_list: list[dict] = []

    logger.info(
        f"System has {len(system)} entities: {len(design)} designed, "
        f"{len(system) - len(design)} context"
    )

    for entity_idx, entity in enumerate(system):
        entry, pointer = _entity_to_boltzgen_yaml(
            entity, entity_idx, chain_ids, pointer, tmp_dir,
            designed=entity_idx in design,
            fixed_pos=(
                fixed_pos.get(entity_idx)
                if fixed_pos is not None else None
            ),
        )
        entities_list.append(entry)

    yaml_data: dict = {"entities": entities_list}

    constraints = _atom_bond_constraints(system, chain_ids)
    if constraints:
        yaml_data["constraints"] = constraints

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    return output_path

# OUTPUT: BoltzGen output tree -> SystemInstance


# 2a. Single design


def _parse_single_design(
    cif_path: Path,
    system: System,
    chain_to_entity: dict[str, int],
    metrics_row: dict | None = None,
    design_id: str | None = None,
) -> SystemInstance:
    """
    Parse one BoltzGen output CIF into a SystemInstance.

    Chains are matched to entities via chain_to_entity;
    designed entities take the CIF's sequence and structure,
    context entities keep their original rep.

    metrics_row: a row from BoltzGen's metrics CSV, populating
    metadata and score (iptm, else ptm, else complex_plddt).
    design_id: BoltzGen's design id, e.g. "design_spec_5";
    required when the CIF filename carries a rank prefix.
    """

    sf = StructureFile(str(cif_path), format="cif")
    full_structure = sf.get_model()

    # Group chains by entity index based on chain_to_entity
    entity_chains: dict[int, list[Structure]] = {}
    for chain_id in full_structure.chains():
        if chain_id not in chain_to_entity:
            logger.warning(
                f"Chain '{chain_id}' in {cif_path.name} "
                f"not in chain_to_entity mapping...skipping"
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
            # Entity missing from CIF: preserve its
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

        # Build the models dict: single chain or
        # homo-oligomer list
        if len(chains) == 1:
            models = {"model_0": chains[0]}
        else:
            models = {"model_0": chains}

        entity_instances.append(
            EntityInstance(rep=seq_array, models=models)
        )

    instance = SystemInstance(entity_instances)

    # Attach metadata. Prefer the caller-supplied id: the
    # Diverse set files are named "rank<N>_design_spec_<M>.cif",
    # so the stem alone would not match the metrics CSV "id".
    if design_id is None:
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

        # Confidence: matches BoltzFold's default of
        # complex_plddt for cross-model consistency
        confidence_val = metrics_row.get("complex_plddt")
        if confidence_val is not None:
            try:
                instance.confidence = float(confidence_val)
            except (ValueError, TypeError):
                pass

    return instance

# 2b. Output-set variants: the two output cases

def _parse_diverse_set(
    ranked_dir: Path,
    system: System,
    chain_to_entity: dict[str, int],
) -> list[SystemInstance]:
    """
    Parse the Diverse set: budget-filtered designs in
    final_<N>_designs/, ranked rank01..rankN, with
    metrics from final_designs_metrics_<N>.csv.
    """
    # Find the final_<N>_designs subdir
    final_dirs = sorted(
        d for d in ranked_dir.glob("final_*_designs")
        if d.is_dir()
    )
    if not final_dirs:
        logger.warning(
            f"No final_<N>_designs directory in "
            f"{ranked_dir}"
        )
        return []
    if len(final_dirs) > 1:
        logger.warning(
            f"Multiple final_<N>_designs directories "
            f"in {ranked_dir}: "
            f"{[d.name for d in final_dirs]}. Using "
            "the first."
        )
    final_designs_dir = final_dirs[0]

    # Find the matching metrics CSV
    metrics_files = sorted(
        ranked_dir.glob("final_designs_metrics_*.csv")
    )
    metrics_by_design_id: dict[str, dict] = {}
    if metrics_files:
        metrics_path = metrics_files[0]
        try:
            df = pd.read_csv(metrics_path)
            if "id" in df.columns:
                metrics_by_design_id = {
                    str(row["id"]): row.to_dict()
                    for _, row in df.iterrows()
                }
                logger.info(
                    f"Loaded metrics for "
                    f"{len(metrics_by_design_id)} "
                    f"Diverse-set designs from "
                    f"{metrics_path.name}"
                )
            else:
                logger.warning(
                    f"{metrics_path.name} has no 'id' "
                    "column"
                )
        except Exception as e:
            logger.warning(
                f"Could not load metrics CSV "
                f"{metrics_path}: {e}"
            )
    else:
        logger.warning(
            f"No final_designs_metrics_*.csv in "
            f"{ranked_dir}"
        )

    # Parse each rank-prefixed CIF
    rank_pattern = re.compile(
        r"rank(\d+)_(design_spec_\d+)\.cif$"
    )

    cif_entries = []
    for p in sorted(final_designs_dir.glob("*.cif")):
        m = rank_pattern.match(p.name)
        if m is None:
            logger.debug(
                f"Skipping unexpected file in "
                f"{final_designs_dir.name}: {p.name}"
            )
            continue
        cif_entries.append(
            (int(m.group(1)), m.group(2), p)
        )

    if not cif_entries:
        logger.warning(
            f"No rank<N>_design_spec_<M>.cif files in "
            f"{final_designs_dir}"
        )
        return []

    cif_entries.sort(key=lambda x: x[0])

    instances: list[SystemInstance] = []
    for rank_num, design_id, cif_path in cif_entries:
        metrics_row = metrics_by_design_id.get(design_id)
        try:
            instance = _parse_single_design(
                cif_path=cif_path,
                system=system,
                chain_to_entity=chain_to_entity,
                metrics_row=metrics_row,
                design_id=design_id,
            )
            if instance.metadata is None:
                instance.metadata = {}
            instance.metadata["boltzgen_rank"] = rank_num
            instances.append(instance)
        except Exception as e:
            logger.warning(
                f"Failed to parse {cif_path.name}: {e}"
            )

    logger.info(
        f"Parsed {len(instances)} BoltzGen designs "
        f"(Diverse set) from {final_designs_dir}"
    )
    return instances


def _parse_all_designs(
    output_dir: Path,
    system: System,
    chain_to_entity: dict[str, int],
) -> list[SystemInstance]:
    """
    Parse the full all_designs_metrics.csv set: every
    design that survived the analysis step. CIFs are
    sourced from intermediate_designs/.
    """
    all_metrics_path = (
        output_dir / "final_ranked_designs" / "all_designs_metrics.csv"
    )
    if not all_metrics_path.exists():
        logger.warning(
            f"all_designs_metrics.csv not found at "
            f"{all_metrics_path}"
        )
        return []

    try:
        df = pd.read_csv(all_metrics_path)
    except Exception as e:
        logger.warning(
            f"Could not load {all_metrics_path}: {e}"
        )
        return []

    if "id" not in df.columns:
        logger.warning(
            "all_designs_metrics.csv has no 'id' column"
        )
        return []

    logger.info(
        f"Loaded metrics for {len(df)} designs from "
        f"all_designs_metrics.csv (full set)"
    )

    intermediate_dir = output_dir / "intermediate_designs"
    if not intermediate_dir.exists():
        logger.warning(
            f"intermediate_designs/ not found at "
            f"{intermediate_dir}"
        )
        return []

    instances: list[SystemInstance] = []
    for _, row in df.iterrows():
        design_id = str(row["id"])
        cif_path = intermediate_dir / f"{design_id}.cif"
        if not cif_path.exists():
            logger.warning(
                f"Metrics row for {design_id} but no "
                f"CIF at {cif_path}"
            )
            continue
        try:
            instance = _parse_single_design(
                cif_path=cif_path,
                system=system,
                chain_to_entity=chain_to_entity,
                metrics_row=row.to_dict(),
            )
            instances.append(instance)
        except Exception as e:
            logger.warning(
                f"Failed to parse {cif_path.name}: {e}"
            )

    logger.info(
        f"Parsed {len(instances)} BoltzGen designs "
        f"(full all_designs set) from "
        f"{intermediate_dir}"
    )
    return instances

# 2c. Entry point


def parse_design_output(
    output_dir: Path,
    system: System,
    return_all: bool = False,
) -> list[SystemInstance]:
    """
    Parse a BoltzGen --output directory into SystemInstances.

    Returns the Diverse set by default: the budget-filtered,
    diversity-selected designs in
    final_ranked_designs/final_<N>_designs/, which is BoltzGen's
    final output (boltzgen.task.filter.filter.Filter). With
    return_all=True, returns every design in
    all_designs_metrics.csv instead, sourcing CIFs from
    intermediate_designs/ - the full post-analysis set, before
    diversity filtering, for custom filtering downstream.

    Chain -> entity routing is derived from system via
    _chain_to_entity_map. Each instance has structures, metrics,
    score and confidence populated.
    """
    chain_to_entity = _chain_to_entity_map(system)
    ranked_dir = output_dir / "final_ranked_designs"

    if not ranked_dir.exists():
        logger.warning(
            f"BoltzGen final_ranked_designs not found "
            f"at {ranked_dir}. Did BoltzGen finish?"
        )
        return []

    if return_all:
        return _parse_all_designs(
            output_dir, system, chain_to_entity
        )
    return _parse_diverse_set(
        ranked_dir, system, chain_to_entity
    )

# 2d. Feeding designs back in as templates


def system_with_design_structures(
    system: System,
    instance: SystemInstance,
    model_key: str = "model_0",
    structure_key: str = "input",
) -> System:
    """
    Copy system with each entity's structures and rep taken from
    instance, bridging generated structures (EntityInstance.models)
    into the template channel that LigandMPNN and BoltzFold read at
    build() time.

    Assumes instance numbering matches system numbering. That holds
    for BoltzGen output, whose handles_insertions and
    handles_deletions are both False, but not for generators that
    design insertions: the promoted rep would be a different length
    while interactions, secondary_structure and atom_bonds still
    carry the original positions.

    Entities with no model under model_key keep their structures.
    """
    new_entities = []
    for entity_idx, template_entity in enumerate(system):
        new_entity = copy.copy(template_entity)
        entity_instance = instance[entity_idx]

        models = entity_instance.models
        if models is not None and model_key in models:
            model = models[model_key]
            # a list is homo-oligomer copies of the chain
            new_entity.structures = {
                structure_key: model[0] if isinstance(model, list) else model
            }
        if entity_instance.rep is not None:
            new_entity.rep = entity_instance.rep.copy()

        new_entities.append(new_entity)

    return type(system)(new_entities)
