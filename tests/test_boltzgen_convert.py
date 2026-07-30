"""
Tests for the evedesign System -> BoltzGen YAML conversion.

Two tiers:

1. Emission tests assert the YAML we produce for each input case
   BoltzGen accepts (design/context entities, ligands, oligomers,
   conditioning blocks, rejections).
2. Round-trip tests feed that YAML through BoltzGen's own
   YamlDesignParser and assert the internal representation the model
   consumes, so the expectations are checked against upstream rather
   than against our reading of it. They skip when the parser or its
   molecule cache is unavailable.
"""
from pathlib import Path

import pytest
import yaml

from evedesign.models.boltz.convert_design import (
    _atom_bond_constraints,
    _entity_to_sequence_spec,
    _secondary_structure_spec,
    parse_design_output,
    system_to_boltzgen_yaml,
)
from evedesign.models.boltz.chains import _get_chain_ids
from evedesign.system import (
    AtomBond,
    Interaction,
    Ligand,
    Protein,
    ResidueBias,
    SecondaryStructure,
    System,
)

# convert_design imports pyyaml, which arrives with the boltzgen extra
pytestmark = pytest.mark.boltzgen


# Sequence spec: how a chain declares what is designed


def test_range_when_min_and_max_set():
    e = Protein(rep=None, min_length=60, max_length=80, id="binder")
    assert _entity_to_sequence_spec(e) == "60..80"


def test_single_length_when_only_min_or_only_max():
    assert _entity_to_sequence_spec(Protein(rep=None, min_length=60, id="binder")) == "60"
    assert _entity_to_sequence_spec(Protein(rep=None, max_length=80, id="binder")) == "80"


def test_length_from_rep_when_no_range():
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    assert _entity_to_sequence_spec(e) == "10..10"


def test_falls_back_to_vanilla_range():
    assert _entity_to_sequence_spec(Protein(rep=None, id="binder")) == "80..140"


# Motif scaffolding: letters kept, numbers designed


@pytest.mark.parametrize(
    "fixed_pos,expected",
    [
        ([4, 5, 6], "3EFG4"),   # ACDEFGHIKL -> keep E,F,G in the middle
        ([1, 2], "AC8"),        # motif at the start
        ([9, 10], "8KL"),       # motif at the end
        (list(range(1, 11)), "ACDEFGHIKL"),  # nothing designed
        (None, "10..10"),       # no motif -> plain length spec
    ],
)
def test_motif_spec_interleaves_letters_and_run_lengths(fixed_pos, expected):
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    assert _entity_to_sequence_spec(e, fixed_pos=fixed_pos) == expected


def test_motif_requires_rep():
    with pytest.raises(ValueError, match="fixed_pos needs rep"):
        _entity_to_sequence_spec(Protein(rep=None, min_length=5, id="binder"), fixed_pos=[1])


def test_motif_position_must_be_in_range():
    e = Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="binder")
    with pytest.raises(ValueError, match=r"outside 1\.\.10"):
        _entity_to_sequence_spec(e, fixed_pos=[99])


def test_fixed_pos_routes_entity_to_the_design_emitter(tmp_path):
    # rep set and no min/max is normally a context entity, but a motif
    # spec contains digits, which BoltzGen counts as designed
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACDEFGHIKL", id="motif"),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml", fixed_pos={1: [4, 5, 6]}
    ).read_text())
    assert spec["entities"][1]["protein"]["sequence"] == "3EFG4"


# Entity kinds


def test_design_and_context_entities(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep=None, min_length=60, max_length=80, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"] == {"id": "A", "sequence": "MKTAYIAKQR"}
    assert entities[1]["protein"] == {"id": "B", "sequence": "60..80"}


@pytest.mark.parametrize(
    "rep_type,rep,key",
    [("smiles", "CCO", "smiles"), ("ccd", "WHL", "ccd")],
)
def test_ligand_with_a_rep_is_emitted_as_a_ligand(tmp_path, rep_type, rep, key):
    system = System([
        Protein(rep=None, min_length=10, id="binder"),
        Ligand(rep=rep, ligand_rep_type=rep_type, id="lig"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert "protein" not in entities[1]
    assert entities[1]["ligand"] == {"id": "B", key: rep}


def test_designable_ligand_without_rep_defaults_to_unk(tmp_path):
    system = System([
        Protein(rep=None, min_length=10, id="binder"),
        Ligand(rep=None, ligand_rep_type="ccd", id="lig"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[1]["ligand"] == {"id": "B", "ccd": "UNK"}


def test_homo_oligomer_id_becomes_a_list(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", copies=2),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["id"] == ["A", "B"]
    assert entities[1]["protein"]["id"] == "C"


def test_context_entity_with_structure_emits_a_file_entry(tmp_path):
    cif = Path("examples/structure_validated_design/target_cache/1g13.cif")
    if not cif.exists():
        pytest.skip("cached 1G13 structure not available")

    from evedesign.structure import StructureFile

    model = StructureFile(str(cif), format="cif").get_model()
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", structures={"x": model}),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]

    file_entry = entities[0]["file"]
    assert file_entry["include"] == [{"chain": {"id": "A"}}]
    assert Path(file_entry["path"]).exists()
    # BoltzGen's mmCIF reader needs _entity_poly_seq, which an
    # AtomArray cannot supply
    assert file_entry["path"].endswith(".pdb")


# Conditioning blocks


def test_secondary_structure_inline_is_a_dense_padded_string():
    e = Protein(
        rep=None, min_length=20, max_length=20,
        secondary_structure=[
            SecondaryStructure(pos=14, type="C"),
            SecondaryStructure(pos=15, type="H"),
            SecondaryStructure(pos=16, type="H"),
            SecondaryStructure(pos=17, type="H"),
            SecondaryStructure(pos=19, type="E"),
        ],
        id="binder",
    )
    # evedesign H/E/C -> boltzgen H/S/L, everything else U
    assert _secondary_structure_spec(e) == "UUUUUUUUUUUUULHHHUSU"



def test_secondary_structure_without_a_known_length_raises():
    e = Protein(rep=None, secondary_structure=[SecondaryStructure(pos=1, type="H")], id="binder")
    with pytest.raises(ValueError, match="needs a known length"):
        _secondary_structure_spec(e)


def test_secondary_structure_anchors_on_the_shortest_length():
    # BoltzGen raises when the string is longer than the sampled chain,
    # so a variable-length entity anchors on min_length
    e = Protein(
        rep=None, min_length=5, max_length=50,
        secondary_structure=[SecondaryStructure(pos=1, type="H")],
        id="binder",
    )
    assert _secondary_structure_spec(e) == "HUUUU"


def test_binding_types_collapse_positions_into_ranges(tmp_path):
    system = System([
        Protein(
            rep="MKTAYIAKQR", id="target",
            interactions=[
                Interaction(id="avoid", pos=[2, 3, 4], avoid=True),
                Interaction(id="want", pos=[8]),
            ],
        ),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {
        "binding": "8", "not_binding": "2..4",
    }


def test_interaction_without_positions_covers_the_whole_entity(tmp_path):
    system = System([
        Protein(rep="MKTA", id="target",
                interactions=[Interaction(id="none", pos=None, avoid=True)]),
        Protein(rep=None, min_length=10, id="binder"),
    ])
    entities = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"]
    assert entities[0]["protein"]["binding_types"] == {"not_binding": "1..4"}


def test_cyclic_flag(tmp_path):
    system = System([Protein(rep=None, min_length=10, cyclic=True, id="binder")])
    assert yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())["entities"][0]["protein"]["cyclic"] is True


def test_atom_bonds_become_top_level_constraints(tmp_path):
    system = System([
        Protein(rep="MCTAYIAKQR", id="target"),
        Protein(
            rep="AAACAAAAAAAA", min_length=12, max_length=12,
            atom_bonds=[AtomBond(
                type="covalent", source_pos=4, source_atom="SG",
                target_entity_id="target", target_pos=2, target_atom="SG",
            )],
        ),
    ])
    spec = yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())
    assert spec["constraints"] == [
        {"bond": {"atom1": ["B", 4, "SG"], "atom2": ["A", 2, "SG"]}}
    ]


def test_no_constraints_key_when_there_are_no_bonds(tmp_path):
    system = System([Protein(rep=None, min_length=10, id="binder")])
    assert "constraints" not in yaml.safe_load(system_to_boltzgen_yaml(
        system, tmp_path / "spec.yaml"
    ).read_text())


# Rejections: fail loudly rather than drop silently


def test_rejects_residue_bias(tmp_path):
    system = System([
        Protein(rep="ACDE", min_length=4, max_length=4,
               residue_bias=[ResidueBias(pos=1, bias={"A": 1.0})],
               id="binder",
           ),
    ])
    with pytest.raises(NotImplementedError, match="residue_bias"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_symmetry(tmp_path):
    system = System([
        Protein(rep=None, min_length=10, symmetry="C", copies=2, id="binder"),
    ])
    with pytest.raises(NotImplementedError, match="symmetry"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_interaction_partner_ids(tmp_path):
    system = System([
        Protein(rep=None, min_length=10,
               interactions=[Interaction(id="s", pos=[1], partner_ids=["x"])]),
    ])
    with pytest.raises(ValueError, match="partner_ids"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_contradictory_binding_positions(tmp_path):
    system = System([
        Protein(rep=None, min_length=10, interactions=[
            Interaction(id="a", pos=[2]),
            Interaction(id="b", pos=[2], avoid=True),
        ]),
    ])
    with pytest.raises(ValueError, match="both"):
        yaml.safe_load(system_to_boltzgen_yaml(
            system, tmp_path / "spec.yaml"
        ).read_text())


def test_rejects_non_covalent_bonds():
    system = System([
        Protein(rep="ACDE", id="x", atom_bonds=[AtomBond(
            type="hydrogen", source_pos=1, source_atom="N",
            target_entity_id="x", target_pos=2, target_atom="O",
        )]),
    ])
    with pytest.raises(ValueError, match="covalent"):
        _atom_bond_constraints(system, _get_chain_ids(system))


def test_rejects_bond_to_unknown_entity():
    system = System([
        Protein(rep="ACDE", id="x", atom_bonds=[AtomBond(
            type="covalent", source_pos=1, source_atom="SG",
            target_entity_id="nope", target_pos=2, target_atom="SG",
        )]),
    ])
    with pytest.raises(ValueError, match="unknown entity"):
        _atom_bond_constraints(system, _get_chain_ids(system))


# Output parsing


def test_parse_returns_empty_when_the_run_did_not_reach_filtering(tmp_path):
    # no final_ranked_designs/ means filtering never ran
    system = System([Protein(rep=None, min_length=10, id="binder")])
    assert parse_design_output(tmp_path, system) == []


# Round-trip: our YAML through BoltzGen's own parser

MOLDIR = Path(
    "~/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/"
    "c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
).expanduser()


def _parse_with_boltzgen(system, tmp_path, **kwargs):
    """Emit our YAML, then parse it with BoltzGen's YamlDesignParser."""
    parser = pytest.importorskip(
        "boltzgen.data.parse.schema", reason="boltzgen not installed"
    ).YamlDesignParser
    if not MOLDIR.exists():
        pytest.skip("boltzgen molecule cache not downloaded")

    path = system_to_boltzgen_yaml(system, tmp_path / "spec.yaml", **kwargs)
    schema = yaml.safe_load(path.read_text())
    target = parser(mol_dir=MOLDIR).parse_boltzgen_schema(
        name="test", schema=schema, mols={}, mol_dir=MOLDIR,
        base_file_path=tmp_path,
    )
    return schema, target


def _residue_slice(target, chain_idx):
    chain = target.structure.chains[chain_idx]
    start = int(chain["res_idx"])
    return start, start + int(chain["res_num"])


def test_roundtrip_motif_yields_the_intended_design_mask(tmp_path):
    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep="ACDEFGHIKL", min_length=10, max_length=10, id="motif"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, fixed_pos={1: [4, 5, 6]})

    start, end = _residue_slice(target, 1)
    mask = target.design_info.res_design_mask[start:end].tolist()
    # 3 designed, E/F/G kept, 4 designed
    assert mask == [True] * 3 + [False] * 3 + [True] * 4


def test_roundtrip_secondary_structure_types(tmp_path):
    from boltzgen.data.const import ss_type_ids

    system = System([
        Protein(rep="MKTAYIAKQR", id="target"),
        Protein(rep=None, min_length=8, max_length=8, id="binder",
                secondary_structure=[
                    SecondaryStructure(pos=2, type="H"),
                    SecondaryStructure(pos=3, type="E"),
                    SecondaryStructure(pos=5, type="C"),
                ]),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path)

    start, end = _residue_slice(target, 1)
    by_id = {v: k for k, v in ss_type_ids.items()}
    types = [by_id[int(v)] for v in target.design_info.res_ss_types[start:end]]
    assert types == [
        "UNSPECIFIED", "HELIX", "SHEET", "UNSPECIFIED", "LOOP",
        "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED",
    ]


def test_roundtrip_binding_types(tmp_path):
    from boltzgen.data.const import binding_type_ids

    system = System([
        Protein(rep="MKTAYIAKQR", id="target", interactions=[
            Interaction(id="avoid", pos=[2, 3, 4], avoid=True),
            Interaction(id="want", pos=[8]),
        ]),
        Protein(rep=None, min_length=6, max_length=6, id="binder"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path)

    start, end = _residue_slice(target, 0)
    by_id = {v: k for k, v in binding_type_ids.items()}
    types = [by_id[int(v)] for v in target.design_info.res_binding_type[start:end]]
    assert types == [
        "UNSPECIFIED", "NOT_BINDING", "NOT_BINDING", "NOT_BINDING",
        "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED", "BINDING",
        "UNSPECIFIED", "UNSPECIFIED",
    ]


def test_roundtrip_covalent_bond_resolves_to_real_atoms(tmp_path):
    # both cysteines must be fixed residues: designed positions are
    # placeholder glycines and have no SG atom
    system = System([
        Protein(rep="MCTAYIAKQR", id="target"),
        Protein(rep="AAACAAAAAAAA", min_length=12, max_length=12, id="binder",
                atom_bonds=[AtomBond(
                    type="covalent", source_pos=4, source_atom="SG",
                    target_entity_id="target", target_pos=2, target_atom="SG",
                )]),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path, fixed_pos={1: [4]})
    assert len(target.structure.bonds) >= 1


def test_roundtrip_structural_target(tmp_path):
    cif = Path("examples/structure_validated_design/target_cache/1g13.cif")
    if not cif.exists():
        pytest.skip("cached 1G13 structure not available")

    import biotite.structure as struc
    from evedesign.structure import Structure, StructureFile

    chain = StructureFile(str(cif), format="cif").get_model().get_chain("A")
    aa = chain.atom_array[struc.filter_amino_acids(chain.atom_array)]
    system = System([
        Protein(rep="MKTAYIAKQR", id="target", structures={"x": Structure(aa)}),
        Protein(rep=None, min_length=10, max_length=10, id="binder"),
    ])
    _, target = _parse_with_boltzgen(system, tmp_path)
    assert len(target.structure.chains) == 2
