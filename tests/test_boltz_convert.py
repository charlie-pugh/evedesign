"""
tests/test_boltz_convert.py

Tests for evedesign/models/boltz/convert.py
"""

import numpy as np
import yaml
import pytest

from evedesign.system import System, SystemInstance, EntityInstance, Protein
from evedesign.models.boltz.convert import (
    _get_chain_ids,
    _boltz_structure_to_atom_array,
    system_instance_to_yaml,
    prediction_to_instance,
)


SEQ = "TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLL"


@pytest.fixture
def single_system():
    return System([Protein(id="prot1", rep=SEQ, first_index=1)])


@pytest.fixture
def multi_system():
    return System([
        Protein(id="A", rep="AAAAAAAAAA", first_index=1),
        Protein(id="B", rep="CCCCCCCCCC", first_index=1),
    ])

class TestGetChainIds:
    def test_single_chain(self):
        assert _get_chain_ids(1) == ["A"]

    def test_three_chains(self):
        assert _get_chain_ids(3) == ["A", "B", "C"]

    def test_26_chains(self):
        ids = _get_chain_ids(26)
        assert len(ids) == 26
        assert ids[0] == "A"
        assert ids[25] == "Z"

    def test_27_chains_uses_two_letters(self):
        ids = _get_chain_ids(27)
        assert ids[26] == "AA"

    def test_zero_chains(self):
        assert _get_chain_ids(0) == []

    def test_chain_ids_are_generated_not_from_entity(self, tmp_path):
        """Chain IDs in YAML are A, B, C... regardless of entity.id"""
        system = System([
            Protein(id="prot_x", rep="AAAAAAAAAA", first_index=1),
            Protein(id="prot_y", rep="CCCCCCCCCC", first_index=1),
        ])
        instance = SystemInstance([
            EntityInstance(rep="AAAAAAAAAA"),
            EntityInstance(rep="CCCCCCCCCC"),
        ])
        out = system_instance_to_yaml(system, instance, tmp_path / "test.yaml")
        data = yaml.safe_load(out.read_text())

        assert data["sequences"][0]["protein"]["id"] == "A"
        assert data["sequences"][1]["protein"]["id"] == "B"

class TestSystemInstanceToYaml:
    def test_single_protein_yaml(self, single_system, tmp_path):
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        out = system_instance_to_yaml(single_system, instance, tmp_path / "test.yaml")

        assert out.exists()
        data = yaml.safe_load(out.read_text())

        assert data["version"] == 1
        assert len(data["sequences"]) == 1
        assert "protein" in data["sequences"][0]
        assert data["sequences"][0]["protein"]["sequence"] == SEQ
        assert data["sequences"][0]["protein"]["id"] == "A"

    def test_uses_instance_sequence_not_entity(self, tmp_path):
        """YAML should contain the instance's sequence, not the entity's."""
        system = System([Protein(id="A", rep="AAAAAAAAAA", first_index=1)])
        instance = SystemInstance([EntityInstance(rep="CCCCCCCCCC")])
        out = system_instance_to_yaml(system, instance, tmp_path / "test.yaml")
        data = yaml.safe_load(out.read_text())

        assert data["sequences"][0]["protein"]["sequence"] == "CCCCCCCCCC"
        
    def test_msa_empty_by_default(self, single_system, tmp_path):
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        out = system_instance_to_yaml(single_system, instance, tmp_path / "test.yaml")
        data = yaml.safe_load(out.read_text())

        assert data["sequences"][0]["protein"]["msa"] == "empty"

    def test_msa_omitted_when_server_enabled(self, single_system, tmp_path):
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        out = system_instance_to_yaml(
            single_system, instance, tmp_path / "test.yaml", use_msa_server=True
        )
        data = yaml.safe_load(out.read_text())

        assert "msa" not in data["sequences"][0]["protein"]

    def test_multi_protein_yaml(self, multi_system, tmp_path):
        instance = SystemInstance([
            EntityInstance(rep="AAAAAAAAAA"),
            EntityInstance(rep="CCCCCCCCCC"),
        ])
        out = system_instance_to_yaml(multi_system, instance, tmp_path / "test.yaml")
        data = yaml.safe_load(out.read_text())

        assert len(data["sequences"]) == 2
        assert data["sequences"][0]["protein"]["id"] == "A"
        assert data["sequences"][0]["protein"]["sequence"] == "AAAAAAAAAA"
        assert data["sequences"][1]["protein"]["id"] == "B"
        assert data["sequences"][1]["protein"]["sequence"] == "CCCCCCCCCC"

    def test_creates_parent_dirs(self, single_system, tmp_path):
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        nested = tmp_path / "a" / "b" / "c" / "test.yaml"
        out = system_instance_to_yaml(single_system, instance, nested)
        assert out.exists()


    def test_returns_path(self, single_system, tmp_path):
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        out = system_instance_to_yaml(single_system, instance, tmp_path / "test.yaml")
        assert out == tmp_path / "test.yaml"



# ── Helpers for mock Boltz StructureV2 ──────────────────────────────────────

def _make_mock_boltz_structure(chain_names, res_per_chain, atoms_per_res=4):
    """
    Build a mock Boltz StructureV2 with protein backbone atoms (N/CA/C/O).

    Returns (boltz_structure, n_total_atoms).
    """
    from boltz.data.types import (
        AtomV2, Residue, Chain, BondV2, Interface, Coords, Ensemble, StructureV2,
    )

    atom_names = ["N", "CA", "C", "O"]
    n_chains = len(chain_names)
    n_total_atoms = n_chains * res_per_chain * atoms_per_res

    atoms = np.zeros(n_total_atoms, dtype=AtomV2)
    residues_list = []
    chains_list = []

    atom_offset = 0
    res_offset = 0

    for chain_idx, chain_name in enumerate(chain_names):
        chain_atom_start = atom_offset
        chain_res_start = res_offset

        for res_i in range(res_per_chain):
            # Fill atoms for this residue
            for a_i in range(atoms_per_res):
                idx = atom_offset + a_i
                atoms[idx]["name"] = atom_names[a_i]
                atoms[idx]["coords"] = [0.0, 0.0, 0.0]
                atoms[idx]["is_present"] = True
                atoms[idx]["bfactor"] = 0.0
                atoms[idx]["plddt"] = 0.0

            residues_list.append((
                "ALA",       # name
                0,           # res_type (protein)
                res_offset,  # res_idx
                atom_offset, # atom_idx
                atoms_per_res, # atom_num
                1,           # atom_center (CA index within residue)
                1,           # atom_disto
                True,        # is_standard
                True,        # is_present
            ))

            atom_offset += atoms_per_res
            res_offset += 1

        chains_list.append((
            chain_name,           # name
            0,                    # mol_type (protein)
            chain_idx,            # entity_id
            0,                    # sym_id
            chain_idx,            # asym_id
            chain_atom_start,     # atom_idx
            res_per_chain * atoms_per_res, # atom_num
            chain_res_start,      # res_idx
            res_per_chain,        # res_num
            0,                    # cyclic_period
        ))

    residues = np.array(residues_list, dtype=Residue)
    chains = np.array(chains_list, dtype=Chain)
    bonds = np.array([], dtype=BondV2)
    interfaces = np.array([], dtype=Interface)
    mask = np.ones(n_chains, dtype=bool)
    coords = np.zeros(n_total_atoms, dtype=Coords)
    ensemble = np.array([], dtype=Ensemble)

    boltz_struct = StructureV2(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
    )
    return boltz_struct, n_total_atoms


def _make_mock_pred_dict(n_atoms, complex_plddt=0.75, ptm=0.8, iptm=0.7):
    """Build a mock pred_dict mimicking Boltz2.predict_step() output."""
    import torch

    coords = torch.randn(1, n_atoms, 3)  # [samples, atoms, 3]
    masks = torch.ones(n_atoms, dtype=torch.bool)

    return {
        "coords": coords,
        "masks": masks,
        "plddt": torch.ones(1, n_atoms) * 0.85,
        "complex_plddt": torch.tensor([complex_plddt]),
        "complex_iplddt": torch.tensor([0.6]),
        "complex_pde": torch.tensor([0.5]),
        "complex_ipde": torch.tensor([0.4]),
        "confidence_score": torch.tensor([0.82]),
        "ptm": torch.tensor([ptm]),
        "iptm": torch.tensor([iptm]),
        "ligand_iptm": torch.tensor([0.0]),
        "protein_iptm": torch.tensor([iptm]),
        "pair_chains_iptm": {
            0: {0: torch.tensor([ptm])},
        },
    }


class TestBoltzStructureToAtomArray:
    """Tests for _boltz_structure_to_atom_array: Boltz StructureV2 → biotite AtomArray."""

    def test_correct_atom_count(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=3)
        coords = np.random.default_rng(0).random((n_atoms, 3)) * 10
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)
        assert len(arr) == n_atoms

    def test_coordinates_match_input(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=2)
        coords = np.random.default_rng(1).random((n_atoms, 3)) * 10
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)
        np.testing.assert_allclose(arr.coord, coords, atol=1e-5)

    def test_chain_ids_match(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A", "B"], res_per_chain=2)
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        unique_chains = sorted(set(arr.chain_id))
        assert unique_chains == ["A", "B"]

    def test_residue_ids_are_one_based(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=3)
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        unique_res = sorted(set(arr.res_id))
        assert unique_res == [1, 2, 3]

    def test_atom_names_are_backbone(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=1)
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        assert list(arr.atom_name) == ["N", "CA", "C", "O"]

    def test_elements_inferred_correctly(self):
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=1)
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        assert list(arr.element) == ["N", "C", "C", "O"]

    def test_multi_chain_atom_separation(self):
        """Each chain's atoms should have the correct chain_id."""
        boltz_struct, n_atoms = _make_mock_boltz_structure(
            ["A", "B"], res_per_chain=2
        )
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        chain_a_atoms = arr[arr.chain_id == "A"]
        chain_b_atoms = arr[arr.chain_id == "B"]
        assert len(chain_a_atoms) == 8  # 2 res * 4 atoms
        assert len(chain_b_atoms) == 8

    def test_non_present_atoms_filtered(self):
        """Atoms with is_present=False should be excluded."""
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=2)
        # Mark first atom as not present
        boltz_struct.atoms[0]["is_present"] = False
        coords = np.zeros((n_atoms, 3))
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        assert len(arr) == n_atoms - 1

    def test_compatible_with_evedesign_structure(self):
        """AtomArray should be wrappable in evedesign Structure."""
        from evedesign.structure import Structure

        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=3)
        coords = np.random.default_rng(2).random((n_atoms, 3)) * 10
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        structure = Structure(arr)
        assert structure.chains() == ["A"]
        res_df = structure.res_df()
        assert len(res_df) == 3

    def test_multi_chain_structure_extractable(self):
        """Should be able to get_chain after wrapping in Structure."""
        from evedesign.structure import Structure

        boltz_struct, n_atoms = _make_mock_boltz_structure(
            ["A", "B"], res_per_chain=2
        )
        coords = np.random.default_rng(3).random((n_atoms, 3)) * 10
        arr = _boltz_structure_to_atom_array(boltz_struct, coords)

        structure = Structure(arr)
        chain_a = structure.get_chain("A")
        chain_b = structure.get_chain("B")
        assert chain_a.chains() == ["A"]
        assert chain_b.chains() == ["B"]
        assert len(chain_a.res_df()) == 2
        assert len(chain_b.res_df()) == 2


class TestPredictionToInstance:
    """Tests for prediction_to_instance: pred_dict tensors → SystemInstance."""

    @pytest.fixture
    def single_chain_setup(self, tmp_path):
        """Single-chain protein with 3 residues."""
        boltz_struct, n_atoms = _make_mock_boltz_structure(["A"], res_per_chain=3)
        structures_dir = tmp_path / "structures"
        structures_dir.mkdir()
        boltz_struct.dump(structures_dir / "fold_0.npz")

        system = System([Protein(id="prot1", rep="AAA", first_index=1)])
        instance = SystemInstance([EntityInstance(rep="AAA")])
        pred_dict = _make_mock_pred_dict(n_atoms, complex_plddt=0.75)

        # Mock batch with record
        from unittest.mock import MagicMock
        record = MagicMock()
        record.id = "fold_0"
        batch = {"record": [record]}

        return pred_dict, batch, structures_dir, system, instance

    @pytest.fixture
    def two_chain_setup(self, tmp_path):
        """Two-chain protein with 2 residues each."""
        boltz_struct, n_atoms = _make_mock_boltz_structure(
            ["A", "B"], res_per_chain=2
        )
        structures_dir = tmp_path / "structures"
        structures_dir.mkdir()
        boltz_struct.dump(structures_dir / "fold_0.npz")

        system = System([
            Protein(id="prot1", rep="AA", first_index=1),
            Protein(id="prot2", rep="AA", first_index=1),
        ])
        instance = SystemInstance([
            EntityInstance(rep="AA"),
            EntityInstance(rep="AA"),
        ])
        pred_dict = _make_mock_pred_dict(n_atoms, complex_plddt=0.82, ptm=0.9)

        from unittest.mock import MagicMock
        record = MagicMock()
        record.id = "fold_0"
        batch = {"record": [record]}

        return pred_dict, batch, structures_dir, system, instance

    def test_returns_system_instance(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        assert isinstance(result, SystemInstance)

    def test_score_from_complex_plddt(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        assert abs(result.score - 0.75) < 1e-5
        assert result.score == result.confidence

    def test_structure_attached_to_entity(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        assert result[0].models is not None
        assert "A" in result[0].models

    def test_structure_has_correct_atoms(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        structure = result[0].models["A"]
        # 3 residues * 4 backbone atoms = 12
        assert len(structure.atom_array) == 12

    def test_coordinates_are_from_prediction(self, single_chain_setup):
        pred_dict, batch, structures_dir, system, instance = single_chain_setup
        result = prediction_to_instance(
            pred_dict, batch, structures_dir, system, instance
        )
        structure = result[0].models["A"]
        # Coords should NOT be all zeros (the mock pred_dict has random coords)
        assert not np.allclose(structure.atom_array.coord, 0.0)

    def test_sequence_preserved(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        assert "".join(result[0].rep) == "AAA"

    def test_confidence_dict_in_metadata(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        conf = result.metadata["boltz_confidence"]
        assert "complex_plddt" in conf
        assert "ptm" in conf
        assert "iptm" in conf
        assert abs(conf["complex_plddt"] - 0.75) < 1e-5

    def test_plddt_array_in_metadata(self, single_chain_setup):
        result = prediction_to_instance(*single_chain_setup)
        plddt = result.metadata["plddt"]
        assert plddt is not None
        assert len(plddt) > 0

    def test_two_chain_both_have_structures(self, two_chain_setup):
        result = prediction_to_instance(*two_chain_setup)
        assert result[0].models is not None
        assert "A" in result[0].models
        assert result[1].models is not None
        assert "B" in result[1].models

    def test_two_chain_structures_are_separate(self, two_chain_setup):
        result = prediction_to_instance(*two_chain_setup)
        struct_a = result[0].models["A"]
        struct_b = result[1].models["B"]
        # Each should only have its own chain
        assert struct_a.chains() == ["A"]
        assert struct_b.chains() == ["B"]

    def test_two_chain_correct_atom_counts(self, two_chain_setup):
        result = prediction_to_instance(*two_chain_setup)
        # 2 residues * 4 atoms each
        assert len(result[0].models["A"].atom_array) == 8
        assert len(result[1].models["B"].atom_array) == 8
