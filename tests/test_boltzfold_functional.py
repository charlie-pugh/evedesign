"""
tests/test_boltzfold_functional.py

End-to-end functional test for BoltzFoldTransformer.
Requires model weights (~3 GB download on first run) and boltz package.

Run with: pytest tests/test_boltzfold_functional.py -v -s
"""

import pytest
import yaml

from evedesign.system import System, SystemInstance, EntityInstance, Protein
from evedesign.models.boltz.convert import system_instance_to_yaml
from evedesign.models.boltzfold import BoltzFoldTransformer

# EcCM - small, fast to fold
SEQ = "TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLLSHRPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVLTQQALLQQH"


@pytest.fixture(scope="module")
def system():
    return System([Protein(id="EcCM", rep=SEQ, first_index=2)])


@pytest.fixture(scope="module")
def instance():
    return SystemInstance([EntityInstance(rep=SEQ)])


@pytest.fixture(scope="module")
def transformer(system):
    t = BoltzFoldTransformer(
        device="cpu",
        sampling_steps=5,
        diffusion_samples=1,
        recycling_steps=3,
        use_msa_server=False,
    )
    t.build(system)
    yield t
    t._delete_model()


class TestSystemInstanceToYaml:
    def test_yaml_content(self, system, instance, tmp_path):
        yaml_path = system_instance_to_yaml(
            system, instance,
            output_path=tmp_path / "input" / "EcCM_fold.yaml",
            use_msa_server=False,
        )

        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())

        assert data["version"] == 1
        assert len(data["sequences"]) == 1
        assert data["sequences"][0]["protein"]["id"] == "A"
        assert data["sequences"][0]["protein"]["sequence"] == SEQ
        assert data["sequences"][0]["protein"]["msa"] == "empty"


class TestTransformRoundTrip:
    @pytest.fixture(scope="class")
    def result(self, transformer, instance):
        results = transformer.transform([instance])
        assert len(results) == 1
        return results[0]

    def test_result_type(self, result):
        assert isinstance(result, SystemInstance)

    def test_sequence_preserved(self, result):
        assert "".join(result[0].rep) == SEQ

    def test_structure_attached(self, result):
        assert result[0].models is not None
        assert "A" in result[0].models
        assert len(result[0].models["A"].atom_array) > 0

    def test_confidence_scores(self, result):
        assert 0.0 < result.score <= 1.0
        assert result.score == result.confidence

    def test_metadata_confidence_dict(self, result):
        assert "boltz_confidence" in result.metadata
        conf = result.metadata["boltz_confidence"]
        assert "complex_plddt" in conf

    def test_plddt_array(self, result):
        plddt = result.metadata.get("plddt")
        if plddt is not None:
            assert plddt.shape[0] > 0
            assert 0.0 <= plddt.mean() <= 1.0
