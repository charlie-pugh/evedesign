"""
tests/test_boltzfold.py

Unit tests for the BoltzFoldTransformer class skeleton.
Tests cover class attributes, init, can_model, build, and ready state.
"""

import pytest

from evedesign.system import System, SystemInstance, EntityInstance, Protein, Entity


SEQ = "TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLL"

@pytest.fixture
def single_protein_system():
    return System([Protein(id="test", rep=SEQ, first_index=1)])


@pytest.fixture
def multi_protein_system():
    return System([
        Protein(id="A", rep="AAAAAAAAAA", first_index=1),
        Protein(id="B", rep="CCCCCCCCCC", first_index=1),
    ])

boltz_available = pytest.importorskip(
    "boltz", reason="boltz package not installed"
)

from evedesign.models.boltzfold import BoltzFoldTransformer  # noqa: E402

class TestClassAttributes:
    def test_name(self):
        assert BoltzFoldTransformer.name == "BoltzFold"

    def test_requires_target(self):
        assert BoltzFoldTransformer.requires_target is True

    def test_requires_fixed_length(self):
        assert BoltzFoldTransformer.requires_fixed_length is True

    def test_handles_deletions(self):
        assert BoltzFoldTransformer.handles_deletions is False

    def test_handles_insertions(self):
        assert BoltzFoldTransformer.handles_insertions is False

    def test_supports_gpu(self):
        assert BoltzFoldTransformer.supports_gpu is True

    def test_requires_gpu(self):
        assert BoltzFoldTransformer.requires_gpu is False


class TestInit:
    def test_default_init(self):
        model = BoltzFoldTransformer()
        assert model.model_dir_path is None
        assert model.batch_size == 1
        assert model.device == "cpu"
        assert model.model is None
        assert model.system is None
        assert model.ready is False

    def test_custom_init(self, tmp_path):
        model = BoltzFoldTransformer(
            model_dir_path=tmp_path,
            batch_size=4,
            device="cpu",
            keep_model_after_build=True,
        )
        assert model.model_dir_path == tmp_path
        assert model.batch_size == 4
        assert model.keep_model_after_build is True

    def test_boltz2_defaults(self):
        model = BoltzFoldTransformer()
        assert model.sampling_steps == 50
        assert model.diffusion_samples == 1
        assert model.recycling_steps == 3
        assert model.use_msa_server is False


class TestCanModel:
    def test_single_protein_ok(self, single_protein_system):
        ok, msg = BoltzFoldTransformer.can_model(single_protein_system)
        assert ok is True
        assert msg == ""

    def test_multi_protein_ok(self, multi_protein_system):
        ok, msg = BoltzFoldTransformer.can_model(multi_protein_system)
        assert ok is True

    def test_rejects_data(self, single_protein_system):
        ok, msg = BoltzFoldTransformer.can_model(single_protein_system, data="something")
        assert ok is False
        assert "data" in msg.lower()

    def test_rejects_empty_system(self):
        # System([]) raises ValueError at construction, so empty systems
        # are already rejected before can_model is ever called
        with pytest.raises(ValueError, match="at least one"):
            System([])

    def test_rejects_non_protein(self):
        system = System([Entity(type="rna", rep="AAAUUU", first_index=1)])
        ok, msg = BoltzFoldTransformer.can_model(system)
        assert ok is False
        assert "protein" in msg.lower()

    def test_rejects_no_sequence(self):
        system = System([Protein(id="empty", first_index=1)])
        ok, msg = BoltzFoldTransformer.can_model(system)
        assert ok is False
        assert "sequence" in msg.lower()


class TestBuild:
    def test_build_sets_system(self, single_protein_system):
        model = BoltzFoldTransformer()
        result = model.build(single_protein_system)
        assert model.ready is True
        assert model.system is single_protein_system
        assert result is model  # returns self for chaining

    def test_not_ready_before_build(self):
        model = BoltzFoldTransformer()
        assert model.ready is False
        with pytest.raises(ValueError):
            model.ready_or_raise()

    def test_build_rejects_invalid_system(self):
        model = BoltzFoldTransformer()
        system = System([Entity(type="rna", rep="AAAUUU", first_index=1)])
        with pytest.raises(ValueError):
            model.build(system)

class TestNotImplementedStubs:
    def test_score_not_implemented(self, single_protein_system):
        model = BoltzFoldTransformer()
        model.build(single_protein_system)
        instance = SystemInstance([EntityInstance(rep=SEQ)])
        with pytest.raises(NotImplementedError):
            model.score([instance])


class TestModelLifecycle:
    def test_delete_model_clears_model(self):
        model = BoltzFoldTransformer()
        model.model = "fake_model"
        model._delete_model()
        assert model.model is None

    def test_load_model_skips_if_loaded(self):
        model = BoltzFoldTransformer()
        model.model = "already_loaded"
        model._load_model()
        assert model.model == "already_loaded"


class TestLoadModel:
    def test_calls_download_and_loads_checkpoint(self, monkeypatch):
        """_load_model should download weights, then load from checkpoint."""
        from unittest.mock import MagicMock, patch, call
        from dataclasses import asdict
        from boltz.main import (
            Boltz2DiffusionParams, PairformerArgsV2,
            MSAModuleArgs, BoltzSteeringParams,
        )

        mock_download = MagicMock()
        mock_boltz2_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_boltz2_cls.load_from_checkpoint.return_value = mock_model_instance

        monkeypatch.setattr("evedesign.models.boltzfold.download_boltz2", mock_download)
        monkeypatch.setattr("evedesign.models.boltzfold.Boltz2", mock_boltz2_cls)

        model = BoltzFoldTransformer()
        model._load_model()

        # download_boltz2 called with cache dir
        mock_download.assert_called_once()
        cache_arg = mock_download.call_args[0][0]
        assert str(cache_arg).endswith(".boltz")

        # Boltz2.load_from_checkpoint called with correct checkpoint path
        mock_boltz2_cls.load_from_checkpoint.assert_called_once()
        ckpt_arg = mock_boltz2_cls.load_from_checkpoint.call_args[0][0]
        assert str(ckpt_arg).endswith("boltz2_conf.ckpt")

        # model moved to device and set to eval
        mock_model_instance.to.assert_called_once_with("cpu")
        mock_model_instance.eval.assert_called_once()

        assert model.model is mock_model_instance

    def test_predict_args_use_instance_params(self, monkeypatch):
        """predict_args passed to checkpoint should reflect instance attributes."""
        from unittest.mock import MagicMock

        mock_download = MagicMock()
        mock_boltz2_cls = MagicMock()
        mock_boltz2_cls.load_from_checkpoint.return_value = MagicMock()

        monkeypatch.setattr("evedesign.models.boltzfold.download_boltz2", mock_download)
        monkeypatch.setattr("evedesign.models.boltzfold.Boltz2", mock_boltz2_cls)

        model = BoltzFoldTransformer()
        model.sampling_steps = 10
        model.diffusion_samples = 3
        model.recycling_steps = 5
        model._load_model()

        call_kwargs = mock_boltz2_cls.load_from_checkpoint.call_args[1]
        predict_args = call_kwargs["predict_args"]
        assert predict_args["sampling_steps"] == 10
        assert predict_args["diffusion_samples"] == 3
        assert predict_args["recycling_steps"] == 5

    def test_diffusion_step_scale(self, monkeypatch):
        """Boltz2DiffusionParams.step_scale should be set to 1.5."""
        from unittest.mock import MagicMock
        from dataclasses import asdict
        from boltz.main import Boltz2DiffusionParams

        mock_download = MagicMock()
        mock_boltz2_cls = MagicMock()
        mock_boltz2_cls.load_from_checkpoint.return_value = MagicMock()

        monkeypatch.setattr("evedesign.models.boltzfold.download_boltz2", mock_download)
        monkeypatch.setattr("evedesign.models.boltzfold.Boltz2", mock_boltz2_cls)

        model = BoltzFoldTransformer()
        model._load_model()

        call_kwargs = mock_boltz2_cls.load_from_checkpoint.call_args[1]
        expected = Boltz2DiffusionParams()
        expected.step_scale = 1.5
        assert call_kwargs["diffusion_process_args"] == asdict(expected)

    def test_load_model_moves_to_device(self, monkeypatch):
        """Model should be moved to the configured device."""
        from unittest.mock import MagicMock

        mock_download = MagicMock()
        mock_boltz2_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_boltz2_cls.load_from_checkpoint.return_value = mock_model_instance

        monkeypatch.setattr("evedesign.models.boltzfold.download_boltz2", mock_download)
        monkeypatch.setattr("evedesign.models.boltzfold.Boltz2", mock_boltz2_cls)

        model = BoltzFoldTransformer(device="cuda")
        model._load_model()

        mock_model_instance.to.assert_called_once_with("cuda")
