import math

import pytest

from evedesign.models import oasis_humanness
from evedesign.models.oasis_humanness import OASisHumanness
from evedesign.system import Entity, EntityInstance, Mutation, Protein, System, SystemInstance


class FakePrombDatabase:
    def compute_peptide_content(self, sequence: str) -> float:
        return sequence.count("W") / len(sequence)


@pytest.fixture
def mock_promb(monkeypatch):
    monkeypatch.setattr(
        oasis_humanness, "init_db", lambda name, verbose=True: FakePrombDatabase()
    )
    monkeypatch.setattr(OASisHumanness, "available", True)


def test_raises_clear_error_without_optional_dependency():
    if oasis_humanness.IMPORT_AVAILABLE:
        pytest.skip("promb is installed")

    with pytest.raises(ImportError, match="promb"):
        OASisHumanness()


def test_can_model_accepts_protein_only_systems():
    valid, reason = OASisHumanness.can_model(System([Protein(rep="AAAAAAAAA")]))
    assert valid, reason

    valid, reason = OASisHumanness.can_model(
        System([Entity(type="dna", rep="AAAAAAAAA", first_index=1)])
    )
    assert not valid
    assert "protein" in reason

    valid, reason = OASisHumanness.can_model(System([Protein(rep="AAAAAAAA")]))
    assert not valid
    assert "at least 9" in reason


def test_scores_selected_chains_with_mean_aggregation(mock_promb):
    system = System([
        Protein(id="heavy", rep="AAAAAAAAA", first_index=1),
        Protein(id="light", rep="WWWWWWWWW", first_index=1),
    ])
    instance = SystemInstance([
        EntityInstance(rep="WWAAAAAAA"),
        EntityInstance(rep="WWWWWWWWW"),
    ])

    scored = OASisHumanness().build(system).score([instance])
    assert scored[0] is not instance
    assert instance.score is None
    assert scored[0].score == pytest.approx((2 / 9 + 1.0) / 2)

    selected = OASisHumanness(entities=[0]).build(system).score([instance])
    assert selected[0].score == pytest.approx(2 / 9)


def test_default_mutation_scorer_returns_relative_scores(mock_promb):
    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    instance = SystemInstance([EntityInstance(rep="AAAAAAAAA")])
    model = OASisHumanness().build(system)

    scored = model.score_mutants(
        instance,
        [[Mutation(entity=0, pos=1, ref="A", to="W")]],
    )

    assert math.isclose(scored[0].score, 1 / 9)


def test_database_is_not_loaded_until_scoring(monkeypatch):
    calls = []

    def spy_init_db(name, verbose=True):
        calls.append(name)
        return FakePrombDatabase()

    monkeypatch.setattr(oasis_humanness, "init_db", spy_init_db)
    monkeypatch.setattr(OASisHumanness, "available", True)

    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    model = OASisHumanness().build(system)
    assert calls == []

    model.score([SystemInstance([EntityInstance(rep="AAAAAAAAA")])])
    assert calls == ["human-oas"]

    model.score([SystemInstance([EntityInstance(rep="AAAAAAAAA")])])
    assert calls == ["human-oas"]


def test_rejects_short_instance_sequences(mock_promb):
    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    model = OASisHumanness().build(system)

    with pytest.raises(ValueError, match="at least 9"):
        model.score([SystemInstance([EntityInstance(rep="AAAAAAAA")])])


def test_score_of_empty_instance_list_is_empty(mock_promb):
    system = System([Protein(rep="AAAAAAAAA", first_index=1)])
    assert OASisHumanness().build(system).score([]) == []


@pytest.mark.skipif(
    not oasis_humanness.IMPORT_AVAILABLE, reason="promb is not installed"
)
def test_real_promb_ranks_humanized_above_murine():
    """End-to-end check against the packaged human-OAS database.

    Trastuzumab's humanized VH must score above its murine (4D5) precursor.
    """
    humanized_vh = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
        "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )
    murine_vh = (
        "QVQLQQSGPELVKPGASVKISCKASGYTFTDYNMDWVKQSHGKSLEWIGDINPNNGGTIYNQKFKG"
        "KATLTVDKSSSTAYMELRSLTSEDTAVYYCARNYYGSSLSMDYWGQGTSVTVSS"
    )

    system = System([Protein(rep=humanized_vh, first_index=1)])
    model = OASisHumanness().build(system)
    scored = model.score([
        SystemInstance([EntityInstance(rep=humanized_vh)]),
        SystemInstance([EntityInstance(rep=murine_vh)]),
    ])

    assert 0.0 <= scored[1].score < scored[0].score <= 1.0
