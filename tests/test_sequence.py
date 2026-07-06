import pytest

from evedesign.sequence import Sequence, Sequences


def _base_sequences():
    hits = [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")]
    return Sequences(seqs=hits, aligned=True, format="a3m")


def test_remap_query_substitution_only():
    # new_query differs only by substitutions -> hits unchanged
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICD")
    assert [s.seq for s in result.seqs] == ["ALSD", "ALCE"]


def test_remap_query_deletion():
    # deletion at column 1 -> that column dropped from every hit
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "V-CD")
    assert [s.seq for s in result.seqs] == ["ASD", "ACE"]


def test_remap_query_insertion():
    # insertion (lowercase 't') between cols 2 and 3 -> gap column added to every hit
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICtD")
    # NOTE: with hits ["ALSD", "ALCE"], hit1's third column is 'S', so it remaps
    # to "ALS-D" (not "ALC-D" as written in the original spec — see report).
    assert [s.seq for s in result.seqs] == ["ALS-D", "ALC-E"]


def test_remap_query_preserves_format_and_ids():
    sequences = _base_sequences()
    result = sequences.remap_query("ALCD", "VICtD")
    assert result.format_ == "a3m"
    assert [s.id_ for s in result.seqs] == ["seq1", "seq2"]


def test_remap_query_unsupported_format():
    hits = [Sequence("ALSD", id="seq1")]
    sequences = Sequences(seqs=hits, aligned=False, format="fasta")
    with pytest.raises(NotImplementedError):
        sequences.remap_query("ALCD", "VICD")


def test_remap_query_wrong_column_count():
    # new_query consumes 5 alignment columns but old_query has 4
    sequences = _base_sequences()
    with pytest.raises(ValueError):
        sequences.remap_query("ALCD", "VICDE")


def test_remap_query_hit_length_mismatch():
    # hit length (3) doesn't match len(old_query) (4)
    hits = [Sequence("ALS", id="seq1")]
    sequences = Sequences(seqs=hits, aligned=True, format="a3m")
    with pytest.raises(ValueError):
        sequences.remap_query("ALCD", "VICD")
