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


# lowercase (A3M insert-state) hits; old_query = "ALCD" (4 match columns)

def _lc(hit):
    return Sequences(seqs=[Sequence(hit, id="h")], aligned=True, format="a3m")


def test_remap_query_lc_substitution():
    # hit "ALsCD" (insert s between cols 1,2), new "VICD" (all substitutions)
    # trace: V->A ; I->L ; C->carry s then C = "sC" ; D->D
    result = _lc("ALsCD").remap_query("ALCD", "VICD")
    assert [s.seq for s in result.seqs] == ["ALsCD"]


def test_remap_query_lc_deletion():
    # hit "ALsCD", new "A-CD" (delete col 1 = L)
    # trace: A->A ; '-'-> no leading lc, take L, drop = "" ; C->carry s then C = "sC" ; D->D
    result = _lc("ALsCD").remap_query("ALCD", "A-CD")
    assert [s.seq for s in result.seqs] == ["AsCD"]


def test_remap_query_lc_deletion_at_insert_boundary():
    # hit "ALsCD", new "AL-D" (delete col 2 = C, which has insert s before it)
    # trace: A->A ; L->L ; '-'-> carry leading s then take C, drop = "s" ; D->D
    result = _lc("ALsCD").remap_query("ALCD", "AL-D")
    assert [s.seq for s in result.seqs] == ["ALsD"]


def test_remap_query_lc_insertion_in_query():
    # hit "ALsCD", new "ALCtD" (insert t between cols 2,3)
    # trace: A->A ; L->L ; C->carry s then C = "sC" ; t-> gap ; D->D
    result = _lc("ALsCD").remap_query("ALCD", "ALCtD")
    assert [s.seq for s in result.seqs] == ["ALsC-D"]


def test_remap_query_lc_trailing_insertion():
    # hit "ALCDy" (trailing insert y), new "ALCD"
    # trace: A,L,C,D consumed ; then trailing lowercase y carried
    result = _lc("ALCDy").remap_query("ALCD", "ALCD")
    assert [s.seq for s in result.seqs] == ["ALCDy"]


def test_remap_query_lc_leading_insertion():
    # hit "xALCD" (leading insert x), new "VLCD"
    # trace: V->carry x then A = "xA" ; L->L ; C->C ; D->D
    result = _lc("xALCD").remap_query("ALCD", "VLCD")
    assert [s.seq for s in result.seqs] == ["xALCD"]


def test_remap_query_lc_too_many_match_columns_raises():
    # hit "ALCDE" has 5 match columns vs old_query's 4
    with pytest.raises(ValueError):
        _lc("ALCDE").remap_query("ALCD", "VICD")


def test_remap_query_lc_too_few_match_columns_raises():
    # hit "ALC" has 3 match columns vs old_query's 4
    with pytest.raises(ValueError):
        _lc("ALC").remap_query("ALCD", "VICD")


def test_remap_query_mixed_clean_and_lowercase_hits():
    hits = [Sequence("ALSD", id="clean"), Sequence("ALsCD", id="lc")]
    sequences = Sequences(seqs=hits, aligned=True, format="a3m")
    result = sequences.remap_query("ALCD", "ALCtD")
    assert [s.seq for s in result.seqs] == ["ALS-D", "ALsC-D"]
