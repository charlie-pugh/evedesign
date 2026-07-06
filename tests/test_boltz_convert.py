from pathlib import Path

from evedesign.models.boltz.convert import _write_a3m
from evedesign.sequence import Sequence, Sequences
from evedesign.system import Entity, EntityInstance


def _read_a3m(path: Path) -> list[tuple[str, str]]:
    """Return [(header, seq), ...] from an A3M file (query first, then hits)."""
    records = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
            elif header is not None:
                records.append((header, line))
                header = None
    return records


def _entity_with_msa(rep: str, hits: list[Sequence]) -> Entity:
    return Entity(
        type="protein",
        rep=rep,
        id="query",
        first_index=1,
        sequences=Sequences(seqs=hits, aligned=True, format="a3m"),
    )


def test_write_a3m_old_query_none_writes_verbatim(tmp_path):
    # Back-compat: no old_query -> hits written exactly as stored
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="VICD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=None)

    records = _read_a3m(out)
    assert records[0] == ("query", "VICD")          # query line is the instance rep
    assert [r[1] for r in records[1:]] == ["ALSD", "ALCE"]  # hits unchanged


def test_write_a3m_old_query_equals_instance_no_remap(tmp_path):
    # old_query == instance rep -> remap skipped, hits verbatim
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="ALCD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    assert records[0] == ("query", "ALCD")
    assert [r[1] for r in records[1:]] == ["ALSD", "ALCE"]


def test_write_a3m_remap_applied_on_deletion(tmp_path):
    # old_query differs (deletion at column 1) -> hits remapped, column dropped
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="V-CD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    # query line is the normalized designed sequence: gap stripped
    assert records[0] == ("query", "VCD")
    # hits remapped: column 1 removed from each
    assert [r[1] for r in records[1:]] == ["ASD", "ACE"]


def test_write_a3m_remap_applied_on_insertion(tmp_path):
    # insertion (lowercase 't') -> gap column added to each hit
    entity = _entity_with_msa("ALCD", [Sequence("ALSD", id="seq1"), Sequence("ALCE", id="seq2")])
    instance = EntityInstance(rep="VICtD")
    out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)

    records = _read_a3m(out)
    # query line normalized: lowercase insertion uppercased
    assert records[0] == ("query", "VICTD")
    assert [r[1] for r in records[1:]] == ["ALS-D", "ALC-E"]


def test_write_a3m_remap_failure_falls_back_verbatim(tmp_path):
    # Malformed input: hit length (3) != len(old_query) (4) -> remap_query raises
    # ValueError; _write_a3m must fall back to verbatim and log a warning.
    entity = _entity_with_msa("ALCD", [Sequence("ALS", id="seq1")])
    instance = EntityInstance(rep="VICD")

    import loguru

    messages: list[str] = []
    handler_id = loguru.logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        out = _write_a3m(entity, instance, tmp_path / "msa" / "A.a3m", old_query=entity.rep)
    finally:
        loguru.logger.remove(handler_id)

    records = _read_a3m(out)
    assert records[0] == ("query", "VICD")
    assert [r[1] for r in records[1:]] == ["ALS"]  # unmodified hit
    assert any("could not remap MSA" in m for m in messages)
