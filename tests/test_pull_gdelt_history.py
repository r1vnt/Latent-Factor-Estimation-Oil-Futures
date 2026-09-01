"""Unit tests for scripts/pull_gdelt_history.py's flush_output(): it must
never read OUT_PATH back before writing (that read-modify-write cycle is
what OOM'd a live run at ~1.29M accumulated rows -- see the OUT_PATH
comment in the source), and the resulting directory-of-parts must still
read back correctly through plain pd.read_parquet(), full and
column-projected, since every downstream consumer relies on that.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_gdelt_history as m1   # noqa: E402


def _record(ts="20200101000000", url="https://x.com/a", themes="OPEC"):
    # Matches flush_output()'s expected column order: V2.1DATE,
    # DocumentIdentifier, text, SourceCommonName, Themes, *TONE_COLS.
    return (ts, url, "body text", "source.com", themes,
           *(["0.1"] * len(m1.TONE_COLS)))


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(m1, "OUT_PATH", tmp_path / "extract")
    monkeypatch.setattr(m1, "_flush_counter", 0)
    return tmp_path


def test_flush_never_reads_existing_output(wired):
    """The whole point of the rewrite: a second flush must not touch the
    first part file at all, let alone read it back into memory."""
    m1.flush_output([_record(url="https://x.com/1")])
    first_part = next(m1.OUT_PATH.iterdir())
    before_mtime = first_part.stat().st_mtime_ns

    def _forbidden(*a, **k):
        raise AssertionError("flush_output read an existing file -- it must not")
    # Scoped to this block only -- undoing it must not also undo the
    # `wired` fixture's own OUT_PATH patch, which a bare monkeypatch.undo()
    # would do, since both share the same fixture-provided MonkeyPatch.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_parquet", _forbidden)
        m1.flush_output([_record(url="https://x.com/2")])

    parts = sorted(m1.OUT_PATH.iterdir())
    assert len(parts) == 2
    assert first_part.stat().st_mtime_ns == before_mtime   # untouched


def test_multiple_flushes_accumulate_and_read_back(wired):
    n1 = m1.flush_output([_record(url="https://x.com/1"), _record(url="https://x.com/2")])
    n2 = m1.flush_output([_record(url="https://x.com/3")])
    assert n1 == 2
    assert n2 == 1

    parts = list(m1.OUT_PATH.iterdir())
    assert len(parts) == 2   # one file per flush call, nothing merged

    full = pd.read_parquet(m1.OUT_PATH)
    assert len(full) == 3
    assert set(full["DocumentIdentifier"]) == {
        "https://x.com/1", "https://x.com/2", "https://x.com/3"}

    projected = pd.read_parquet(m1.OUT_PATH, columns=["timestamp", "DocumentIdentifier"])
    assert list(projected.columns) == ["timestamp", "DocumentIdentifier"]
    assert len(projected) == 3


def test_intra_batch_duplicates_dropped_but_not_cross_batch(wired):
    """Duplicates within one flush call are still removed (cheap, and free
    from the DataFrame construction that already happens); duplicates
    across two separate flush calls are NOT -- that dedup now happens at
    read time in every downstream consumer instead, the tradeoff that
    makes bounded-memory writes possible."""
    n = m1.flush_output([_record(url="https://x.com/dup"), _record(url="https://x.com/dup")])
    assert n == 1

    m1.flush_output([_record(url="https://x.com/dup")])
    full = pd.read_parquet(m1.OUT_PATH)
    assert len(full) == 2   # one from each flush; not deduped across parts


def test_part_filenames_unique_within_a_run(wired):
    for i in range(5):
        m1.flush_output([_record(url=f"https://x.com/{i}")])
    parts = list(m1.OUT_PATH.iterdir())
    assert len(parts) == 5
    assert len({p.name for p in parts}) == 5
