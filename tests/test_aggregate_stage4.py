"""Unit tests for scripts/aggregate_stage4.py's coverage-normalisation and
missing-day handling: a day with zero fetched files must come out as an
explicit NaN row (never absent, never a fabricated zero), and a partial-
coverage day's I_t must be scaled up to what full coverage would show.
"""
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate_stage4 as s4          # noqa: E402
import pull_gdelt_history as m1        # noqa: E402


def _write_stage1_manifest(path, day_to_ok_hours):
    """day_to_ok_hours: {"YYYYMMDD": n_ok_files}. One synthetic hourly
    timestamp per ok file, all status 'ok'."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "status", "n_rows"])
        for day, n_ok in day_to_ok_hours.items():
            for hour in range(n_ok):
                w.writerow([f"{day}{hour:02d}0000", "ok", 10])


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point every module-level path/window constant the pipeline touches
    at an isolated tmp_path, so this test can't read or clobber the live
    data/ directory."""
    monkeypatch.setattr(m1, "MANIFEST_PATH", tmp_path / "stage1_manifest.csv")
    monkeypatch.setattr(s4, "STAGE1_OUTPUT", tmp_path / "stage1_extract.parquet")
    monkeypatch.setattr(s4, "STAGE2_MANIFEST", tmp_path / "stage2_manifest.csv")
    monkeypatch.setattr(s4, "STAGE3_LOG", tmp_path / "stage3_log.parquet")
    monkeypatch.setattr(s4, "OUT_PATH", tmp_path / "stage4_daily.parquet")
    monkeypatch.setattr(s4, "WINDOW_START", dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))
    # Exclusive, matching pull_gdelt_history.py's END convention: the last
    # actual day in the window is 2024-01-03, one day before this.
    monkeypatch.setattr(s4, "WINDOW_END", dt.datetime(2024, 1, 4, tzinfo=dt.timezone.utc))
    return tmp_path


def _url(i):
    return f"https://example.com/story-{i}"


def test_full_partial_and_zero_coverage_days(wired):
    tmp_path = wired

    # Day 1 (2024-01-01): full coverage, 24/24 files.
    # Day 2 (2024-01-02): partial coverage, 12/24 files.
    # Day 3 (2024-01-03): zero coverage -- absent from the stage1 manifest
    # entirely, exactly as a genuinely unfetched day would be.
    _write_stage1_manifest(m1.MANIFEST_PATH, {"20240101": 24, "20240102": 12})

    s1_rows = [
        {"DocumentIdentifier": _url(1), "Themes": "OPEC;ECON_OILPRICE",
        **{c: 1.0 for c in m1.TONE_COLS}},
        {"DocumentIdentifier": _url(2), "Themes": "ENV_NATURALGAS",
        **{c: -1.0 for c in m1.TONE_COLS}},
        {"DocumentIdentifier": _url(3), "Themes": "FUELPRICES",
        **{c: 0.0 for c in m1.TONE_COLS}},
    ]
    pd.DataFrame(s1_rows).to_parquet(s4.STAGE1_OUTPUT, index=False)

    # Stage 2 manifest: raw (pre-normalisation) I_t for the two days that
    # were actually scanned.
    with open(s4.STAGE2_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "I_t", "n_candidates", "fetch_failed", "no_text",
                   "lang_skip", "duplicate", "scored"])
        w.writerow(["20240101", 240, 100, 30, 5, 5, 0, 60])
        w.writerow(["20240102", 120, 100, 40, 5, 5, 0, 50])

    s3_rows = [
        {"date": "20240101", "url_hash": s4.url_hash(_url(1)),
        "finbert_score": 0.8, "outcome": "scored"},
        {"date": "20240101", "url_hash": s4.url_hash(_url(2)),
        "finbert_score": -0.4, "outcome": "scored"},
        {"date": "20240102", "url_hash": s4.url_hash(_url(3)),
        "finbert_score": 0.2, "outcome": "scored"},
    ]
    pd.DataFrame(s3_rows).to_parquet(s4.STAGE3_LOG, index=False)

    s4.main()

    out = pd.read_parquet(s4.OUT_PATH)
    out = out.set_index("date")

    # Every day in the window is a row, including the untouched one.
    assert len(out) == 3
    assert pd.Timestamp("2024-01-03") in out.index

    d1 = out.loc[pd.Timestamp("2024-01-01")]
    d2 = out.loc[pd.Timestamp("2024-01-02")]
    d3 = out.loc[pd.Timestamp("2024-01-03")]

    # Full-coverage day: I_t passes through unscaled, coverage == 1.0.
    assert d1["files_present"] == 24
    assert d1["coverage"] == pytest.approx(1.0)
    assert d1["I_t"] == pytest.approx(240.0)
    assert d1["n_t"] == 2
    assert d1["sent_mean"] == pytest.approx((0.8 + -0.4) / 2)
    assert d1["theme_opec_policy"] == pytest.approx(0.5)   # 1 of 2 scored docs

    # Partial-coverage day: raw I_t=120 at 12/24 files scales to 240, the
    # same reading a full day would have shown at this rate.
    assert d2["files_present"] == 12
    assert d2["coverage"] == pytest.approx(0.5)
    assert d2["I_t"] == pytest.approx(240.0)
    assert d2["n_t"] == 1

    # Zero-coverage day: NaN everywhere, not zero, not absent.
    assert d3["files_present"] == 0
    assert d3["coverage"] == pytest.approx(0.0)
    assert np.isnan(d3["I_t"])
    assert np.isnan(d3["n_t"])
    assert np.isnan(d3["sent_mean"])
    assert np.isnan(d3["intensity"])
    for cat in s4.THEME_CATEGORIES:
        assert np.isnan(d3[f"theme_{cat}"])


def test_zero_coverage_never_becomes_a_zero_even_with_a_stage2_row(wired):
    """A day that stage 2 attempted and logged I_t=0 for (ran, found no
    candidates) is a different situation from a day stage 1 never fetched
    at all -- but both must come out NaN here, since eq. (24) has no way
    to use "confirmed zero volume" any differently from "no reading," and
    fabricating a distinction downstream that doesn't exist upstream would
    be worse than collapsing them."""
    tmp_path = wired
    _write_stage1_manifest(m1.MANIFEST_PATH, {})   # nothing fetched at all

    pd.DataFrame([{"DocumentIdentifier": _url(1), "Themes": "OPEC",
                  **{c: 0.0 for c in m1.TONE_COLS}}]).to_parquet(
        s4.STAGE1_OUTPUT, index=False)

    with open(s4.STAGE2_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "I_t", "n_candidates", "fetch_failed", "no_text",
                   "lang_skip", "duplicate", "scored"])
        w.writerow(["20240101", 0, 0, 0, 0, 0, 0, 0])

    pd.DataFrame(columns=["date", "url_hash", "finbert_score",
                          "outcome"]).to_parquet(s4.STAGE3_LOG, index=False)

    s4.main()

    out = pd.read_parquet(s4.OUT_PATH).set_index("date")
    d1 = out.loc[pd.Timestamp("2024-01-01")]
    assert d1["files_present"] == 0
    assert np.isnan(d1["I_t"])


def test_intensity_standardised_within_year_not_globally(wired, monkeypatch):
    """Two years at wildly different I_t scales (10^2 vs 10^5) -- if
    standardisation were global, the small-scale year's z-scores would be
    driven far from 0 by the huge-scale year dominating the combined
    mean/std. Standardised per year instead, every year with exactly two
    valid days has a clean, scale-independent signature: +-1/sqrt(2),
    regardless of the year's absolute I_t level. That signature landing
    identically in both years is direct evidence the grouping is per-year,
    not global -- explore/08_retention_rate_by_year.py found real
    non-trending year-to-year retention swings (47%-81%) that a global
    standardisation would otherwise leak into gamma in eq. (24) as spurious
    year-level steps."""
    tmp_path = wired
    monkeypatch.setattr(s4, "WINDOW_START", dt.datetime(2023, 12, 30, tzinfo=dt.timezone.utc))
    monkeypatch.setattr(s4, "WINDOW_END", dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc))

    _write_stage1_manifest(m1.MANIFEST_PATH, {
        "20231230": 24, "20231231": 24, "20240101": 24, "20240102": 24,
    })
    pd.DataFrame([{"DocumentIdentifier": _url(1), "Themes": "OPEC",
                  **{c: 0.0 for c in m1.TONE_COLS}}]).to_parquet(
        s4.STAGE1_OUTPUT, index=False)

    with open(s4.STAGE2_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "I_t", "n_candidates", "fetch_failed", "no_text",
                   "lang_skip", "duplicate", "scored"])
        # 2023: I_t in the hundreds. 2024: I_t in the hundreds of thousands.
        w.writerow(["20231230", 100, 0, 0, 0, 0, 0, 0])
        w.writerow(["20231231", 300, 0, 0, 0, 0, 0, 0])
        w.writerow(["20240101", 100000, 0, 0, 0, 0, 0, 0])
        w.writerow(["20240102", 300000, 0, 0, 0, 0, 0, 0])

    pd.DataFrame(columns=["date", "url_hash", "finbert_score",
                          "outcome"]).to_parquet(s4.STAGE3_LOG, index=False)

    s4.main()

    out = pd.read_parquet(s4.OUT_PATH).set_index("date")
    z = 1 / (2 ** 0.5)
    for lo, hi in [("2023-12-30", "2023-12-31"), ("2024-01-01", "2024-01-02")]:
        vals = sorted([out.loc[pd.Timestamp(lo), "intensity"],
                       out.loc[pd.Timestamp(hi), "intensity"]])
        assert vals[0] == pytest.approx(-z, abs=1e-6)
        assert vals[1] == pytest.approx(z, abs=1e-6)
