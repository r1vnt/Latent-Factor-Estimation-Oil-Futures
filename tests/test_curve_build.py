"""Panel-assembly tests.

The rules on the log transform matter more than they look: a filter that
silently floors or drops a bad price corrupts every downstream likelihood
without raising an error. Each rule below is pinned by a test that would
fail if a future edit "helpfully" changed the behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

DT = 1.0 / 252.0


def _base_prices(n: int = 60, m: int = 4, start: str = "2019-01-02",
                  seed: int = 0) -> pd.DataFrame:
    """A clean wide panel: m nearby-contract columns, small daily noise."""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    front = 60.0 + np.cumsum(rng.normal(0, 0.3, n))
    cols = {i: front + (i - 1) * 0.5 for i in range(1, m + 1)}
    return pd.DataFrame(cols, index=dates)


# ------------------------------------------------------------ rule 1 & 2
def test_nonpositive_price_becomes_nan_without_dropping_the_row():
    wide = _base_prices()
    bad_date = wide.index[10]
    wide.loc[bad_date, 1] = -12.5

    panel = Panel.from_prices(wide, verbose=False)
    i = panel.dates.get_loc(bad_date)

    assert not np.isfinite(panel.log_f[i, 0])   # the bad cell: NaN
    assert np.isfinite(panel.log_f[i, 1])       # sibling contract: untouched
    assert bad_date in panel.dates              # row itself: retained


def test_all_nan_row_keeps_its_date():
    """Dropping rows makes dt uneven; the filter has no exact treatment
    for that. A row with nothing observable is still a date."""
    wide = _base_prices()
    bad_date = wide.index[20]
    wide.loc[bad_date, :] = 0.0

    panel = Panel.from_prices(wide, verbose=False)

    assert len(panel.dates) == len(wide)
    assert bad_date in panel.dates
    i = panel.dates.get_loc(bad_date)
    assert not np.isfinite(panel.log_f[i]).any()


# ------------------------------------------------------------------ rule 3
def test_log_finite_mask_matches_source_finite_positive_mask_exactly():
    """Guards against a future 'fix' that floors instead of NaN-ing.

    If a non-positive price were floored to some tiny epsilon before the
    log, this cell would become a finite (very negative) number and the
    mask equality below would break.
    """
    wide = _base_prices()
    wide.iloc[5, 0] = -3.0
    wide.iloc[15, 2] = 0.0
    price = wide.to_numpy(dtype=float)

    panel = Panel.from_prices(wide, verbose=False)

    expected = np.isfinite(price) & (price > 0)
    assert np.array_equal(np.isfinite(panel.log_f), expected)


# --------------------------------------------------------------- screens
def test_low_price_screen_flags_the_unstable_band():
    wide = _base_prices()
    wide.iloc[8, 0] = 3.25   # positive, below the $5 default

    panel = Panel.from_prices(wide, verbose=False)

    assert panel.flags["low_price"][8, 0]
    assert panel.flags["low_price"].sum() == 1


def test_stale_screen_flags_runs_at_the_threshold_not_below_it():
    wide = _base_prices()
    wide.iloc[10:15, 1] = 55.0   # 5 identical settlements -> stale
    wide.iloc[30:34, 2] = 60.0   # 4 identical settlements -> not yet

    panel = Panel.from_prices(wide, verbose=False)

    assert panel.flags["stale"][10:15, 1].all()
    assert not panel.flags["stale"][30:34, 2].any()


def test_outlier_return_screen_uses_mad_not_stdev():
    wide = _base_prices(n=120)
    # A permanent level shift from day 50 onward: exactly one abnormal
    # return, at the shift itself. (A one-day spike-and-revert would flag
    # *two* returns -- the jump out and the jump back -- which is correct
    # but would make a weaker test of the threshold itself.)
    wide.iloc[50:, 0] = wide.iloc[50:, 0] * 1.5

    panel = Panel.from_prices(wide, verbose=False)

    assert panel.flags["outlier_return"][50, 0]
    assert panel.flags["outlier_return"][:, 0].sum() == 1   # noise stays quiet


# ------------------------------------------------------------- QualityReport
def test_quality_report_counts_and_dates_match_the_flags():
    wide = _base_prices()
    wide.iloc[8, 0] = 3.25
    wide.iloc[10:15, 1] = 55.0

    panel = Panel.from_prices(wide, verbose=False)
    q = panel.quality

    assert q.flag_counts["low_price"] == int(panel.flags["low_price"].sum())
    assert q.flag_counts["stale"] == int(panel.flags["stale"].sum())
    assert wide.index[8] in q.flag_dates["low_price"]
    assert all(d in q.flag_dates["stale"] for d in wide.index[10:15])
    assert q.n_cells == panel.n * panel.m


# ------------------------------------------------------------------- slice
def test_slice_subsets_every_array_together():
    """If a future edit to slice() forgets an array -- especially a flag
    added after this was written -- this is the test that catches it."""
    wide = _base_prices(n=80)
    wide.iloc[8, 0] = 3.25
    wide.iloc[10:15, 1] = 55.0
    wide.iloc[50, 2] = wide.iloc[49, 2] * 1.6

    panel = Panel.from_prices(wide, verbose=False)
    start, end = wide.index[20], wide.index[60]
    sub = panel.slice(start, end)
    mask = (panel.dates >= start) & (panel.dates <= end)

    assert list(sub.dates) == list(panel.dates[mask])
    assert np.array_equal(sub.log_f, panel.log_f[mask], equal_nan=True)
    assert np.array_equal(sub.tau, panel.tau[mask])
    assert np.array_equal(sub.roll, panel.roll[mask])
    assert set(sub.flags) == set(panel.flags)
    for name in panel.flags:
        assert np.array_equal(sub.flags[name], panel.flags[name][mask]), name


# ------------------------------------------------------------- long vs wide
def test_long_format_input_matches_wide_format_input():
    wide = _base_prices(n=40)
    long = (wide.rename_axis("date").reset_index()
                .melt(id_vars="date", var_name="contract", value_name="settle"))

    panel_wide = Panel.from_prices(wide, verbose=False)
    panel_long = Panel.from_prices(long, verbose=False)

    assert np.array_equal(panel_wide.log_f, panel_long.log_f, equal_nan=True)
    assert np.array_equal(panel_wide.tau, panel_long.tau)
    assert list(panel_wide.dates) == list(panel_long.dates)
    assert panel_wide.labels == panel_long.labels


# --------------------------------------------------------- the test that matters most
def test_negative_price_excluded_end_to_end_through_the_filter():
    """Proves the exclusion is exact end-to-end: a non-positive price is
    dropped from the likelihood, not interpolated or silently filled
    somewhere between the panel and the filter."""
    wide = _base_prices(n=300, m=4)
    wide.iloc[123, 1] = -8.0

    panel = Panel.from_prices(wide, verbose=False)
    ll, x_filt, P_filt, resid, n_obs = run_spec(
        REGISTRY["S0"], SS_OIL, panel.log_f, panel.tau, DT
    )

    assert n_obs == int(np.isfinite(panel.log_f).sum())
    assert np.isfinite(ll)
