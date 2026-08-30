"""Contract calendar tests.

An error in tau is an error in the factor loading exp(-kappa*tau), and it does
not average out -- it grows across each roll cycle and biases kappa, which is
the model's central economic read-out. These are pinned against dates that can
be checked independently against exchange records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.expiry import (MONTH_CODES, cl_expiry, contract_chain,  # noqa: E402
                              contract_code, is_business_day,
                              nearby_expiries, roll_indicator,
                              time_to_maturity)

# Independently verifiable last trading days for NYMEX CL.
KNOWN = [
    ((2020, 5), "2020-04-21"),   # CLK20: settled negative on this date
    ((2024, 1), "2023-12-19"),   # 25 Dec is a holiday -> steps back
    ((2024, 12), "2024-11-20"),  # 25 Nov is an ordinary business day
    ((2024, 6), "2024-05-21"),
    ((2023, 7), "2023-06-20"),
    ((2022, 3), "2022-02-22"),
]


@pytest.mark.parametrize("ym,expected", KNOWN)
def test_known_expiries(ym, expected):
    assert cl_expiry(*ym).strftime("%Y-%m-%d") == expected


def test_contract_codes():
    assert contract_code(2025, 12) == "CLZ25"
    assert contract_code(2020, 5) == "CLK20"
    assert MONTH_CODES[0] == "F" and MONTH_CODES[-1] == "Z"


# ------------------------------------------------------------ calendar rules
def test_expiry_is_always_a_business_day():
    for y in range(2015, 2027):
        for m in range(1, 13):
            assert is_business_day(cl_expiry(y, m))


def test_expiry_precedes_delivery_month():
    for y in range(2015, 2027):
        for m in range(1, 13):
            assert cl_expiry(y, m) < pd.Timestamp(year=y, month=m, day=1)


def test_good_friday_is_a_holiday_and_columbus_day_is_not():
    """NYMEX is not the federal calendar. Using USFederalHolidayCalendar
    unmodified moves expiries, and does so silently."""
    assert not is_business_day("2024-03-29")   # Good Friday
    assert is_business_day("2024-10-14")       # Columbus Day
    assert is_business_day("2024-11-11")       # Veterans Day
    assert not is_business_day("2024-12-25")   # Christmas


def test_expiries_are_strictly_increasing():
    chain = contract_chain("2018-01-01", "2026-12-01")
    exp = [c.expiry for c in chain]
    assert all(b > a for a, b in zip(exp, exp[1:]))


# ------------------------------------------------------- maturity structure
@pytest.fixture(scope="module")
def dates():
    return pd.bdate_range("2019-01-02", "2023-12-29")


def test_tau_is_ordered_and_non_negative(dates):
    tau = time_to_maturity(dates, 6)
    assert np.all(tau >= 0)
    assert np.all(np.diff(tau, axis=1) > 0)


def test_tau_shrinks_within_a_roll_cycle_and_jumps_at_roll(dates):
    """The whole reason Z_t and d_t are time-varying."""
    front = time_to_maturity(dates, 1)[:, 0]
    d = np.diff(front)
    assert np.mean(d < 0) > 0.9          # mostly shrinking day to day
    assert np.sum(d > 0) >= 55           # roughly monthly jumps over 5 years


def test_front_contract_reaches_expiry(dates):
    """tau = 0 must occur: the front contract is held to its last trading day."""
    assert time_to_maturity(dates, 1)[:, 0].min() == 0.0


def test_nearby_expiries_are_the_next_m_contracts(dates):
    exp = nearby_expiries(dates, 4)
    assert np.all(exp[:, 0] <= exp[:, 1])
    d64 = pd.DatetimeIndex(dates).to_numpy("datetime64[ns]")
    assert np.all(exp[:, 0] >= d64)      # front never already expired


def test_business_and_calendar_bases_are_close_but_not_identical(dates):
    """Both bases are annualised (252 business days ~ 365.25 calendar days), so
    they are comparable rather than ordered -- neither dominates.

    They are not interchangeable either. Holiday clustering moves them apart by
    up to ~5 days of annualised maturity, worth ~2% in the front-contract
    loading. The basis used here must match the ``dt`` in the transition
    equation; mixing them biases kappa quietly.
    """
    cal = time_to_maturity(dates, 4, basis="calendar")
    bus = time_to_maturity(dates, 4, basis="business")
    diff_days = (bus - cal) * 365.25

    assert np.corrcoef(cal.ravel(), bus.ravel())[0, 1] > 0.99
    assert abs(diff_days.mean()) < 1.0        # no systematic drift
    assert np.abs(diff_days).max() < 10.0     # but a real gap around holidays
    assert np.mean(np.abs(diff_days) > 3.0) > 0.02   # and it is not negligible


def test_unknown_basis_rejected(dates):
    with pytest.raises(ValueError):
        time_to_maturity(dates[:20], 2, basis="actual/360")


# ------------------------------------------------------------ roll indicator
def test_roll_indicator_marks_days_before_expiry(dates):
    ri = roll_indicator(dates, window=3)
    assert 0.05 < ri.mean() < 0.25
    front = nearby_expiries(dates, 1)[:, 0]
    d64 = pd.DatetimeIndex(dates).to_numpy("datetime64[ns]")
    assert ri[d64 == front].all()        # expiry day itself is always flagged


def test_roll_indicator_is_float_for_variance_scaling(dates):
    """Consumed directly as the regressor in scale_variances (eq. 25)."""
    assert roll_indicator(dates).dtype == float


# ------------------------------------------- the error the rule of thumb made
def test_reconstructed_rule_is_materially_wrong(dates):
    """Guards against anyone reintroducing the 'about the 20th' shortcut.

    The prototype approximated expiry as the first of the delivery month less
    eleven days. Against the exchange calendar that is wrong by about ten days
    on average, which is a several-percent error in the short-run loading.
    """
    idx = pd.DatetimeIndex(dates)
    tau = time_to_maturity(idx, 1)[:, 0]
    approx = ((idx.to_period("M") + 1).to_timestamp()
              - pd.Timedelta(days=11) - idx).days.to_numpy(float) / 365.25
    err_days = np.abs(approx - tau) * 365.25
    assert err_days.mean() > 5.0

    kappa = 1.49
    rel = np.abs(np.exp(-kappa * approx) - np.exp(-kappa * tau)) / np.exp(-kappa * tau)
    assert rel.mean() > 0.02
