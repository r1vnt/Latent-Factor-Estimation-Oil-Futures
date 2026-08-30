"""NYMEX WTI (CL) contract expiry calendar.

Time to maturity is not a detail. The short-run factor loads on the curve as
``exp(-kappa*tau)``, so an error in ``tau`` is an error in the loading, and the
error grows across each roll cycle rather than averaging out. The prototype
reconstructed expiries from a rule of thumb ("about the 20th of the preceding
month"), which is wrong by up to several days. This module implements the
actual exchange rule.

**CME rule (NYMEX Light Sweet Crude Oil, CL).** Trading in the current delivery
month ceases on the third business day prior to the twenty-fifth calendar day
of the month preceding the delivery month. If that twenty-fifth calendar day is
not a business day, trading ceases on the third business day prior to the last
business day preceding the twenty-fifth.

Verified against three known expiries in ``tests/test_expiry.py``, including
CLK20 -- the May 2020 contract that settled negative on 20 April 2020, expiring
21 April 2020.

**Business days here are NYMEX business days**, which are not US federal
banking days. NYMEX observes Good Friday, which the federal calendar does not,
and trades on Columbus Day and Veterans Day, which the federal calendar treats
as holidays. Using ``USFederalHolidayCalendar`` unmodified shifts expiries and
is a silent error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday,
                                    USFederalHolidayCalendar)
from pandas.tseries.offsets import CustomBusinessDay

#: CME month codes, January through December.
MONTH_CODES = "FGHJKMNQUVXZ"

TRADING_DAYS_PER_YEAR = 252.0
DAYS_PER_YEAR = 365.25


class NymexHolidayCalendar(AbstractHolidayCalendar):
    """NYMEX holidays: US federal, less Columbus/Veterans, plus Good Friday."""

    rules = [
        r for r in USFederalHolidayCalendar.rules
        if r.name not in ("Columbus Day", "Veterans Day")
    ] + [GoodFriday]


_CAL = NymexHolidayCalendar()
NYMEX_BDAY = CustomBusinessDay(calendar=_CAL)


def _holidays(start="1990-01-01", end="2035-12-31") -> pd.DatetimeIndex:
    return _CAL.holidays(pd.Timestamp(start), pd.Timestamp(end))


def is_business_day(ts) -> bool:
    ts = pd.Timestamp(ts).normalize()
    return bool(ts.weekday() < 5 and ts not in _holidays())


def contract_code(year: int, month: int, prefix: str = "CL") -> str:
    """CME contract code, e.g. ``contract_code(2025, 12) -> 'CLZ25'``."""
    return f"{prefix}{MONTH_CODES[month - 1]}{year % 100:02d}"


def cl_expiry(year: int, month: int) -> pd.Timestamp:
    """Last trading day for the CL contract delivering in ``year``-``month``.

    Implements the exchange rule stated in the module docstring.
    """
    delivery = pd.Timestamp(year=year, month=month, day=1)
    prior = delivery - pd.DateOffset(months=1)
    ref = pd.Timestamp(year=prior.year, month=prior.month, day=25)

    if not is_business_day(ref):
        # step back to the last business day preceding the 25th
        while not is_business_day(ref):
            ref -= pd.Timedelta(days=1)

    # third business day prior to ref
    return (ref - 3 * NYMEX_BDAY).normalize()


@dataclass(frozen=True)
class Contract:
    """One listed contract."""

    year: int
    month: int
    code: str
    expiry: pd.Timestamp

    @property
    def delivery(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=self.month, day=1)


def contract_chain(start, end, prefix: str = "CL") -> list[Contract]:
    """Every monthly contract with a delivery month in ``[start, end]``."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    while cur <= e:
        out.append(Contract(cur.year, cur.month,
                            contract_code(cur.year, cur.month, prefix),
                            cl_expiry(cur.year, cur.month)))
        cur += pd.DateOffset(months=1)
    return out


def nearby_expiries(dates, m: int, prefix: str = "CL") -> np.ndarray:
    """Expiry dates of the ``m`` nearby contracts on each observation date.

    The *n*-th nearby on date ``d`` is the *n*-th contract whose last trading
    day is on or after ``d``. Contracts are dropped from the front on their
    expiry date, which is what generates the roll.

    Returns an (len(dates), m) array of ``datetime64[ns]``.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    chain = contract_chain(dates[0] - pd.DateOffset(months=1),
                           dates[-1] + pd.DateOffset(months=m + 3), prefix)
    exp = np.array([c.expiry.to_datetime64() for c in chain])
    exp.sort()

    out = np.empty((len(dates), m), dtype="datetime64[ns]")
    d64 = dates.to_numpy("datetime64[ns]")
    for i, d in enumerate(d64):
        idx = np.searchsorted(exp, d, side="left")
        if idx + m > len(exp):
            raise ValueError("contract chain too short; extend the end date")
        out[i] = exp[idx:idx + m]
    return out


def time_to_maturity(dates, m: int, prefix: str = "CL",
                     basis: str = "calendar") -> np.ndarray:
    """(n, m) time to maturity in years for the ``m`` nearby contracts.

    ``basis='calendar'`` divides calendar days by 365.25, which matches the
    continuous-time convention of the pricing formula. ``basis='business'``
    counts NYMEX business days over 252, which some implementations prefer;
    the choice must match the ``dt`` used in the transition equation, and
    mixing them is a quiet source of bias in ``kappa``.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    exp = nearby_expiries(dates, m, prefix)

    if basis == "calendar":
        days = (exp - dates.to_numpy("datetime64[ns]")[:, None]
                ).astype("timedelta64[D]").astype(float)
        tau = days / DAYS_PER_YEAR
    elif basis == "business":
        hol = _holidays().to_numpy("datetime64[D]")
        d0 = dates.to_numpy("datetime64[D]")
        tau = np.empty(exp.shape)
        for j in range(exp.shape[1]):
            tau[:, j] = np.busday_count(
                d0, exp[:, j].astype("datetime64[D]"), holidays=hol
            ) / TRADING_DAYS_PER_YEAR
    else:
        raise ValueError("basis must be 'calendar' or 'business'")

    # A contract expiring today has tau = 0, which makes exp(-kappa*tau) = 1
    # and is legitimate; negative tau is not.
    if np.any(tau < 0):
        raise ValueError("negative time to maturity -- contract chain misaligned")
    return tau


def roll_indicator(dates, m: int = 1, window: int = 3,
                   prefix: str = "CL") -> np.ndarray:
    """Roll-window indicator for the measurement-variance inflation of eq. (25).

    Marks the ``window`` business days up to and including each front-contract
    expiry, when open interest migrates to the next contract and settlement
    quality deteriorates. Returned as float so it can be used directly as the
    regressor in ``scale_variances``.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    front = nearby_expiries(dates, 1, prefix)[:, 0]
    hol = _holidays().to_numpy("datetime64[D]")
    bdays = np.busday_count(dates.to_numpy("datetime64[D]"),
                            front.astype("datetime64[D]"), holidays=hol)
    return (bdays < window).astype(float)


def summarise(dates, m: int, prefix: str = "CL") -> str:
    """One-line sanity report for a constructed maturity grid."""
    tau = time_to_maturity(dates, m, prefix)
    return (f"{len(dates)} dates x {m} contracts | "
            f"tau front {tau[:, 0].min():.4f}-{tau[:, 0].max():.4f}y | "
            f"tau back {tau[:, -1].min():.4f}-{tau[:, -1].max():.4f}y | "
            f"rolls {int(np.sum(np.diff(tau[:, 0]) > 0))}")
