"""Panel assembly: raw settlement prices -> the Panel every study consumes.

The log transform obeys three rules, in order of precedence:

1. A non-positive price becomes NaN in that cell -- never floored, never
   dropped as a row.
2. The date is always kept, even when every price on it is NaN. ``dt`` in
   the transition equation is fixed; the filter drops missing observations
   exactly (state/kalman.py) but has no treatment at all for a variable-
   length dt, which is what dropping a date would silently produce.
3. ``log_f`` is finite in a cell if and only if the source price was finite
   and positive there. No code path may write a finite log price from a
   non-positive source -- including by flooring it first.

Three screens flag, but never remove, suspect cells: a low-price band where
the log is numerically unstable, runs of a stale (repeated) settlement, and
log-return outliers by a MAD-based threshold rather than a stdev-based one,
since a stdev estimate is exactly what those outliers would inflate. Flags
are stored as boolean arrays so they can enter the conditional analysis as
covariates later, rather than silently deleting data a downstream study
might have wanted to condition on.

A fourth, derived flag -- ``bad_price`` -- lives in level space rather than
return space. A single bad settlement at day t shows up as *two* flagged
returns (the jump into t and the jump back out at t+1), so conditioning on
``outlier_return`` directly as a price-quality covariate mislabels t+1,
whose own settlement was fine. ``bad_price`` isolates day t by requiring
both adjacent returns flagged with opposite sign -- the spike-and-revert
signature -- which a genuine, persistent repricing does not produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .expiry import roll_indicator, time_to_maturity

DEFAULT_MIN_PRICE = 5.0
DEFAULT_STALE_RUN = 5
DEFAULT_MAD_K = 8.0
DEFAULT_ROLL_WINDOW = 3
MAD_TO_SD = 1.4826  # consistency constant: robust sigma from MAD under normality


@dataclass
class QualityReport:
    """Cell counts and affected dates for missingness and each screen."""

    n_dates: int
    n_contracts: int
    n_cells: int
    n_missing: int
    flag_counts: dict[str, int]
    flag_dates: dict[str, pd.DatetimeIndex]

    def __str__(self) -> str:
        lines = [
            f"Panel: {self.n_dates} dates x {self.n_contracts} contracts "
            f"({self.n_cells} cells)",
            f"  missing/excluded: {self.n_missing} "
            f"({self.n_missing / max(self.n_cells, 1):.2%})",
        ]
        for name, count in self.flag_counts.items():
            lines.append(f"  {name}: {count} cells on "
                         f"{len(self.flag_dates[name])} dates")
        return "\n".join(lines)


def _build_quality(dates: pd.DatetimeIndex, log_f: np.ndarray,
                    flags: dict[str, np.ndarray]) -> QualityReport:
    n, m = log_f.shape
    flag_counts = {name: int(mask.sum()) for name, mask in flags.items()}
    flag_dates = {name: dates[np.any(mask, axis=1)] for name, mask in flags.items()}
    return QualityReport(
        n_dates=n, n_contracts=m, n_cells=n * m,
        n_missing=int((~np.isfinite(log_f)).sum()),
        flag_counts=flag_counts, flag_dates=flag_dates,
    )


def _to_wide(data: pd.DataFrame, date_col: str, contract_col: str,
             price_col: str) -> pd.DataFrame:
    """Long (date, contract, price) rows -> wide, or pass a wide frame
    through. Either way, columns end up sorted by nearby rank ascending."""
    if contract_col in data.columns and price_col in data.columns:
        long = data[[date_col, contract_col, price_col]].copy()
        long[date_col] = pd.to_datetime(long[date_col])
        wide = long.pivot(index=date_col, columns=contract_col, values=price_col)
    else:
        wide = data.copy()
        if date_col in wide.columns:
            wide = wide.set_index(date_col)
        wide.index = pd.to_datetime(wide.index)

    try:
        order = np.argsort(pd.to_numeric(wide.columns).to_numpy())
        wide = wide.iloc[:, order]
    except (TypeError, ValueError):
        pass  # non-numeric labels: trust the caller's column order

    wide.index = pd.DatetimeIndex(wide.index).normalize()
    return wide.sort_index()


def _log_transform(price: np.ndarray) -> np.ndarray:
    """Rules 1 and 3: log is finite in a cell iff the source was too."""
    ok = np.isfinite(price) & (price > 0)
    log_f = np.full(price.shape, np.nan)
    log_f[ok] = np.log(price[ok])
    return log_f


def _stale_mask(col: np.ndarray, min_run: int) -> np.ndarray:
    """Runs of >= min_run consecutive identical, non-missing settlements.

    A gap breaks a run rather than bridging it: two three-day runs either
    side of a missing observation are two incidents, not one six-day run.
    """
    s = pd.Series(col)
    valid = s.notna()
    new_run = (s != s.shift()) | (valid != valid.shift())
    run_id = new_run.cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (valid & (run_len >= min_run)).to_numpy()


def _log_returns(log_col: np.ndarray) -> np.ndarray:
    ret = np.full(log_col.shape, np.nan)
    ret[1:] = np.diff(log_col)
    return ret


def _outlier_mask(ret: np.ndarray, k: float) -> np.ndarray:
    """Log returns beyond k robust SDs, MAD-based rather than stdev-based.

    The outliers this is meant to catch are exactly what would inflate a
    stdev threshold; the median absolute deviation is not moved by them.
    """
    med = np.nanmedian(ret)
    mad = np.nanmedian(np.abs(ret - med))
    robust_sd = MAD_TO_SD * mad
    if not np.isfinite(robust_sd) or robust_sd == 0.0:
        return np.zeros(ret.shape, dtype=bool)
    return np.isfinite(ret) & (np.abs(ret - med) > k * robust_sd)


def _bad_price_mask(outlier: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Isolate the bad *price*, not the two returns it contaminates.

    A one-day bad settlement at day t shows up as two flagged returns: the
    jump into t and the jump back out at t+1, in opposite directions. That
    signature -- ret[t] and ret[t+1] both flagged, opposite sign -- pins the
    error to day t specifically, which is what a price-quality covariate
    needs to condition on (H4 interacts the news effect with price-channel
    noise; mislabelling t+1 would condition on a settlement that was fine).

    A persistent level shift flags only ret[t], with a normal-sized ret[t+1]
    once the new level is established -- correctly not flagged here, since
    it is a real repricing, not a measurement error.
    """
    n = outlier.shape[0]
    bad = np.zeros(n, dtype=bool)
    if n < 2:
        return bad
    both = outlier[:-1] & outlier[1:]
    opposite = np.sign(ret[:-1]) != np.sign(ret[1:])
    bad[:-1] = both & opposite
    return bad


@dataclass
class Panel:
    """The price panel every study consumes.

    ``log_f`` and ``tau`` are (n, m): n dates x m nearby contracts. ``flags``
    holds one boolean (n, m) array per screen -- screens flag, they never
    remove, so a flagged cell is still a real number in ``log_f`` unless
    rule 1 already excluded it.
    """

    log_f: np.ndarray
    tau: np.ndarray
    dates: pd.DatetimeIndex
    labels: list[str]
    roll: np.ndarray
    flags: dict[str, np.ndarray]
    quality: QualityReport
    source: str = "unknown"

    @property
    def n(self) -> int:
        return len(self.dates)

    @property
    def m(self) -> int:
        return len(self.labels)

    @classmethod
    def from_prices(cls, data: pd.DataFrame, *, date_col: str = "date",
                     contract_col: str = "contract", price_col: str = "settle",
                     m: int | None = None, prefix: str = "CL",
                     basis: str = "calendar", min_price: float = DEFAULT_MIN_PRICE,
                     stale_run: int = DEFAULT_STALE_RUN,
                     mad_k: float = DEFAULT_MAD_K,
                     roll_window: int = DEFAULT_ROLL_WINDOW,
                     source: str = "unknown", verbose: bool = True) -> "Panel":
        """Build a Panel from long ``(date, contract, price)`` rows or a wide
        (date-indexed, one column per nearby contract) frame.

        ``contract_col`` values (long input) or column labels (wide input)
        are nearby ranks -- 1 is the front contract -- sorted ascending, so
        ``tau`` and ``roll`` below are genuinely about the Nth nearby
        contract, matching ``curve.expiry``.
        """
        wide = _to_wide(data, date_col, contract_col, price_col)
        if m is not None:
            wide = wide.iloc[:, :m]

        price = wide.to_numpy(dtype=float)
        dates = pd.DatetimeIndex(wide.index)
        labels = [f"{prefix}{i}" for i in range(1, wide.shape[1] + 1)]

        log_f = _log_transform(price)
        tau = time_to_maturity(dates, wide.shape[1], prefix=prefix, basis=basis)
        roll = roll_indicator(dates, window=roll_window, prefix=prefix)

        returns = np.column_stack([_log_returns(log_f[:, j])
                                   for j in range(log_f.shape[1])])
        outlier = np.column_stack([_outlier_mask(returns[:, j], mad_k)
                                   for j in range(log_f.shape[1])])
        flags = {
            "low_price": (price > 0) & (price < min_price),
            "stale": np.column_stack([
                _stale_mask(price[:, j], stale_run) for j in range(price.shape[1])
            ]),
            "outlier_return": outlier,
            # The level-space counterpart of outlier_return: pins a bad
            # settlement to the one day it actually happened on, rather than
            # also implicating the day after via the reversion it caused.
            "bad_price": np.column_stack([
                _bad_price_mask(outlier[:, j], returns[:, j])
                for j in range(log_f.shape[1])
            ]),
        }

        quality = _build_quality(dates, log_f, flags)
        if verbose:
            print(quality)

        return cls(log_f=log_f, tau=tau, dates=dates, labels=labels, roll=roll,
                   flags=flags, quality=quality, source=source)

    def slice(self, start, end) -> "Panel":
        """Subset every array -- including every entry in ``flags`` -- to
        ``[start, end]`` inclusive, together."""
        mask = ((self.dates >= pd.Timestamp(start))
                & (self.dates <= pd.Timestamp(end)))
        dates = self.dates[mask]
        log_f = self.log_f[mask]
        tau = self.tau[mask]
        roll = self.roll[mask]
        flags = {name: arr[mask] for name, arr in self.flags.items()}
        quality = _build_quality(dates, log_f, flags)
        return Panel(log_f=log_f, tau=tau, dates=dates, labels=list(self.labels),
                    roll=roll, flags=flags, quality=quality, source=self.source)
