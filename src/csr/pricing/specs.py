"""The pre-registered specification lattice (Section XI-D).

Committed **before** estimation begins. Fixing the candidate set in advance is
what separates a nested-model comparison from a search over specifications, and
it is the reason the likelihood-ratio statistics reported later are meaningful.

    S0   B = 0, G = 0     baseline, prices only
    SD   G = 0            news drives the factors
    SM   B = 0            news measures the innovations
    SDM  unrestricted     both channels

Each spec knows its own parent and parameter count, so degrees of freedom for
every LR test are derived rather than hand-maintained. Hand-maintained degrees
of freedom are a classic silent error: the test still runs and still returns a
p-value.

Exact nesting is asserted in ``tests/test_nesting.py``. If SM with G = 0 fails
to reproduce S0's likelihood, every downstream statistic is void, and the
failure produces no error message of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..state.augment import (append_news_block, augment_prior, augment_system,
                             news_variance)
from ..state.fast import filter_uv
from .twofactor import TwoFactorParams, base_system, initial_state, scale_variances


@dataclass(frozen=True)
class Spec:
    """One point on the lattice."""

    name: str
    driver: bool          # B != 0  (news enters the transition)
    measurement: bool     # G != 0  (news enters the observation)
    parent: str | None = None
    asymmetric: bool = False   # separate B+ / B- on observed feature sign
    note: str = ""

    def n_news_params(self, p: int) -> int:
        """Free parameters contributed by the news channels."""
        k = 0
        if self.driver:
            k += (2 * p) * (2 if self.asymmetric else 1)   # B is (2, p)
        if self.measurement:
            k += 2 * p        # G is (p, 2)
            k += p            # intercepts a_n
        return k


REGISTRY: dict[str, Spec] = {
    "S0": Spec("S0", driver=False, measurement=False, parent=None,
               note="baseline, prices only"),
    "SD": Spec("SD", driver=True, measurement=False, parent="S0",
               note="news drives the factors"),
    "SM": Spec("SM", driver=False, measurement=True, parent="S0",
               note="news measures the innovations"),
    "SDM": Spec("SDM", driver=True, measurement=True, parent="SD",
                note="both channels"),
    # Asymmetric variant of SD (Section XI-G); nests SD with B+ = B-.
    #
    # Asymmetry lives in the DRIVER channel, not the measurement channel, and
    # the distinction is not cosmetic. In the driver equation u_t is an *input*,
    # so splitting it at zero and giving each half its own loading is linear and
    # nests exactly, because u+ + u- = u. In the measurement equation u_t is an
    # *observation*; splitting it produces two rows, one of which reports a
    # structural zero on any given day. The filter cannot tell that zero apart
    # from a genuine reading of "the shock was zero" and updates the state on
    # it, so the restriction B+ = B- does not recover the parent. That failure
    # is silent -- it was caught only by the nesting assertion in
    # tests/test_nesting.py, which is the argument for having it.
    #
    # Measurement-side asymmetry, which is closer to the inventory story, would
    # have to condition on the sign of the latent shock and therefore requires
    # a nonlinear filter. It is out of scope here by choice, not oversight.
    "SD_ASYM": Spec("SD_ASYM", driver=True, measurement=False, parent="SD",
                    asymmetric=True,
                    note="asymmetric driver loadings on observed feature sign"),
}

#: Specifications on which inference is reported. Committed before estimation.
PREREGISTERED = ("S0", "SD", "SM", "SDM")


def split_sign(u: np.ndarray) -> np.ndarray:
    """Split observed features at zero into (u+, u-); Section XI-G eq. (26).

    The split is on the *observed* feature, never on the latent innovation: a
    loading that depended on the sign of an unobserved state would make the
    measurement equation nonlinear in the state and the Kalman recursions
    inexact.
    """
    u = np.atleast_2d(np.asarray(u, float))
    return np.concatenate([np.maximum(u, 0.0), np.minimum(u, 0.0)], axis=1)


@dataclass
class NewsData:
    """Everything the news channels consume."""

    features: np.ndarray            # (n, p) standardised, for B and G
    counts: np.ndarray              # (n,)   RAW article counts, post-dedup
    dispersion: np.ndarray          # (n,)   within-day sentiment dispersion
    intensity: np.ndarray | None = None   # (n,) standardised, for gamma
    release: np.ndarray | None = None     # (n,) scheduled-release indicator
    rv_lag: np.ndarray | None = None      # (n,) LAGGED realised volatility
    roll: np.ndarray | None = None        # (n,) roll-window indicator

    @property
    def p(self) -> int:
        return np.atleast_2d(self.features).shape[1]


def _driver_term(B: np.ndarray | None, u: np.ndarray, n: int, k: int):
    """Exogenous transition contribution B u_{t-1}, lagged one session."""
    if B is None:
        return np.zeros((n, k))
    lagged = np.zeros_like(u)
    lagged[1:] = u[:-1]
    Bu = np.zeros((n, k))
    Bu[:, :2] = lagged @ np.asarray(B, float).T
    return Bu


def run_spec(spec: Spec, p_params: TwoFactorParams, log_f, tau, dt,
             news: NewsData | None = None,
             B: np.ndarray | None = None,
             G: np.ndarray | None = None,
             a_n: np.ndarray | None = None,
             lag: int = 1):
    """Build the system for ``spec`` and run the filter.

    Returns ``(loglik, x_filt, P_filt, resid, n_obs)``.

    The augmented four-dimensional state is used throughout, including for S0,
    so that all specifications share one code path. The shock block is inert
    when no measurement channel loads on it, so S0 through this path is
    numerically identical to the two-dimensional baseline.
    """
    log_f = np.atleast_2d(np.asarray(log_f, float))
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = log_f.shape

    sys = base_system(p_params, tau, dt)
    if news is not None:
        sys = scale_variances(sys, p_params, intensity=news.intensity,
                              release=news.release, rv_lag=news.rv_lag,
                              roll=news.roll)
    else:
        sys = scale_variances(sys, p_params)

    aug = augment_system(sys)
    x0, P0 = initial_state(log_f, tau, p_params)
    Q2_0 = sys["Q"][0] if np.asarray(sys["Q"]).ndim == 3 else sys["Q"]
    x0a, P0a = augment_prior(x0, P0, Q2_0)

    y = log_f
    if spec.measurement:
        if news is None or G is None:
            raise ValueError(f"{spec.name} requires news data and G")
        feats = np.atleast_2d(news.features)
        p_eff = feats.shape[1]
        # A day's feature reading and its own dispersion/counts must be
        # NaN together, not independently -- append_news_block() shifts
        # both by the same lag, so if their NaN patterns ever disagreed, a
        # shifted-in valid observation could still land on a shifted-in NaN
        # variance (or vice versa) despite the shift itself being correct,
        # silently poisoning the log-likelihood and every filtered state
        # after it exactly the way an unshifted R used to. Caught here,
        # once, rather than downstream as an unexplained NaN loglik.
        feat_missing = np.isnan(feats).any(axis=1)
        other_missing = np.isnan(np.asarray(news.dispersion, float)) | \
                        np.isnan(np.asarray(news.counts, float))
        if not np.array_equal(feat_missing, other_missing):
            raise ValueError(
                "NewsData.features, .dispersion, and .counts must share the "
                "same per-day coverage: a day with a feature reading must "
                "also have its own dispersion/counts, and a day without one "
                "must lack all three -- otherwise the measurement variance "
                "attached to a lag-shifted observation can silently "
                "describe the wrong day."
            )
        R = news_variance(news.dispersion, news.counts, p_eff)
        if a_n is None:
            a_n = np.zeros(p_eff)
        aug, y_news = append_news_block(aug, feats, G, a_n, R, lag=lag)
        y = np.concatenate([log_f, y_news], axis=1)

    Bu = np.zeros((n, 4))
    if spec.driver:
        if news is None or B is None:
            raise ValueError(f"{spec.name} requires news data and B")
        u = split_sign(news.features) if spec.asymmetric \
            else np.atleast_2d(news.features)
        Bu = _driver_term(B, u, n, 4)

    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(y),
        np.ascontiguousarray(aug["Z"]),
        np.ascontiguousarray(aug["d"]),
        np.ascontiguousarray(aug["T"]),
        np.ascontiguousarray(aug["c"]),
        np.ascontiguousarray(aug["Q"]),
        np.ascontiguousarray(aug["Hdiag"]),
        np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0a),
        np.ascontiguousarray(P0a),
    )
    return float(ll), xf, Pf, resid, int(n_obs)


def lr_degrees_of_freedom(child: str, parent: str, p: int) -> int:
    """Degrees of freedom for the LR test of ``parent`` nested in ``child``."""
    return (REGISTRY[child].n_news_params(p)
            - REGISTRY[parent].n_news_params(p))
