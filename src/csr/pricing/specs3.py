"""The pre-registered specification lattice, ported to the three-factor base
(csr.pricing.threefactor) -- news drives/measures the two persistence
classes (chi1, chi2), never the permanent level xi.

Reuses Spec/REGISTRY/NewsData/split_sign/_driver_term from specs.py
unchanged -- those are dimension-agnostic (Spec just counts parameters
from p; _driver_term writes into Bu[:, :2], which is chi1/chi2 in this
base exactly as it was chi/xi in the two-factor one, so it needs no
change at all). What the two-factor run_spec cannot do unmodified is the
measurement channel: append_news_block hard-codes a 4-dim augmented state
with the shock block at [2:4]. Here the augmented state is 5-dim (chi1,
chi2, xi, w_chi1, w_chi2) with the shock block at [3:5] -- a new pair of
functions, not a generalisation of the pinned two-factor ones, matching
this project's own stated reason for keeping filter_uv and
filter_uv_contributions separate (state/fast.py's docstring): a change
made for one caller has no business risking every existing assertion
pinned against the other.

WHY XI IS EXCLUDED FROM THE SHOCK BLOCK. The whole point of exposing
"today's shock" as a separate state (state/augment.py's own module
docstring) is that news is evidence about what just happened, not the
accumulated level. xi is the permanent component by construction
(mu_xi_star drift, no mean reversion) -- news explaining today's
transitory or cyclical move is economically coherent; news explaining
today's contribution to a permanent level shift is not a claim this
lattice is designed to test. Only chi1 (fast/cyclical) and chi2
(slow/structural) get a shock-exposure row.
"""

from __future__ import annotations

import numpy as np

from ..state.fast import filter_uv
from .fit3 import _raw_system3
from .smoothmeas import SmoothMeasParams
from .specs import NewsData, Spec, _driver_term, news_variance, split_sign
from .threefactor import ThreeFactorParams, initial_state3

__all__ = ["augment_system3_news", "augment_prior3_news", "append_news_block3",
          "run_spec3"]


def augment_system3_news(sys: dict) -> dict:
    """Lift the 3-dim (chi1, chi2, xi) system to 5-dim, exposing (w_chi1,
    w_chi2) as their own state dims. Mirrors state.augment.augment_system's
    2->4 lift, but with a real difference from it, not just a resize: the
    two-factor lift's Q2 IS the full base block (both original states get
    exposed), so copying it once serves both the base sub-block and the
    shock duplication. Here only chi1/chi2 are exposed, xi is not, so the
    two copies are of DIFFERENT things -- the full 3x3 base block (xi's
    own variance and its correlation with chi1/chi2 included) goes into
    Q5[:3,:3] unchanged, and only the chi1/chi2 2x2 sub-block gets
    duplicated into the new shock dims. Conflating these (copying just the
    2x2 sub-block into Q5[:3,:3] too) silently zeroes xi's own process
    variance -- caught by explore/38's S0 sanity check landing 31000 nats
    away from the known-good price-only fit before any news spec was
    trusted."""
    n, m = sys["n"], sys["m"]
    T3, c3, Z3 = sys["T"], np.asarray(sys["c"], float), sys["Z"]
    Q3 = sys["Q"]
    if Q3.ndim == 2:
        Q3 = np.repeat(Q3[None, :, :], n, axis=0)

    T5 = np.zeros((5, 5))
    T5[:3, :3] = T3

    c5 = np.zeros((n, 5))
    c5[:, :3] = c3 if c3.ndim == 2 else c3[None, :]

    Q5 = np.zeros((n, 5, 5))
    Q5[:, :3, :3] = Q3          # full base block, xi included, untouched
    Q2 = Q3[:, :2, :2]           # chi1, chi2 sub-block, for the shock copy
    Q5[:, :2, 3:5] = Q2
    Q5[:, 3:5, :2] = Q2
    Q5[:, 3:5, 3:5] = Q2

    Z5 = np.zeros((n, m, 5))
    Z5[:, :, :3] = Z3

    return {"T": T5, "c": c5, "Q": Q5, "Z": Z5, "d": sys["d"],
           "Hdiag": sys["Hdiag"], "n": n, "m": m, "k": 5}


def augment_prior3_news(x0: np.ndarray, P0: np.ndarray, Q2_0: np.ndarray):
    """x0, P0 are (3,), (3,3) from initial_state3. Q2_0 is the (2,2)
    chi1/chi2 sub-block of the base Q (the shock block's own prior
    variance -- overwritten by the first prediction step regardless,
    same note as augment_prior)."""
    x5 = np.zeros(5)
    x5[:3] = x0
    P5 = np.zeros((5, 5))
    P5[:3, :3] = P0
    P5[3:5, 3:5] = Q2_0
    return x5, P5


def append_news_block3(aug: dict, news: np.ndarray, G: np.ndarray,
                       a_n: np.ndarray, R_diag: np.ndarray,
                       lag: int = 1) -> tuple[dict, np.ndarray]:
    """append_news_block, for the 5-dim (shock block at [3:5]) augmented
    state instead of the two-factor's 4-dim (shock block at [2:4]) one.
    Identical lag-shift and NaN-coverage-pairing logic; see
    state.augment.append_news_block's docstring for the reasoning, not
    repeated here."""
    n, m = aug["n"], aug["m"]
    news = np.atleast_2d(np.asarray(news, float))
    p = news.shape[1]
    G = np.asarray(G, float).reshape(p, 2)
    R_diag = np.atleast_2d(np.asarray(R_diag, float))
    if R_diag.shape[0] == 1 and n != 1:
        R_diag = np.repeat(R_diag, n, axis=0)

    def _shift(a: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan)
        if lag >= 0:
            if lag < n:
                out[: n - lag] = a[lag:]
        else:
            s = -lag
            if s < n:
                out[s:] = a[: n - s]
        return out

    y_news = _shift(news)
    R_shifted = _shift(R_diag)

    Z_news = np.zeros((n, p, 5))
    Z_news[:, :, 3:5] = G[None, :, :]

    Z = np.concatenate([aug["Z"], Z_news], axis=1)
    d = np.concatenate([aug["d"], np.repeat(np.asarray(a_n, float)[None, :], n, 0)],
                       axis=1)
    Hdiag = np.concatenate([aug["Hdiag"], R_shifted], axis=1)

    out = dict(aug)
    out.update({"Z": Z, "d": d, "Hdiag": Hdiag, "m": m + p,
               "m_price": m, "p_news": p})
    return out, y_news


def run_spec3(spec: Spec, p_params: ThreeFactorParams, smooth: SmoothMeasParams,
             log_f, tau, dt, news: NewsData | None = None,
             B: np.ndarray | None = None, G: np.ndarray | None = None,
             a_n: np.ndarray | None = None, lag: int = 1, rv_lag=None,
             weight: np.ndarray | None = None):
    """Three-factor analogue of specs.run_spec. Always augments to 5-dim
    (shock block inert when spec.measurement is False), matching
    run_spec's own "one shared code path for every spec" convention.

    ``weight`` (n, m), if given, divides the PRICE rows' Hdiag only --
    applied before any news rows are appended, so it can never touch
    the news measurement variance R. News rows stay Gaussian throughout;
    nothing in this project's news machinery treats them any other way
    (news_variance's R is already the sampling variance of an observed
    mean, not a free scale competing with a t-density). weight=1
    reproduces the unweighted likelihood exactly, same convention as
    fit3.weighted_filter3.

    Returns (loglik, x_filt, resid, n_obs).
    """
    log_f = np.atleast_2d(np.asarray(log_f, float))
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = log_f.shape

    sys = _raw_system3(p_params, smooth, tau, dt, rv_lag=rv_lag)
    if weight is not None:
        sys["Hdiag"] = sys["Hdiag"] / weight
    x0, P0 = initial_state3(log_f, tau, p_params)
    aug = augment_system3_news(sys)
    Q2_0 = sys["Q"][0, :2, :2] if np.asarray(sys["Q"]).ndim == 3 else sys["Q"][:2, :2]
    x0a, P0a = augment_prior3_news(x0, P0, Q2_0)

    y = log_f
    if spec.measurement:
        if news is None or G is None:
            raise ValueError(f"{spec.name} requires news data and G")
        feats = np.atleast_2d(news.features)
        p_eff = feats.shape[1]
        feat_missing = np.isnan(feats).any(axis=1)
        other_missing = np.isnan(np.asarray(news.dispersion, float)) | \
                        np.isnan(np.asarray(news.counts, float))
        if not np.array_equal(feat_missing, other_missing):
            raise ValueError(
                "NewsData.features, .dispersion, and .counts must share the "
                "same per-day coverage -- see specs.run_spec's identical check."
            )
        R = news_variance(news.dispersion, news.counts, p_eff)
        if a_n is None:
            a_n = np.zeros(p_eff)
        aug, y_news = append_news_block3(aug, feats, G, a_n, R, lag=lag)
        y = np.concatenate([log_f, y_news], axis=1)

    Bu = np.zeros((n, 5))
    if spec.driver:
        if news is None or B is None:
            raise ValueError(f"{spec.name} requires news data and B")
        u = split_sign(news.features) if spec.asymmetric \
            else np.atleast_2d(news.features)
        Bu = _driver_term(B, u, n, 5)

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
    return float(ll), xf, resid, int(n_obs)
