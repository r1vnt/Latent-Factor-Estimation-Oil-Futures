"""MLE fitting for the three-factor model (csr.pricing.threefactor), with the
smooth measurement curve (csr.pricing.smoothmeas) wired in from the start --
never the free per-contract sigma_meas that produced the Chen-Scott collapse
fit.py's two-factor path had to be guarded against.

Mirrors csr.pricing.fit's Gaussian-then-Student-t-EM structure exactly, over
ThreeFactorParams instead of TwoFactorParams. Two departures, both structural:

* No 2->4 state augmentation. csr.state.augment.augment_system is hard-coded
  for a two-dimensional base state (it lifts to 4 for the news shock block);
  the three-factor base state is already 3-dimensional and no news channel is
  used here, so this calls csr.state.fast.filter_uv directly on
  base_system3's raw (T, c, Q, Z, d) -- filter_uv is generic in the state
  dimension k, so this needs no filter changes, only skips a lift that does
  not apply.
* rho12 is fixed at 0, not fit. Free would give 12 structural parameters
  against ~11 intended; the two mean-reverting factors' own shocks being
  uncorrelated is the natural default when they are separated because they
  serve different maturity ranges, not because they are believed to share a
  common driver. ThreeFactorParams.rho12 remains settable directly for
  anyone who wants to test that assumption outside the packed vector.

Structural theta layout (indices 0-11), then curve (12-13 linear, 12-14
quadratic):

    0: log(gap)       gap = kappa1 - kappa2 > 0, so kappa1 = kappa2 + exp(gap)
                       -- the reparameterisation that makes label-switching
                       unreachable by construction, not just checked after
                       the fact by ThreeFactorParams.__post_init__.
    1: log(kappa2)
    2: log(sigma1)
    3: log(sigma2)
    4: log(sigma_xi)
    5: atanh(rho1x)
    6: atanh(rho2x)
    7: mu_xi
    8: mu_xi_star
    9: lambda1
    10: lambda2
    11: gamma_v        -- bounded to +-GAMMA_V_MAX; process-noise scaling
                          (eq. 24) driven by whatever series the caller
                          passes as rv_lag, inert whenever rv_lag=None.
                          As of the front-spread horse race (explore/47),
                          this stays on lagged RV -- a separately
                          identified, genuine Q-side effect -- even after
                          eta_v below moved off it.
    12: a, 13: b [, 14: c]     -- smoothmeas.SmoothMeasParams
    [next: eta_v, next+1: lambda_v]  -- front-concentrated measurement-
                          noise inflation (eq. 25 extension), driven by
                          whatever series the caller passes as h_driver
                          (defaults to rv_lag when h_driver is omitted,
                          for exact backward compatibility). explore/47
                          found lagged RV explains none of this channel's
                          target once |front spread| is controlled for;
                          h_driver exists so the H-side (eta_v) and Q-side
                          (gamma_v) channels can be driven by different,
                          independently-chosen series instead of forcing
                          one series to explain both. Only present when
                          rv_extension=True; inert whenever h_driver=None
                          regardless. Index depends on quadratic (14 or
                          15) -- always compute via pack3_smooth/
                          unpack3_smooth, never hard-coded.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..state.fast import filter_uv
from .smoothmeas import SmoothMeasParams, hdiag_from_tau
from .threefactor import (ThreeFactorParams, base_system3, initial_state3,
                          scale_variances3)

STUDENT_T_DF = 5.0  # matches fit.py's Sec. XI-F choice; not jointly estimated

# Economic bounds for the two RV-driven variance-inflation channels, on a
# standardised log-RV driver: a value of 3 already implies e^3 ~ 20x
# variance at a 1-SD-in-log-vol day, which is already an extreme reading
# for a "how much should one bad day widen measurement/process noise"
# multiplier. Bounded BY CONSTRUCTION (gamma_v/eta_v = MAX*tanh(z)), not
# by clipping the exponent -- a clip is a numerical overflow guard being
# asked to do an economic constraint's job, and that mismatch is exactly
# what manufactured the plateau every multi-start ran into (explore/44).
GAMMA_V_MAX = 3.0
ETA_V_MAX = 3.0


def pack3_smooth(p: ThreeFactorParams, a: float, b: float, c: float = 0.0,
                 quadratic: bool = True, eta_v: float = 0.0, lambda_v: float = 0.0,
                 rv_extension: bool = False) -> np.ndarray:
    gap = p.kappa1 - p.kappa2
    curve = [a, b, c] if quadratic else [a, b]
    extra = curve + ([np.arctanh(np.clip(eta_v / ETA_V_MAX, -0.999, 0.999)), lambda_v]
                    if rv_extension else [])
    return np.concatenate([[
        np.log(max(gap, 1e-10)), np.log(p.kappa2),
        np.log(p.sigma1), np.log(p.sigma2), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho1x, -0.999, 0.999)),
        np.arctanh(np.clip(p.rho2x, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda1, p.lambda2,
        np.arctanh(np.clip(p.gamma_v / GAMMA_V_MAX, -0.999, 0.999)),
    ], extra])


def unpack3_smooth(theta: np.ndarray, quadratic: bool = True,
                   rv_extension: bool = False
                   ) -> tuple[ThreeFactorParams, SmoothMeasParams]:
    e = np.exp(np.clip(theta[:5], -50.0, 50.0))
    gap, kappa2, sigma1, sigma2, sigma_xi = e
    p = ThreeFactorParams(
        kappa1=float(kappa2 + gap), kappa2=float(kappa2),
        sigma1=float(sigma1), sigma2=float(sigma2), sigma_xi=float(sigma_xi),
        rho12=0.0, rho1x=float(np.tanh(theta[5])), rho2x=float(np.tanh(theta[6])),
        mu_xi=float(theta[7]), mu_xi_star=float(theta[8]),
        lambda1=float(theta[9]), lambda2=float(theta[10]),
        gamma_v=float(GAMMA_V_MAX * np.tanh(theta[11])),
    )
    idx = 12
    a, b = float(theta[idx]), float(theta[idx + 1])
    idx += 2
    c = float(theta[idx]) if quadratic else 0.0
    idx += 1 if quadratic else 0
    eta_v = float(ETA_V_MAX * np.tanh(theta[idx])) if rv_extension else 0.0
    lambda_v = float(theta[idx + 1]) if rv_extension else 0.0
    smooth = SmoothMeasParams(a=a, b=b, c=c, quadratic=quadratic,
                              eta_v=eta_v, lambda_v=lambda_v)
    return p, smooth


def _raw_system3(p: ThreeFactorParams, smooth: SmoothMeasParams,
                 tau: np.ndarray, dt: float, rv_lag=None, h_driver=None):
    """rv_lag drives the Q-side gamma_v channel (scale_variances3); h_driver
    drives the H-side eta_v channel (hdiag_from_tau). Independent by design
    -- the horse race in explore/47 found lagged RV explains none of the
    front-contract measurement-variance mismatch once |front spread| is
    controlled for (RV's own coefficient collapses to p=0.48 alongside it),
    while gamma_v's RV-driven Q-side effect is separately identified and
    genuine. Defaulting h_driver to rv_lag when omitted keeps every existing
    caller (explore/44-46, tests) byte-for-byte unchanged."""
    if h_driver is None:
        h_driver = rv_lag
    tau = np.atleast_2d(np.asarray(tau, float))
    sys = base_system3(p, tau, dt)
    sys = scale_variances3(sys, p, rv_lag=rv_lag)
    sys["Hdiag"] = hdiag_from_tau(smooth, tau, rv_lag=h_driver)
    return sys


def weighted_filter3(p: ThreeFactorParams, smooth: SmoothMeasParams,
                     log_f: np.ndarray, tau: np.ndarray, dt: float,
                     weight: np.ndarray, rv_lag=None, h_driver=None):
    """Direct filter_uv call on the raw 3-dim system -- no news augmentation.
    weight == 1 is the plain Gaussian S0-equivalent likelihood."""
    sys = _raw_system3(p, smooth, tau, dt, rv_lag=rv_lag, h_driver=h_driver)
    x0, P0 = initial_state3(log_f, tau, p)
    n, k = sys["n"], 3
    Hdiag = sys["Hdiag"] / weight
    Bu = np.zeros((n, k))
    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f, dtype=np.float64),
        np.ascontiguousarray(sys["Z"], dtype=np.float64),
        np.ascontiguousarray(sys["d"], dtype=np.float64),
        np.ascontiguousarray(sys["T"], dtype=np.float64),
        np.ascontiguousarray(sys["c"], dtype=np.float64),
        np.ascontiguousarray(sys["Q"], dtype=np.float64),
        np.ascontiguousarray(Hdiag, dtype=np.float64),
        np.ascontiguousarray(Bu, dtype=np.float64),
        np.ascontiguousarray(x0, dtype=np.float64),
        np.ascontiguousarray(P0, dtype=np.float64),
    )
    return float(ll), xf, resid, int(n_obs)


def student_t_weights3(resid: np.ndarray, df: float) -> np.ndarray:
    w = (df + 1.0) / (df + resid ** 2)
    return np.where(np.isfinite(resid), w, 1.0)


def _neg_loglik3(theta, log_f, tau, dt, quadratic, rv_lag=None, rv_extension=False,
                 h_driver=None):
    try:
        p, smooth = unpack3_smooth(theta, quadratic, rv_extension)
    except ValueError:
        # rho1x, rho2x are each tanh-bounded to (-1, 1) independently, but
        # PSD of the (rho12=0, rho1x, rho2x) correlation matrix needs the
        # joint constraint rho1x^2 + rho2x^2 <= 1 -- a transient BFGS
        # line-search step can violate that even though neither coordinate
        # alone is out of range. ThreeFactorParams.__post_init__ correctly
        # rejects it; treat it as a very bad point (matching the
        # non-finite-ll convention below), not a fatal error -- the
        # optimiser needs to be able to step through this region, not die
        # in it (found the hard way: explore/28's battery crashed here on
        # "drop C5-C12" after 19/26 fits had already completed).
        return 1e10
    ll, *_ = weighted_filter3(p, smooth, log_f, tau, dt, np.ones_like(log_f),
                              rv_lag=rv_lag, h_driver=h_driver)
    return -ll if np.isfinite(ll) else 1e10


def fit3_gaussian_smooth(log_f: np.ndarray, tau: np.ndarray, dt: float,
                         theta0: np.ndarray, quadratic: bool = True, rv_lag=None,
                         rv_extension: bool = False, h_driver=None):
    res = minimize(_neg_loglik3, theta0,
                   args=(log_f, tau, dt, quadratic, rv_lag, rv_extension, h_driver),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        res2 = minimize(_neg_loglik3, res.x,
                        args=(log_f, tau, dt, quadratic, rv_lag, rv_extension, h_driver),
                        method="Nelder-Mead",
                        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat, smooth_hat = unpack3_smooth(res.x, quadratic, rv_extension)
    ll, x_filt, resid, n_obs = weighted_filter3(
        p_hat, smooth_hat, log_f, tau, dt, np.ones_like(log_f), rv_lag=rv_lag, h_driver=h_driver)
    return p_hat, smooth_hat, x_filt, resid, n_obs, ll


def _neg_loglik3_weighted(theta, log_f, tau, dt, weight, quadratic, rv_lag=None,
                          rv_extension=False, h_driver=None):
    try:
        p, smooth = unpack3_smooth(theta, quadratic, rv_extension)
    except ValueError:
        return 1e10  # see _neg_loglik3's identical guard for why
    ll, _, _, _ = weighted_filter3(p, smooth, log_f, tau, dt, weight,
                                   rv_lag=rv_lag, h_driver=h_driver)
    return -ll if np.isfinite(ll) else 1e10


def fit3_student_t_smooth(log_f: np.ndarray, tau: np.ndarray, dt: float,
                          df: float, theta0: np.ndarray,
                          n_outer: int = 200, tol: float = 1e-6,
                          quadratic: bool = True, rv_lag=None,
                          rv_extension: bool = False, h_driver=None):
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    p, smooth = unpack3_smooth(theta, quadratic, rv_extension)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik3_weighted, theta,
                       args=(log_f, tau, dt, weight, quadratic, rv_lag, rv_extension, h_driver),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p, smooth = unpack3_smooth(theta, quadratic, rv_extension)
        _, _, resid_raw, _ = weighted_filter3(p, smooth, log_f, tau, dt,
                                              np.ones_like(log_f), rv_lag=rv_lag, h_driver=h_driver)
        new_weight = student_t_weights3(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = weighted_filter3(
        p, smooth, log_f, tau, dt, np.ones_like(log_f), rv_lag=rv_lag, h_driver=h_driver)
    return p, smooth, x_filt, resid_raw, weight, it + 1, ll
