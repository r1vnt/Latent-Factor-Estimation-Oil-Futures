"""MLE fitting for the four-factor model (csr.pricing.fourfactor), mirroring
fit3.py's structure exactly, one factor further.

Structural theta layout (indices 0-14), then curve (15-16 linear, 15-17
quadratic):

    0: log(gap12)      gap12 = kappa1 - kappa2 > 0
    1: log(gap23)      gap23 = kappa2 - kappa3 > 0
    2: log(kappa3)     kappa3 free; kappa2 = kappa3 + exp(gap23);
                       kappa1 = kappa2 + exp(gap12) -- unreachable label
                       switching by construction, same discipline fit3.py
                       uses for two rates.
    3: log(sigma1)
    4: log(sigma2)
    5: log(sigma3)
    6: log(sigma_xi)
    7: atanh(rho1x)
    8: atanh(rho2x)
    9: atanh(rho3x)
    10: mu_xi
    11: mu_xi_star
    12: lambda1
    13: lambda2
    14: lambda3
    15: a, 16: b [, 17: c]     -- smoothmeas.SmoothMeasParams

rho12 = rho13 = rho23 = 0, not fit -- fourfactor.py's structural choice,
not this module's. The independent tanh bounds on rho1x/rho2x/rho3x do NOT
imply their joint PSD constraint (rho1x^2+rho2x^2+rho3x^2 <= 1); a BFGS
line-search step can violate it even though each coordinate alone looks
fine (this is exactly what crashed the three-factor battery once already
-- see fit3.py's _neg_loglik3). Both neg-loglik functions here catch that
ValueError from the very first version, not after finding it the hard way.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..state.fast import filter_uv
from .fourfactor import FourFactorParams, base_system4, initial_state4
from .smoothmeas import SmoothMeasParams, hdiag_from_tau

STUDENT_T_DF = 5.0


def pack4_smooth(p: FourFactorParams, a: float, b: float, c: float = 0.0,
                 quadratic: bool = True) -> np.ndarray:
    gap12 = p.kappa1 - p.kappa2
    gap23 = p.kappa2 - p.kappa3
    extra = [a, b, c] if quadratic else [a, b]
    return np.concatenate([[
        np.log(max(gap12, 1e-10)), np.log(max(gap23, 1e-10)), np.log(p.kappa3),
        np.log(p.sigma1), np.log(p.sigma2), np.log(p.sigma3), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho1x, -0.999, 0.999)),
        np.arctanh(np.clip(p.rho2x, -0.999, 0.999)),
        np.arctanh(np.clip(p.rho3x, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda1, p.lambda2, p.lambda3,
    ], extra])


def unpack4_smooth(theta: np.ndarray, quadratic: bool = True
                   ) -> tuple[FourFactorParams, SmoothMeasParams]:
    e = np.exp(np.clip(theta[:7], -50.0, 50.0))
    gap12, gap23, kappa3, sigma1, sigma2, sigma3, sigma_xi = e
    kappa2 = kappa3 + gap23
    kappa1 = kappa2 + gap12
    p = FourFactorParams(
        kappa1=float(kappa1), kappa2=float(kappa2), kappa3=float(kappa3),
        sigma1=float(sigma1), sigma2=float(sigma2), sigma3=float(sigma3),
        sigma_xi=float(sigma_xi),
        rho1x=float(np.tanh(theta[7])), rho2x=float(np.tanh(theta[8])),
        rho3x=float(np.tanh(theta[9])),
        mu_xi=float(theta[10]), mu_xi_star=float(theta[11]),
        lambda1=float(theta[12]), lambda2=float(theta[13]), lambda3=float(theta[14]),
    )
    smooth = SmoothMeasParams(a=float(theta[15]), b=float(theta[16]),
                              c=float(theta[17]) if quadratic else 0.0,
                              quadratic=quadratic)
    return p, smooth


def _raw_system4(p: FourFactorParams, smooth: SmoothMeasParams,
                 tau: np.ndarray, dt: float):
    tau = np.atleast_2d(np.asarray(tau, float))
    sys = base_system4(p, tau, dt)
    sys["Hdiag"] = hdiag_from_tau(smooth, tau)
    return sys


def weighted_filter4(p: FourFactorParams, smooth: SmoothMeasParams,
                     log_f: np.ndarray, tau: np.ndarray, dt: float,
                     weight: np.ndarray):
    sys = _raw_system4(p, smooth, tau, dt)
    x0, P0 = initial_state4(log_f, tau, p)
    n, k = sys["n"], 4
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


def student_t_weights4(resid: np.ndarray, df: float) -> np.ndarray:
    w = (df + 1.0) / (df + resid ** 2)
    return np.where(np.isfinite(resid), w, 1.0)


def _neg_loglik4(theta, log_f, tau, dt, quadratic):
    try:
        p, smooth = unpack4_smooth(theta, quadratic)
    except ValueError:
        return 1e10
    ll, *_ = weighted_filter4(p, smooth, log_f, tau, dt, np.ones_like(log_f))
    return -ll if np.isfinite(ll) else 1e10


def fit4_gaussian_smooth(log_f: np.ndarray, tau: np.ndarray, dt: float,
                         theta0: np.ndarray, quadratic: bool = True):
    res = minimize(_neg_loglik4, theta0, args=(log_f, tau, dt, quadratic),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        res2 = minimize(_neg_loglik4, res.x, args=(log_f, tau, dt, quadratic),
                        method="Nelder-Mead",
                        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat, smooth_hat = unpack4_smooth(res.x, quadratic)
    ll, x_filt, resid, n_obs = weighted_filter4(
        p_hat, smooth_hat, log_f, tau, dt, np.ones_like(log_f))
    return p_hat, smooth_hat, x_filt, resid, n_obs, ll


def _neg_loglik4_weighted(theta, log_f, tau, dt, weight, quadratic):
    try:
        p, smooth = unpack4_smooth(theta, quadratic)
    except ValueError:
        return 1e10
    ll, _, _, _ = weighted_filter4(p, smooth, log_f, tau, dt, weight)
    return -ll if np.isfinite(ll) else 1e10


def fit4_student_t_smooth(log_f: np.ndarray, tau: np.ndarray, dt: float,
                          df: float, theta0: np.ndarray,
                          n_outer: int = 200, tol: float = 1e-6,
                          quadratic: bool = True):
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    p, smooth = unpack4_smooth(theta, quadratic)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik4_weighted, theta,
                       args=(log_f, tau, dt, weight, quadratic),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p, smooth = unpack4_smooth(theta, quadratic)
        _, _, resid_raw, _ = weighted_filter4(p, smooth, log_f, tau, dt, np.ones_like(log_f))
        new_weight = student_t_weights4(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = weighted_filter4(
        p, smooth, log_f, tau, dt, np.ones_like(log_f))
    return p, smooth, x_filt, resid_raw, weight, it + 1, ll
