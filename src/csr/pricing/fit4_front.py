"""MLE fitting for fourfactor_front.FourFactorFrontParams -- rung 2 of the
escalation ladder (see fourfactor_front.py's module docstring for why
this is a different object from the already-refuted fourfactor.py chi1).

Mirrors fit3.py's Gaussian-then-Student-t-EM structure exactly. kappa_f_Q
is NOT in theta -- fixed at KAPPA_F_Q_ANCHOR (or whatever the caller
passes explicitly), never estimated, by design.

theta layout (15, +1/+2 for the curve):
    0: log(gap23)     gap23 = kappa2 - kappa3 > 0
    1: log(kappa3)
    2: log(sigma_f)
    3: log(sigma2)
    4: log(sigma3)
    5: log(sigma_xi)
    6: atanh(rho_fx)
    7: atanh(rho2x)
    8: atanh(rho3x)
    9: mu_xi
    10: mu_xi_star
    11: lambda_f
    12: lambda2
    13: lambda3
    14: log(kappa_f_P)   -- the essentially-affine free parameter
    15: a, 16: b [, 17: c]   -- smoothmeas.SmoothMeasParams
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .fit3 import _neg_t_loglik_df, student_t_weights3
from .fourfactor_front import (
    KAPPA_F_Q_ANCHOR,
    FourFactorFrontParams,
    base_system4_front,
    initial_state4_front,
)
from .smoothmeas import SmoothMeasParams, hdiag_from_tau
from ..state.fast import filter_uv, filter_uv_contributions


def pack4_front(p: FourFactorFrontParams, a: float, b: float, c: float = 0.0,
                quadratic: bool = True) -> np.ndarray:
    gap23 = p.kappa2 - p.kappa3
    curve = [a, b, c] if quadratic else [a, b]
    return np.concatenate([[
        np.log(max(gap23, 1e-10)), np.log(p.kappa3),
        np.log(p.sigma_f), np.log(p.sigma2), np.log(p.sigma3), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho_fx, -0.999, 0.999)),
        np.arctanh(np.clip(p.rho2x, -0.999, 0.999)),
        np.arctanh(np.clip(p.rho3x, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda_f, p.lambda2, p.lambda3,
        np.log(max(p.kappa_f_P, 1e-10)),
    ], curve])


def unpack4_front(theta: np.ndarray, quadratic: bool = True,
                  kappa_f_q: float = KAPPA_F_Q_ANCHOR
                  ) -> tuple[FourFactorFrontParams, SmoothMeasParams]:
    e = np.exp(np.clip(theta[:6], -50.0, 50.0))
    gap23, kappa3, sigma_f, sigma2, sigma3, sigma_xi = e
    kappa_f_p = float(np.exp(np.clip(theta[14], -50.0, 50.0)))
    p = FourFactorFrontParams(
        kappa2=float(kappa3 + gap23), kappa3=float(kappa3),
        sigma_f=float(sigma_f), sigma2=float(sigma2), sigma3=float(sigma3), sigma_xi=float(sigma_xi),
        kappa_f_Q=float(kappa_f_q), kappa_f_P=kappa_f_p,
        rho_fx=float(np.tanh(theta[6])), rho2x=float(np.tanh(theta[7])), rho3x=float(np.tanh(theta[8])),
        mu_xi=float(theta[9]), mu_xi_star=float(theta[10]),
        lambda_f=float(theta[11]), lambda2=float(theta[12]), lambda3=float(theta[13]),
    )
    idx = 15
    a, b = float(theta[idx]), float(theta[idx + 1])
    c = float(theta[idx + 2]) if quadratic else 0.0
    smooth = SmoothMeasParams(a=a, b=b, c=c, quadratic=quadratic)
    return p, smooth


def _raw_system4_front(p: FourFactorFrontParams, smooth: SmoothMeasParams, tau, dt: float) -> dict:
    tau = np.atleast_2d(np.asarray(tau, float))
    sys = base_system4_front(p, tau, dt)
    sys["Hdiag"] = hdiag_from_tau(smooth, tau)
    return sys


def weighted_filter4_front(p: FourFactorFrontParams, smooth: SmoothMeasParams,
                           log_f: np.ndarray, tau: np.ndarray, dt: float, weight: np.ndarray):
    sys = _raw_system4_front(p, smooth, tau, dt)
    x0, P0 = initial_state4_front(log_f, tau, p)
    n = sys["n"]
    Hdiag = sys["Hdiag"] / weight
    Bu = np.zeros((n, 4))
    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f, dtype=np.float64), np.ascontiguousarray(sys["Z"], dtype=np.float64),
        np.ascontiguousarray(sys["d"], dtype=np.float64), np.ascontiguousarray(sys["T"], dtype=np.float64),
        np.ascontiguousarray(sys["c"], dtype=np.float64), np.ascontiguousarray(sys["Q"], dtype=np.float64),
        np.ascontiguousarray(Hdiag, dtype=np.float64), np.ascontiguousarray(Bu, dtype=np.float64),
        np.ascontiguousarray(x0, dtype=np.float64), np.ascontiguousarray(P0, dtype=np.float64))
    return float(ll), xf, resid, int(n_obs)


def filter4_front_contributions(p: FourFactorFrontParams, smooth: SmoothMeasParams,
                                log_f: np.ndarray, tau: np.ndarray, dt: float):
    """Like weighted_filter4_front but via filter_uv_contributions (ll_cell
    exposed) -- for diagnostics (Ljung-Box, ll decomposition, per-contract
    breakdowns) that need the per-cell log(F) recoverable, same pattern as
    fit3.py's callers use directly on filter_uv_contributions rather than
    adding a parallel weighted_filter3-style wrapper for it."""
    sys = _raw_system4_front(p, smooth, tau, dt)
    x0, P0 = initial_state4_front(log_f, tau, p)
    n = sys["n"]
    return filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys["Z"]), np.ascontiguousarray(sys["d"]),
        np.ascontiguousarray(sys["T"]), np.ascontiguousarray(sys["c"]), np.ascontiguousarray(sys["Q"]),
        np.ascontiguousarray(sys["Hdiag"]), np.zeros((n, 4)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))


def _neg_loglik4_front(theta, log_f, tau, dt, quadratic, kappa_f_q=KAPPA_F_Q_ANCHOR):
    try:
        p, smooth = unpack4_front(theta, quadratic, kappa_f_q)
    except ValueError:
        # rho_fx/rho2x/rho3x joint-PSD violation on a transient line-search
        # step -- see fit3._neg_loglik3's identical guard for why this is a
        # very-bad-point return, not a fatal error.
        return 1e10
    ll, *_ = weighted_filter4_front(p, smooth, log_f, tau, dt, np.ones_like(log_f))
    return -ll if np.isfinite(ll) else 1e10


def fit4_front_gaussian(log_f: np.ndarray, tau: np.ndarray, dt: float, theta0: np.ndarray,
                        quadratic: bool = True, kappa_f_q: float = KAPPA_F_Q_ANCHOR):
    res = minimize(_neg_loglik4_front, theta0, args=(log_f, tau, dt, quadratic, kappa_f_q),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        res2 = minimize(_neg_loglik4_front, res.x, args=(log_f, tau, dt, quadratic, kappa_f_q),
                        method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat, smooth_hat = unpack4_front(res.x, quadratic, kappa_f_q)
    ll, x_filt, resid, n_obs = weighted_filter4_front(p_hat, smooth_hat, log_f, tau, dt, np.ones_like(log_f))
    return p_hat, smooth_hat, x_filt, resid, n_obs, ll


def _neg_loglik4_front_weighted(theta, log_f, tau, dt, weight, quadratic, kappa_f_q=KAPPA_F_Q_ANCHOR):
    try:
        p, smooth = unpack4_front(theta, quadratic, kappa_f_q)
    except ValueError:
        return 1e10
    ll, _, _, _ = weighted_filter4_front(p, smooth, log_f, tau, dt, weight)
    return -ll if np.isfinite(ll) else 1e10


def fit4_front_student_t(log_f: np.ndarray, tau: np.ndarray, dt: float, df: float, theta0: np.ndarray,
                         n_outer: int = 200, tol: float = 1e-6, quadratic: bool = True,
                         kappa_f_q: float = KAPPA_F_Q_ANCHOR, estimate_df: bool = False,
                         df_bounds: tuple = (2.1, 100.0)):
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    p, smooth = unpack4_front(theta, quadratic, kappa_f_q)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik4_front_weighted, theta,
                       args=(log_f, tau, dt, weight, quadratic, kappa_f_q),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p, smooth = unpack4_front(theta, quadratic, kappa_f_q)
        _, _, resid_raw, _ = weighted_filter4_front(p, smooth, log_f, tau, dt, np.ones_like(log_f))
        if estimate_df:
            z_finite = resid_raw[np.isfinite(resid_raw)]
            res_df = minimize_scalar(_neg_t_loglik_df, bounds=df_bounds, method="bounded", args=(z_finite,))
            df = float(res_df.x)
        new_weight = student_t_weights3(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = weighted_filter4_front(p, smooth, log_f, tau, dt, np.ones_like(log_f))
    return p, smooth, x_filt, resid_raw, weight, it + 1, ll, df
