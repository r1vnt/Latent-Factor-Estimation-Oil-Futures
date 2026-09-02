"""Real-data MLE fitting for S0 -- Gaussian and Student-t (Sec. XI-F).

Gaussian MLE (``fit_s0_gaussian``) is the baseline. A Gaussian likelihood
gives quadratic weight to a handful of extreme innovations, which on the
real EIA panel is April 2020: excluding that one month moves kappa
substantially (explore/02_fit_s0.py has the comparison), and a fit that
sensitive to a single month is not one to build Study A's filtered state
on. Student-t observation noise (``fit_s0_student_t``) down-weights
exactly those extreme residuals instead of excluding the month outright.

No closed-form filter exists for a Student-t observation model, so this
uses the standard EM/scale-mixture construction: a Student-t(df) residual
is a Gaussian scale mixture, y_t | lambda_t ~ N(., H_t / lambda_t),
lambda_t ~ Gamma(df/2, df/2). E-step: given the current fit, compute each
residual's posterior mean weight from the model's own (unweighted, raw
H_t) standardised innovation -- exactly the IRLS weight used in robust
regression. M-step: refit holding those weights fixed, i.e. an ordinary
Gaussian MLE against Hdiag / weight. Iterate. ``df`` is fixed, not jointly
estimated -- a reasonable first pass, not a final answer (see
explore/02_fit_s0.py's df diagnostics for why estimating it freely on the
full sample would just re-absorb April 2020 into the tail instead).

This is the same fitting logic explore/02_fit_s0.py validated directly
against real data (bypass-checked there against run_spec at weight = 1,
pinned here as tests/test_fit.py::test_weighted_filter_at_weight_one_...).
Extracted into the package because Study A (csr/study/recovery.py) needs
the fitted parameters as an input, and duplicating MLE-fitting code
between two places is exactly the kind of drift hazard this project has
otherwise avoided throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..state.augment import augment_prior, augment_system
from ..state.fast import filter_uv
from .specs import REGISTRY, run_spec
from .twofactor import TwoFactorParams, base_system, initial_state, scale_variances

# One tick ($0.01) at a representative $70/bbl, in log terms: measurement
# error cannot be smaller than the price grid it's quoted on. An unfloored
# fit let two contracts collapse to ~1e-9 -- a boundary solution, which
# invalidates Hessian-based standard errors for every parameter, not just
# theirs (asymptotic normality assumes an interior optimum).
MIN_SIGMA_MEAS = 0.01 / 70.0

#: Sec. XI-F's fixed Student-t degrees of freedom. Not jointly estimated --
#: see the module docstring and explore/02_fit_s0.py's df diagnostics.
STUDENT_T_DF = 5.0


def pack(p: TwoFactorParams, min_sigma_meas: float = MIN_SIGMA_MEAS) -> np.ndarray:
    """Unconstrained reparameterisation for plain BFGS: log for positive
    scales, atanh for rho, log-slack-above-floor for sigma_meas."""
    slack = np.maximum(np.asarray(p.sigma_meas, float) - min_sigma_meas, 1e-8)
    return np.concatenate([[
        np.log(p.kappa), np.log(p.sigma_chi), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda_chi,
    ], np.log(slack)])


def unpack(theta: np.ndarray, m: int,
          min_sigma_meas: float = MIN_SIGMA_MEAS) -> TwoFactorParams:
    # Clipped before exp(): line search proposes wild steps in log-space
    # transiently; this keeps them finite (and therefore penalisable by the
    # NLL) instead of overflowing to inf.
    e = np.exp(np.clip(theta, -50.0, 50.0))
    return TwoFactorParams(
        kappa=e[0], sigma_chi=e[1], sigma_xi=e[2], rho=np.tanh(theta[3]),
        mu_xi=theta[4], mu_xi_star=theta[5], lambda_chi=theta[6],
        sigma_meas=min_sigma_meas + e[7:7 + m],
    )


def _neg_loglik(theta, log_f, tau, dt, m, min_sigma_meas):
    p = unpack(theta, m, min_sigma_meas)
    ll, *_ = run_spec(REGISTRY["S0"], p, log_f, tau, dt)
    return -ll if np.isfinite(ll) else 1e10


def fit_s0_gaussian(log_f: np.ndarray, tau: np.ndarray, dt: float, m: int,
                    theta0: np.ndarray, min_sigma_meas: float = MIN_SIGMA_MEAS):
    """Gaussian MLE for S0.

    Returns ``(p_hat, hess_inv, x_filt, resid, n_obs, ll)``. ``hess_inv`` is
    BFGS's own inverse-Hessian approximation, usable directly for delta-
    method standard errors -- see explore/02_fit_s0.py for that derivation
    per parameter; not reproduced here since Study A only needs the point
    estimate to initialise the Student-t fit.
    """
    res = minimize(_neg_loglik, theta0, args=(log_f, tau, dt, m, min_sigma_meas),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        # Polish from wherever BFGS stalled -- a fresh Nelder-Mead pass from
        # the same point distinguishes real non-convergence from a flat or
        # degenerate optimum BFGS's line search gave up on.
        res2 = minimize(_neg_loglik, res.x, args=(log_f, tau, dt, m, min_sigma_meas),
                        method="Nelder-Mead",
                        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat = unpack(res.x, m, min_sigma_meas)
    ll, x_filt, P_filt, resid, n_obs = run_spec(REGISTRY["S0"], p_hat, log_f, tau, dt)
    return p_hat, res.hess_inv, x_filt, resid, n_obs, ll


def _raw_system(p: TwoFactorParams, log_f: np.ndarray, tau: np.ndarray, dt: float):
    """The augmented S0 system (no news) -- mirrors run_spec's S0 path."""
    sys = scale_variances(base_system(p, tau, dt), p)
    aug = augment_system(sys)
    x0, P0 = initial_state(log_f, tau, p)
    Q2_0 = sys["Q"][0] if np.asarray(sys["Q"]).ndim == 3 else sys["Q"]
    x0a, P0a = augment_prior(x0, P0, Q2_0)
    return aug, x0a, P0a


def weighted_filter(p: TwoFactorParams, log_f: np.ndarray, tau: np.ndarray, dt: float,
                    weight: np.ndarray):
    """Gaussian filter with Hdiag / weight. weight == 1 must reproduce
    run_spec's plain S0 filter exactly -- pinned in tests/test_fit.py
    before this is trusted for anything built on top of it."""
    aug, x0a, P0a = _raw_system(p, log_f, tau, dt)
    n = aug["n"]
    Hdiag = aug["Hdiag"] / weight
    Bu = np.zeros((n, 4))
    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f), np.ascontiguousarray(aug["Z"]),
        np.ascontiguousarray(aug["d"]), np.ascontiguousarray(aug["T"]),
        np.ascontiguousarray(aug["c"]), np.ascontiguousarray(aug["Q"]),
        np.ascontiguousarray(Hdiag), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0a), np.ascontiguousarray(P0a),
    )
    return float(ll), xf, resid, int(n_obs)


def student_t_weights(resid: np.ndarray, df: float) -> np.ndarray:
    """E-step: IRLS weight from the model's own unweighted residual.
    Missing cells get weight 1 (inert: divides Hdiag by 1, i.e. a no-op --
    filter_uv skips them on the observation side regardless)."""
    w = (df + 1.0) / (df + resid ** 2)
    return np.where(np.isfinite(resid), w, 1.0)


def _neg_loglik_weighted(theta, log_f, tau, dt, m, weight, min_sigma_meas):
    p = unpack(theta, m, min_sigma_meas)
    ll, _, _, _ = weighted_filter(p, log_f, tau, dt, weight)
    return -ll if np.isfinite(ll) else 1e10


def fit_s0_student_t(log_f: np.ndarray, tau: np.ndarray, dt: float, m: int,
                     df: float, theta0: np.ndarray,
                     min_sigma_meas: float = MIN_SIGMA_MEAS,
                     n_outer: int = 30, tol: float = 1e-6):
    """EM/scale-mixture Student-t(df) fit of S0.

    ``theta0`` should come from :func:`fit_s0_gaussian` (``pack(p_hat)``) --
    starting the EM loop from a converged Gaussian fit rather than raw
    defaults, matching how this was validated in explore/02_fit_s0.py.

    Returns ``(p_hat, x_filt, resid, weight, n_iters)``. ``resid`` and
    ``x_filt`` are from the FINAL unweighted (weight=1) filter pass at the
    converged parameters -- the raw scale the returned ``weight`` was
    itself computed from, not the reweighted one, so residual diagnostics
    and the filtered state stay on a directly interpretable scale.
    """
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    p = unpack(theta, m, min_sigma_meas)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik_weighted, theta,
                       args=(log_f, tau, dt, m, weight, min_sigma_meas),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p = unpack(theta, m, min_sigma_meas)
        _, _, resid_raw, _ = weighted_filter(p, log_f, tau, dt, np.ones_like(log_f))
        new_weight = student_t_weights(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = weighted_filter(p, log_f, tau, dt, np.ones_like(log_f))
    return p, x_filt, resid_raw, weight, it + 1
