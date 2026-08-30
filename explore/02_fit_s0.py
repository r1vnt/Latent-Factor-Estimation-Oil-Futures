# %% [markdown]
# # S0 fit on the real EIA panel
#
# The estimator (Phase 3) and a real panel (Phase 1) have never met. One
# `run_spec` call plus MLE tells us more than further building would:
#
# 1. What kappa comes out as, and its standard error -- converts "four
#    contracts barely identify kappa" from an assertion into a measurement.
# 2. The s_i (measurement-error stdevs) per contract -- the real-data
#    counterpart to the 62% noise-to-signal figure from simulation.
# 3. Whether standardised innovations look standard normal -- exact on
#    simulated data; the size of the gap on WTI is informative.
#
# Run with and without the April 2020 window (negative-WTI), per Sec. XII-A:
# that regime was never going to fit the pricing model, and burying it in
# one full-sample number would hide which gate actually failed.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec                 # noqa: E402
from csr.pricing.twofactor import (TwoFactorParams, base_system,  # noqa: E402
                                   initial_state, scale_variances)
from csr.simulate import SS_OIL                                   # noqa: E402
from csr.state.augment import augment_prior, augment_system       # noqa: E402
from csr.state.fast import filter_uv                              # noqa: E402

DT = 1.0 / 252.0   # one NYMEX business day
DATA = Path(__file__).resolve().parents[1] / "data" / "eia_cl_futures.parquet"

# One tick ($0.01) at a representative $70/bbl, in log terms: measurement
# error cannot be smaller than the price grid it's quoted on. The first fit
# let two contracts collapse to ~1e-9 -- a boundary solution, not a small
# estimate -- which invalidates the Hessian-based SEs for every parameter,
# not just theirs (asymptotic normality assumes an interior optimum).
MIN_SIGMA_MEAS = 0.01 / 70.0

# %% [markdown]
# ## Load the panel
#
# `basis="business"` because `DT` above is a business-day step. The
# annualisation clock for `tau` has to match `dt`'s clock or kappa picks up
# a quiet bias -- see the `basis` note in `curve/expiry.py`.

# %%
df = pd.read_parquet(DATA)
panel = Panel.from_prices(df, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia")
m = panel.m

# %% [markdown]
# ## MLE machinery
#
# Unconstrained reparameterisation (log for positive scales, atanh for rho)
# so plain BFGS applies and its `hess_inv` can be read directly for standard
# errors via the delta method -- no need for a separate numerical Hessian.
# `sigma_meas` is floored at `MIN_SIGMA_MEAS`, not merely logged: a plain log
# still lets the optimiser drive it to zero asymptotically, which is exactly
# the boundary problem the floor exists to rule out.

# %%
def pack(p: TwoFactorParams) -> np.ndarray:
    slack = np.maximum(np.asarray(p.sigma_meas, float) - MIN_SIGMA_MEAS, 1e-8)
    return np.concatenate([[
        np.log(p.kappa), np.log(p.sigma_chi), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda_chi,
    ], np.log(slack)])


def unpack(theta: np.ndarray, m: int) -> TwoFactorParams:
    # Clipped before exp(): line search proposes wild steps in log-space
    # transiently; this keeps them finite (and therefore penalisable by the
    # NLL) instead of overflowing to inf.
    e = np.exp(np.clip(theta, -50.0, 50.0))
    return TwoFactorParams(
        kappa=e[0], sigma_chi=e[1], sigma_xi=e[2], rho=np.tanh(theta[3]),
        mu_xi=theta[4], mu_xi_star=theta[5], lambda_chi=theta[6],
        sigma_meas=MIN_SIGMA_MEAS + e[7:7 + m],
    )


def neg_loglik(theta: np.ndarray, log_f: np.ndarray, tau: np.ndarray, m: int) -> float:
    p = unpack(theta, m)
    ll, *_ = run_spec(REGISTRY["S0"], p, log_f, tau, DT)
    return -ll if np.isfinite(ll) else 1e10


def fit_s0(log_f: np.ndarray, tau: np.ndarray, m: int, label: str):
    theta0 = pack(SS_OIL)
    res = minimize(neg_loglik, theta0, args=(log_f, tau, m),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        # Polish from wherever BFGS stalled -- a fresh Nelder-Mead pass from
        # the same point tells us whether it's still moving (real
        # non-convergence) or sitting on a flat/degenerate optimum.
        res2 = minimize(neg_loglik, res.x, args=(log_f, tau, m),
                        method="Nelder-Mead",
                        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        print(f"  [{label}] BFGS: {res.message} (nit={res.nit}, nfev={res.nfev}); "
             f"polish Nelder-Mead moved NLL by {res.fun - res2.fun:+.6f}")
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat = unpack(res.x, m)
    se_theta = np.sqrt(np.diag(res.hess_inv))

    # Delta method: natural-scale SE from the transformed-scale SE.
    se_kappa = p_hat.kappa * se_theta[0]
    se_sigma_chi = p_hat.sigma_chi * se_theta[1]
    se_sigma_xi = p_hat.sigma_xi * se_theta[2]
    se_rho = (1.0 - p_hat.rho ** 2) * se_theta[3]
    # d(sigma_meas)/d(theta) = exp(theta) = sigma_meas - floor (the slack
    # above the floor), not sigma_meas itself now that it's floored.
    se_sigma_meas = (p_hat.sigma_meas - MIN_SIGMA_MEAS) * se_theta[7:7 + m]
    half_life = p_hat.half_life_days
    se_half_life = (np.log(2.0) / p_hat.kappa ** 2) * se_kappa * 365.25

    ll, x_filt, P_filt, resid, n_obs = run_spec(REGISTRY["S0"], p_hat, log_f, tau, DT)
    r = resid[np.isfinite(resid)]

    print(f"\n=== {label} ===")
    print(f"converged: {res.success}   n_obs: {n_obs}   loglik: {ll:.1f}")
    print(f"kappa       {p_hat.kappa:9.4f}  +/- {se_kappa:.4f}")
    print(f"half-life   {half_life:9.1f}d +/- {se_half_life:.1f}d")
    print(f"sigma_chi   {p_hat.sigma_chi:9.4f}  +/- {se_sigma_chi:.4f}")
    print(f"sigma_xi    {p_hat.sigma_xi:9.4f}  +/- {se_sigma_xi:.4f}")
    print(f"rho         {p_hat.rho:9.4f}  +/- {se_rho:.4f}")
    print(f"mu_xi       {p_hat.mu_xi:9.4f}")
    print(f"mu_xi*      {p_hat.mu_xi_star:9.4f}")
    print(f"lambda_chi  {p_hat.lambda_chi:9.4f}")
    for i in range(m):
        at_floor = " (at floor)" if p_hat.sigma_meas[i] < 1.5 * MIN_SIGMA_MEAS else ""
        print(f"s_{i + 1} ({panel.labels[i]})   {p_hat.sigma_meas[i]:.6e}  "
             f"+/- {se_sigma_meas[i]:.2e}{at_floor}")
    print(f"innovations: mean {r.mean():+.4f}  sd {r.std():.4f}  "
         f"n={r.size}  |z|>3: {np.mean(np.abs(r) > 3):.2%}")
    return p_hat, res, resid


# %% [markdown]
# ## Fit 1: full sample

# %%
p_full, res_full, resid_full = fit_s0(panel.log_f, panel.tau, m, "FULL SAMPLE")

# %% [markdown]
# ## Fit 2: excluding April 2020
#
# Masked, not sliced: keeps `dt` fixed at one business day throughout and
# lets the filter's exact missing-observation handling treat the month as a
# gap, rather than concatenating two independently-tau'd stretches.

# %%
covid = (panel.dates >= "2020-04-01") & (panel.dates <= "2020-04-30")
log_f_ex = panel.log_f.copy()
log_f_ex[covid] = np.nan
print(f"\nmasking {int(covid.sum())} dates in April 2020 "
     f"({int(np.isfinite(panel.log_f[covid]).sum())} finite cells removed)")

p_ex, res_ex, resid_ex = fit_s0(log_f_ex, panel.tau, m, "EXCLUDING APRIL 2020")

# %% [markdown]
# ## rho: chi/xi confounding, or a real secular-shock regime?
#
# Over a 1-4 month spread the two loadings (e^{-kappa*tau} vs a flat 1.0) are
# close enough that a large sigma_xi could be the two factors partly standing
# in for each other -- in which case a strongly negative rho (compensating
# for the confounding) is the tell. Neither fit shows that:
#
#     rho, full sample     : see printed value above (~ +0.11)
#     rho, excluding Apr20 : see printed value above (~ -0.09)
#
# Both are modest and far from the -0.7-or-beyond signature of confounding.
# That reads as the secular-shock hypothesis for sigma_xi being live rather
# than an artifact -- tentatively, since every SE above still comes from a
# fit with at least one sigma_meas pinned at its floor (see below).

# %% [markdown]
# ## Did the floor actually fix anything?
#
# It did not move kappa. Full sample: 3.9424 vs 3.9422 unfloored: p_full's
# kappa is essentially unchanged from the original (unfloored) run, and
# likewise for the ex-April fit. What the floor *did* do is relocate the
# boundary from a numerically pathological ~1e-9 to an economically
# defensible ~1.4e-4 -- but CL4 (full sample) and CL2 + CL4 (ex-April) still
# land exactly on it, confirmed by their own SEs collapsing to ~1e-14-1e-18
# (the BFGS curvature in that direction is numerically zero at an active
# constraint). Two of four contracts are still being priced almost exactly
# by the other two under a 2-factor model. Read plainly: the floor makes the
# *symptom* defensible, it does not touch the underlying cause -- four
# contracts genuinely can't support more than two factors' worth of
# information here, floor or no floor.

# %% [markdown]
# ## Robust spec: Student-t observation noise (Sec. XI-F)
#
# A Gaussian likelihood gives quadratic weight to a handful of extreme
# innovations, which is one mechanism that could make kappa this sensitive
# to a single month. Student-t errors down-weight exactly those.
#
# No closed-form filter exists for a Student-t observation model, so this
# uses the standard EM/scale-mixture construction: a Student-t(df) residual
# is a Gaussian scale mixture, y_t | lambda_t ~ N(., H_t / lambda_t),
# lambda_t ~ Gamma(df/2, df/2). E-step: given the current fit, compute each
# residual's posterior mean weight from the model's own (unweighted, raw
# H_t) standardised innovation -- exactly the IRLS weight used in robust
# regression. M-step: refit holding those weights fixed, i.e. an ordinary
# Gaussian MLE against Hdiag / weight. Iterate. `df` is fixed, not jointly
# estimated -- a reasonable first pass, not a final answer.
#
# This bypasses `run_spec` to get at Hdiag directly; the sanity check right
# below pins the bypass against `run_spec` at weight = 1 before trusting it.

# %%
STUDENT_T_DF = 5.0


def _raw_system(p: TwoFactorParams, log_f: np.ndarray, tau: np.ndarray):
    """The augmented S0 system (no news) -- mirrors run_spec's S0 path."""
    sys = scale_variances(base_system(p, tau, DT), p)
    aug = augment_system(sys)
    x0, P0 = initial_state(log_f, tau, p)
    Q2_0 = sys["Q"][0] if np.asarray(sys["Q"]).ndim == 3 else sys["Q"]
    x0a, P0a = augment_prior(x0, P0, Q2_0)
    return aug, x0a, P0a


def weighted_filter(p: TwoFactorParams, log_f: np.ndarray, tau: np.ndarray,
                    weight: np.ndarray):
    """Gaussian filter with Hdiag / weight. weight == 1 is the plain S0 filter."""
    aug, x0a, P0a = _raw_system(p, log_f, tau)
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
    return float(ll), resid, int(n_obs)


# Bypass sanity check: weight = 1 must reproduce run_spec's S0 exactly.
_ll_chk, _resid_chk, _n_chk = weighted_filter(p_full, panel.log_f, panel.tau,
                                              np.ones_like(panel.log_f))
_ll_ref, _, _, _, _n_ref = run_spec(REGISTRY["S0"], p_full, panel.log_f, panel.tau, DT)
assert abs(_ll_chk - _ll_ref) < 1e-6 and _n_chk == _n_ref, "weighted_filter bypass diverged from run_spec"
print(f"\nbypass check OK: weighted_filter(weight=1) matches run_spec to "
     f"{abs(_ll_chk - _ll_ref):.2e}")


def student_t_weights(resid: np.ndarray, df: float) -> np.ndarray:
    """E-step: IRLS weight from the model's own unweighted residual.
    Missing cells get weight 1 (inert: divides Hdiag by 1, i.e. no-op)."""
    w = (df + 1.0) / (df + resid ** 2)
    return np.where(np.isfinite(resid), w, 1.0)


def neg_loglik_weighted(theta, log_f, tau, m, weight):
    p = unpack(theta, m)
    ll, resid, n_obs = weighted_filter(p, log_f, tau, weight)
    return -ll if np.isfinite(ll) else 1e10


def fit_student_t(log_f, tau, m, df, label, theta0, n_outer=30, tol=1e-6):
    weight = np.ones_like(log_f)
    theta = theta0.copy()
    p = unpack(theta, m)
    for it in range(n_outer):
        res = minimize(neg_loglik_weighted, theta, args=(log_f, tau, m, weight),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p = unpack(theta, m)
        _, resid_raw, _ = weighted_filter(p, log_f, tau, np.ones_like(log_f))
        new_weight = student_t_weights(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, resid_raw, n_obs = weighted_filter(p, log_f, tau, np.ones_like(log_f))
    r = resid_raw[np.isfinite(resid_raw)]
    print(f"\n=== {label} (Student-t df={df:.0f}, {it + 1} EM iters, "
         f"final weight shift {shift:.2e}) ===")
    print(f"kappa       {p.kappa:9.4f}")
    print(f"half-life   {p.half_life_days:9.1f}d")
    print(f"sigma_chi   {p.sigma_chi:9.4f}")
    print(f"sigma_xi    {p.sigma_xi:9.4f}")
    print(f"rho         {p.rho:9.4f}")
    for i in range(m):
        print(f"s_{i + 1} ({panel.labels[i]})   {p.sigma_meas[i]:.6e}")
    print(f"raw residuals (modeled as t_{df:.0f}, not N(0,1) -- var t_df = "
         f"{df / (df - 2):.2f} at this df): mean {r.mean():+.4f}  sd {r.std():.4f}  "
         f"n={r.size}  |z|>3: {np.mean(np.abs(r) > 3):.2%}")
    print("[point estimates only -- no formal SEs for this EM fit yet]")
    return p, resid_raw, weight


# %% [markdown]
# ## The testable prediction
#
# Under Student-t errors, full-sample kappa should move toward the ex-April
# value (~1.47) *without excluding anything* -- because the down-weighted
# April 2020 innovations stop dominating the fit quadratically.

# %%
p_t_full, resid_t_full, weight_t_full = fit_student_t(
    panel.log_f, panel.tau, m, STUDENT_T_DF, "FULL SAMPLE", theta0=pack(p_full),
)

print(f"\nkappa comparison:")
print(f"  Gaussian, full sample      : {p_full.kappa:7.4f}  (half-life {p_full.half_life_days:6.1f}d)")
print(f"  Gaussian, excluding Apr20  : {p_ex.kappa:7.4f}  (half-life {p_ex.half_life_days:6.1f}d)")
print(f"  Student-t, full sample     : {p_t_full.kappa:7.4f}  (half-life {p_t_full.half_life_days:6.1f}d)")

covid_weight = weight_t_full[covid]
covid_weight = covid_weight[np.isfinite(covid_weight)]
print(f"\nApril 2020 EM weights (1.0 = full Gaussian trust, -> 0 = down-weighted): "
     f"mean {covid_weight.mean():.3f}  min {covid_weight.min():.3f}, "
     f"vs overall mean {np.nanmean(weight_t_full):.3f}")
