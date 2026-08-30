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

from csr.curve.build import Panel                       # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec        # noqa: E402
from csr.pricing.twofactor import TwoFactorParams        # noqa: E402
from csr.simulate import SS_OIL                          # noqa: E402

DT = 1.0 / 252.0   # one NYMEX business day
DATA = Path(__file__).resolve().parents[1] / "data" / "eia_cl_futures.parquet"

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

# %%
def pack(p: TwoFactorParams) -> np.ndarray:
    return np.concatenate([[
        np.log(p.kappa), np.log(p.sigma_chi), np.log(p.sigma_xi),
        np.arctanh(np.clip(p.rho, -0.999, 0.999)),
        p.mu_xi, p.mu_xi_star, p.lambda_chi,
    ], np.log(np.asarray(p.sigma_meas, float))])


def unpack(theta: np.ndarray, m: int) -> TwoFactorParams:
    return TwoFactorParams(
        kappa=np.exp(theta[0]), sigma_chi=np.exp(theta[1]),
        sigma_xi=np.exp(theta[2]), rho=np.tanh(theta[3]),
        mu_xi=theta[4], mu_xi_star=theta[5], lambda_chi=theta[6],
        sigma_meas=np.exp(theta[7:7 + m]),
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
    se_sigma_meas = p_hat.sigma_meas * se_theta[7:7 + m]
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
        print(f"s_{i + 1} ({panel.labels[i]})   {p_hat.sigma_meas[i]:.6e}  "
             f"+/- {se_sigma_meas[i]:.2e}")
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
