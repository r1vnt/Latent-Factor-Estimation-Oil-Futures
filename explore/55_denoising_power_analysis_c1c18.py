# %% [markdown]
# # Is the 0/18 null informative, or underpowered? A per-contract power calc.
#
# The theory (gap ~ H) makes a CONTRACT-SPECIFIC prediction of effect
# size, and that size varies enormously across the curve -- H at the
# front is huge in absolute terms (a real, specific, checkable claim);
# H at the back is a tiny fraction of that contract's own daily-move
# variance. A test with fixed power on ~1,900-3,200 observations can
# resolve the front's predicted effect and be structurally incapable of
# resolving the back's, however real it is. "0/18 significant" conflates
# those two very different situations unless the power is checked.
#
# For each contract: MDE = minimum detectable effect (two-sided
# alpha=0.01, 80% power) given the ACTUAL HAC standard error of the loss
# differential at this sample size; predicted effect = fitted H (the
# theory's own prediction for the gap); power_at_H = actual power to
# detect an effect of exactly that size, at that SE. detectable = H > MDE.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
HAC_LAGS = 20
ALPHA = 0.01   # two-sided, matching every DM significance flag used so far
POWER_TARGET = 0.80
Z_ALPHA2 = float(norm.ppf(1.0 - ALPHA / 2.0))
Z_POWER = float(norm.ppf(POWER_TARGET))
MDE_FACTOR = Z_ALPHA2 + Z_POWER
print(f"alpha={ALPHA} (two-sided, z={Z_ALPHA2:.4f})  target power={POWER_TARGET} (z={Z_POWER:.4f})  "
     f"MDE factor={MDE_FACTOR:.4f}")

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                          c=before["curve"]["c"], quadratic=True)

sys3 = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)
ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))

Z_arr, d_arr = sys3["Z"], sys3["d"]
pred = np.einsum('tia,ta->ti', Z_arr, xp) + d_arr
e_honest = log_f - pred
e_honest[0, :] = np.nan

rw_error = np.full_like(log_f, np.nan)
rw_error[1:] = np.diff(log_f, axis=0)

mean_H = np.nanmean(sys3["Hdiag"], axis=0)

# %% [markdown]
# ## Per-contract power calculation

# %%
print(f"\n{'=' * 118}\nPER-CONTRACT POWER: is the theory-predicted gap (=H) detectable at this sample size?\n{'=' * 118}")
hdr = (f"{'':4s}{'n':>6s}{'MSE(rw)':>12s}{'fitted H':>10s}{'H/MSE(rw)':>11s}"
      f"{'SE(d)':>10s}{'MDE':>10s}{'MDE/MSE(rw)':>13s}{'power@H':>9s}{'detectable':>12s}{'gap/H':>8s}")
print(hdr)
results = []
for j in range(18):
    d_loss = e_honest[:, j] ** 2 - rw_error[:, j] ** 2   # = MSE(filter) - MSE(rw) in expectation
    d_loss = d_loss[np.isfinite(d_loss)]
    nobs = len(d_loss)
    X = np.ones((nobs, 1))
    res = sm.OLS(d_loss, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    se = float(res.bse[0])

    mse_rw = float(np.nanmean(rw_error[:, j] ** 2))
    mse_f = float(np.nanmean(e_honest[:, j] ** 2))
    gap = mse_rw - mse_f
    h_j = float(mean_H[j])
    h_frac = h_j / mse_rw
    mde = MDE_FACTOR * se
    mde_frac = mde / mse_rw
    power_at_h = float(norm.cdf(h_j / se - Z_ALPHA2)) if se > 0 else float("nan")
    detectable = h_j > mde
    gap_over_h = gap / h_j if h_j > 0 else float("nan")

    results.append({"contract": j + 1, "n": nobs, "mse_rw": mse_rw, "fitted_H": h_j,
                    "H_frac_of_MSE_rw": h_frac, "SE_d": se, "MDE": mde, "MDE_frac_of_MSE_rw": mde_frac,
                    "power_at_H": power_at_h, "detectable": bool(detectable), "gap_over_H": gap_over_h})
    print(f"C{j + 1:2d}{nobs:6d}{mse_rw * 1e6:12.4f}{h_j * 1e6:10.4f}{h_frac * 100:10.3f}%"
         f"{se * 1e6:10.4f}{mde * 1e6:10.4f}{mde_frac * 100:12.3f}%{power_at_h:9.3f}"
         f"{'YES' if detectable else 'no':>12s}{gap_over_h:8.2f}")
print("(MSE(rw)/fitted H/SE/MDE columns in units of 1e-6 log-price^2)")

# %% [markdown]
# ## Is the predicted effect (H) largest, as a fraction of MSE, at the front?

# %%
h_fracs = [r["H_frac_of_MSE_rw"] for r in results]
print(f"\n{'=' * 90}\nH/MSE(rw) BY CONTRACT (is the predicted effect largest at the front?)\n{'=' * 90}")
for r in results:
    print(f"  C{r['contract']:2d}: H/MSE(rw) = {r['H_frac_of_MSE_rw'] * 100:.3f}%   "
         f"power@H = {r['power_at_H']:.3f}   detectable = {r['detectable']}")
front_biggest = h_fracs[0] == max(h_fracs)
print(f"\nfront contract (C1) has the largest H/MSE(rw) fraction: {front_biggest}")
print(f"contract with the largest H/MSE(rw) fraction: C{h_fracs.index(max(h_fracs)) + 1} "
     f"({max(h_fracs) * 100:.3f}%)")

# %%
out = {"alpha": ALPHA, "power_target": POWER_TARGET, "mde_factor": MDE_FACTOR, "results": results}
out_path = ROOT / "results" / "c1c18_denoising_power_analysis.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
