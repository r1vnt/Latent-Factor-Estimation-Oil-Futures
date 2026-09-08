# %% [markdown]
# # Turn on gamma_v (eq. 24's lagged-RV process-noise channel) on C1-C18
#
# The claim to test: if the process-noise-scaling channel already in the
# spec lattice absorbs the volatility clustering explore/35-36 confirmed
# is real and independent of the mean dynamics, that's the fix -- one fit,
# no new model. Ported from twofactor.py to threefactor.py (gamma_v field
# + scale_variances3, both mirroring the existing two-factor pattern
# exactly) since the three-factor pipeline bypassed scale_variances
# entirely up to now.
#
# RV proxy: C1's own realised vol (5-day rolling std of log returns),
# lagged 1 day (never contemporaneous -- see twofactor.scale_variances's
# docstring for why), standardised to zero mean / unit variance so gamma_v
# is identified from TIME-VARIATION in vol, not conflated with sigma1/
# sigma2's baseline level (mirrors this project's existing convention:
# `intensity` is standardised for the same reason, per augment.py's
# comment on why raw article counts and standardised intensity are kept
# separate).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    STUDENT_T_DF,
    fit3_gaussian_smooth,
    fit3_student_t_smooth,
    pack3_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
RV_WINDOW = 5

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts")

# %% [markdown]
# ## Build the lagged, standardised RV series from C1

# %%
log_c1 = panel.log_f[:, 0]
ret_c1 = np.diff(log_c1, prepend=np.nan)
rv = pd.Series(ret_c1).rolling(RV_WINDOW, min_periods=RV_WINDOW).std().to_numpy()
rv_lag = np.roll(rv, 1)
rv_lag[0] = np.nan
ok = np.isfinite(rv_lag)
rv_lag_std = np.where(ok, (rv_lag - np.nanmean(rv_lag)) / np.nanstd(rv_lag), 0.0)
print(f"RV series: {ok.sum()}/{len(rv_lag)} valid days "
     f"(first {(~ok).sum()} days are NaN warm-up, filled with 0 -- inert "
     f"since gamma_v's exp(gamma_v*0)=1 there, a no-op, not a fabricated value)")

# %% [markdown]
# ## Refit C1-C18, gamma_v free, warm-started from the existing three-factor baseline

# %%
three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_init = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
    gamma_v=0.0,
)
c3 = three_factor["curve"]
theta0 = pack3_smooth(p_init, c3["a"], c3["b"], c3["c"], quadratic=True)

p_g, smooth_g, _, _, _, ll_g = fit3_gaussian_smooth(
    panel.log_f, panel.tau, DT, theta0, quadratic=True, rv_lag=rv_lag_std)
print(f"Gaussian done  ll={ll_g:.2f}  gamma_v={p_g.gamma_v:.4f}  "
     f"k1={p_g.kappa1:.4f}  k2={p_g.kappa2:.4f}")

abc_g = fit_smooth_to_vector(sigma_from_tau(smooth_g, panel.tau[0]), panel.tau[0], quadratic=True)
p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll = fit3_student_t_smooth(
    panel.log_f, panel.tau, DT, STUDENT_T_DF,
    theta0=pack3_smooth(p_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True),
    n_outer=200, quadratic=True, rv_lag=rv_lag_std)
converged = n_iters < 200
print(f"Student-t EM done ({n_iters}/200 iters, "
     f"{'CONVERGED' if converged else 'HIT CAP'})")
print(f"kappa1={p_hat.kappa1:.4f}  kappa2={p_hat.kappa2:.4f}  gamma_v={p_hat.gamma_v:.4f}")
print(f"ll = {ll:.2f}  (three-factor without gamma_v was {three_factor['ll']:.2f}, "
     f"gain={ll - three_factor['ll']:.2f})")

# %% [markdown]
# ## Decisive check: Ljung-Box on squared standardised innovations

# %%
print(f"\n{'=' * 90}\nLJUNG-BOX ON SQUARED INNOVATIONS, gamma_v fit\n{'=' * 90}")
LAGS = [1, 5, 10, 20]
sq_results = []
for j in range(18):
    z = resid[:, j]
    z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z ** 2, lags=LAGS, return_df=True)
    row = {"contract": j + 1}
    for lag in LAGS:
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    sq_results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"  C{j + 1:2d}: {flags}")

sig1 = [r["contract"] for r in sq_results if r["lb_pvalue_1"] < 0.01]
print(f"\nsignificant (p<0.01) at lag 1: {sig1}  (before gamma_v: all 18)")

print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if len(sig1) <= 3:
    print("gamma_v ABSORBS the clustering -- squared innovations are now "
         "roughly white. This is the fix, one parameter, already in the "
         "spec lattice.")
elif len(sig1) < 15:
    print("gamma_v PARTIALLY absorbs the clustering -- fewer contracts "
         "significant than before, but not clean. Worth keeping, not a "
         "complete answer.")
else:
    print("gamma_v does NOT absorb the clustering -- a single common "
         "front-contract-RV-driven scalar on (chi1, chi2)'s process noise "
         "is not the right functional form. The clustering is real "
         "(explore/35-36) but this specific channel, this specific proxy, "
         "doesn't reach it.")

# %%
out = {
    "gamma_v": p_hat.gamma_v, "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2,
    "ll": ll, "ll_without_gamma_v": three_factor["ll"], "n_em_iters": n_iters,
    "converged": converged, "ljung_box_squared": sq_results,
}
(ROOT / "results" / "c1c18_gamma_v.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_gamma_v.json")
