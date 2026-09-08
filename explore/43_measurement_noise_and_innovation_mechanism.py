# %% [markdown]
# # Two follow-up checks on the Study A3 result
#
# 1. Is real WTI front-spread measurement noise a small fraction of
#    spread variation (the user's "prices are cleaner than the
#    simulation assumed" hypothesis), the way the simulation's 62%
#    figure would predict if real noise were much smaller? Computed as
#    sqrt(s1^2 + s2^2) / std(front spread), same C1/C2 pair the naive
#    label uses, on the fitted three-factor smooth curve.
#
# 2. Does the filter's own raw innovation at C1/C2 predict realised vol
#    BETTER than either label -- the proposed mechanism for why naive
#    beats filtered specifically on the realised_vol target at long
#    horizons (the discarded component IS the signal for a volatility
#    target, even though it's noise for a level/state target).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402
from csr.study.recovery3 import realised_vol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

full_3f = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p = ThreeFactorParams(
    kappa1=full_3f["kappa1"], kappa2=full_3f["kappa2"], sigma1=full_3f["sigma1"],
    sigma2=full_3f["sigma2"], sigma_xi=full_3f["sigma_xi"], rho12=0.0,
    rho1x=full_3f["rho1x"], rho2x=full_3f["rho2x"], mu_xi=full_3f["mu_xi"],
    mu_xi_star=full_3f["mu_xi_star"], lambda1=full_3f["lambda1"], lambda2=full_3f["lambda2"],
    gamma_v=0.0,
)
smooth = SmoothMeasParams(a=full_3f["curve"]["a"], b=full_3f["curve"]["b"],
                          c=full_3f["curve"]["c"], quadratic=True)

# %% [markdown]
# ## 1. s(tau) at C1/C2 vs. front spread SD

# %%
s_all = sigma_from_tau(smooth, tau)
s1, s2 = s_all[:, 0], s_all[:, 1]
spread = log_f[:, 0] - log_f[:, 1]
spread_std = float(np.nanstd(spread))
combined_noise = float(np.sqrt(np.mean(s1 ** 2 + s2 ** 2)))
ratio = combined_noise / spread_std

print(f"{'=' * 78}\n1. MEASUREMENT NOISE / SPREAD-SD RATIO\n{'=' * 78}")
print(f"s1 (C1) mean: {s1.mean() * 1e4:.2f}bp   s2 (C2) mean: {s2.mean() * 1e4:.2f}bp")
print(f"front spread SD: {spread_std * 1e4:.2f}bp")
print(f"combined sqrt(s1^2+s2^2): {combined_noise * 1e4:.2f}bp")
print(f"ratio to spread SD: {ratio:.4f} ({ratio * 100:.1f}%)  vs. simulation's 62%")
print("CAVEAT: uses the model's fitted s(tau); C1's fitted s has been flagged all "
     "session as likely inflated by unresolved front-contract misspecification, "
     "not necessarily clean settlement noise -- this may not cleanly separate "
     "'market is clean' from 'model still doesn't fit the front'.")

# %% [markdown]
# ## 2. Innovation-predicts-volatility mechanism check

# %%
sys3 = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)
n = sys3["n"]
ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((n, 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))

logF = -2 * ll_cell - LOG_2PI - resid ** 2
F = np.exp(logF)
v_raw = resid * np.sqrt(F)   # back out the raw (unstandardised) innovation
v_mag = np.sqrt(v_raw[:, 0] ** 2 + v_raw[:, 1] ** 2)

print(f"\n{'=' * 78}\n2. INNOVATION-MAGNITUDE VS. LABEL REGRESSIONS, realised_vol\n{'=' * 78}")
label_r2 = {22: 0.0152, 66: 0.0204, 132: 0.0489, 252: 0.0620}   # from explore/42, full panel
results = {}
for h in (22, 66, 132, 252):
    y = realised_vol(log_f, h)
    valid = np.isfinite(y) & np.isfinite(v_mag)
    X = sm.add_constant(v_mag[valid], has_constant="add")
    res = sm.OLS(y[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
    results[h] = {"n_obs": int(res.nobs), "coef": float(res.params[1]),
                 "p": float(res.pvalues[1]), "r_squared": float(res.rsquared)}
    print(f"h={h:4d}  v_mag-only: n={int(res.nobs)}  coef={res.params[1]:+.4f} "
         f"(p={res.pvalues[1]:.4g})  R2={res.rsquared:.4f}   "
         f"vs. label-regression R2={label_r2[h]:.4f}   "
         f"{'INNOVATION WINS' if res.rsquared > label_r2[h] else 'labels win'}")

# %%
out = {
    "measurement_noise_ratio": {"s1_mean_bp": float(s1.mean() * 1e4),
                               "s2_mean_bp": float(s2.mean() * 1e4),
                               "spread_std_bp": spread_std * 1e4,
                               "combined_noise_bp": combined_noise * 1e4,
                               "ratio": ratio, "simulation_assumed": 0.62},
    "innovation_mechanism": results,
    "label_regression_r2_for_comparison": label_r2,
}
out_path = ROOT / "results" / "c1c18_study_a3_followups.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
