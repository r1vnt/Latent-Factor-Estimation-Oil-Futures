# %% [markdown]
# # Decomposing the +32,334 nat gain: better prediction, or claimed lower F?
#
# Correction to explore/29: `resid` returned by filter_uv / filter_uv_
# contributions is ALREADY v / sqrt(F) -- the exact one-step-ahead
# prediction-variance-standardised innovation (state/fast.py:139, 259),
# not a raw residual. explore/29's SD comparison was therefore already a
# standardised-residual comparison (good), but its Ljung-Box run divided
# that by sqrt(Hdiag) a second time (wrong -- corrected below, using
# `resid` as-is).
#
# Per-cell log(F) is recoverable exactly, with no filter changes, from
# filter_uv_contributions's own two outputs:
#
#   ll_cell = -0.5*(log(2*pi) + log(F) + resid_std^2)
#   =>  log(F) = -2*ll_cell - log(2*pi) - resid_std^2
#
# This splits the likelihood gain per contract into two pieces:
#   -0.5*resid_std^2   "is the point prediction actually more accurate,
#                       relative to the model's own claimed uncertainty?"
#   -0.5*log(F)         "does the model just claim a smaller F?"
# A model that improves the first while holding or shrinking the second is
# a genuine improvement. A model that shrinks F while resid_std^2 gets
# WORSE is overconfident -- claiming more precision than it delivers,
# which inflates the likelihood without inflating accuracy.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit import _raw_system  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402
from csr.pricing.twofactor import TwoFactorParams  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)

# %% [markdown]
# ## Rebuild both models' filters via filter_uv_contributions directly

# %%
two_factor = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s2, c2 = two_factor["structural"], two_factor["curve"]
p2 = TwoFactorParams(
    kappa=s2["kappa"], sigma_chi=s2["sigma_chi"], sigma_xi=s2["sigma_xi"],
    rho=s2["rho"], mu_xi=s2["mu_xi"], mu_xi_star=s2["mu_xi_star"],
    lambda_chi=s2["lambda_chi"],
    sigma_meas=np.exp(c2["a"] + c2["b"] * panel.tau + c2["c"] * panel.tau ** 2),
)
aug2, x0a2, P0a2 = _raw_system(p2, panel.log_f, panel.tau, DT)
n2 = aug2["n"]
ll2, xf2, Pf2, xp2, Pp2, resid2, ll_cell2, n_obs2 = filter_uv_contributions(
    np.ascontiguousarray(panel.log_f), np.ascontiguousarray(aug2["Z"]),
    np.ascontiguousarray(aug2["d"]), np.ascontiguousarray(aug2["T"]),
    np.ascontiguousarray(aug2["c"]), np.ascontiguousarray(aug2["Q"]),
    np.ascontiguousarray(aug2["Hdiag"]), np.zeros((n2, 4)),
    np.ascontiguousarray(x0a2), np.ascontiguousarray(P0a2),
)

three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p3 = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
)
smooth3 = SmoothMeasParams(a=three_factor["curve"]["a"], b=three_factor["curve"]["b"],
                           c=three_factor["curve"]["c"], quadratic=True)
sys3 = _raw_system3(p3, smooth3, panel.tau, DT)
from csr.pricing.threefactor import initial_state3  # noqa: E402
x03, P03 = initial_state3(panel.log_f, panel.tau, p3)
n3 = sys3["n"]
ll3, xf3, Pf3, xp3, Pp3, resid3, ll_cell3, n_obs3 = filter_uv_contributions(
    np.ascontiguousarray(panel.log_f), np.ascontiguousarray(sys3["Z"]),
    np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
    np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((n3, 3)),
    np.ascontiguousarray(x03), np.ascontiguousarray(P03),
)

print(f"ll2={ll2:.2f}  ll3={ll3:.2f}  gain={ll3 - ll2:.2f}  "
     f"(cross-check vs explore/29: 229642.91 / 261977.28 / 32334.37)")

# %% [markdown]
# ## Back out log(F) per cell, decompose the gain per contract

# %%
logF2 = -2.0 * ll_cell2 - LOG_2PI - resid2 ** 2
logF3 = -2.0 * ll_cell3 - LOG_2PI - resid3 ** 2

print(f"\n{'=' * 100}\nPER-CONTRACT DECOMPOSITION: resid_std^2 (calibration) vs log(F) (claimed uncertainty)\n{'=' * 100}")
print(f"{'':4s}{'resid_std^2':>14s}{'':>3s}{'mean logF':>12s}{'':>3s}{'ll/obs':>10s}   "
     f"{'  2F->3F  '}{'d(resid^2)':>12s}{'d(logF)':>10s}{'d(ll/obs)':>11s}")
rows = []
for j in range(18):
    r2 = resid2[:, j]; r3 = resid3[:, j]
    ok2, ok3 = np.isfinite(r2), np.isfinite(r3)
    m_r2, m_r3 = np.nanmean(r2 ** 2), np.nanmean(r3 ** 2)
    m_lf2, m_lf3 = np.nanmean(logF2[:, j]), np.nanmean(logF3[:, j])
    m_ll2, m_ll3 = np.nanmean(ll_cell2[:, j]), np.nanmean(ll_cell3[:, j])
    rows.append({"contract": j + 1, "resid2_sq": m_r2, "resid3_sq": m_r3,
                "logF2": m_lf2, "logF3": m_lf3, "ll2": m_ll2, "ll3": m_ll3})
    print(f"C{j + 1:2d}  2F: {m_r2:6.3f} logF={m_lf2:6.2f} ll={m_ll2:7.3f}   "
         f"3F: {m_r3:6.3f} logF={m_lf3:6.2f} ll={m_ll3:7.3f}   "
         f"d(resid^2)={m_r3 - m_r2:+7.3f}  d(logF)={m_lf3 - m_lf2:+7.3f}  "
         f"d(ll)={m_ll3 - m_ll2:+7.3f}")

pooled_resid2 = np.nanmean([r["resid2_sq"] for r in rows])
pooled_resid3 = np.nanmean([r["resid3_sq"] for r in rows])
pooled_logF2 = np.nanmean([r["logF2"] for r in rows])
pooled_logF3 = np.nanmean([r["logF3"] for r in rows])
print(f"\npooled (unweighted mean across contracts):")
print(f"  resid_std^2: 2F={pooled_resid2:.4f}  3F={pooled_resid3:.4f}  "
     f"delta={pooled_resid3 - pooled_resid2:+.4f}  "
     f"({'WORSE calibration under 3F' if pooled_resid3 > pooled_resid2 else 'better calibration under 3F'})")
print(f"  mean logF:   2F={pooled_logF2:.4f}  3F={pooled_logF3:.4f}  "
     f"delta={pooled_logF3 - pooled_logF2:+.4f}  "
     f"(F ratio 3F/2F = {np.exp(pooled_logF3 - pooled_logF2):.4f})")

# %% [markdown]
# ## Is the F-shrinkage concentrated where the smooth curve claims lowest noise?

# %%
from csr.pricing.smoothmeas import sigma_from_tau  # noqa: E402
s3_bp = sigma_from_tau(smooth3, panel.tau[0]) * 1e4
s2_bp = np.exp(c2["a"] + c2["b"] * panel.tau[0] + c2["c"] * panel.tau[0] ** 2) * 1e4
print(f"\n{'=' * 100}\nCurve trough vs. where d(logF) is most negative\n{'=' * 100}")
for j in range(18):
    print(f"C{j + 1:2d}: s2={s2_bp[j]:8.3f}bp  s3={s3_bp[j]:8.3f}bp  "
         f"d(logF)={rows[j]['logF3'] - rows[j]['logF2']:+7.3f}  "
         f"d(resid^2)={rows[j]['resid3_sq'] - rows[j]['resid2_sq']:+7.3f}")

# %% [markdown]
# ## Corrected Ljung-Box: resid3 IS already standardised, don't divide again

# %%
print(f"\n{'=' * 100}\nCORRECTED LJUNG-BOX (resid3 used as-is, no double standardisation)\n{'=' * 100}")
LAGS = [1, 5, 10, 20]
lb_results = []
for j in range(18):
    z = resid3[:, j]
    z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z, lags=LAGS, return_df=True)
    row = {"contract": j + 1, "n": len(z)}
    for lag in LAGS:
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    lb_results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"  C{j + 1:2d} (n={row['n']:4d}):  {flags}")

sig1 = [r["contract"] for r in lb_results if r["lb_pvalue_1"] < 0.01]
sig5 = [r["contract"] for r in lb_results if r["lb_pvalue_5"] < 0.01]
print(f"\nsignificant (p<0.01) at lag 1: {sig1}")
print(f"significant (p<0.01) at lag 5: {sig5}")

# %%
out = {
    "ll2": ll2, "ll3": ll3, "gain": ll3 - ll2,
    "per_contract": rows,
    "pooled": {"resid2_sq": pooled_resid2, "resid3_sq": pooled_resid3,
              "logF2": pooled_logF2, "logF3": pooled_logF3},
    "ljung_box_corrected": lb_results,
}
out_path = ROOT / "results" / "c1c18_ll_decomposition.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
