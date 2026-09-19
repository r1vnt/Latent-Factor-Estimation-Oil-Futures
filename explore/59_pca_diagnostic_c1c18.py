# %% [markdown]
# # PCA diagnostic: is 3 factors the ceiling, or is the exponential loading the cost?
#
# H_combo exceeding MSE(rw) outright (up to 4405% on the deepest
# butterflies) isn't "poor fit" -- it's a claim that measurement error
# exceeds total observed variation, which would make the spread
# unobservable. It isn't; both legs are settlement prices verified to
# the tick against EIA on 99.94% of days. So H is globally overstated,
# and ML is booking PERSISTENT curve-shape misfit as noise because the
# model has nowhere else to put it. Two candidates, no model/filter,
# minutes of arithmetic:
#
#   1. Not enough factors -- 3 states can't span 18 maturities' daily
#      variation (ruled less likely already: 4 factors added nothing).
#   2. Wrong loading shape -- 3 states are plenty, but the affine family
#      {e^{-k1*tau}, e^{-k2*tau}, 1} is too rigid, and the no-arbitrage
#      restriction is what costs the forecast.
#
# Four numbers per contract, on daily CHANGES (levels are dominated by
# the unit root and tell you nothing):
#   1. total variance of dy_i
#   2. residual variance after projecting onto 3 UNCONSTRAINED PCs
#      (the ceiling any 3-factor linear model could reach)
#   3. residual variance after projecting onto the AFFINE model's own
#      fixed loadings [e^{-k1*tau_j}, e^{-k2*tau_j}, 1] at the fitted
#      k1=1.887, k2=0.389 -- cross-sectional OLS at each date, so this
#      isolates the LOADING SHAPE's spanning power, independent of the
#      temporal (AR-1-like) dynamics already tested by the forecast runs.
#   4. 2*H_i, the model's claimed noise in CHANGE units (H is a level
#      variance; Var(eps_t - eps_{t-1}) = 2H for iid measurement noise --
#      the factor of two is the bridge, not a rounding choice).
#
# Complete-case rows only (a handful of dates, e.g. around 2020-04-20,
# have an undefined front-month log price and are dropped from both the
# PCA and the affine regression for a fair comparison).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

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
Z_arr = sys3["Z"]
mean_H = np.nanmean(sys3["Hdiag"], axis=0)
print(f"fitted kappa1={p.kappa1:.4f}  kappa2={p.kappa2:.4f}")

# %% [markdown]
# ## Daily changes, complete cases only

# %%
dy = np.diff(log_f, axis=0)          # (n-1, 18); dy[i] = log_f[i+1] - log_f[i]
Z_at_change = Z_arr[1:, :, :]         # loadings dated at the SAME date as the change
complete = np.all(np.isfinite(dy), axis=1)
n_complete = int(complete.sum())
print(f"complete-case dates: {n_complete}/{len(dy)} ({len(dy) - n_complete} dropped)")

dy_c = dy[complete]
Z_c = Z_at_change[complete]
dy_centered = dy_c - dy_c.mean(axis=0, keepdims=True)

# %% [markdown]
# ## 1/2. PCA-3: the ceiling any unconstrained 3-factor linear model reaches

# %%
U, S, Vt = np.linalg.svd(dy_centered, full_matrices=False)
loadings_pca = Vt[:3, :].T                     # (18, 3)
scores_pca = U[:, :3] * S[:3]                  # (n_complete, 3)
dy_hat_pca = scores_pca @ Vt[:3, :]
resid_pca = dy_centered - dy_hat_pca

total_var = np.var(dy_centered, axis=0, ddof=1)
pca_resid_var = np.var(resid_pca, axis=0, ddof=1)
pct_explained_pca_total = float(1.0 - (S[3:] ** 2).sum() / (S ** 2).sum())
print(f"\n{'=' * 60}\nAGGREGATE: variance explained by 3 unconstrained PCs = {pct_explained_pca_total:.4%}\n{'=' * 60}")
print("(term-structure panels typically give 95-99% for 3 -- benchmark check)")

# %% [markdown]
# ## 3. Affine model's own fixed loadings: cross-sectional OLS at each date

# %%
resid_affine = np.empty_like(dy_c)
for i in range(n_complete):
    X = Z_c[i]                        # (18, 3), this date's actual e^{-k*tau} loadings
    beta, *_ = np.linalg.lstsq(X, dy_c[i], rcond=None)
    resid_affine[i] = dy_c[i] - X @ beta
affine_resid_var = np.var(resid_affine, axis=0, ddof=1)
# aggregate: variance-weighted, matching the PCA aggregate's construction
pct_explained_affine_total = float(1.0 - affine_resid_var.sum() / total_var.sum())
print(f"AGGREGATE: variance explained by the affine model's own 3 loadings = "
     f"{pct_explained_affine_total:.4%}")

# %% [markdown]
# ## Per-contract comparison: PCA residual vs. affine residual vs. 2H

# %%
print(f"\n{'=' * 130}\nPER-CONTRACT: total var, PCA-3 residual, affine-3 residual, 2H, and the diagnostic ratios\n{'=' * 130}")
print(f"{'':4s}{'total var':>11s}{'PCA resid':>11s}{'affine resid':>13s}{'2H':>10s}"
     f"{'aff/PCA':>9s}{'2H/max(res)':>12s}{'PCA %exp':>9s}{'aff %exp':>9s}")
rows = []
for j in range(18):
    h2 = float(2.0 * mean_H[j])
    ratio_aff_pca = float(affine_resid_var[j] / pca_resid_var[j]) if pca_resid_var[j] > 0 else float("nan")
    max_res = max(pca_resid_var[j], affine_resid_var[j])
    ratio_2h = float(h2 / max_res) if max_res > 0 else float("nan")
    pct_pca_j = float(1.0 - pca_resid_var[j] / total_var[j])
    pct_aff_j = float(1.0 - affine_resid_var[j] / total_var[j])
    rows.append({"contract": j + 1, "total_var": float(total_var[j]), "pca3_resid_var": float(pca_resid_var[j]),
                "affine3_resid_var": float(affine_resid_var[j]), "twoH": h2,
                "affine_over_pca": ratio_aff_pca, "twoH_over_max_resid": ratio_2h,
                "pct_explained_pca": pct_pca_j, "pct_explained_affine": pct_aff_j})
    print(f"C{j + 1:2d}{total_var[j] * 1e6:11.4f}{pca_resid_var[j] * 1e6:11.4f}{affine_resid_var[j] * 1e6:13.4f}"
         f"{h2 * 1e6:10.4f}{ratio_aff_pca:9.3f}{ratio_2h:12.3f}{pct_pca_j * 100:8.2f}%{pct_aff_j * 100:8.2f}%")
print("(total var/resid/2H in units of 1e-6 log-price^2 per day)")

median_ratio_aff_pca = float(np.nanmedian([r["affine_over_pca"] for r in rows]))
median_ratio_2h = float(np.nanmedian([r["twoH_over_max_resid"] for r in rows]))
print(f"\nmedian affine/PCA residual ratio across 18 contracts: {median_ratio_aff_pca:.3f}")
print(f"median 2H/max(residual) ratio across 18 contracts: {median_ratio_2h:.3f}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if median_ratio_aff_pca > 3.0:
    print(f"Affine residual >> PCA residual (median ratio {median_ratio_aff_pca:.2f}x) -- three "
         f"states are plenty ({pct_explained_pca_total:.1%} explained unconstrained); the "
         f"exponential loading family {{e^-k1*tau, e^-k2*tau, 1}} is what can't fit the curve. "
         f"The no-arbitrage restriction is the cost, quotable at roughly "
         f"{pct_explained_pca_total - pct_explained_affine_total:.1%} of daily variance.")
elif median_ratio_aff_pca < 1.5:
    print(f"Affine residual ~ PCA residual (median ratio {median_ratio_aff_pca:.2f}x) -- the "
         f"loadings are about as good as any 3-factor linear model gets. H being globally "
         f"inflated is NOT explained by the loading shape -- genuinely surprising, worth chasing.")
else:
    print(f"Affine residual moderately exceeds PCA residual (median ratio {median_ratio_aff_pca:.2f}x) "
         f"-- a real but partial loading-shape cost, not the whole story.")

if median_ratio_2h > 3.0:
    print(f"\n2H >> max(PCA, affine) residual (median ratio {median_ratio_2h:.2f}x) -- confirms "
         f"curve-wide what the spread test showed at C1: the model books more noise than there "
         f"is unexplained variation to begin with, at essentially every maturity, not just the front.")

# %%
out = {
    "n_complete": n_complete, "n_total": len(dy),
    "pct_explained_pca3_aggregate": pct_explained_pca_total,
    "pct_explained_affine3_aggregate": pct_explained_affine_total,
    "median_affine_over_pca_ratio": median_ratio_aff_pca,
    "median_2H_over_max_resid_ratio": median_ratio_2h,
    "per_contract": rows,
}
out_path = ROOT / "results" / "c1c18_pca_diagnostic.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
