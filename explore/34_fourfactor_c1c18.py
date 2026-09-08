# %% [markdown]
# # Four-factor model, C1-C18: the fast class, warm-started from the ACF rate
#
# kappa_fast starting value is explore/33's ACF-implied rate (12.3214,
# half-life ~20.5 trading days), not a guess. Old three-factor (kappa1,
# kappa2) become the new (kappa2, kappa3) via from_three_factor.

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit4 import (  # noqa: E402
    STUDENT_T_DF,
    fit4_gaussian_smooth,
    fit4_student_t_smooth,
    pack4_smooth,
)
from csr.pricing.fourfactor import from_three_factor  # noqa: E402
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
FLOOR_BP = 0.01 / 70.0 * 1e4

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts")

three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p3 = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
)
c3 = three_factor["curve"]

fast_diag = json.loads((ROOT / "results" / "c1c18_fast_class_diagnosis.json").read_text())
KAPPA_FAST0 = fast_diag["fitted_kappa_fast"]
print(f"three-factor baseline: k1={p3.kappa1:.4f}  k2={p3.kappa2:.4f}  "
     f"ll={three_factor['ll']:.2f}")
print(f"ACF-implied kappa_fast warm start: {KAPPA_FAST0:.4f}")

# %% [markdown]
# ## Gaussian pre-fit

# %%
t0 = time.time()
p4_init = from_three_factor(p3, kappa_fast=KAPPA_FAST0)
theta0 = pack4_smooth(p4_init, c3["a"], c3["b"], c3["c"], quadratic=True)
p4_g, smooth4_g, _, _, _, ll_g = fit4_gaussian_smooth(
    panel.log_f, panel.tau, DT, theta0, quadratic=True)
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian done ({t1 - t0:.1f}s)  ll={ll_g:.2f}")
print(f"  k1={p4_g.kappa1:.4f}  k2={p4_g.kappa2:.4f}  k3={p4_g.kappa3:.4f}")

# %% [markdown]
# ## Student-t EM

# %%
abc_g = fit_smooth_to_vector(sigma_from_tau(smooth4_g, panel.tau[0]), panel.tau[0], quadratic=True)
theta0_em = pack4_smooth(p4_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True)
p4_hat, smooth4_hat, x_filt4, resid4, weight4, n_iters4, ll4 = fit4_student_t_smooth(
    panel.log_f, panel.tau, DT, STUDENT_T_DF, theta0_em, n_outer=200, quadratic=True)
t2 = time.time()
converged = n_iters4 < 200

s_bp = (sigma_from_tau(smooth4_hat, panel.tau[0]) * 1e4).round(4)
pinned = [j + 1 for j in range(18) if abs(s_bp[j] - FLOOR_BP) < 0.05]

print(f"[{time.strftime('%H:%M:%S')}] Student-t EM done ({t2 - t1:.1f}s, "
     f"{n_iters4}/200 iters, {'CONVERGED' if converged else 'HIT CAP'})  total={t2 - t0:.1f}s")
print(f"k1={p4_hat.kappa1:.4f} (half-life {p4_hat.half_life_fast_days:.1f}d)")
print(f"k2={p4_hat.kappa2:.4f} (half-life {p4_hat.half_life_medium_days:.1f}d)")
print(f"k3={p4_hat.kappa3:.4f} (half-life {p4_hat.half_life_slow_days:.1f}d)")
print(f"sigma1={p4_hat.sigma1:.4f}  sigma2={p4_hat.sigma2:.4f}  sigma3={p4_hat.sigma3:.4f}  "
     f"sigma_xi={p4_hat.sigma_xi:.4f}")
print(f"rho1x={p4_hat.rho1x:.4f}  rho2x={p4_hat.rho2x:.4f}  rho3x={p4_hat.rho3x:.4f}  "
     f"(sum of squares={p4_hat.rho1x**2+p4_hat.rho2x**2+p4_hat.rho3x**2:.4f}, must be <=1)")
print(f"curve: a={smooth4_hat.a:.4f}  b={smooth4_hat.b:.4f}  c={smooth4_hat.c:.4f}")
print(f"s(bp), day 0: {s_bp.tolist()}")
print(f"pinned: {pinned if pinned else 'none'}")
print(f"ll = {ll4:.2f}  (three-factor was {three_factor['ll']:.2f}, "
     f"gain={ll4 - three_factor['ll']:.2f})")

# %% [markdown]
# ## Redundancy checks: any pair of rates collapsing together?

# %%
print(f"\n{'=' * 90}\nREDUNDANCY CHECKS\n{'=' * 90}")
r12 = p4_hat.kappa1 / p4_hat.kappa2
r23 = p4_hat.kappa2 / p4_hat.kappa3
print(f"k1/k2 = {r12:.2f}  {'WARNING: close to 1' if r12 < 1.3 else 'separated'}")
print(f"k2/k3 = {r23:.2f}  {'WARNING: close to 1' if r23 < 1.3 else 'separated'}")

# %% [markdown]
# ## Ljung-Box on the new innovations -- did adding the fast class whiten them?

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
print(f"\n{'=' * 90}\nLJUNG-BOX, four-factor standardised innovations\n{'=' * 90}")
LAGS = [1, 5, 10, 20]
lb_results = []
for j in range(18):
    z = resid4[:, j]
    z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z, lags=LAGS, return_df=True)
    row = {"contract": j + 1}
    for lag in LAGS:
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    lb_results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"  C{j + 1:2d}: {flags}")

sig1 = [r["contract"] for r in lb_results if r["lb_pvalue_1"] < 0.01]
sig5 = [r["contract"] for r in lb_results if r["lb_pvalue_5"] < 0.01]
print(f"\nsignificant (p<0.01) at lag 1: {sig1}")
print(f"significant (p<0.01) at lag 5: {sig5}")
print(f"(three-factor: 18/18 at both lag 1 and lag 5)")

# %%
out = {
    "kappa_fast_warm_start": KAPPA_FAST0,
    "kappa1": p4_hat.kappa1, "kappa2": p4_hat.kappa2, "kappa3": p4_hat.kappa3,
    "half_life_fast_days": p4_hat.half_life_fast_days,
    "half_life_medium_days": p4_hat.half_life_medium_days,
    "half_life_slow_days": p4_hat.half_life_slow_days,
    "sigma1": p4_hat.sigma1, "sigma2": p4_hat.sigma2, "sigma3": p4_hat.sigma3,
    "sigma_xi": p4_hat.sigma_xi,
    "rho1x": p4_hat.rho1x, "rho2x": p4_hat.rho2x, "rho3x": p4_hat.rho3x,
    "mu_xi": p4_hat.mu_xi, "mu_xi_star": p4_hat.mu_xi_star,
    "lambda1": p4_hat.lambda1, "lambda2": p4_hat.lambda2, "lambda3": p4_hat.lambda3,
    "curve": {"a": smooth4_hat.a, "b": smooth4_hat.b, "c": smooth4_hat.c},
    "s_bp_day0": s_bp.tolist(), "pinned": pinned,
    "n_em_iters": n_iters4, "converged": converged, "ll": ll4, "seconds": t2 - t0,
    "ljung_box": lb_results,
}
out_path = ROOT / "results" / "c1c18_fourfactor.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
