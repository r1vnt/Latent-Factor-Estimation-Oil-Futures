# %% [markdown]
# # Three-factor model, C1-C18: multi-start over kappa_ratio, then fit
#
# Warm-started from the two-factor C1-C18 result (kappa=0.7574) via
# from_two_factor, multi-start over kappa_ratio in {2,4,8,16} since that
# ratio is the parameter this warm start knows least about. Smooth
# quadratic curve throughout (base_system3 deliberately returns Hdiag as
# ones; fit3.py builds it from hdiag_from_tau, never a free per-contract
# vector). rho12 fixed at 0 (fit3.py's structural choice, not this
# script's).

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    STUDENT_T_DF,
    fit3_gaussian_smooth,
    fit3_student_t_smooth,
    pack3_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import from_two_factor  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
FLOOR_BP = 0.01 / 70.0 * 1e4
KAPPA_RATIOS = [2.0, 4.0, 8.0, 16.0]

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

# Two-factor C1-C18 baseline, for the warm start and the comparison table.
two_factor = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
p2 = SS_OIL.replace(kappa=two_factor["structural"]["kappa"])
A0, B0, C0 = two_factor["curve"]["a"], two_factor["curve"]["b"], two_factor["curve"]["c"]

# %% [markdown]
# ## Multi-start over kappa_ratio (Gaussian stage only)

# %%
print(f"\n{'=' * 90}\nMULTI-START, kappa_ratio in {KAPPA_RATIOS}\n{'=' * 90}")
starts = []
for ratio in KAPPA_RATIOS:
    t0 = time.time()
    p3_init = from_two_factor(p2, kappa_ratio=ratio)
    theta0 = pack3_smooth(p3_init, A0, B0, C0, quadratic=True)
    p_g, smooth_g, _, _, _, ll_g = fit3_gaussian_smooth(
        panel.log_f, panel.tau, DT, theta0, quadratic=True)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] ratio={ratio:5.1f}  ll={ll_g:12.2f}  "
         f"k1={p_g.kappa1:7.4f}  k2={p_g.kappa2:7.4f}  "
         f"k1/k2={p_g.kappa1 / p_g.kappa2:6.2f}  ({dt:.1f}s)")
    starts.append({"ratio": ratio, "p": p_g, "smooth": smooth_g, "ll": ll_g})

best = max(starts, key=lambda s: s["ll"])
print(f"\nbest start: ratio={best['ratio']}  ll={best['ll']:.2f}  "
     f"k1={best['p'].kappa1:.4f}  k2={best['p'].kappa2:.4f}")
ll_spread = max(s["ll"] for s in starts) - min(s["ll"] for s in starts)
print(f"ll spread across starts: {ll_spread:.4f}  "
     f"({'stable -- starts agree' if ll_spread < 1.0 else 'starts disagree, check individually'})")

# %% [markdown]
# ## Student-t EM from the best start

# %%
print(f"\n{'=' * 90}\nSTUDENT-T EM, C1-C18, three-factor\n{'=' * 90}")
t0 = time.time()
abc_g = fit_smooth_to_vector(sigma_from_tau(best["smooth"], panel.tau[0]),
                             panel.tau[0], quadratic=True)
theta0_em = pack3_smooth(best["p"], abc_g.a, abc_g.b, abc_g.c, quadratic=True)
p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll = fit3_student_t_smooth(
    panel.log_f, panel.tau, DT, STUDENT_T_DF, theta0_em, n_outer=200, quadratic=True)
dt = time.time() - t0
converged = n_iters < 200

s_bp = (sigma_from_tau(smooth_hat, panel.tau[0]) * 1e4).round(4)
pinned = [j + 1 for j in range(18) if abs(s_bp[j] - FLOOR_BP) < 0.05]

print(f"[{time.strftime('%H:%M:%S')}] done ({dt:.1f}s, {n_iters}/200 EM iters, "
     f"{'CONVERGED' if converged else 'HIT CAP'})")
print(f"kappa1={p_hat.kappa1:.4f} (half-life {p_hat.half_life_fast_days:.1f}d)   "
     f"kappa2={p_hat.kappa2:.4f} (half-life {p_hat.half_life_slow_days:.1f}d)   "
     f"ratio={p_hat.kappa1 / p_hat.kappa2:.2f}")
print(f"sigma1={p_hat.sigma1:.4f}  sigma2={p_hat.sigma2:.4f}  sigma_xi={p_hat.sigma_xi:.4f}")
print(f"rho1x={p_hat.rho1x:.4f}  rho2x={p_hat.rho2x:.4f}")
print(f"curve: a={smooth_hat.a:.4f}  b={smooth_hat.b:.4f}  c={smooth_hat.c:.4f}")
print(f"s(bp), day 0: {s_bp.tolist()}")
print(f"pinned at floor: {pinned if pinned else 'none'}")
print(f"ll = {ll:.2f}  (two-factor C1-C18 ll was {two_factor.get('ll', float('nan'))})")

# %% [markdown]
# ## Redundancy check: are k1, k2 actually separated, or collapsing together?

# %%
print(f"\n{'=' * 90}\nREDUNDANCY CHECK\n{'=' * 90}")
if p_hat.kappa1 / p_hat.kappa2 < 1.3:
    print(f"WARNING: k1/k2 = {p_hat.kappa1 / p_hat.kappa2:.2f} -- close to 1, "
         f"the second factor may be redundant (ratio driven by the warm "
         f"start / prior rather than the data).")
else:
    print(f"k1/k2 = {p_hat.kappa1 / p_hat.kappa2:.2f} -- factors are separated, "
         f"not collapsing toward each other.")

# %% [markdown]
# ## Conditioning: three-factor loadings vs. two-factor, on the realised tau grid

# %%
def z_diagnostics3(tau, k1, k2, m):
    tau_mean = np.nanmean(tau, axis=0)
    Z = np.column_stack([np.exp(-k1 * tau_mean), np.exp(-k2 * tau_mean), np.ones(m)])
    eig = np.linalg.eigvalsh(Z.T @ Z)
    return eig.min(), eig.min() / m, np.linalg.cond(Z)


def z_diagnostics2(tau, k, m):
    tau_mean = np.nanmean(tau, axis=0)
    Z = np.column_stack([np.exp(-k * tau_mean), np.ones(m)])
    eig = np.linalg.eigvalsh(Z.T @ Z)
    return eig.min(), eig.min() / m, np.linalg.cond(Z)


lmin3, lmin3_pc, cond3 = z_diagnostics3(panel.tau, p_hat.kappa1, p_hat.kappa2, 18)
lmin2, lmin2_pc, cond2 = z_diagnostics2(panel.tau, two_factor["structural"]["kappa"], 18)
print(f"\n{'=' * 90}\nCONDITIONING, C1-C18, realised tau grid\n{'=' * 90}")
print(f"three-factor (k1={p_hat.kappa1:.4f}, k2={p_hat.kappa2:.4f}): "
     f"lambda_min={lmin3:.4f}  lambda_min/contract={lmin3_pc:.4f}  cond(Z)={cond3:.2f}")
print(f"two-factor   (kappa={two_factor['structural']['kappa']:.4f}):            "
     f"lambda_min={lmin2:.4f}  lambda_min/contract={lmin2_pc:.4f}  cond(Z)={cond2:.2f}")

# %%
out = {
    "kappa_ratio_starts": [{"ratio": s["ratio"], "ll": s["ll"],
                           "k1": s["p"].kappa1, "k2": s["p"].kappa2} for s in starts],
    "best_ratio": best["ratio"], "ll_spread_across_starts": ll_spread,
    "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2,
    "half_life_fast_days": p_hat.half_life_fast_days,
    "half_life_slow_days": p_hat.half_life_slow_days,
    "sigma1": p_hat.sigma1, "sigma2": p_hat.sigma2, "sigma_xi": p_hat.sigma_xi,
    "rho1x": p_hat.rho1x, "rho2x": p_hat.rho2x,
    "mu_xi": p_hat.mu_xi, "mu_xi_star": p_hat.mu_xi_star,
    "lambda1": p_hat.lambda1, "lambda2": p_hat.lambda2,
    "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
    "s_bp_day0": s_bp.tolist(), "pinned": pinned,
    "n_em_iters": n_iters, "converged": converged, "ll": ll, "seconds": dt,
    "conditioning": {"three_factor": {"lambda_min": lmin3, "lambda_min_pc": lmin3_pc, "cond": cond3},
                     "two_factor": {"lambda_min": lmin2, "lambda_min_pc": lmin2_pc, "cond": cond2}},
}
out_path = ROOT / "results" / "c1c18_threefactor.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
