# %% [markdown]
# # Three-factor battery, C1-C18: does k1/k2 slide the way a single kappa did?
#
# Phase 0 revision: rates are NOT fixed a priori -- explore/27's multi-start
# was bit-identical across an 8x kappa_ratio range, so this tests the FREE
# three-factor model's own k1/k2 stability directly, same structure as
# explore/26's two-factor battery (18 leave-one-out + 3 block-outs + 2
# sub-periods + 3 cold-start checks), warm-started from the explore/27
# baseline (k1=1.887, k2=0.389).
#
# Acceptance target (from the plan): k1 and k2 spread across the three
# block-outs under ~0.15 each, against the 0.42 a single kappa showed.
# Multi-start tests the optimiser; block-outs test the specification --
# orthogonal checks, per the two-factor precedent on this exact panel.
#
# Bonus diagnostic (cheap, x_filt already computed for every fit): filtered
# chi1/chi2 correlation between each LOO/block-out fit and the full-panel
# baseline, on the shared date range (LOO and block-outs keep all 3254
# dates, only columns drop) -- the state-level version of the same question.

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
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
FLOOR_BP = 0.01 / 70.0 * 1e4

# %% [markdown]
# ## Reassemble panel, load the C1-C18 three-factor baseline

# %%
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

baseline = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_warm = ThreeFactorParams(
    kappa1=baseline["kappa1"], kappa2=baseline["kappa2"],
    sigma1=baseline["sigma1"], sigma2=baseline["sigma2"], sigma_xi=baseline["sigma_xi"],
    rho12=0.0, rho1x=baseline["rho1x"], rho2x=baseline["rho2x"],
    mu_xi=baseline["mu_xi"], mu_xi_star=baseline["mu_xi_star"],
    lambda1=baseline["lambda1"], lambda2=baseline["lambda2"],
)
A_WARM, B_WARM, C_WARM = baseline["curve"]["a"], baseline["curve"]["b"], baseline["curve"]["c"]
print(f"baseline: k1={p_warm.kappa1:.4f}  k2={p_warm.kappa2:.4f}  ll={baseline['ll']:.2f}")

p_cold = ThreeFactorParams(kappa1=2.0, kappa2=0.5, sigma1=0.3, sigma2=0.2, sigma_xi=0.2,
                           rho12=0.0, rho1x=0.0, rho2x=0.0, mu_xi=0.0, mu_xi_star=0.0,
                           lambda1=0.0, lambda2=0.0)
A_COLD, B_COLD, C_COLD = np.log(0.005), 0.0, 0.0


def fit_threefactor(log_f, tau, label, cold=False):
    t0 = time.time()
    m = log_f.shape[1]
    p_start = p_cold if cold else p_warm
    a0, b0, c0_ = (A_COLD, B_COLD, C_COLD) if cold else (A_WARM, B_WARM, C_WARM)
    theta0 = pack3_smooth(p_start, a0, b0, c0_, quadratic=True)
    p_g, smooth_g, _, _, _, ll_g = fit3_gaussian_smooth(log_f, tau, DT, theta0, quadratic=True)
    abc_g = fit_smooth_to_vector(sigma_from_tau(smooth_g, tau[0]), tau[0], quadratic=True)
    theta0_em = pack3_smooth(p_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True)
    p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll, _ = fit3_student_t_smooth(
        log_f, tau, DT, STUDENT_T_DF, theta0_em, n_outer=200, quadratic=True)
    s_bp = (sigma_from_tau(smooth_hat, tau[0]) * 1e4).round(4)
    pinned = [j + 1 for j in range(m) if abs(s_bp[j] - FLOOR_BP) < 0.05]
    dt = time.time() - t0
    tag = " [COLD]" if cold else ""
    ratio = p_hat.kappa1 / p_hat.kappa2
    print(f"[{time.strftime('%H:%M:%S')}] {label:26s}{tag:7s} m={m:2d}  "
         f"k1={p_hat.kappa1:7.4f}  k2={p_hat.kappa2:7.4f}  ratio={ratio:5.2f}  "
         f"ll={ll:12.2f}  pinned={pinned if pinned else 'none':10}  "
         f"n_iters={n_iters:3d}  ({dt:.1f}s)")
    return {"label": label, "cold": cold, "m": m, "kappa1": p_hat.kappa1,
           "kappa2": p_hat.kappa2, "ratio": ratio, "ll": ll,
           "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
           "pinned": pinned, "n_iters": n_iters, "seconds": dt,
           "x_filt": x_filt}


results = []
t_start = time.time()

# %% [markdown]
# ## Leave-one-contract-out (18 fits)

# %%
print(f"\n{'=' * 90}\nLEAVE-ONE-OUT (18 fits)\n{'=' * 90}")
for drop in range(18):
    cols = [j for j in range(18) if j != drop]
    r = fit_threefactor(panel.log_f[:, cols], panel.tau[:, cols], f"drop C{drop + 1}")
    r["cols"] = cols
    results.append(r)

# %% [markdown]
# ## Block-outs (3 fits)

# %%
print(f"\n{'=' * 90}\nBLOCK-OUTS (3 fits)\n{'=' * 90}")
blocks = {
    "drop C1-C4": [j for j in range(18) if j not in range(0, 4)],
    "drop C5-C12": [j for j in range(18) if j not in range(4, 12)],
    "drop C13-C18": [j for j in range(18) if j not in range(12, 18)],
}
for label, cols in blocks.items():
    r = fit_threefactor(panel.log_f[:, cols], panel.tau[:, cols], label)
    r["cols"] = cols
    results.append(r)

# %% [markdown]
# ## Sub-period splits (2 fits)

# %%
print(f"\n{'=' * 90}\nSUB-PERIOD SPLITS (2 fits)\n{'=' * 90}")
panel_early = panel.slice("2010-01-01", "2016-12-31")
panel_late = panel.slice("2017-01-01", "2022-11-21")
for label, sub in [("C1-C18, 2010-2016", panel_early), ("C1-C18, 2017-2022", panel_late)]:
    r = fit_threefactor(sub.log_f, sub.tau, label)
    r["cols"] = list(range(18))
    results.append(r)

# %% [markdown]
# ## Cold-start basin check (3 fits)

# %%
print(f"\n{'=' * 90}\nCOLD-START SPOT CHECK (3 fits)\n{'=' * 90}")
cold_configs = [
    ("C1-C18 baseline", list(range(18)), panel.log_f, panel.tau),
    ("drop C5-C12", blocks["drop C5-C12"], panel.log_f[:, blocks["drop C5-C12"]],
    panel.tau[:, blocks["drop C5-C12"]]),
    ("drop C1", [j for j in range(18) if j != 0],
    panel.log_f[:, [j for j in range(18) if j != 0]],
    panel.tau[:, [j for j in range(18) if j != 0]]),
]
for label, cols, log_f_sub, tau_sub in cold_configs:
    r = fit_threefactor(log_f_sub, tau_sub, label, cold=True)
    r["cols"] = cols
    results.append(r)

t_total = time.time() - t_start
print(f"\ntotal battery wall-clock: {t_total:.1f}s ({t_total / 60:.1f} min)")

# %% [markdown]
# ## Summary: k1/k2 spread against the 0.15 target, plus state correlations

# %%
print(f"\n{'=' * 90}\nBATTERY SUMMARY\n{'=' * 90}")
loo = [r for r in results if r["label"].startswith("drop C") and r["m"] == 17 and not r["cold"]]
block = [r for r in results if r["label"].startswith("drop C") and r["m"] != 17 and not r["cold"]]
subperiod = [r for r in results if "201" in r["label"] and not r["cold"]]
cold = [r for r in results if r["cold"]]

print(f"baseline: k1={p_warm.kappa1:.4f}  k2={p_warm.kappa2:.4f}")

k1_loo = [r["kappa1"] for r in loo]; k2_loo = [r["kappa2"] for r in loo]
print(f"\nLOO spread (18 fits): k1 {max(k1_loo) - min(k1_loo):.4f} "
     f"[{min(k1_loo):.4f}, {max(k1_loo):.4f}]   "
     f"k2 {max(k2_loo) - min(k2_loo):.4f} [{min(k2_loo):.4f}, {max(k2_loo):.4f}]")

k1_block = [r["kappa1"] for r in block]; k2_block = [r["kappa2"] for r in block]
block_k1_spread = max(k1_block) - min(k1_block)
block_k2_spread = max(k2_block) - min(k2_block)
print(f"\nblock-out spread (3 fits): k1 {block_k1_spread:.4f} "
     f"[{min(k1_block):.4f}, {max(k1_block):.4f}]   "
     f"k2 {block_k2_spread:.4f} [{min(k2_block):.4f}, {max(k2_block):.4f}]")
for r in block:
    print(f"  {r['label']:14s} k1={r['kappa1']:8.4f}  k2={r['kappa2']:8.4f}  ratio={r['ratio']:5.2f}")

print(f"\nACCEPTANCE TEST (target: block-out spread < 0.15 for both k1 and k2, "
     f"vs. the two-factor single-kappa spread of 0.42):")
k1_pass = block_k1_spread < 0.15
k2_pass = block_k2_spread < 0.15
print(f"  k1: {block_k1_spread:.4f}  {'PASS' if k1_pass else 'FAIL'}")
print(f"  k2: {block_k2_spread:.4f}  {'PASS' if k2_pass else 'FAIL'}")
print(f"  OVERALL: {'PASS -- the third factor fixed the sliding defect' if k1_pass and k2_pass else 'FAIL -- sliding persists; report as a finding about WTI, do not override'}")

k1_sub = [r["kappa1"] for r in subperiod]; k2_sub = [r["kappa2"] for r in subperiod]
print(f"\nsub-period: k1 {max(k1_sub) - min(k1_sub):.4f} [{min(k1_sub):.4f}, {max(k1_sub):.4f}]   "
     f"k2 {max(k2_sub) - min(k2_sub):.4f} [{min(k2_sub):.4f}, {max(k2_sub):.4f}]")

print(f"\ncold vs. warm basin-trapping check:")
for c in cold:
    warm_match = next((r for r in results if r["label"] == c["label"] and not r["cold"]), None)
    if warm_match is None and c["label"] == "C1-C18 baseline":
        wk1, wk2, wll = p_warm.kappa1, p_warm.kappa2, baseline["ll"]
    else:
        wk1, wk2, wll = warm_match["kappa1"], warm_match["kappa2"], warm_match["ll"]
    print(f"  {c['label']:20s}  cold: k1={c['kappa1']:7.4f} k2={c['kappa2']:7.4f} ll={c['ll']:12.2f}   "
         f"warm: k1={wk1:7.4f} k2={wk2:7.4f} ll={wll:12.2f}   "
         f"delta_ll={c['ll'] - wll:+.3f}")

all_pinned = [r for r in results if r["pinned"]]
print(f"\npinned contracts across all {len(results)} fits: "
     f"{'NONE' if not all_pinned else all_pinned}")

# %% [markdown]
# ## Filtered-state correlation, LOO + block-out fits vs. baseline (full dates)

# %%
print(f"\n{'=' * 90}\nFILTERED STATE CORRELATION vs. baseline (chi1, chi2)\n{'=' * 90}")
_, _, x_filt_baseline, _, _, _, _, _ = fit3_student_t_smooth(
    panel.log_f, panel.tau, DT, STUDENT_T_DF,
    pack3_smooth(p_warm, A_WARM, B_WARM, C_WARM, quadratic=True),
    n_outer=200, quadratic=True)
corrs = []
for r in loo + block:
    c1 = np.corrcoef(r["x_filt"][:, 0], x_filt_baseline[:, 0])[0, 1]
    c2 = np.corrcoef(r["x_filt"][:, 1], x_filt_baseline[:, 1])[0, 1]
    corrs.append({"label": r["label"], "corr_chi1": c1, "corr_chi2": c2})
    print(f"  {r['label']:14s}  corr(chi1)={c1:.4f}  corr(chi2)={c2:.4f}")
min_c1 = min(c["corr_chi1"] for c in corrs)
min_c2 = min(c["corr_chi2"] for c in corrs)
print(f"\nworst-case correlation: chi1={min_c1:.4f}  chi2={min_c2:.4f}  "
     f"(target: > 0.95)  {'PASS' if min_c1 > 0.95 and min_c2 > 0.95 else 'FAIL'}")

# %%
for r in results:
    r.pop("x_filt", None)
out_path = ROOT / "results" / "c1c18_threefactor_battery.json"
out_path.write_text(json.dumps({"results": results, "state_correlations": corrs,
                               "total_seconds": t_total,
                               "acceptance": {"k1_block_spread": block_k1_spread,
                                             "k2_block_spread": block_k2_spread,
                                             "k1_pass": k1_pass, "k2_pass": k2_pass}},
                              indent=2, default=str))
print(f"\nsaved to {out_path}")
