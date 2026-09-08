# %% [markdown]
# # Runbook step 2, C1-C8: refit under the smooth measurement curve
#
# explore/16-19's free per-contract sigma_meas left C7 pinned at
# MIN_SIGMA_MEAS even after the Student-t EM stage (the "still pinned"
# branch of the gate -- see results/c1c8_student_t_svector.json). Per the
# runbook, this replaces the free m-entry sigma_meas vector with the
# 2-parameter curve s(tau) = exp(a + b*tau) (csr.pricing.smoothmeas)
# before scaling to C1-C18.
#
# The free-s_i fit is rerun here (not loaded from the saved JSON) because
# that JSON only kept kappa/sigma_meas, not the full TwoFactorParams --
# and the LR test below needs both models' Gaussian log-likelihood at
# their own converged point, evaluated by the same function, in the same
# run, so the comparison is apples-to-apples. Cheap either way: the whole
# C1-C8 free fit took ~2 minutes with a good warm start (explore/17's
# confirmed kappa=1.5015), not the 14.7h explore/19 warned about.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import (  # noqa: E402
    MIN_SIGMA_MEAS,
    STUDENT_T_DF,
    fit_s0_gaussian,
    fit_s0_gaussian_smooth,
    fit_s0_student_t,
    fit_s0_student_t_smooth,
    pack,
    pack_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, spread_bound  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
KAPPA0 = 1.5015  # explore/17's confirmed C1-C8 Gaussian-stage optimum
FLOOR_BP = MIN_SIGMA_MEAS * 1e4


def pinned_count(s_bp: np.ndarray, atol: float = 0.05) -> list[int]:
    return [i + 1 for i, s in enumerate(s_bp) if abs(s - FLOOR_BP) < atol]


log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")
print(f"[{time.strftime('%H:%M:%S')}] deep panel (C1-C8): {log_f.shape}, "
     f"floor={FLOOR_BP:.4f}bp")

# %% [markdown]
# ## Free s_i (baseline this nests)

# %%
t0 = time.time()
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))
p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f, tau, DT, M, theta0=pack(p_init))
p_free, _, _, _, n_iters_free = fit_s0_student_t(
    log_f, tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss))
ll_free, *_ = run_spec(REGISTRY["S0"], p_free, log_f, tau, DT)
t1 = time.time()
s_free_bp = (p_free.sigma_meas * 1e4).round(4)
pinned_free = pinned_count(s_free_bp)
print(f"[{time.strftime('%H:%M:%S')}] free s_i   done ({t1 - t0:.1f}s, "
     f"{n_iters_free} EM iters)  kappa={p_free.kappa:.4f}  ll={ll_free:.2f}")
print(f"  s_free(bp): {s_free_bp.tolist()}")
print(f"  pinned at floor: {pinned_free if pinned_free else 'none'}  "
     f"(gate expectation: <= k = 2 factors)")

# %% [markdown]
# ## Smooth curve, warm-started off the free fit's own sigma_meas

# %%
a0b0 = fit_smooth_to_vector(p_free.sigma_meas, tau)
print(f"\n[{time.strftime('%H:%M:%S')}] smooth warm start: a={a0b0.a:.4f}  "
     f"b={a0b0.b:.4f}  (from fit_smooth_to_vector on the free s_i vector)")

t2 = time.time()
# Structural warm start from the free fit's own converged params (not just
# kappa -- p_free already has sigma_chi/sigma_xi/rho/mu_xi/mu_xi_star/
# lambda_chi at a good optimum for this exact data); pack_smooth ignores
# p_free.sigma_meas entirely and uses a0b0 for the curve instead.
theta0_smooth = pack_smooth(p_free, a0b0.a, a0b0.b)
p_gauss_smooth, _, _, _, _, _ = fit_s0_gaussian_smooth(log_f, tau, DT, theta0=theta0_smooth)
# Recover (a, b) from the converged Gaussian-stage curve to warm-start the EM
# stage -- sigma_meas is already exactly log-linear in tau at this point, so
# this regression recovers the fitted (a, b) essentially exactly rather than
# approximating them.
ab_gauss = fit_smooth_to_vector(p_gauss_smooth.sigma_meas[0], tau[0])
p_smooth, _, _, _, n_iters_smooth = fit_s0_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF,
    theta0=pack_smooth(p_gauss_smooth, ab_gauss.a, ab_gauss.b))
ll_smooth, *_ = run_spec(REGISTRY["S0"], p_smooth, log_f, tau, DT)
t3 = time.time()

ab_final = fit_smooth_to_vector(p_smooth.sigma_meas[0], tau[0])
s_smooth_bp = (p_smooth.sigma_meas[0] * 1e4).round(4)
pinned_smooth = pinned_count(s_smooth_bp)
print(f"[{time.strftime('%H:%M:%S')}] smooth done ({t3 - t2:.1f}s, "
     f"{n_iters_smooth} EM iters)  kappa={p_smooth.kappa:.4f}  ll={ll_smooth:.2f}  "
     f"a={ab_final.a:.4f}  b={ab_final.b:.4f}")
print(f"  s_smooth(bp), day 0: {s_smooth_bp.tolist()}")
print(f"  pinned at floor: {pinned_smooth if pinned_smooth else 'none'}  "
     f"(gate expectation: zero)")
print(f"  spread_bound (max/min achievable ratio at this b): "
     f"{spread_bound(ab_final, tau[0]):.2f}  vs realised max/min = "
     f"{s_smooth_bp.max() / s_smooth_bp.min():.2f}")

# %% [markdown]
# ## LR test: free-s_i nests the smooth curve
#
# 2*(ll_free - ll_smooth) ~ chi2(m - 2) under H0 (smooth curve adequate).
# Both log-likelihoods are the same plain Gaussian S0 likelihood
# (run_spec), evaluated at each model's own Student-t-EM-converged point
# estimate -- the EM reweighting is how each point was found, not part of
# the likelihood being compared, so this is a standard nested-model LR
# regardless of which stage produced the estimate.

# %%
df = M - 2
LR = 2.0 * (ll_free - ll_smooth)
pval = float(chi2.sf(LR, df))
print(f"\n{'=' * 78}\nLR TEST (free s_i vs. smooth curve), C1-C8\n{'=' * 78}")
print(f"ll_free={ll_free:.4f}  ll_smooth={ll_smooth:.4f}  LR={LR:.4f}  df={df}  "
     f"p={pval:.4g}")
if LR < 0:
    print("NOTE: LR < 0 -- the smooth curve's own optimizer found a HIGHER "
         "likelihood than the free fit's boundary optimum. Not a nesting "
         "violation (the free fit is degenerate, not the true unrestricted "
         "MLE) but flag it rather than reporting a negative chi2 stat at "
         "face value.")
elif pval < 0.05:
    print("REJECTED at 5%: the smooth curve is a statistically worse fit than "
         "the free vector. Try quadratic=True (a hump is still degenerate-"
         "proof) before concluding smoothness itself is wrong.")
else:
    print("NOT REJECTED at 5%: smoothness is not rejected by the free-s_i "
         "comparison.")

print(f"\n{'=' * 78}\nGATE VERDICT\n{'=' * 78}")
print(f"free s_i pinned:     {pinned_free if pinned_free else 'none'}")
print(f"smooth curve pinned: {pinned_smooth if pinned_smooth else 'none'}")
if not pinned_smooth:
    print("PASS -- smooth curve carries zero pinned contracts. Proceed to "
         "C1-C18.")
else:
    print("FAIL -- smooth curve still pins a contract; the parameterisation "
         "is not doing its job. Do not proceed to C1-C18 -- investigate "
         "before scaling up.")

# %%
out = {
    "free": {"kappa": p_free.kappa, "ll": ll_free, "s_bp": s_free_bp.tolist(),
             "pinned": pinned_free, "n_em_iters": n_iters_free,
             "seconds": t1 - t0},
    "smooth": {"kappa": p_smooth.kappa, "ll": ll_smooth, "a": ab_final.a,
              "b": ab_final.b, "s_bp_day0": s_smooth_bp.tolist(),
              "pinned": pinned_smooth, "n_em_iters": n_iters_smooth,
              "seconds": t3 - t2},
    "lr_test": {"LR": LR, "df": df, "p_value": pval},
    "floor_bp": FLOOR_BP,
}
out_path = ROOT / "results" / "c1c8_smooth_vs_free.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
