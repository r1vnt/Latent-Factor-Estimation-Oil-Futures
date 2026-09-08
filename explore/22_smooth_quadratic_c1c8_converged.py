# %% [markdown]
# # Runbook step 2, C1-C8: quadratic curve with EM actually converged
#
# explore/21's quadratic fit hit fit_s0_student_t_smooth's n_outer=30 cap
# without the EM shift dropping below tol -- so ll_quad=94834.56 (LR=159.1
# vs. the free fit) may not be a converged value. This reruns the same
# fit with n_outer raised well past where it stopped, so the stopping
# point this time is the tol criterion, not the cap, before trusting the
# LR gap as real rather than an artifact of stopping early. At ~4s/outer
# iteration (122.4s / 30 iters last time), 200 iterations is ~13 minutes
# -- still cheap at this panel's warm-started wall-clock.

# %%
import time
from pathlib import Path
import sys

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
from csr.pricing.smoothmeas import fit_smooth_to_vector  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
KAPPA0 = 1.5015
FLOOR_BP = MIN_SIGMA_MEAS * 1e4
N_OUTER_CAP = 200


def pinned_count(s_bp: np.ndarray, atol: float = 0.05) -> list[int]:
    return [i + 1 for i, s in enumerate(s_bp) if abs(s - FLOOR_BP) < atol]


log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")

# %% [markdown]
# ## Free s_i (same baseline as explore/20-21)

# %%
t0 = time.time()
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))
p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f, tau, DT, M, theta0=pack(p_init))
p_free, _, _, _, n_iters_free = fit_s0_student_t(
    log_f, tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss))
ll_free, *_ = run_spec(REGISTRY["S0"], p_free, log_f, tau, DT)
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] free s_i done ({t1 - t0:.1f}s)  "
     f"kappa={p_free.kappa:.4f}  ll={ll_free:.2f}")

# %% [markdown]
# ## Quadratic curve, n_outer raised to {N_OUTER_CAP}

# %%
ab0 = fit_smooth_to_vector(p_free.sigma_meas, tau, quadratic=True)
t2 = time.time()
theta0_quad = pack_smooth(p_free, ab0.a, ab0.b, ab0.c, quadratic=True)
p_gauss_quad, _, _, _, _, _ = fit_s0_gaussian_smooth(
    log_f, tau, DT, theta0=theta0_quad, quadratic=True)
abc_gauss = fit_smooth_to_vector(p_gauss_quad.sigma_meas[0], tau[0], quadratic=True)
p_quad, _, _, _, n_iters_quad = fit_s0_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF,
    theta0=pack_smooth(p_gauss_quad, abc_gauss.a, abc_gauss.b, abc_gauss.c, quadratic=True),
    quadratic=True, n_outer=N_OUTER_CAP)
ll_quad, *_ = run_spec(REGISTRY["S0"], p_quad, log_f, tau, DT)
t3 = time.time()

converged = n_iters_quad < N_OUTER_CAP
abc_final = fit_smooth_to_vector(p_quad.sigma_meas[0], tau[0], quadratic=True)
s_quad_bp = (p_quad.sigma_meas[0] * 1e4).round(4)
pinned_quad = pinned_count(s_quad_bp)
print(f"[{time.strftime('%H:%M:%S')}] quadratic done ({t3 - t2:.1f}s, "
     f"{n_iters_quad}/{N_OUTER_CAP} EM iters, "
     f"{'CONVERGED (tol reached)' if converged else 'HIT CAP -- still not converged'})  "
     f"kappa={p_quad.kappa:.4f}  ll={ll_quad:.2f}  "
     f"a={abc_final.a:.4f}  b={abc_final.b:.4f}  c={abc_final.c:.4f}")
print(f"  s_quad(bp), day 0: {s_quad_bp.tolist()}")
print(f"  pinned at floor: {pinned_quad if pinned_quad else 'none'}")

# %% [markdown]
# ## LR test vs. free s_i, at the (now converged, or re-flagged) ll_quad

# %%
df_vs_free = M - 3
LR_vs_free = 2.0 * (ll_free - ll_quad)
pval_vs_free = float(chi2.sf(LR_vs_free, df_vs_free))

print(f"\n{'=' * 78}\nLR TEST, C1-C8, quadratic vs. free s_i (this run)\n{'=' * 78}")
print(f"ll_free={ll_free:.4f}  ll_quad={ll_quad:.4f}  LR={LR_vs_free:.4f}  "
     f"df={df_vs_free}  p={pval_vs_free:.4g}  "
     f"{'REJECTED' if pval_vs_free < 0.05 else 'not rejected'} at 5%")
print(f"vs. explore/21's uncapped-at-30 result: ll_quad was 94834.5561, "
     f"LR was 159.1267 -- delta this run: "
     f"d(ll_quad)={ll_quad - 94834.5561:+.4f}  d(LR)={LR_vs_free - 159.1267:+.4f}")

print(f"\n{'=' * 78}\nGATE VERDICT\n{'=' * 78}")
print(f"EM convergence: {'reached tol' if converged else 'DID NOT converge even at n_outer=' + str(N_OUTER_CAP)}")
print(f"pinned: {pinned_quad if pinned_quad else 'none'}")
if not pinned_quad and pval_vs_free >= 0.05:
    print("PASS -- zero pinned, not rejected vs. free s_i at a converged fit.")
elif not pinned_quad:
    print("STILL REJECTED even converged -- the gap is real, not a stopping-"
         "early artifact. Decide separately whether it's economically "
         "material (concentrated at C7, the free fit's own still-floor-"
         "adjacent contract) before scaling to C1-C18.")
else:
    print("FAIL -- pinned contract reappeared. Investigate.")

# %%
import json  # noqa: E402

out = {
    "n_outer_cap": N_OUTER_CAP,
    "n_iters_quad": n_iters_quad,
    "converged": converged,
    "free": {"kappa": p_free.kappa, "ll": ll_free},
    "quadratic": {"kappa": p_quad.kappa, "ll": ll_quad, "a": abc_final.a,
                 "b": abc_final.b, "c": abc_final.c,
                 "s_bp_day0": s_quad_bp.tolist(), "pinned": pinned_quad,
                 "seconds": t3 - t2},
    "lr_vs_free": {"LR": LR_vs_free, "df": df_vs_free, "p_value": pval_vs_free},
}
out_path = ROOT / "results" / "c1c8_quadratic_converged.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
