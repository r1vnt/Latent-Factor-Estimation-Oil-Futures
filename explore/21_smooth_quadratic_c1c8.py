# %% [markdown]
# # Runbook step 2, C1-C8: does a quadratic curve rescue the smooth fit?
#
# explore/20's linear curve s(tau) = exp(a + b*tau) eliminated the C7
# floor-pinning (pinned: none) but was decisively rejected against the
# free-s_i vector: LR=8692.0 on df=6, p~0 (results/c1c8_smooth_vs_free.json).
# The free vector isn't monotone -- it dips to 2.64bp at C4 then rises back
# to 8.67bp at C5 -- and a monotone log-linear curve cannot represent that
# shape at all, which is the obvious suspect for a rejection this large,
# not necessarily "smoothness is wrong". csr.pricing.smoothmeas's own
# quadratic=True option (a hump term, still degenerate-proof per its
# structural tests) is the documented next check before concluding that.
#
# Reruns the free fit again here (not reloaded) for the same reason as
# explore/20: the LR test needs both log-likelihoods from the same run.

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
from csr.pricing.smoothmeas import fit_smooth_to_vector, spread_bound  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
KAPPA0 = 1.5015
FLOOR_BP = MIN_SIGMA_MEAS * 1e4


def pinned_count(s_bp: np.ndarray, atol: float = 0.05) -> list[int]:
    return [i + 1 for i, s in enumerate(s_bp) if abs(s - FLOOR_BP) < atol]


log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")

# %% [markdown]
# ## Free s_i (same baseline as explore/20)

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
# ## Quadratic curve, warm-started off the free fit (c0 = 0)

# %%
ab0 = fit_smooth_to_vector(p_free.sigma_meas, tau, quadratic=True)
print(f"\n[{time.strftime('%H:%M:%S')}] quadratic warm start: a={ab0.a:.4f}  "
     f"b={ab0.b:.4f}  c={ab0.c:.4f}")

t2 = time.time()
theta0_quad = pack_smooth(p_free, ab0.a, ab0.b, ab0.c, quadratic=True)
p_gauss_quad, _, _, _, _, _ = fit_s0_gaussian_smooth(
    log_f, tau, DT, theta0=theta0_quad, quadratic=True)
abc_gauss = fit_smooth_to_vector(p_gauss_quad.sigma_meas[0], tau[0], quadratic=True)
p_quad, _, _, _, n_iters_quad = fit_s0_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF,
    theta0=pack_smooth(p_gauss_quad, abc_gauss.a, abc_gauss.b, abc_gauss.c, quadratic=True),
    quadratic=True)
ll_quad, *_ = run_spec(REGISTRY["S0"], p_quad, log_f, tau, DT)
t3 = time.time()

abc_final = fit_smooth_to_vector(p_quad.sigma_meas[0], tau[0], quadratic=True)
s_quad_bp = (p_quad.sigma_meas[0] * 1e4).round(4)
pinned_quad = pinned_count(s_quad_bp)
print(f"[{time.strftime('%H:%M:%S')}] quadratic done ({t3 - t2:.1f}s, "
     f"{n_iters_quad} EM iters)  kappa={p_quad.kappa:.4f}  ll={ll_quad:.2f}  "
     f"a={abc_final.a:.4f}  b={abc_final.b:.4f}  c={abc_final.c:.4f}")
print(f"  s_quad(bp), day 0: {s_quad_bp.tolist()}")
print(f"  pinned at floor: {pinned_quad if pinned_quad else 'none'}  "
     f"(gate expectation: zero)")

# %% [markdown]
# ## LR tests: quadratic vs. free (df = m-3), and linear vs. quadratic (df = 1)

# %%
df_vs_free = M - 3
LR_vs_free = 2.0 * (ll_free - ll_quad)
pval_vs_free = float(chi2.sf(LR_vs_free, df_vs_free))

# explore/20's linear result, for the nested linear-vs-quadratic test.
LL_LINEAR = 90568.1124  # results/c1c8_smooth_vs_free.json: smooth.ll
LR_vs_linear = 2.0 * (ll_quad - LL_LINEAR)
pval_vs_linear = float(chi2.sf(LR_vs_linear, 1))

print(f"\n{'=' * 78}\nLR TESTS, C1-C8\n{'=' * 78}")
print(f"quadratic vs. free s_i:   ll_free={ll_free:.4f}  ll_quad={ll_quad:.4f}  "
     f"LR={LR_vs_free:.4f}  df={df_vs_free}  p={pval_vs_free:.4g}  "
     f"{'REJECTED' if pval_vs_free < 0.05 else 'not rejected'} at 5%")
print(f"quadratic vs. linear:     ll_linear={LL_LINEAR:.4f}  ll_quad={ll_quad:.4f}  "
     f"LR={LR_vs_linear:.4f}  df=1  p={pval_vs_linear:.4g}  "
     f"{'quadratic term justified' if pval_vs_linear < 0.05 else 'quadratic term NOT justified'} at 5%")

print(f"\n{'=' * 78}\nGATE VERDICT\n{'=' * 78}")
print(f"quadratic curve pinned: {pinned_quad if pinned_quad else 'none'}")
if not pinned_quad and pval_vs_free >= 0.05:
    verdict = "PASS -- quadratic curve carries zero pinned contracts and is not rejected vs. free s_i. Use quadratic=True going into C1-C18."
elif not pinned_quad:
    verdict = "PARTIAL -- zero pinned contracts, but still statistically rejected vs. free s_i. The free vector's shape may need more flexibility than a hump (or C7's near-floor value is itself not a smooth-curve-representable feature); flag before scaling to C1-C18 rather than silently proceeding."
else:
    verdict = "FAIL -- quadratic curve still pins a contract. Investigate before scaling up."
print(verdict)

# %%
import json  # noqa: E402

out = {
    "free": {"kappa": p_free.kappa, "ll": ll_free},
    "quadratic": {"kappa": p_quad.kappa, "ll": ll_quad, "a": abc_final.a,
                 "b": abc_final.b, "c": abc_final.c,
                 "s_bp_day0": s_quad_bp.tolist(), "pinned": pinned_quad,
                 "n_em_iters": n_iters_quad, "seconds": t3 - t2},
    "lr_vs_free": {"LR": LR_vs_free, "df": df_vs_free, "p_value": pval_vs_free},
    "lr_vs_linear": {"LR": LR_vs_linear, "df": 1, "p_value": pval_vs_linear},
    "floor_bp": FLOOR_BP,
}
out_path = ROOT / "results" / "c1c8_quadratic_vs_free.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
