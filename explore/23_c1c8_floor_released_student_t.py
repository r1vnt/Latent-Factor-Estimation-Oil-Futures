# %% [markdown]
# # Is free-fit C7 (1.43bp) a genuine Student-t optimum, or the Gaussian
# # Chen-Scott collapse surviving under the floor's clamp?
#
# smoothmeas.py's own docstring records explore/18's Guard 2 finding:
# releasing MIN_SIGMA_MEAS at the GAUSSIAN stage sent C5 to 0.0021bp and
# C7 to 0.0233bp -- 60-700x below a tick -- the Chen-Scott collapse.
# Guard 2 only tested the Gaussian stage; every C1-C8 Student-t fit since
# (explore/16, 20, 21, 22) has used the DEFAULT floor throughout, so C7
# landing at 1.4289bp -- indistinguishable from the floor -- has never
# been checked against what happens with the floor off at the Student-t
# stage too. If C7 collapses toward the same ~0.02bp region under EM
# reweighting, its floored value is the same degenerate artifact clamped
# at a different wall, not a genuine feature the quadratic curve is
# failing to match. per_contract_loglik decomposes exactly this: whether
# a likelihood gain from releasing the floor is spread across contracts
# (a real structural improvement) or concentrated in one or two (the
# collapse signature, per its own docstring).

# %%
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import (  # noqa: E402
    MIN_SIGMA_MEAS,
    STUDENT_T_DF,
    fit_s0_gaussian,
    fit_s0_student_t,
    pack,
    per_contract_loglik,
)
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
KAPPA0 = 1.5015
DEFAULT_FLOOR = MIN_SIGMA_MEAS

log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))

# %% [markdown]
# ## Default floor (reproduces explore/16/20/21/22's free fit)

# %%
t0 = time.time()
p_gauss_default, _, _, _, _, _ = fit_s0_gaussian(
    log_f, tau, DT, M, theta0=pack(p_init, DEFAULT_FLOOR), min_sigma_meas=DEFAULT_FLOOR)
p_default, _, _, _, n_iters_default = fit_s0_student_t(
    log_f, tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss_default, DEFAULT_FLOOR),
    min_sigma_meas=DEFAULT_FLOOR)
t1 = time.time()
s_default_bp = (p_default.sigma_meas * 1e4).round(4)
print(f"[{time.strftime('%H:%M:%S')}] default floor ({DEFAULT_FLOOR * 1e4:.4f}bp) "
     f"done ({t1 - t0:.1f}s)  kappa={p_default.kappa:.4f}")
print(f"  s(bp): {s_default_bp.tolist()}")

# %% [markdown]
# ## Floor released entirely at BOTH stages (Gaussian and Student-t EM)

# %%
t2 = time.time()
p_gauss_rel, _, _, _, _, _ = fit_s0_gaussian(
    log_f, tau, DT, M, theta0=pack(p_init, 0.0), min_sigma_meas=0.0)
s_gauss_rel_bp = (p_gauss_rel.sigma_meas * 1e4).round(6)
print(f"\n[{time.strftime('%H:%M:%S')}] Gaussian, floor released: "
     f"kappa={p_gauss_rel.kappa:.4f}  s(bp)={s_gauss_rel_bp.tolist()}  "
     f"(Guard 2 comparison: C5=0.0021bp, C7=0.0233bp)")

p_rel, _, _, _, n_iters_rel = fit_s0_student_t(
    log_f, tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss_rel, 0.0), min_sigma_meas=0.0)
t3 = time.time()
s_rel_bp = (p_rel.sigma_meas * 1e4).round(6)
print(f"[{time.strftime('%H:%M:%S')}] Student-t, floor released, done "
     f"({t3 - t2:.1f}s, {n_iters_rel} EM iters)  kappa={p_rel.kappa:.4f}")
print(f"  s(bp): {s_rel_bp.tolist()}")

# %% [markdown]
# ## Per-contract log-likelihood decomposition of the floor-release gain

# %%
ll_contract_default = per_contract_loglik(p_default, log_f, tau, DT)
ll_contract_rel = per_contract_loglik(p_rel, log_f, tau, DT)
gain = ll_contract_rel - ll_contract_default

print(f"\n{'=' * 78}\nPER-CONTRACT LOG-LIKELIHOOD GAIN, floor released vs. default\n{'=' * 78}")
for i in range(M):
    flag = "  <-- C5/C7" if i in (4, 6) else ""
    print(f"  C{i + 1}: default={s_default_bp[i]:9.4f}bp  released={s_rel_bp[i]:9.6f}bp  "
         f"ll_gain={gain[i]:8.3f}{flag}")
total_gain = gain.sum()
gain_at_5_7 = gain[4] + gain[6]
print(f"\ntotal ll gain: {total_gain:.3f}   gain at C5+C7: {gain_at_5_7:.3f}  "
     f"({100 * gain_at_5_7 / total_gain:.1f}% of the total, if positive)")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
collapsed = [i + 1 for i in range(M) if s_rel_bp[i] < 0.3]
print(f"positions below 0.3bp when the Student-t-stage floor is released: "
     f"{collapsed if collapsed else 'none'}")
if 7 in collapsed or (s_rel_bp[6] < s_default_bp[6] / 3):
    print("C7 COLLAPSES further once the Student-t-stage floor is off too -- "
         "its floored value (1.4289bp) is the same Chen-Scott mechanism "
         "Guard 2 found at the Gaussian stage, just clamped at a different "
         "wall by MIN_SIGMA_MEAS. The quadratic curve's residual LR gap "
         "against this contract is not a real feature to chase; the free "
         "vector's C7 is not a legitimate target.")
else:
    print("C7 does NOT collapse further under Student-t reweighting with the "
         "floor off -- its floored value may be closer to a genuine local "
         "optimum than the Gaussian-stage collapse was, in which case the "
         "quadratic curve's residual LR gap deserves more weight.")

# %%
import json  # noqa: E402

out = {
    "default_floor_bp": s_default_bp.tolist(),
    "released_gaussian_bp": s_gauss_rel_bp.tolist(),
    "released_student_t_bp": s_rel_bp.tolist(),
    "per_contract_ll_gain": gain.tolist(),
    "collapsed_positions_below_0p3bp": collapsed,
}
out_path = ROOT / "results" / "c1c8_floor_released_student_t.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
