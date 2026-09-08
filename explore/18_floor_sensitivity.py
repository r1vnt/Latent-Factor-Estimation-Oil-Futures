# %% [markdown]
# # Guard 2: is the C1-C8 floor mildly binding, or is the fit degenerate?
#
# explore/17's Gaussian-stage multistart found s_5, s_7 pinned exactly at
# MIN_SIGMA_MEAS = 0.01/70 = 1.4286bp (one tick). That floor being active
# is not automatically pathological -- the S0P investigation's actual
# degeneracy signature was s_i collapsing to ~1e-9, five orders of
# magnitude below a tick, absurd on its face. One tick of settlement noise
# on a deferred contract is a defensible number to land on by itself.
#
# This releases the bound (min_sigma_meas=0.0, so unpack's floor term
# contributes nothing and sigma_meas_i = exp(theta_i) is free to go
# wherever the likelihood wants) and refits the Gaussian stage only --
# per the diagnosis in pricing_error.py Sec X-F, the boundary-seeking
# behaviour is a Gaussian-likelihood phenomenon (a Gaussian fit obliged to
# accommodate April 2020 finds "price the contract exactly" a cheap way to
# do it); the Student-t stage is what removes it, so if the released floor
# doesn't run away here, it will not run away after EM either. Warm-started
# from explore/17's confirmed optimum (kappa=1.5015), since that multistart
# already established the optimum doesn't depend on the starting point.
#
#   released bound settles ~0.5-1.2bp  -> mildly binding, harmless, move on
#   released bound runs to ~1e-9        -> degenerate; a third factor would
#                                          chase this, not real curvature

# %%
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import _neg_loglik, pack, unpack  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
DEFAULT_FLOOR = 0.01 / 70.0

log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")
print(f"deep panel (C1-C8): {log_f.shape}")
print(f"default floor: {DEFAULT_FLOOR * 1e4:.4f}bp")

# Warm start: explore/17's confirmed C1-C8 Gaussian optimum (kappa=1.5015,
# identical across all 7 starting points spanning a 10x range).
p_init = SS_OIL.replace(kappa=1.5015, sigma_meas=np.full(M, 0.005))
theta0_default_floor = pack(p_init, min_sigma_meas=DEFAULT_FLOOR)

# %% [markdown]
# ## Refit at the default floor (sanity check -- should reproduce explore/17)

# %%
t0 = time.time()
res_default = minimize(_neg_loglik, theta0_default_floor,
                       args=(log_f, tau, DT, M, DEFAULT_FLOOR),
                       method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
p_default = unpack(res_default.x, M, DEFAULT_FLOOR)
print(f"\n[{time.strftime('%H:%M:%S')}] default floor ({DEFAULT_FLOOR*1e4:.2f}bp): "
     f"{time.time()-t0:.1f}s  success={res_default.success}  fun={res_default.fun:.4f}  "
     f"kappa={p_default.kappa:.4f}")
print(f"  s(bp): {(p_default.sigma_meas * 1e4).round(3).tolist()}")

# %% [markdown]
# ## Refit with the bound released entirely (min_sigma_meas=0.0)

# %%
theta0_released = pack(p_init, min_sigma_meas=0.0)
t0 = time.time()
res_rel = minimize(_neg_loglik, theta0_released,
                   args=(log_f, tau, DT, M, 0.0),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
p_rel = unpack(res_rel.x, M, 0.0)
print(f"\n[{time.strftime('%H:%M:%S')}] released floor (0.0bp): "
     f"{time.time()-t0:.1f}s  success={res_rel.success}  fun={res_rel.fun:.4f}  "
     f"kappa={p_rel.kappa:.4f}")
print(f"  s(bp): {(p_rel.sigma_meas * 1e4).round(6).tolist()}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 78}\nGUARD 2 VERDICT\n{'=' * 78}")
was_pinned = np.isclose(p_default.sigma_meas * 1e4, DEFAULT_FLOOR * 1e4, atol=0.05)
print(f"positions pinned at default floor: "
     f"{[f'C{i+1}' for i in range(M) if was_pinned[i]]}")
for i in range(M):
    if was_pinned[i]:
        released_bp = p_rel.sigma_meas[i] * 1e4
        verdict = ("MILDLY BINDING -- settled near a tick, harmless"
                  if released_bp > 0.3 else
                  "DEGENERATE -- ran toward zero, floor was masking a real problem")
        print(f"  C{i+1}: default={p_default.sigma_meas[i]*1e4:.3f}bp  "
             f"released={released_bp:.6f}bp  -> {verdict}")
