# %% [markdown]
# # Is the C1-C8 kappa=1.2192 optimum genuine, or a BFGS-path artifact?
#
# explore/16's deep-panel refit found kappa=1.2192 (vs 1.4490 on this same
# Kaggle panel's own C1-C4), with s_1/s_2 (147bp/78bp) far larger than the
# C1-C4 fit's s_1 alone (58bp) -- worse, not better, than the "more
# contracts pins down kappa" story predicted. Before trusting that as a
# real finding motivating 2.7 (a third factor), rule out the simpler
# explanation: BFGS's path-dependence on an ill-conditioned surface landed
# in a different place than another starting point would.
#
# BFGS-only (no Nelder-Mead fallback) -- explore/16's own diagnostic
# (scratchpad/diagnose_slow_fit.py, not committed, ad hoc) already showed
# Nelder-Mead's "success" improves the log-likelihood by 0.0064 over
# BFGS's "precision loss" stopping point on the C1-C4 problem, at a cost
# of 20 extra minutes -- for a stability check across starting points,
# that precision is not worth 6-7x the wall-clock, so this uses BFGS alone
# and reports raw convergence diagnostics rather than silently trusting
# them.

# %%
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import MIN_SIGMA_MEAS, _neg_loglik, pack, unpack  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8

log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")
print(f"deep panel: {log_f.shape}")

KAPPA_STARTS = [0.3, 0.6, 0.9, 1.2192, 1.5, 2.0, 3.0]
results = []

for k0 in KAPPA_STARTS:
    p_init = SS_OIL.replace(kappa=k0, sigma_meas=np.full(M, SS_OIL.sigma_meas.mean()))
    theta0 = pack(p_init)
    t0 = time.time()
    res = minimize(_neg_loglik, theta0, args=(log_f, tau, DT, M, MIN_SIGMA_MEAS),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    dt = time.time() - t0
    p_hat = unpack(res.x, M, MIN_SIGMA_MEAS)
    s_bp = (p_hat.sigma_meas * 1e4).round(1).tolist()
    print(f"[{time.strftime('%H:%M:%S')}] kappa0={k0:5.2f} -> kappa_hat={p_hat.kappa:8.4f}  "
         f"fun={res.fun:12.4f}  success={res.success!s:5s}  nit={res.nit:4d}  "
         f"nfev={res.nfev:5d}  time={dt:7.1f}s  s(bp)={s_bp}")
    results.append({"kappa0": k0, "kappa_hat": p_hat.kappa, "fun": res.fun,
                    "success": res.success, "nit": res.nit, "nfev": res.nfev,
                    "time": dt, "s_bp": s_bp, "p_hat": p_hat})

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 78}\nMULTI-START SUMMARY\n{'=' * 78}")
funs = np.array([r["fun"] for r in results])
kappas = np.array([r["kappa_hat"] for r in results])
best = results[int(np.argmin(funs))]
print(f"best fun={best['fun']:.4f}  at kappa0={best['kappa0']}  kappa_hat={best['kappa_hat']:.4f}")
print(f"fun spread across all starts: {funs.max() - funs.min():.4f}")
print(f"kappa_hat spread across all starts: {kappas.max() - kappas.min():.4f}  "
     f"(values: {kappas.round(4).tolist()})")

near_best = np.abs(funs - funs.min()) < 1.0
print(f"\nstarts landing within 1.0 nats of the best fun: {near_best.sum()}/{len(results)}")
if near_best.sum() >= len(results) - 1 and (kappas[near_best].max() - kappas[near_best].min()) < 0.1:
    print("VERDICT: STABLE -- independent starts converge to the same optimum. "
         "The large s_1/s_2 pattern on the deep panel is a genuine feature of "
         "the fit, not a BFGS starting-point artifact.")
else:
    print("VERDICT: UNSTABLE -- starts disagree materially. The deep-panel "
         "kappa=1.2192 / s_1=147bp result cannot be trusted without more "
         "work (e.g. a proper profile likelihood) before building on it.")
