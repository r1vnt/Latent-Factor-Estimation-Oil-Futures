# %% [markdown]
# # Profile likelihood in kappa, C1-C18, smooth quadratic curve
#
# explore/24 found kappa=0.7574 at C1-C18 vs. 1.2445 at C1-C8 (smooth
# quadratic, both fits clean: zero pinned, EM converged) -- not the
# stabilisation-with-depth the panel-widening program was testing for.
# Before running the full perturbation battery on that number, profile
# the likelihood in kappa directly: is 0.7574 a sharp optimum, or does a
# wide range of kappa fit nearly as well (in which case "kappa moved"
# is not surprising -- it was barely identified to begin with, and the
# battery would just be re-discovering that same flatness 24+ times)?
#
# Same convention as csr.pricing.pricing_error.profile_likelihood_phi
# (Runbook 1.5): profile the plain Gaussian (weight=1) likelihood, not
# the full Student-t EM, refitting every other parameter (structural +
# curve) at each fixed kappa from the SAME base start every time (not
# sequentially warm-started point-to-point) -- a diagnostic of
# identification, not a re-estimate.

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import pack_smooth, unpack_smooth  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.pricing.twofactor import TwoFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

log_f = np.load(ROOT / "data" / "deep_panel_c18_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_c18_tau.npy")
fit = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s = fit["structural"]
c = fit["curve"]
print(f"loaded C1-C18 fit: kappa_hat={s['kappa']:.4f}  a={c['a']:.4f}  "
     f"b={c['b']:.4f}  c={c['c']:.4f}")

p_hat = TwoFactorParams(
    kappa=s["kappa"], sigma_chi=s["sigma_chi"], sigma_xi=s["sigma_xi"],
    rho=s["rho"], mu_xi=s["mu_xi"], mu_xi_star=s["mu_xi_star"],
    lambda_chi=s["lambda_chi"], sigma_meas=np.full(log_f.shape[1], 0.005),
)
theta0_base = pack_smooth(p_hat, c["a"], c["b"], c["c"], quadratic=True)[1:]  # drop log(kappa)


def _neg_ll_fixed_kappa(theta_reduced, kappa_fixed):
    theta_full = np.concatenate([[np.log(kappa_fixed)], theta_reduced])
    p = unpack_smooth(theta_full, tau, quadratic=True)
    ll, *_ = run_spec(REGISTRY["S0"], p, log_f, tau, DT)
    return -ll if np.isfinite(ll) else 1e10


KAPPA_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7574, 0.8, 0.9, 1.0,
                       1.1, 1.2, 1.2445, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])

# %%
t0 = time.time()
lls, successes, nits = [], [], []
for k in KAPPA_GRID:
    tk0 = time.time()
    res = minimize(_neg_ll_fixed_kappa, theta0_base, args=(k,),
                   method="BFGS", options={"maxiter": 500})
    lls.append(-res.fun)
    successes.append(bool(res.success))
    nits.append(int(res.nit))
    print(f"[{time.strftime('%H:%M:%S')}] kappa={k:7.4f}  ll={-res.fun:12.4f}  "
         f"success={res.success!s:5s}  nit={res.nit:4d}  ({time.time() - tk0:.1f}s)")

lls = np.array(lls)
t1 = time.time()
print(f"\ntotal profile time: {t1 - t0:.1f}s")

# %% [markdown]
# ## Shape

# %%
argmax_idx = int(np.argmax(lls))
argmax_kappa = float(KAPPA_GRID[argmax_idx])
ll_range = float(lls.max() - lls.min())
# 2 log-likelihood units ~ a rough 1-parameter 95% CI radius (chi2(1) 95%
# critical value / 2 = 1.92), for an eyeball sense of the "flat enough to
# contain both point estimates" question -- not a formal interval.
within_2ll = KAPPA_GRID[lls > lls.max() - 1.92]

print(f"\n{'=' * 78}\nPROFILE SHAPE\n{'=' * 78}")
print(f"argmax: kappa={argmax_kappa:.4f}  ll={lls.max():.4f}")
print(f"range across grid: {ll_range:.4f} nats")
print(f"kappa values within ~2 ll-units of the max (rough 95% band): "
     f"[{within_2ll.min():.4f}, {within_2ll.max():.4f}]" if len(within_2ll)
     else "none (grid too coarse or profile very peaked)")
print(f"\nC1-C8 point estimate  (1.2445): ll at nearest grid point "
     f"({KAPPA_GRID[np.argmin(np.abs(KAPPA_GRID - 1.2445))]:.4f}) = "
     f"{lls[np.argmin(np.abs(KAPPA_GRID - 1.2445))]:.4f}  "
     f"(delta from max: {lls[np.argmin(np.abs(KAPPA_GRID - 1.2445))] - lls.max():+.4f})")
print(f"C1-C18 point estimate (0.7574): ll at nearest grid point "
     f"({KAPPA_GRID[np.argmin(np.abs(KAPPA_GRID - 0.7574))]:.4f}) = "
     f"{lls[np.argmin(np.abs(KAPPA_GRID - 0.7574))]:.4f}  "
     f"(delta from max: {lls[np.argmin(np.abs(KAPPA_GRID - 0.7574))] - lls.max():+.4f})")

if 1.2445 in within_2ll or (within_2ll.min() <= 1.2445 <= within_2ll.max() if len(within_2ll) else False):
    verdict = ("FLAT -- the C1-C8 value (1.2445) sits within ~2 ll-units of the "
              "C1-C18 optimum. kappa is weakly identified at this depth too; "
              "the 'drift' is consistent with sampling along a shallow ridge, "
              "not two genuinely different optima.")
else:
    verdict = ("SHARP -- the C1-C8 value is well outside the ~2 ll-unit band "
              "around the C1-C18 optimum. The drift looks like a real shift "
              "in the best-supported kappa, not just flat-profile noise.")
print(f"\nVERDICT: {verdict}")

# %%
out = {
    "kappa_grid": KAPPA_GRID.tolist(), "loglik": lls.tolist(),
    "success": successes, "nit": nits,
    "argmax_kappa": argmax_kappa, "ll_range": ll_range,
    "within_2ll_band": within_2ll.tolist() if len(within_2ll) else [],
}
out_path = ROOT / "results" / "c1c18_kappa_profile.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
