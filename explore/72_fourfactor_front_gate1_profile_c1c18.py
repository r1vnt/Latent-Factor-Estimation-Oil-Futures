# %% [markdown]
# # Gate 1: profile likelihood over kappa_f_P, restriction kappa_f_P=kappa_f_Q as the null
#
# lambda_f1 (equivalently kappa_f_P, given kappa_f_Q fixed) is notoriously
# weakly identified in affine models -- this is the test that decides
# whether the point estimate (8.28) means anything. kappa_f_P=kappa_f_Q
# (=12.3, completely affine for this factor) is a plain nested
# restriction with no unidentified nuisance parameter under the null --
# unlike the factor-count test, no Davies problem, a genuine chi2(1)
# reference applies.
#
# Grid spans the null (12.3) down toward near-unit-root, plus points
# above 12.3 to check for a second mode. At each grid point, kappa_f_P is
# HELD FIXED and every other parameter is re-optimised (a proper profile,
# not just re-evaluating the unrestricted optimum's other params).
#
# Falsifier, stated in advance: LR fails to reject at 5%, or the profile
# shows a second mode of comparable height -> the P/Q divergence is
# noise, and rung 2 reduces to "a fourth factor with a front-concentrated
# loading," a materially weaker claim than essentially-affine risk premia.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit4_front import _neg_loglik4_front, pack4_front  # noqa: E402
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]
c0 = before["curve"]

# Warm start at the unrestricted optimum
p_opt = FourFactorFrontParams(
    kappa2=best["kappa2_hat"], kappa3=best["kappa3_hat"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta_opt_full = pack4_front(p_opt, c0["a"], c0["b"], c0["c"], quadratic=True)
KAPPA_F_P_IDX = 14
reduced_idx = [i for i in range(len(theta_opt_full)) if i != KAPPA_F_P_IDX]


def _neg_loglik_profile(theta_reduced, log_kappa_f_p_fixed):
    theta_full = np.empty(len(theta_opt_full))
    theta_full[reduced_idx] = theta_reduced
    theta_full[KAPPA_F_P_IDX] = log_kappa_f_p_fixed
    return _neg_loglik4_front(theta_full, log_f, tau, DT, True)


# %% [markdown]
# ## Profile grid

# %%
GRID = [20.0, 15.0, KAPPA_F_Q_ANCHOR, 10.0, 8.28, 6.0, 4.0, 2.0, 1.0, 0.3, 0.05]
theta0_reduced = theta_opt_full[reduced_idx]

profile = []
for kf_p in GRID:
    res = minimize(_neg_loglik_profile, theta0_reduced, args=(np.log(kf_p),),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    ll = -res.fun
    profile.append({"kappa_f_P": kf_p, "ll": ll})
    print(f"kappa_f_P={kf_p:7.3f} (fixed)  ->  ll={ll:.3f}")

# %% [markdown]
# ## LR test: null is kappa_f_P = kappa_f_Q (the completely-affine restriction)

# %%
ll_null = next(r["ll"] for r in profile if abs(r["kappa_f_P"] - KAPPA_F_Q_ANCHOR) < 1e-9)
ll_unrestricted = best["ll"]
LR = 2.0 * (ll_unrestricted - ll_null)
p_value = float(chi2.sf(LR, df=1))

print(f"\n{'=' * 90}\nGATE 1 VERDICT\n{'=' * 90}")
print(f"ll(null, kappa_f_P=kappa_f_Q={KAPPA_F_Q_ANCHOR}) = {ll_null:.3f}")
print(f"ll(unrestricted, kappa_f_P={best['kappa_f_P_hat']:.3f}) = {ll_unrestricted:.3f}")
print(f"LR = {LR:.3f}  df=1  p={p_value:.4g}")

lls = [r["ll"] for r in profile]
max_ll = max(lls)
second_mode = any(abs(r["ll"] - max_ll) < 2.0 and abs(r["kappa_f_P"] - best["kappa_f_P_hat"]) > 2.0
                 for r in profile)

if p_value < 0.05 and not second_mode:
    print("\nREJECTED at 5%, single mode -- the P/Q divergence is distinguishable from zero. "
         "Rung 2's essentially-affine claim survives Gate 1.")
else:
    print("\nGate 1 FAILS -- either the restriction is not rejected, or the profile shows a "
         "second mode of comparable height. The P/Q divergence is noise; rung 2 reduces to "
         "'a fourth factor with a front-concentrated loading,' not an essentially-affine finding.")

# %%
out = {"profile": profile, "ll_null": ll_null, "ll_unrestricted": ll_unrestricted,
      "LR": LR, "p_value": p_value, "second_mode_detected": bool(second_mode)}
out_path = ROOT / "results" / "c1c18_fourfactor_front_gate1_profile.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
