# %% [markdown]
# # Optional fit: pin kappa2/kappa3 at the S3 baseline, refit only the 4th factor
#
# Rung 2's unrestricted fit let kappa2 move 1.89 -> 2.18 (kappa3 barely
# moved, 0.389 -> 0.385). The concern: kappa2's movement is what "frees"
# it to help the back of the curve (C5-C18, 37.6% of the gain) -- i.e.
# part of rung 2's gain is generic reallocation of curvature across ALL
# three original factors, not something specific to the new 4th factor.
# Pinning kappa2/kappa3 at their exact baseline S3 values and refitting
# ONLY sigma_f, sigma2, sigma3, sigma_xi, the rho's, mu_xi's, lambda's,
# kappa_f_P and the curve isolates what the 4th factor alone can buy
# without that reallocation margin. If the back-end (C5-C18) gain mostly
# vanishes here, that confirms "front factor absorbs front misfit, which
# frees kappa2 to reoptimise for the back" as the mechanism -- not
# generic flexibility. If the back-end gain survives even with kappa2/
# kappa3 frozen, the reallocation story is wrong and something else is
# going on.
#
# Falsifier stated in advance: back-end gain within ~20% of the
# unrestricted fit's 4928.76 nats -> reallocation is NOT the mechanism.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF, student_t_weights3  # noqa: E402
from csr.pricing.fit4_front import (  # noqa: E402
    _neg_loglik4_front_weighted,
    filter4_front_contributions,
    pack4_front,
    unpack4_front,
    weighted_filter4_front,
)
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

KAPPA2_BASELINE = before["kappa1"]   # 1.8871, old kappa1 -> new kappa2
KAPPA3_BASELINE = before["kappa2"]   # 0.3890, old kappa2 -> new kappa3
GAP23_BASELINE = KAPPA2_BASELINE - KAPPA3_BASELINE
print(f"pinning kappa2={KAPPA2_BASELINE:.4f}  kappa3={KAPPA3_BASELINE:.4f}  (gap={GAP23_BASELINE:.4f})")

# Warm start at the unrestricted rung-2 optimum, then overwrite kappa2/kappa3
p_init = FourFactorFrontParams(
    kappa2=KAPPA2_BASELINE, kappa3=KAPPA3_BASELINE,
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta_full_init = pack4_front(p_init, c0["a"], c0["b"], c0["c"], quadratic=True)
PINNED_IDX = [0, 1]
pinned_vals = np.array([np.log(GAP23_BASELINE), np.log(KAPPA3_BASELINE)])
reduced_idx = [i for i in range(len(theta_full_init)) if i not in PINNED_IDX]


def _rebuild(theta_reduced):
    theta_full = np.empty(len(theta_full_init))
    theta_full[PINNED_IDX] = pinned_vals
    theta_full[reduced_idx] = theta_reduced
    return theta_full


def _neg_loglik_reduced(theta_reduced, weight):
    theta_full = _rebuild(theta_reduced)
    return _neg_loglik4_front_weighted(theta_full, log_f, tau, DT, weight, True)


# %% [markdown]
# ## Student-t EM, kappa2/kappa3 excluded from the free parameter set

# %%
weight = np.ones_like(log_f)
theta_reduced = theta_full_init[reduced_idx]
n_outer, tol = 200, 1e-6
for it in range(n_outer):
    res = minimize(_neg_loglik_reduced, theta_reduced, args=(weight,),
                   method="BFGS", options={"maxiter": 300})
    theta_reduced = res.x
    theta_full = _rebuild(theta_reduced)
    p_hat, smooth_hat = unpack4_front(theta_full, True)
    _, _, resid_raw, _ = weighted_filter4_front(p_hat, smooth_hat, log_f, tau, DT, np.ones_like(log_f))
    new_weight = student_t_weights3(resid_raw, STUDENT_T_DF)
    shift = float(np.nanmax(np.abs(new_weight - weight)))
    weight = new_weight
    if shift < tol:
        break

ll, x_filt, resid_raw, n_obs = weighted_filter4_front(p_hat, smooth_hat, log_f, tau, DT, np.ones_like(log_f))
print(f"\nconverged in {it + 1} EM iterations")
print(f"ll={ll:.2f}  kappa2(pinned)={p_hat.kappa2:.4f}  kappa3(pinned)={p_hat.kappa3:.4f}  "
     f"kappa_f_P={p_hat.kappa_f_P:.4f}  sigma_f={p_hat.sigma_f:.4f}")

# %% [markdown]
# ## Per-contract gain vs the S3 baseline (same decomposition as explore/71)

# %%
ll_pinned, xf, Pf, xp, Pp, resid, ll_cell_pinned, n_obs_c = filter4_front_contributions(
    p_hat, smooth_hat, log_f, tau, DT)
print(f"cross-check: ll={ll_pinned:.2f}  (EM return {ll:.2f})")

ll_before = before["ll"]
ll_cell_before_sum = ll_before  # pooled total; per-contract breakdown needs the S3 cell decomposition
# Re-derive S3 per-contract ll via its own contributions call, matching explore/71's method exactly.
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

p3_before = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"], sigma2=before["sigma2"],
    sigma_xi=before["sigma_xi"], rho12=0.0, rho1x=before["rho1x"], rho2x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
)
smooth3_before = SmoothMeasParams(a=c0["a"], b=c0["b"], c=c0["c"], quadratic=True)
sys3 = _raw_system3(p3_before, smooth3_before, tau, DT)
x0_3, P0_3 = initial_state3(log_f, tau, p3_before)
ll3, xf3, Pf3, xp3, Pp3, resid3, ll_cell3, n_obs3 = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0_3), np.ascontiguousarray(P0_3))
print(f"S3 cross-check: ll={ll3:.2f}  (saved {before['ll']:.2f})")

per_contract_gain_pinned = [float(np.nansum(ll_cell_pinned[:, j]) - np.nansum(ll_cell3[:, j])) for j in range(18)]
front4_gain_pinned = sum(per_contract_gain_pinned[:4])
back_gain_pinned = sum(per_contract_gain_pinned[4:])
total_gain_pinned = ll_pinned - ll3

print(f"\n{'=' * 90}\nPER-CONTRACT GAIN, KAPPA2/KAPPA3 PINNED AT BASELINE\n{'=' * 90}")
for j in range(18):
    print(f"C{j + 1:<3d} gain={per_contract_gain_pinned[j]:+10.2f}")
print(f"\nfront4 (C1-C4) gain: {front4_gain_pinned:+.2f}  ({100 * front4_gain_pinned / total_gain_pinned:.1f}%)")
print(f"back (C5-C18) gain:  {back_gain_pinned:+.2f}  ({100 * back_gain_pinned / total_gain_pinned:.1f}%)")
print(f"total gain: {total_gain_pinned:+.2f}  (unrestricted rung-2 total: {best['ll'] - before['ll']:+.2f})")

# %% [markdown]
# ## Verdict against the pre-stated falsifier

# %%
UNRESTRICTED_BACK_GAIN = 4928.76
ratio = back_gain_pinned / UNRESTRICTED_BACK_GAIN
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"pinned back-end (C5-C18) gain: {back_gain_pinned:.2f} nats  vs unrestricted: {UNRESTRICTED_BACK_GAIN:.2f} nats  "
     f"(ratio={ratio:.1%})")
if ratio < 0.20:
    print("Back-end gain collapses once kappa2/kappa3 are frozen at baseline -- CONFIRMS the "
         "reallocation mechanism: kappa2's movement (1.89->2.18) is what let the model help the "
         "back of the curve, not the 4th factor itself. Rung 2's back-end gain is generic "
         "curvature reallocation across the pre-existing factors, not a genuine front-factor effect.")
elif ratio > 0.80:
    print("Back-end gain survives essentially intact even with kappa2/kappa3 frozen -- the "
         "reallocation story is WRONG. The back-end gain is not explained by kappa2's movement; "
         "something else (likely still 'generic added flexibility from a 4th state', consistent "
         "with Gate 2's placebo result) is doing the work.")
else:
    print(f"Partial: back-end gain is reduced but not eliminated ({ratio:.1%} of the unrestricted "
         f"figure survives) -- kappa2's reallocation explains PART of the back-end gain, but not "
         f"all of it.")

# %%
out = {
    "kappa2_pinned": KAPPA2_BASELINE, "kappa3_pinned": KAPPA3_BASELINE,
    "ll": ll_pinned, "kappa_f_P_hat": p_hat.kappa_f_P, "sigma_f_hat": p_hat.sigma_f,
    "per_contract_gain": per_contract_gain_pinned,
    "front4_gain": front4_gain_pinned, "back_gain": back_gain_pinned, "total_gain": total_gain_pinned,
    "unrestricted_back_gain": UNRESTRICTED_BACK_GAIN, "ratio_to_unrestricted_back_gain": ratio,
    "n_em_iters": it + 1,
}
out_path = ROOT / "results" / "c1c18_fourfactor_front_kappa23_pinned.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
