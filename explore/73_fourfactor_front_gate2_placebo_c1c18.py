# %% [markdown]
# # Gate 2: the placebo -- does ANY fourth-factor shape buy a comparable gain?
#
# Real account: a fourth state changes the joint Kalman gain everywhere,
# so C5-C18 gain even with negligible direct loading is expected.
# Competing account: the fourth factor is just generic flexibility, and
# the front story is decoration. Same parameter count, same essentially-
# affine mechanism (kappa_placebo_Q FIXED, kappa_placebo_P FREE), same
# star correlation, same warm-start discipline -- the ONLY thing that
# changes is the loading shape: 1-e^{-k*tau} (back-loaded: ~0 at C1,
# rising toward the deep end) instead of e^{-k*tau} (front-loaded).
#
# A(tau)'s contribution from this factor is dropped -- there is no valid
# no-arbitrage closed form for E^Q[x_T] under an arbitrarily-chosen
# loading shape (only e^{-k*tau} is the actual conditional expectation of
# an OU state under Q; 1-e^{-k*tau} is a deliberately mismatched shape
# for exactly this diagnostic and was never meant to price anything).
# Implemented via A_tau4_front(p.replace(sigma_f=0, lambda_f=0), tau) --
# the SAME nesting property already tested (sigma_f=lambda_f=0 zeroes a
# factor's A(tau) contribution exactly) used here to zero out THIS
# factor's contribution while Z/T/Q keep its real, nonzero sigma_f/kappa.
#
# Falsifier, stated in advance: placebo gain within ~20% of the front
# version's 13,102 nats -> four factors fit better than three, full
# stop, and the front story is decoration.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.fourfactor_front import (  # noqa: E402
    A_tau4_front,
    FourFactorFrontParams,
    base_system4_front,
    initial_state4_front,
)
from csr.pricing.smoothmeas import SmoothMeasParams, hdiag_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv, filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
KAPPA_PLACEBO_Q = 1.5   # back-loaded shape parameter -- see module docstring

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_before = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth_before = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                                 c=before["curve"]["c"], quadratic=True)
sys3 = _raw_system3(p_before, smooth_before, tau, DT)
x0_3, P0_3 = initial_state3(log_f, tau, p_before)
ll_before, *_ = filter_uv(np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
                          np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
                          np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
                          np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
                          np.ascontiguousarray(x0_3), np.ascontiguousarray(P0_3))
print(f"S3 baseline: ll={ll_before:.2f}")

placebo_loading = 1.0 - np.exp(-KAPPA_PLACEBO_Q * tau[0])
front_loading = np.exp(-12.3 * tau[0])
print(f"placebo (back-loaded) profile at day0:  {np.round(placebo_loading, 3)}")
print(f"front-loaded profile at day0 (rung 2):  {np.round(front_loading, 3)}")

# %% [markdown]
# ## Build the placebo system: same T/Q/state, swapped Z, A(tau) minus this factor

# %%
def _raw_system4_placebo(p: FourFactorFrontParams, smooth: SmoothMeasParams, tau, dt: float):
    tau2 = np.atleast_2d(np.asarray(tau, float))
    sys = base_system4_front(p, tau2, dt)   # T, Q, c unchanged -- shape-independent
    sys["Z"][:, :, 0] = 1.0 - np.exp(-p.kappa_f_Q * tau2)   # swap: back-loaded, not front
    p_no_f_contribution = p.replace(sigma_f=0.0, lambda_f=0.0)
    d_without_f = A_tau4_front(p_no_f_contribution, tau2)   # exact zero-out, same nesting property
    sys["d"] = d_without_f
    sys["Hdiag"] = hdiag_from_tau(smooth, tau2)
    return sys


def _neg_loglik_placebo(theta_reduced, reduced_idx, theta_full_template, kappa_placebo_q):
    from csr.pricing.fit4_front import unpack4_front
    theta_full = theta_full_template.copy()
    theta_full[reduced_idx] = theta_reduced
    try:
        p, smooth = unpack4_front(theta_full, True, kappa_placebo_q)
    except ValueError:
        return 1e10
    sys = _raw_system4_placebo(p, smooth, tau, DT)
    x0, P0 = initial_state4_front(log_f, tau, p)
    n = sys["n"]
    ll, *_ = filter_uv(np.ascontiguousarray(log_f), np.ascontiguousarray(sys["Z"]),
                       np.ascontiguousarray(sys["d"]), np.ascontiguousarray(sys["T"]),
                       np.ascontiguousarray(sys["c"]), np.ascontiguousarray(sys["Q"]),
                       np.ascontiguousarray(sys["Hdiag"]), np.zeros((n, 4)),
                       np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return -ll if np.isfinite(ll) else 1e10


# %% [markdown]
# ## Fit the placebo, same warm start and multi-start discipline

# %%
from csr.pricing.fit4_front import pack4_front  # noqa: E402

c0 = before["curve"]
STARTS = [KAPPA_PLACEBO_Q, 3.0, 1.0, 0.3, 0.03]
results = []
for k_p_start in STARTS:
    p_start = FourFactorFrontParams(
        kappa2=before["kappa1"], kappa3=before["kappa2"],
        sigma_f=0.05, sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
        kappa_f_Q=KAPPA_PLACEBO_Q, kappa_f_P=k_p_start,
        rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
        mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
        lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
    )
    theta0 = pack4_front(p_start, c0["a"], c0["b"], c0["c"], quadratic=True)
    res = minimize(_neg_loglik_placebo, theta0,
                   args=(np.arange(len(theta0)), theta0, KAPPA_PLACEBO_Q),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    ll = -res.fun
    from csr.pricing.fit4_front import unpack4_front
    p_hat, smooth_hat = unpack4_front(res.x, True, KAPPA_PLACEBO_Q)
    results.append({"start": k_p_start, "ll": ll, "kappa_placebo_P_hat": p_hat.kappa_f_P,
                    "sigma_placebo_hat": p_hat.sigma_f})
    print(f"start={k_p_start:6.3f}  ->  ll={ll:.2f}  kappa_placebo_P_hat={p_hat.kappa_f_P:.4f}  "
         f"sigma_placebo_hat={p_hat.sigma_f:.4f}")

# %% [markdown]
# ## Verdict

# %%
best_placebo = max(results, key=lambda r: r["ll"])
front_gain = 275079.68 - ll_before   # rung 2's own gain, from explore/70/71
placebo_gain = best_placebo["ll"] - ll_before

print(f"\n{'=' * 90}\nGATE 2 VERDICT\n{'=' * 90}")
print(f"front-loaded (rung 2) gain: {front_gain:.2f} nats")
print(f"back-loaded (placebo) gain: {placebo_gain:.2f} nats")
print(f"placebo / front ratio: {placebo_gain / front_gain:.1%}")

if placebo_gain > 0.8 * front_gain:
    print("\nFAILS: the placebo delivers a comparable gain -- four factors fit better than "
         "three, regardless of shape. The front story is decoration; rung 2 does not survive.")
else:
    print("\nPASSES: the placebo gain is materially smaller -- the front-concentrated shape is "
         "doing real, shape-specific work, not just adding a fourth degree of freedom.")

# %%
out = {"kappa_placebo_Q": KAPPA_PLACEBO_Q, "starts": results, "front_gain": front_gain,
      "placebo_gain": placebo_gain, "ratio": placebo_gain / front_gain}
out_path = ROOT / "results" / "c1c18_fourfactor_front_gate2_placebo.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
