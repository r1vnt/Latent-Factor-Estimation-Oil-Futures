# %% [markdown]
# # Simulation-based calibration benchmark: is the target 1.0 or 1.667?
#
# Settling the bookkeeping question empirically rather than by reasoning
# about it: simulate from each model's own fitted parameters -- genuine
# Student-t(5) observation noise via numpy's own standard_t (which has the
# correct built-in variance df/(df-2) = 1.667, no manual Gamma-mixture
# construction needed), Gaussian process noise via T/c/Q -- then filter
# that synthetic panel with the SAME fitted parameters at weight=1, the
# same procedure used on the real data throughout this investigation. The
# model is correctly specified by construction on its own simulated data,
# so whatever E[resid_std^2] comes out there is the real benchmark,
# whichever bookkeeping convention is actually in force.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.pricing.twofactor import TwoFactorParams, base_system, initial_state  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
STUDENT_T_DF = 5.0
LOG_2PI = float(np.log(2.0 * np.pi))
N_REPS = 5
rng = np.random.default_rng(20260907)

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
n, m = panel.n, 18
tau = panel.tau


def simulate(T, c, Q, Z, d, Hdiag, k, df, rng):
    x = np.zeros(k)
    log_f = np.empty((n, m))
    for t in range(n):
        c_t = c if c.ndim == 1 else c[t]
        Q_t = Q if Q.ndim == 2 else Q[t]
        w = rng.multivariate_normal(np.zeros(k), Q_t)
        x = T @ x + c_t + w
        log_f[t] = d[t] + Z[t] @ x + rng.standard_t(df, size=m) * np.sqrt(Hdiag[t])
    return log_f


def filter_resid_sq(log_f_sim, T, c, Q, Z, d, Hdiag, x0, P0, k):
    Bu = np.zeros((n, k))
    c_full = c if c.ndim == 2 else np.tile(c, (n, 1))
    Q_full = Q if Q.ndim == 3 else np.tile(Q, (n, 1, 1))
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f_sim), np.ascontiguousarray(Z),
        np.ascontiguousarray(d), np.ascontiguousarray(T),
        np.ascontiguousarray(c_full), np.ascontiguousarray(Q_full),
        np.ascontiguousarray(Hdiag), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return float(np.nanmean(resid ** 2))


# %% [markdown]
# ## Two-factor

# %%
two_factor = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s2, c2 = two_factor["structural"], two_factor["curve"]
p2 = TwoFactorParams(
    kappa=s2["kappa"], sigma_chi=s2["sigma_chi"], sigma_xi=s2["sigma_xi"],
    rho=s2["rho"], mu_xi=s2["mu_xi"], mu_xi_star=s2["mu_xi_star"],
    lambda_chi=s2["lambda_chi"],
    sigma_meas=np.exp(c2["a"] + c2["b"] * tau + c2["c"] * tau ** 2),
)
sys2 = base_system(p2, tau, DT)  # raw 2-dim, price-only -- augmentation is inert for this
x0_2, P0_2 = initial_state(panel.log_f, tau, p2)

reps2 = []
for rep in range(N_REPS):
    log_f_sim = simulate(sys2["T"], sys2["c"], sys2["Q"], sys2["Z"], sys2["d"],
                         sys2["Hdiag"], 2, STUDENT_T_DF, rng)
    reps2.append(filter_resid_sq(log_f_sim, sys2["T"], sys2["c"], sys2["Q"],
                                 sys2["Z"], sys2["d"], sys2["Hdiag"], x0_2, P0_2, 2))
print(f"two-factor simulated E[resid_std^2] over {N_REPS} reps: "
     f"{np.mean(reps2):.4f}  (range [{min(reps2):.4f}, {max(reps2):.4f}])")

# %% [markdown]
# ## Three-factor

# %%
three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p3 = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
)
smooth3 = SmoothMeasParams(a=three_factor["curve"]["a"], b=three_factor["curve"]["b"],
                           c=three_factor["curve"]["c"], quadratic=True)
sys3 = _raw_system3(p3, smooth3, tau, DT)
x0_3, P0_3 = initial_state3(panel.log_f, tau, p3)

reps3 = []
for rep in range(N_REPS):
    log_f_sim = simulate(sys3["T"], sys3["c"], sys3["Q"], sys3["Z"], sys3["d"],
                         sys3["Hdiag"], 3, STUDENT_T_DF, rng)
    reps3.append(filter_resid_sq(log_f_sim, sys3["T"], sys3["c"], sys3["Q"],
                                 sys3["Z"], sys3["d"], sys3["Hdiag"], x0_3, P0_3, 3))
print(f"three-factor simulated E[resid_std^2] over {N_REPS} reps: "
     f"{np.mean(reps3):.4f}  (range [{min(reps3):.4f}, {max(reps3):.4f}])")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"theoretical t5 target if weights are NOT folded into Hdiag: {5 / 3:.4f}")
print(f"theoretical Gaussian target if weights ARE folded in: 1.0000")
print(f"two-factor:   simulated benchmark={np.mean(reps2):.4f}   real data resid_std^2=1.2491")
print(f"three-factor: simulated benchmark={np.mean(reps3):.4f}   real data resid_std^2=1.3761")

bench = (np.mean(reps2) + np.mean(reps3)) / 2
print(f"\nusing simulated benchmark ~{bench:.3f}:")
print(f"  two-factor:   {1.2491:.4f} vs {bench:.4f}  -> "
     f"{'under-dispersed' if 1.2491 < bench else 'over-dispersed'} by "
     f"{100 * abs(1.2491 - bench) / bench:.1f}%")
print(f"  three-factor: {1.3761:.4f} vs {bench:.4f}  -> "
     f"{'under-dispersed' if 1.3761 < bench else 'over-dispersed'} by "
     f"{100 * abs(1.3761 - bench) / bench:.1f}%")
if abs(1.3761 - bench) < abs(1.2491 - bench):
    print("\nTHREE-FACTOR IS BETTER CALIBRATED (closer to the correctly-specified "
         "benchmark) -- the earlier 'overconfidence' reading was wrong; both "
         "models are under-dispersed relative to a genuine t5, and three-factor "
         "moves toward correct calibration, not away from it.")
else:
    print("\nTWO-FACTOR IS BETTER CALIBRATED -- the overconfidence reading stands.")

# %%
out = {"two_factor_sim_benchmark": reps2, "three_factor_sim_benchmark": reps3,
      "real_resid_sq": {"two_factor": 1.2491, "three_factor": 1.3761}}
(ROOT / "results" / "c1c18_calibration_benchmark_sim.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_calibration_benchmark_sim.json")
