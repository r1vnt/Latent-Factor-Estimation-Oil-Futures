# %% [markdown]
# # Letting nu move: does the front-spread channel's tail-thinning hold up?
#
# explore/48-50 found the (bug-fixed) front-spread H channel collapses
# pooled excess kurtosis from 34.3 to 1.86 (near-Gaussian) and pulls mean
# resid_std^2 to ~1.10 -- far below the df=5 benchmark of 1.667, but close
# to the df=inf (Gaussian) benchmark of 1.0. That pattern was inferred
# from kurtosis, not estimated directly. fit3.py now supports estimate_df
# (CM-step 2 of the ECM for the t distribution -- see its docstring):
# warm-start from explore/48's fixed-nu=5 optimum and let nu float.
#
# Fix 5's own criterion: if nu rises well above 5, the channel is doing
# genuine tail-thinning work, not just inflating variance. If nu stays at
# or below 5, it isn't.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3, fit3_student_t_smooth, pack3_smooth  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))
LAGS = [1, 5, 10, 20]
MIN_HISTORY = 60

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")

# %% [markdown]
# ## Rebuild the (corrected) front-spread driver, warm-start from explore/48

# %%
spread = log_f[:, 0] - log_f[:, 1]
abs_spread_lag = np.concatenate([[np.nan], np.abs(spread[:-1])])
fs_driver = np.zeros(n)
running_sum, running_sumsq, count = 0.0, 0.0, 0
last_valid = 0.0
for t in range(n):
    if count >= MIN_HISTORY:
        mean = running_sum / count
        var = running_sumsq / count - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        if np.isfinite(abs_spread_lag[t]):
            last_valid = (abs_spread_lag[t] - mean) / std
        fs_driver[t] = last_valid
    if np.isfinite(abs_spread_lag[t]):
        running_sum += abs_spread_lag[t]
        running_sumsq += abs_spread_lag[t] ** 2
        count += 1

fc = json.loads((ROOT / "results" / "c1c18_frontspread_channel.json").read_text())["student_t_em"]
fcp = fc["params"]
p_warm = ThreeFactorParams(
    kappa1=fc["kappa1"], kappa2=fc["kappa2"], sigma1=fcp["sigma1"], sigma2=fcp["sigma2"],
    sigma_xi=fcp["sigma_xi"], rho12=0.0, rho1x=fcp["rho1x"], rho2x=fcp["rho2x"],
    mu_xi=fcp["mu_xi"], mu_xi_star=fcp["mu_xi_star"], lambda1=fcp["lambda1"], lambda2=fcp["lambda2"],
    gamma_v=fc["gamma_v"],
)
smooth_warm = SmoothMeasParams(a=fcp["curve"]["a"], b=fcp["curve"]["b"], c=fcp["curve"]["c"],
                               quadratic=True, eta_v=fc["eta_v"], lambda_v=fcp["curve"]["lambda_v"])
print(f"warm start (fixed nu=5): ll={fc['ll']:.2f}  k1={p_warm.kappa1:.4f}  k2={p_warm.kappa2:.4f}  "
     f"eta_v={fc['eta_v']:.4f}  gamma_v={fc['gamma_v']:.4f}")

# %% [markdown]
# ## Let nu float (CM-step 2 active), starting from nu=5

# %%
theta0 = pack3_smooth(p_warm, smooth_warm.a, smooth_warm.b, smooth_warm.c, quadratic=True,
                      eta_v=smooth_warm.eta_v, lambda_v=smooth_warm.lambda_v, rv_extension=True)
DF_BOUNDS = (2.1, 100.0)
p_hat, smooth_hat, x_filt, resid_final, weight, n_iters, ll_hat, nu_hat = fit3_student_t_smooth(
    log_f, tau, DT, 5.0, theta0, quadratic=True, rv_lag=rv_lag, rv_extension=True,
    h_driver=fs_driver, estimate_df=True, df_bounds=DF_BOUNDS)

binding = "AT THE BOUND -- data wants even more, 100 is not the answer" if nu_hat > DF_BOUNDS[1] * 0.95 else "interior"
print(f"\n{'=' * 90}\nJOINT NU ESTIMATE\n{'=' * 90}")
print(f"ll={ll_hat:.2f}  (fixed-nu=5 warm start: {fc['ll']:.2f})  n_em_iters={n_iters}")
print(f"nu_hat = {nu_hat:.3f}   ({binding})")
print(f"k1={p_hat.kappa1:.4f}  k2={p_hat.kappa2:.4f}  eta_v={smooth_hat.eta_v:.4f}  gamma_v={p_hat.gamma_v:.4f}")

correct_benchmark = nu_hat / (nu_hat - 2.0) if nu_hat > 2.0 else float("inf")
print(f"correct calibration benchmark at nu_hat: {correct_benchmark:.4f}  "
     f"(df=5 benchmark was 1.667; Gaussian benchmark is 1.0)")

if nu_hat > 5.0 + 1.0:
    print("\nnu rose meaningfully above 5 -- the front-spread channel is doing genuine "
         "tail-thinning work, confirming the kurtosis-based inference directly.")
elif nu_hat <= 5.0:
    print("\nnu did NOT rise above 5 even when free -- the earlier kurtosis-based "
         "inference was wrong; the channel is not doing genuine tail-thinning work "
         "in the sense Fix 5 meant.")
else:
    print("\nnu moved up only marginally -- a weak, not a clean, confirmation.")

# %% [markdown]
# ## Recompute calibration and Ljung-Box under the correctly-specified nu

# %%
sys3 = _raw_system3(p_hat, smooth_hat, tau, DT, rv_lag=rv_lag, h_driver=fs_driver)
x0, P0 = initial_state3(log_f, tau, p_hat)
ll_check, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
print(f"\ncross-check: ll={ll_check:.2f} (EM return {ll_hat:.2f})")

mean_resid2 = float(np.nanmean(resid ** 2))
excess_kurt = float(_kurtosis(resid[np.isfinite(resid)].ravel(), fisher=True, bias=False))
print(f"\n{'=' * 90}\nCALIBRATION UNDER nu_hat={nu_hat:.3f}\n{'=' * 90}")
print(f"mean resid_std^2: {mean_resid2:.4f}  (correct benchmark: {correct_benchmark:.4f}, "
     f"vs. Gaussian's 1.0)")
print(f"excess kurtosis (pooled): {excess_kurt:.4f}")

print(f"\n{'=' * 90}\nLJUNG-BOX ON SQUARED INNOVATIONS, nu_hat={nu_hat:.3f}\n{'=' * 90}")
sig_by_lag = {}
for lag in LAGS:
    sig = []
    for j in range(18):
        z = resid[:, j]
        z = z[np.isfinite(z)]
        lb = acorr_ljungbox(z ** 2, lags=[lag], return_df=True)
        if lb.loc[lag, "lb_pvalue"] < 0.01:
            sig.append(j + 1)
    sig_by_lag[lag] = sig
    print(f"lag={lag:3d}: {len(sig):2d}/18 significant   {sig}")

# %%
out = {
    "nu_hat": nu_hat, "nu_binding": nu_hat > DF_BOUNDS[1] * 0.95, "ll": ll_hat, "n_em_iters": n_iters,
    "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2, "eta_v": smooth_hat.eta_v, "gamma_v": p_hat.gamma_v,
    "correct_benchmark": correct_benchmark, "mean_resid2": mean_resid2, "excess_kurtosis": excess_kurt,
    "ljung_box_squared_significant_by_lag": {str(k): v for k, v in sig_by_lag.items()},
    # Full params, saved this time (explore/45's first pass omitted these and
    # had to be rerun to get them) -- needed to warm-start or reuse this
    # exact fit (e.g. Study A re-run) without refitting.
    "params": {
        "sigma1": p_hat.sigma1, "sigma2": p_hat.sigma2, "sigma_xi": p_hat.sigma_xi,
        "rho1x": p_hat.rho1x, "rho2x": p_hat.rho2x, "mu_xi": p_hat.mu_xi,
        "mu_xi_star": p_hat.mu_xi_star, "lambda1": p_hat.lambda1, "lambda2": p_hat.lambda2,
        "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c,
                 "lambda_v": smooth_hat.lambda_v},
    },
}
out_path = ROOT / "results" / "c1c18_joint_nu_estimation.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
