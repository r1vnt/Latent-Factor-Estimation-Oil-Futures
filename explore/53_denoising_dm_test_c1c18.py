# %% [markdown]
# # The direct test of the denoising claim: filter vs. random walk, full panel
#
# Study A compared two near-identical sign objects (L_obs, a smoothed
# version of L_obs) on 390 disagreement days and found nothing informative
# -- an arithmetic property of the comparison, not a finding about WTI.
# This is the test that was always more fundamental: the framework's
# premise (observed = state + noise) makes a sharp, label-free,
# horizon-free prediction on the FULL panel (~55k obs, not 390):
#
#   The filter's one-step-ahead prediction error v_t should have SMALLER
#   MSE than the random walk's error (observed_t - observed_{t-1}),
#   because today's observed price carries measurement noise that does
#   not persist into tomorrow's reference point the way it does for a
#   naive random-walk forecast built directly off observed prices.
#
# Quantitative prediction, not just directional: MSE(random walk) -
# MSE(filter) should be approximately H_j, the fitted per-contract
# measurement variance -- since the random-walk error's noise component
# is Var(noise_t - noise_{t-1}) = 2H (independent noise draws
# differenced) while the filter's one-step error's noise component is
# just H (today's fresh draw only), and the STATE-uncertainty component
# of each is comparable in steady state. With s1 (front contract) fitted
# at ~172.8bp -- 82.9% of the front spread's SD (explore/43) -- that
# predicted gap is large and specific, not a rounding-level effect.
#
# Primary test: the plain price-only three-factor state (results/
# c1c18_threefactor.json, no RV/front-spread channels) -- the object the
# 82.9% figure and the framework's premise both refer to directly.
# Secondary (bonus, since the arrays are already in hand): the front-
# spread + joint-nu state, to see whether the "materially better state
# estimate" (explore/48-51) is also a materially better one-step
# forecaster, not just a better-calibrated one.
#
# DM test: HAC-adjusted t-test of the loss differential's mean, via the
# standard "regress d_t on a constant, HAC SE" trick already used
# throughout this session's regressions (explore/43/47/48) -- the
# constant's t-stat IS the DM statistic. maxlags=20, matching the
# longest lag already tested for squared-innovation autocorrelation
# (Ljung-Box has confirmed it's real and persistent, so an HAC estimator
# with only 1-2 lags would understate the loss differential's true
# variance).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))
HAC_LAGS = 20
MIN_HISTORY = 60

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n


def _filter_with_contributions(p, smooth, rv_lag_arg, h_driver_arg):
    sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag_arg, h_driver=h_driver_arg)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell, sys3["Hdiag"]


def _raw_innovation(resid, ll_cell):
    logF = -2.0 * ll_cell - LOG_2PI - resid ** 2
    F = np.exp(logF)
    return resid * np.sqrt(F)


def _dm_test(e_filter, e_rw):
    """d_t = e_filter^2 - e_rw^2, per contract. Negative mean d with a
    significant HAC t-stat means the filter's squared error is smaller --
    i.e. the filter beats the random walk."""
    rows = []
    for j in range(e_filter.shape[1]):
        d = e_filter[:, j] ** 2 - e_rw[:, j] ** 2
        d = d[np.isfinite(d)]
        X = np.ones((len(d), 1))
        res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        rows.append({"contract": j + 1, "n": len(d), "mean_d": float(res.params[0]),
                    "dm_stat": float(res.tvalues[0]), "p_value": float(res.pvalues[0])})
    return rows


def run_test(label, p, smooth, rv_lag_arg, h_driver_arg):
    ll, resid, ll_cell, Hdiag = _filter_with_contributions(p, smooth, rv_lag_arg, h_driver_arg)
    v = _raw_innovation(resid, ll_cell)

    rw_error = np.full_like(log_f, np.nan)
    rw_error[1:] = np.diff(log_f, axis=0)
    v_aligned = v.copy()
    v_aligned[0, :] = np.nan   # drop day 0 from both -- no "yesterday" for the RW, diffuse P0 for the filter

    mean_H = np.nanmean(Hdiag, axis=0)
    dm_rows = _dm_test(v_aligned, rw_error)

    print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
    print(f"{'':4s}{'MSE(filter)':>14s}{'MSE(rw)':>14s}{'gap':>12s}{'fitted H':>12s}"
         f"{'gap/H':>8s}{'DM stat':>10s}{'p-value':>10s}")
    results = []
    for j in range(18):
        mse_f = float(np.nanmean(v_aligned[:, j] ** 2))
        mse_rw = float(np.nanmean(rw_error[:, j] ** 2))
        gap = mse_rw - mse_f
        h_j = float(mean_H[j])
        ratio = gap / h_j if h_j > 0 else float("nan")
        dm = dm_rows[j]
        results.append({"contract": j + 1, "mse_filter": mse_f, "mse_rw": mse_rw, "gap": gap,
                        "fitted_H": h_j, "gap_over_H": ratio, "dm_stat": dm["dm_stat"],
                        "p_value": dm["p_value"], "n": dm["n"]})
        flag = "FILTER WINS" if (gap > 0 and dm["p_value"] < 0.01) else (
               "RW WINS" if (gap < 0 and dm["p_value"] < 0.01) else "no sig. difference")
        print(f"C{j + 1:2d} {mse_f * 1e6:14.4f}{mse_rw * 1e6:14.4f}{gap * 1e6:12.4f}"
             f"{h_j * 1e6:12.4f}{ratio:8.3f}{dm['dm_stat']:10.3f}{dm['p_value']:10.4g}  {flag}")
    print("(MSE/gap/H columns in units of 1e-6 log-price^2, i.e. roughly bp^2/100)")

    n_filter_wins = sum(1 for r in results if r["gap"] > 0 and r["p_value"] < 0.01)
    n_rw_wins = sum(1 for r in results if r["gap"] < 0 and r["p_value"] < 0.01)
    ratios = [r["gap_over_H"] for r in results if np.isfinite(r["gap_over_H"])]
    print(f"\nfilter wins (p<0.01): {n_filter_wins}/18   rw wins (p<0.01): {n_rw_wins}/18   "
         f"mean gap/H ratio: {np.mean(ratios):.3f}")
    return results


# %% [markdown]
# ## Primary: the plain price-only three-factor state (no RV/front-spread channels)

# %%
before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_plain = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth_plain = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                                c=before["curve"]["c"], quadratic=True)
results_plain = run_test("PRIMARY: plain price-only three-factor state", p_plain, smooth_plain, None, None)

# %% [markdown]
# ## Secondary (bonus): the front-spread + joint-nu state

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

rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")
fit = json.loads((ROOT / "results" / "c1c18_joint_nu_estimation.json").read_text())
fp = fit["params"]
p_fs = ThreeFactorParams(
    kappa1=fit["kappa1"], kappa2=fit["kappa2"], sigma1=fp["sigma1"], sigma2=fp["sigma2"],
    sigma_xi=fp["sigma_xi"], rho12=0.0, rho1x=fp["rho1x"], rho2x=fp["rho2x"],
    mu_xi=fp["mu_xi"], mu_xi_star=fp["mu_xi_star"], lambda1=fp["lambda1"], lambda2=fp["lambda2"],
    gamma_v=fit["gamma_v"],
)
smooth_fs = SmoothMeasParams(a=fp["curve"]["a"], b=fp["curve"]["b"], c=fp["curve"]["c"], quadratic=True,
                             eta_v=fit["eta_v"], lambda_v=fp["curve"]["lambda_v"])
results_fs = run_test("SECONDARY: front-spread + joint-nu state", p_fs, smooth_fs, rv_lag, fs_driver)

# %%
out = {"primary_plain_state": results_plain, "secondary_frontspread_state": results_fs}
out_path = ROOT / "results" / "c1c18_denoising_dm_test.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
