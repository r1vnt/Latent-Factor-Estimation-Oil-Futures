# %% [markdown]
# # Is explore/53's "forecast" order-dependent? And the honest version.
#
# state/fast.py's own docstring (filter_uv_contributions, lines 169-175)
# already says this, in so many words: within one time step, sequential
# univariate updating means contract i>0's v and F are computed against
# the state AFTER contracts 0..i-1's same-day updates, not against the
# raw one-step-ahead prediction. explore/53 used `resid` (equivalently,
# v = resid*sqrt(F)) as if it were a uniform one-step-ahead forecast
# error for every contract -- correct for contract 0 (C1, absorbed
# first, nothing conditions it), increasingly NOT correct for later
# positions, which partly measure same-day cross-sectional interpolation
# given earlier contracts already absorbed that day.
#
# Two checks:
#   1. Reverse the column order (C18..C1 instead of C1..C18) and refilter
#      (no refit). Total log-likelihood must match to 1e-10 (sequential
#      updating is exact regardless of order -- this is a pure
#      re-ordering, not a new model). If the MSE-by-contract pattern
#      flips with it, explore/53's pattern is order, not economics.
#   2. The honest one-step-ahead error, shared across all contracts on
#      a date, using x_pred (the state predicted through T,Q from the
#      PRIOR day's end, before ANY of today's observations are seen):
#      e[t,i] = y[t,i] - (Z[t,i,:] . x_pred[t,:] + d[t,i]).
#      filter_uv_contributions already returns x_pred -- no new fit.

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
HAC_LAGS = 20

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                          c=before["curve"]["c"], quadratic=True)


def _filter(log_f_arg, tau_arg):
    sys3 = _raw_system3(p, smooth, tau_arg, DT)
    x0, P0 = initial_state3(log_f_arg, tau_arg, p)
    return filter_uv_contributions(
        np.ascontiguousarray(log_f_arg), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0)), sys3


# %% [markdown]
# ## Check 1: reverse processing order

# %%
(ll_fwd, xf_fwd, Pf_fwd, xp_fwd, Pp_fwd, resid_fwd, ll_cell_fwd, nobs_fwd), sys3_fwd = _filter(log_f, tau)

log_f_rev = log_f[:, ::-1].copy()
tau_rev = tau[:, ::-1].copy()
(ll_rev, xf_rev, Pf_rev, xp_rev, Pp_rev, resid_rev, ll_cell_rev, nobs_rev), sys3_rev = _filter(log_f_rev, tau_rev)

print(f"{'=' * 90}\nCHECK 1: ORDER REVERSAL\n{'=' * 90}")
print(f"ll forward = {ll_fwd:.6f}")
print(f"ll reversed = {ll_rev:.6f}")
print(f"|difference| = {abs(ll_fwd - ll_rev):.2e}  (must be ~1e-10 if this is a pure re-ordering)")

print(f"\n{'contract':>10s}{'std-resid^2 fwd':>18s}{'std-resid^2 rev':>18s}{'ratio':>10s}")
for j in range(18):
    mse_fwd_j = float(np.nanmean(resid_fwd[:, j] ** 2))
    # contract j sits at reversed-array position (17 - j) after [:, ::-1]
    mse_rev_j = float(np.nanmean(resid_rev[:, 17 - j] ** 2))
    ratio = mse_rev_j / mse_fwd_j if mse_fwd_j > 0 else float("nan")
    print(f"C{j + 1:<9d}{mse_fwd_j:18.6f}{mse_rev_j:18.6f}{ratio:10.3f}")
print("(standardised resid -- mixes raw error with the model's own order-dependent F "
     "claim, so not a clean test by itself; see the RAW v^2 version below.)")

# Raw (unstandardised) v^2 -- isolates actual prediction-error magnitude from
# the model's own shifting uncertainty claim, which the standardised version
# above conflates. v = resid * sqrt(F), F recovered from ll_cell exactly as
# throughout this session (explore/30 etc).
LOG_2PI = float(np.log(2.0 * np.pi))
F_fwd = np.exp(-2.0 * ll_cell_fwd - LOG_2PI - resid_fwd ** 2)
F_rev = np.exp(-2.0 * ll_cell_rev - LOG_2PI - resid_rev ** 2)
v_fwd = resid_fwd * np.sqrt(F_fwd)
v_rev = resid_rev * np.sqrt(F_rev)

print(f"\n{'contract':>10s}{'raw v^2 fwd':>18s}{'raw v^2 rev':>18s}{'ratio':>10s}")
for j in range(18):
    mse_fwd_j = float(np.nanmean(v_fwd[:, j] ** 2))
    mse_rev_j = float(np.nanmean(v_rev[:, 17 - j] ** 2))
    ratio = mse_rev_j / mse_fwd_j if mse_fwd_j > 0 else float("nan")
    print(f"C{j + 1:<9d}{mse_fwd_j * 1e6:18.4f}{mse_rev_j * 1e6:18.4f}{ratio:10.3f}")
print("(units of 1e-6 log-price^2; this is the apples-to-apples version of explore/53's metric)")

flips = (float(np.nanmean(v_fwd[:, 0] ** 2)) > float(np.nanmean(v_rev[:, 17] ** 2))) and \
       (float(np.nanmean(v_fwd[:, 17] ** 2)) < float(np.nanmean(v_rev[:, 0] ** 2)))
print(f"\nC1's raw v^2 shrinks when processed last, C18's raw v^2 grows when processed first: {flips}")

# %% [markdown]
# ## Check 2: the honest, shared-x_pred one-step-ahead forecast error

# %%
Z_fwd, d_fwd = sys3_fwd["Z"], sys3_fwd["d"]
pred = np.einsum('tia,ta->ti', Z_fwd, xp_fwd) + d_fwd
e_honest = log_f - pred
e_honest[0, :] = np.nan   # x_pred[0] is one transition off the diffuse initial prior, not a real "yesterday"

rw_error = np.full_like(log_f, np.nan)
rw_error[1:] = np.diff(log_f, axis=0)

Hdiag_fwd = sys3_fwd["Hdiag"]
mean_H = np.nanmean(Hdiag_fwd, axis=0)

print(f"\n{'=' * 100}\nCHECK 2: HONEST ONE-STEP-AHEAD FORECAST (shared x_pred, all contracts)\n{'=' * 100}")
print(f"{'':4s}{'MSE(filter)':>14s}{'MSE(rw)':>14s}{'gap':>12s}{'fitted H':>12s}"
     f"{'gap/H':>8s}{'DM stat':>10s}{'p-value':>10s}")
results = []
for j in range(18):
    d_loss = e_honest[:, j] ** 2 - rw_error[:, j] ** 2
    d_loss = d_loss[np.isfinite(d_loss)]
    X = np.ones((len(d_loss), 1))
    res = sm.OLS(d_loss, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    mse_f = float(np.nanmean(e_honest[:, j] ** 2))
    mse_rw = float(np.nanmean(rw_error[:, j] ** 2))
    gap = mse_rw - mse_f
    h_j = float(mean_H[j])
    ratio = gap / h_j if h_j > 0 else float("nan")
    dm_stat, p_value = float(res.tvalues[0]), float(res.pvalues[0])
    results.append({"contract": j + 1, "mse_filter": mse_f, "mse_rw": mse_rw, "gap": gap,
                    "fitted_H": h_j, "gap_over_H": ratio, "dm_stat": dm_stat, "p_value": p_value,
                    "n": int(res.nobs)})
    flag = "FILTER WINS" if (gap > 0 and p_value < 0.01) else (
           "RW WINS" if (gap < 0 and p_value < 0.01) else "no sig. difference")
    print(f"C{j + 1:2d} {mse_f * 1e6:14.4f}{mse_rw * 1e6:14.4f}{gap * 1e6:12.4f}"
         f"{h_j * 1e6:12.4f}{ratio:8.3f}{dm_stat:10.3f}{p_value:10.4g}  {flag}")
print("(MSE/gap/H columns in units of 1e-6 log-price^2)")

n_filter_wins = sum(1 for r in results if r["gap"] > 0 and r["p_value"] < 0.01)
n_rw_wins = sum(1 for r in results if r["gap"] < 0 and r["p_value"] < 0.01)
ratios = [r["gap_over_H"] for r in results if np.isfinite(r["gap_over_H"])]
print(f"\nfilter wins (p<0.01): {n_filter_wins}/18   rw wins (p<0.01): {n_rw_wins}/18   "
     f"mean gap/H ratio: {np.mean(ratios):.3f}")
print(f"gap/H by contract: {[round(r['gap_over_H'], 2) for r in results]}")
print("expectation if the honest test vindicates the denoising claim: gap/H near 1, "
     "advantage concentrated at the front (largest H), not the back.")

# %%
out = {
    "order_reversal": {
        "ll_forward": ll_fwd, "ll_reversed": ll_rev, "ll_abs_diff": abs(ll_fwd - ll_rev),
        "pattern_flips_raw_v2": bool(flips),
        "raw_v2_fwd": [float(np.nanmean(v_fwd[:, j] ** 2)) for j in range(18)],
        "raw_v2_rev_same_contract": [float(np.nanmean(v_rev[:, 17 - j] ** 2)) for j in range(18)],
    },
    "honest_forecast": results,
}
out_path = ROOT / "results" / "c1c18_order_reversal_and_honest_forecast.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
