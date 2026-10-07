# %% [markdown]
# # C1-C4 H3a robustness check: is the front-end RMSE gap outlier-driven?
#
# Pre-registered in
# preregistrations/2026-10-07_c1_robustness_reporting_rule.md, written
# before this ran. No fitting: reuses the exact fitted S3 params
# (results/c1c18_threefactor.json) and the exact leave-one-out
# construction of explore/65 (filter_uv on contract i NaN'd, Nelson-Siegel
# refit on the other 17 per day), restricted to C1-C4, to recover the
# per-day error arrays that explore/65 computed but did not save.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.curve.expiry import nearby_expiries  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
TARGET_CONTRACTS = [0, 1, 2, 3]  # C1-C4, zero-indexed

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau, dates, n = panel.log_f, panel.tau, panel.dates, panel.n

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

sys3_full = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)

# %% [markdown]
# ## Filter side: leave-one-out, C1-C4 only

# %%
filter_pred = np.full((n, 18), np.nan)
for i in TARGET_CONTRACTS:
    log_f_loo = log_f.copy()
    log_f_loo[:, i] = np.nan
    ll, x_filt, P_filt, x_pred, P_pred, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f_loo), np.ascontiguousarray(sys3_full["Z"]),
        np.ascontiguousarray(sys3_full["d"]), np.ascontiguousarray(sys3_full["T"]),
        np.ascontiguousarray(sys3_full["c"]), np.ascontiguousarray(sys3_full["Q"]),
        np.ascontiguousarray(sys3_full["Hdiag"]), np.zeros((n, 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    filter_pred[:, i] = np.einsum("ta,ta->t", sys3_full["Z"][:, i, :], x_filt) + sys3_full["d"][:, i]
    print(f"  contract {i + 1:2d} left out: filter ll={ll:.2f}  n_obs={n_obs}")

# %% [markdown]
# ## Curve side: same per-day Nelson-Siegel refit as explore/65, C1-C4 only

# %%
def _ns_basis(t, lam):
    x = t / lam
    x = np.where(x <= 1e-8, 1e-8, x)
    slope = (1.0 - np.exp(-x)) / x
    curve = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curve])


LAMBDA_GRID = np.linspace(0.05, 3.0, 30)
curve_pred = np.full((n, 18), np.nan)

for t in range(n):
    tau_t = tau[t, :]
    y_t = log_f[t, :]
    valid_t = np.isfinite(y_t)
    if valid_t.sum() < 6:
        continue

    best_lam, best_sse = None, np.inf
    for lam in LAMBDA_GRID:
        X = _ns_basis(tau_t[valid_t], lam)
        beta, *_ = np.linalg.lstsq(X, y_t[valid_t], rcond=None)
        sse = float(np.sum((y_t[valid_t] - X @ beta) ** 2))
        if sse < best_sse:
            best_sse, best_lam = sse, lam

    for i in TARGET_CONTRACTS:
        if not valid_t[i]:
            continue
        others = valid_t.copy()
        others[i] = False
        if others.sum() < 5:
            continue
        X_others = _ns_basis(tau_t[others], best_lam)
        beta_i, *_ = np.linalg.lstsq(X_others, y_t[others], rcond=None)
        x_i = _ns_basis(tau_t[i:i + 1], best_lam)
        curve_pred[t, i] = float((x_i @ beta_i)[0])

print(f"curve-side fit complete: {np.isfinite(curve_pred[:, TARGET_CONTRACTS]).sum():,} predictions")

# %% [markdown]
# ## Cross-check against the already-reported Table h3a RMSE before anything else

# %%
reported = json.loads((ROOT / "results" / "c1c18_h3a_leaveoneout.json").read_text())
reported_by_contract = {r["contract"]: r for r in reported["per_contract"]}

RMSE_TOL_PCT = 0.5  # percent
for i in TARGET_CONTRACTS:
    err_f = log_f[:, i] - filter_pred[:, i]
    err_c = log_f[:, i] - curve_pred[:, i]
    # Table h3a's RMSE is over days where BOTH the filter and the NS
    # refit are defined (NS needs >=6 valid contracts that day); matching
    # that mask, not just isfinite(err_f), is what makes "RMSE (full)"
    # below identical to the already-published figure.
    valid = np.isfinite(err_f) & np.isfinite(err_c)
    rmse_f = float(np.sqrt(np.mean(err_f[valid] ** 2))) * 1e4
    rec = reported_by_contract[i + 1]["rmse_filter_bp"]
    pct_diff = 100 * (rmse_f / rec - 1)
    print(f"C{i + 1}: recomputed RMSE={rmse_f:.3f}bp  Table h3a={rec:.3f}bp  diff={pct_diff:+.3f}%")
    if abs(pct_diff) > RMSE_TOL_PCT:
        raise RuntimeError(
            f"C{i + 1} recomputed filter RMSE does not match Table h3a within {RMSE_TOL_PCT}% "
            f"(recomputed={rmse_f:.3f}bp, reported={rec:.3f}bp) -- this is not a re-derivation "
            f"of the same leave-one-out array. Refusing to report robustness figures.")
print("RMSE cross-check OK on all four contracts: this is the same error array Table h3a used.\n")

# %% [markdown]
# ## Robustness statistics: MAE, ex-COVID RMSE, ex-near-expiry RMSE

# %%
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-31")
covid_mask = (dates >= COVID_START) & (dates <= COVID_END)

exp_dates = nearby_expiries(dates, 18, prefix="CL")  # (n, 18) datetime64
hol = pd.tseries.offsets.CustomBusinessDay
from csr.curve.expiry import _holidays  # noqa: E402
hol_arr = _holidays().to_numpy("datetime64[D]")
d0 = dates.to_numpy("datetime64[D]")
bdays_to_expiry = np.empty((n, 18))
for j in range(18):
    bdays_to_expiry[:, j] = np.busday_count(d0, exp_dates[:, j].astype("datetime64[D]"), holidays=hol_arr)
near_expiry_mask = bdays_to_expiry < 5  # per-contract, final 5 sessions before ITS OWN expiry

print(f"{'=' * 100}\nC1-C4 ROBUSTNESS: RMSE vs MAE, and under two exclusions (bp)\n{'=' * 100}")
header = f"{'':4s}{'RMSE (full)':>13s}{'MAE (full)':>12s}{'MAE/RMSE':>10s}{'n_covid':>9s}{'RMSE ex-COVID':>15s}{'n_nearexp':>11s}{'RMSE ex-nearexp':>17s}"
print(header)

rows = []
for i in TARGET_CONTRACTS:
    err_f = log_f[:, i] - filter_pred[:, i]
    err_c = log_f[:, i] - curve_pred[:, i]
    valid = np.isfinite(err_f) & np.isfinite(err_c)  # matches Table h3a's mask exactly

    rmse_full = float(np.sqrt(np.mean(err_f[valid] ** 2))) * 1e4
    mae_full = float(np.mean(np.abs(err_f[valid]))) * 1e4
    ratio = mae_full / rmse_full

    excl_covid = valid & ~covid_mask
    n_covid_dropped = int((valid & covid_mask).sum())
    rmse_ex_covid = float(np.sqrt(np.mean(err_f[excl_covid] ** 2))) * 1e4

    excl_nearexp = valid & ~near_expiry_mask[:, i]
    n_nearexp_dropped = int((valid & near_expiry_mask[:, i]).sum())
    rmse_ex_nearexp = float(np.sqrt(np.mean(err_f[excl_nearexp] ** 2))) * 1e4

    print(f"C{i + 1:2d}{rmse_full:13.3f}{mae_full:12.3f}{ratio:10.4f}{n_covid_dropped:9d}"
         f"{rmse_ex_covid:15.3f}{n_nearexp_dropped:11d}{rmse_ex_nearexp:17.3f}")
    rows.append({
        "contract": i + 1, "rmse_full_bp": rmse_full, "mae_full_bp": mae_full,
        "mae_over_rmse": ratio, "n_covid_dropped": n_covid_dropped,
        "rmse_ex_covid_bp": rmse_ex_covid, "n_nearexpiry_dropped": n_nearexp_dropped,
        "rmse_ex_nearexpiry_bp": rmse_ex_nearexp,
    })

# %% [markdown]
# ## Verdict against the pre-registered falsifier

# %%
c1_ratio = rows[0]["mae_over_rmse"]
print(f"\n{'=' * 100}\nVERDICT (pre-registered threshold: MAE/RMSE at C1 < 0.2)\n{'=' * 100}")
print(f"C1 MAE/RMSE = {c1_ratio:.4f}")
if c1_ratio < 0.2:
    verdict = "OUTLIER_DOMINATED"
    print("Below threshold: the Table h3a RMSE gap at C1 is outlier-dominated. "
         "Exclusion-adjusted RMSE becomes the primary figure per the pre-registered rule.")
else:
    verdict = "PERVASIVE"
    print("At or above threshold: the gap is pervasive, not a handful of sessions. "
         "The original Table h3a RMSE stands as primary; these are supplementary only.")

# %%
out = {"rows": rows, "covid_window": ["2020-03-01", "2020-05-31"],
      "near_expiry_window_sessions": 5, "c1_mae_over_rmse": c1_ratio, "verdict": verdict}
out_path = ROOT / "results" / "c1c18_c1_robustness_check.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
