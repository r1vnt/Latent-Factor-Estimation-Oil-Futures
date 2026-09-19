# %% [markdown]
# # H3a: cross-sectional leave-one-out, no dynamics
#
# Pre-registered in preregistrations/2026-09-19_h1_h2_h3_revisions.md.
# H3b (one-step-ahead forecasting) is a joint test of the measurement
# model and the transition model -- it rejected the conjunction. The
# measurement premise (18 contracts carry redundant, independently-
# contaminated reads of a common state) is a CROSS-SECTIONAL claim and
# needs no dynamics to test: does the filtered state price a held-out
# contract on the SAME day better than a flexible cross-sectional fit to
# the other 17, with no filter and no transition equation at all?
#
# Filter side: leave contract i out of the OBSERVATION set entirely (all
# n dates, that one column NaN'd), filter the other 17 with the EXISTING
# fitted S3 structural params (no re-estimation -- "no dynamics enter"
# means the transition equation does the state-propagation work here,
# same as always; what's being tested is whether that state, built
# from the other 17 contracts, prices contract i better than a same-day
# curve fit with no temporal information at all).
#
# Curve side: Nelson-Siegel-style 3-factor shape (level, slope,
# curvature; single shared decay lambda), fit PER DAY, no dynamics, no
# temporal linkage across days whatsoever. lambda is fit once per day
# using all 18 contracts (a stated simplification -- refitting lambda
# per left-out-contract-per-day was judged not worth the cost, since a
# single day's curve shape is unlikely to hinge on which one of 18
# points is excluded); the three level/slope/curvature loadings ARE
# refit on just the other 17 each time, so contract i's own value never
# enters its own prediction. Deliberately NOT linear interpolation
# (over-fits trivially) or an exact spline through all points (same
# Chen-Scott-style pathology already diagnosed once this project) --
# Nelson-Siegel is over-determined (3-4 params from 17 points), so it
# cannot fit noise the way an interpolant would.

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
from csr.state.fast import filter_uv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

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

# %% [markdown]
# ## Filter side: 18 leave-one-out filter passes (contract i NaN'd throughout)

# %%
sys3_full = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)

filter_pred = np.full((n, 18), np.nan)
for i in range(18):
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
# ## Curve side: Nelson-Siegel, lambda fit once per day, loadings refit per left-out contract

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

    for i in range(18):
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

print(f"curve-side fit complete: {np.isfinite(curve_pred).sum():,} (day, contract) predictions")

# %% [markdown]
# ## Compare: filter LOO vs. Nelson-Siegel LOO, per contract

# %%
print(f"\n{'=' * 90}\nH3a: FILTER vs. NELSON-SIEGEL, SAME-DAY LEAVE-ONE-OUT PREDICTION\n{'=' * 90}")
print(f"{'':4s}{'RMSE filter (bp)':>18s}{'RMSE NS (bp)':>15s}{'gain (%)':>10s}{'n':>7s}")
rows = []
n_filter_wins = 0
for i in range(18):
    err_f = log_f[:, i] - filter_pred[:, i]
    err_c = log_f[:, i] - curve_pred[:, i]
    valid = np.isfinite(err_f) & np.isfinite(err_c)
    rmse_f = float(np.sqrt(np.mean(err_f[valid] ** 2))) * 1e4
    rmse_c = float(np.sqrt(np.mean(err_c[valid] ** 2))) * 1e4
    gain = 100 * (1 - rmse_f / rmse_c)
    n_filter_wins += rmse_f < rmse_c
    rows.append({"contract": i + 1, "rmse_filter_bp": rmse_f, "rmse_ns_bp": rmse_c,
                "gain_pct": gain, "n": int(valid.sum())})
    print(f"C{i + 1:2d}{rmse_f:18.3f}{rmse_c:15.3f}{gain:10.2f}%{int(valid.sum()):7d}")

pooled_f = float(np.sqrt(np.nanmean((log_f - filter_pred) ** 2))) * 1e4
pooled_c = float(np.sqrt(np.nanmean((log_f - curve_pred) ** 2))) * 1e4
print(f"\npooled RMSE: filter={pooled_f:.3f}bp  NS={pooled_c:.3f}bp  "
     f"gain={100 * (1 - pooled_f / pooled_c):.2f}%")
print(f"filter beats NS on {n_filter_wins}/18 contracts")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if n_filter_wins >= 14 and pooled_f < pooled_c:
    print("H3a SUPPORTED: the filter prices held-out contracts same-day better than a flexible, "
         "dynamics-free cross-sectional fit -- the cross-sectional measurement-error claim holds "
         "up on its own, independent of H3b's temporal rejection.")
elif n_filter_wins <= 4 and pooled_f > pooled_c:
    print("H3a NOT SUPPORTED: the flexible cross-sectional fit prices held-out contracts as well "
         "or better -- the cross-sectional claim doesn't survive either, a stronger and more "
         "complete negative than H3b alone.")
else:
    print("MIXED: no clean sweep either way.")

# %%
out = {"per_contract": rows, "pooled_rmse_filter_bp": pooled_f, "pooled_rmse_ns_bp": pooled_c,
      "n_filter_wins": n_filter_wins}
out_path = ROOT / "results" / "c1c18_h3a_leaveoneout.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
