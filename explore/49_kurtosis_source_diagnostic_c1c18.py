# %% [markdown]
# # Why did kurtosis rise even though squared-innovation autocorrelation improved?
#
# explore/48's front-spread channel is a much better result than explore/46's
# RV channel on every OTHER axis: 3x the LR gain (8057 vs 2612 nats), 67%
# of it genuine vs. 46%, Ljung-Box on squared innovations actually moves
# (18/18 -> 14-15/18, the first real movement all session), and |front
# spread|'s own predictive power for the target is fully absorbed (R^2
# 0.557 -> 0.007). But pooled excess kurtosis rose almost identically to
# before (34.3 -> 55.46 vs. 34.3 -> 54.3), which by the user's own
# diagnostic ("if the channel is widening on the right days, excess
# kurtosis must fall") reads as "still the wrong days."
#
# Those two facts are only in tension if the SAME days drive both. Check
# directly: pull the largest-|z| observations under the new fit, and see
# whether they are (a) the same days that were already extreme before --
# a denominator effect, where the bulk of ordinary days got tighter around
# zero and a few unrelated outliers became relatively more prominent by
# comparison, not newly mis-widened -- or (b) genuinely new/different
# days, and whether THOSE days have low front-spread readings (confirming
# "wrong variable, even at the tail") or high ones (a narrower problem:
# under-widening at the extreme tail specifically, fixable in eta_v/
# lambda_v's shape, not a sign the variable itself is wrong).

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
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
dates = panel.dates
n = panel.n

rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")

spread = log_f[:, 0] - log_f[:, 1]
abs_spread_lag = np.concatenate([[np.nan], np.abs(spread[:-1])])
fs_driver = np.zeros(n)
running_sum, running_sumsq, count = 0.0, 0.0, 0
last_valid = 0.0
MIN_HISTORY = 60
for t in range(n):
    if count >= MIN_HISTORY:
        mean = running_sum / count
        var = running_sumsq / count - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        if np.isfinite(abs_spread_lag[t]):
            last_valid = (abs_spread_lag[t] - mean) / std
        fs_driver[t] = last_valid   # carried forward through NaN days, see explore/48
    if np.isfinite(abs_spread_lag[t]):
        running_sum += abs_spread_lag[t]
        running_sumsq += abs_spread_lag[t] ** 2
        count += 1

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

fc = json.loads((ROOT / "results" / "c1c18_frontspread_channel.json").read_text())["student_t_em"]
fcp = fc["params"]
p_after = ThreeFactorParams(
    kappa1=fc["kappa1"], kappa2=fc["kappa2"], sigma1=fcp["sigma1"], sigma2=fcp["sigma2"],
    sigma_xi=fcp["sigma_xi"], rho12=0.0, rho1x=fcp["rho1x"], rho2x=fcp["rho2x"],
    mu_xi=fcp["mu_xi"], mu_xi_star=fcp["mu_xi_star"], lambda1=fcp["lambda1"], lambda2=fcp["lambda2"],
    gamma_v=fc["gamma_v"],
)
smooth_after = SmoothMeasParams(a=fcp["curve"]["a"], b=fcp["curve"]["b"], c=fcp["curve"]["c"],
                                quadratic=True, eta_v=fc["eta_v"], lambda_v=fcp["curve"]["lambda_v"])


def _filter(p, smooth, rv_lag_arg, h_driver_arg):
    sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag_arg, h_driver=h_driver_arg)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return resid


resid_before = _filter(p_before, smooth_before, None, None)
resid_after = _filter(p_after, smooth_after, rv_lag, fs_driver)

# %% [markdown]
# ## Top 30 |z| under the new fit -- same days as before, or new ones?

# %%
TOP_N = 30
flat_after = []
for j in range(18):
    for t in range(n):
        z = resid_after[t, j]
        if np.isfinite(z):
            flat_after.append((abs(z), t, j))
flat_after.sort(reverse=True)
top_after = flat_after[:TOP_N]

flat_before = []
for j in range(18):
    for t in range(n):
        z = resid_before[t, j]
        if np.isfinite(z):
            flat_before.append((abs(z), t, j))
flat_before.sort(reverse=True)
top_before_set = {(t, j) for _, t, j in flat_before[:TOP_N]}

print(f"{'=' * 100}\nTOP {TOP_N} |z| UNDER THE NEW (front-spread) FIT\n{'=' * 100}")
print(f"{'date':>12s}{'contract':>10s}{'|z| after':>12s}{'|z| before':>12s}"
     f"{'fs_driver':>12s}{'rv_driver':>12s}{'in before-top30':>18s}")
overlap = 0
for absz, t, j in top_after:
    was_before = (t, j) in top_before_set
    overlap += was_before
    print(f"{str(dates[t].date()):>12s}  C{j + 1:<8d}{absz:12.3f}{abs(resid_before[t, j]):12.3f}"
         f"{fs_driver[t]:12.3f}{rv_lag[t]:12.3f}{'YES' if was_before else 'no':>18s}")

print(f"\noverlap with before's own top {TOP_N}: {overlap}/{TOP_N}")

# %% [markdown]
# ## Are the new top-30 days high-front-spread days, or not?

# %%
fs_of_top_after = np.array([fs_driver[t] for _, t, j in top_after])
fs_all_active = fs_driver[fs_driver != 0]
print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
print(f"fs_driver at the new top-{TOP_N} |z| days: mean={fs_of_top_after.mean():.3f}  "
     f"median={np.median(fs_of_top_after):.3f}")
print(f"fs_driver overall (active days):           mean={fs_all_active.mean():.3f}  "
     f"median={np.median(fs_all_active):.3f}  p95={np.percentile(fs_all_active, 95):.3f}")
pct_low_fs = float(np.mean(fs_of_top_after < np.percentile(fs_all_active, 50)))
print(f"fraction of new top-{TOP_N} days BELOW the median fs_driver reading: {pct_low_fs:.2%}")
if pct_low_fs > 0.5:
    print("\nThe worst days under the new fit are NOT high-front-spread days -- "
         "consistent with a denominator effect: fitting the channel tightened "
         "the bulk of ordinary days, and a distinct, unrelated set of outliers "
         "(genuine jumps the front-spread driver has no information about) "
         "became relatively more prominent in the pooled kurtosis statistic "
         "without the model actually widening H incorrectly on them.")
else:
    print("\nThe worst days under the new fit ARE elevated-front-spread days -- "
         "the channel is under-widening even on the days it should fire hardest "
         "on. A tail-shape problem (eta_v/lambda_v's functional form under-"
         "responds at the extreme end), not evidence the variable itself is wrong.")

# %%
out = {
    "top_after": [{"date": str(dates[t].date()), "contract": j + 1, "abs_z_after": float(absz),
                  "abs_z_before": float(abs(resid_before[t, j])), "fs_driver": float(fs_driver[t]),
                  "rv_driver": float(rv_lag[t]), "in_before_top30": (t, j) in top_before_set}
                 for absz, t, j in top_after],
    "overlap_with_before_top30": overlap,
    "fs_driver_mean_at_new_top30": float(fs_of_top_after.mean()),
    "fs_driver_median_overall": float(np.median(fs_all_active)),
    "pct_new_top30_below_median_fs": pct_low_fs,
}
out_path = ROOT / "results" / "c1c18_kurtosis_source_diagnostic.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
