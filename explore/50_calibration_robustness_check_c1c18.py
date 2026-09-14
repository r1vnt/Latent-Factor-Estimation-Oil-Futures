# %% [markdown]
# # Is the calibration drift (1.667 -> 1.0998) broad, or one week's artifact?
#
# The fixed front-spread channel dropped pooled excess kurtosis from 34.3
# to 1.86 -- a clean pass on the sharpest diagnostic. But mean resid_std^2
# moved FURTHER from the 1.667 t5 benchmark (1.3731 -> 1.0998), and that
# fit also now prices the 2020-04-21 crash aftermath almost exactly right
# (the whole point of the bug fix). A single week fit near-perfectly can
# drag a pooled mean down a lot on its own. Recompute mean resid_std^2 and
# excess kurtosis excluding 2020-04-13 to 2020-04-27 (two weeks bracketing
# the negative-price event) for both fits, to see whether the calibration
# drift is a broad, every-day phenomenon or concentrated in that window.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis

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
        fs_driver[t] = last_valid
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

crash_mask = (dates >= "2020-04-13") & (dates <= "2020-04-27")
print(f"crash window: {int(crash_mask.sum())} dates excluded from the robustness check")

for label, resid in (("before", resid_before), ("after", resid_after)):
    full = resid[np.isfinite(resid)]
    ex_crash = resid[~crash_mask][np.isfinite(resid[~crash_mask])]
    m_full = float(np.mean(full ** 2))
    m_ex = float(np.mean(ex_crash ** 2))
    k_full = float(_kurtosis(full, fisher=True, bias=False))
    k_ex = float(_kurtosis(ex_crash, fisher=True, bias=False))
    print(f"\n{label}: n_full={len(full)}  n_ex_crash={len(ex_crash)}")
    print(f"  mean resid_std^2:   full={m_full:.4f}   ex-crash={m_ex:.4f}   (t5 benchmark: 1.667)")
    print(f"  excess kurtosis:    full={k_full:.4f}   ex-crash={k_ex:.4f}")

out = {"crash_dates_excluded": int(crash_mask.sum())}
for label, resid in (("before", resid_before), ("after", resid_after)):
    full = resid[np.isfinite(resid)]
    ex_crash = resid[~crash_mask][np.isfinite(resid[~crash_mask])]
    out[label] = {
        "mean_resid2_full": float(np.mean(full ** 2)), "mean_resid2_ex_crash": float(np.mean(ex_crash ** 2)),
        "excess_kurt_full": float(_kurtosis(full, fisher=True, bias=False)),
        "excess_kurt_ex_crash": float(_kurtosis(ex_crash, fisher=True, bias=False)),
    }
(ROOT / "results" / "c1c18_calibration_robustness.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to results/c1c18_calibration_robustness.json")
