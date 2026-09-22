# %% [markdown]
# # Why does the essentially-affine fit gain 3142 nats? Front-specific, or diffuse?
#
# The multi-start result (explore/68) is stable but points the WRONG way:
# kappa1_phys=4.37 > kappa1^Q=1.89 (faster physical reversion, 58-day
# half-life vs 134 -- the opposite of "closer to a random walk"). Before
# reading anything into that direction, check what the gain is actually
# doing: per-contract residual comparison against the S3 baseline. If
# it's concentrated at C1 (even in the wrong direction), this is still
# about the front. If it's diffuse across the whole curve, this is more
# likely picking up general volatility-clustering structure that has
# nothing to do with the specific front-contract diagnosis -- exactly
# the "traded a finding for a fitted line" risk flagged in advance.

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
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))

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

ea = json.loads((ROOT / "results" / "c1c18_essentially_affine_multistart.json").read_text())
best = ea["starts"][ea["best_start_idx"]]
p_ea = p_before.replace(kappa1_phys=best["kappa1_phys_hat"], kappa2_phys=best["kappa2_phys_hat"])
# curve/other structural params: re-use the S3 baseline's own (the EA fit warm-started from
# them and only kappa1_phys/kappa2_phys plus the curve were free to move during optimisation;
# reload the ACTUAL fitted curve+structural params by refitting is unnecessary -- re-derive
# via a quick refit at the converged kappa_phys to get the exact accompanying params)
from csr.pricing.fit3 import fit3_gaussian_smooth, pack3_smooth  # noqa: E402
theta0 = pack3_smooth(p_ea, before["curve"]["a"], before["curve"]["b"], before["curve"]["c"],
                      quadratic=True, ea_extension=True)
p_ea, smooth_ea, x_filt_ea, resid_ea_direct, n_obs_ea, ll_ea = fit3_gaussian_smooth(
    log_f, tau, DT, theta0, quadratic=True, ea_extension=True)
print(f"re-fit check: ll={ll_ea:.2f}  (multistart reported {best['ll']:.2f})  "
     f"k1_phys={p_ea.kappa1_phys:.4f}  k2_phys={p_ea.kappa2_phys:.4f}")


def _filter_with_contributions(p, smooth):
    sys3 = _raw_system3(p, smooth, tau, DT)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
        np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell


ll_before, resid_before, ll_cell_before = _filter_with_contributions(p_before, smooth_before)
ll_ea2, resid_ea, ll_cell_ea = _filter_with_contributions(p_ea, smooth_ea)
print(f"cross-check: ll_before={ll_before:.2f}  ll_ea={ll_ea2:.2f}")

# %% [markdown]
# ## Per-contract residual comparison: front-concentrated, or diffuse?

# %%
print(f"\n{'=' * 100}\nPER-CONTRACT: mean resid_std^2 (standardised), before vs essentially-affine\n{'=' * 100}")
print(f"{'':4s}{'resid2 before':>15s}{'resid2 EA':>13s}{'d(resid2)':>12s}{'ll gain':>10s}")
rows = []
total_gain = 0.0
for j in range(18):
    m_before = float(np.nanmean(resid_before[:, j] ** 2))
    m_ea = float(np.nanmean(resid_ea[:, j] ** 2))
    ll_j_before = float(np.nansum(ll_cell_before[:, j]))
    ll_j_ea = float(np.nansum(ll_cell_ea[:, j]))
    gain = ll_j_ea - ll_j_before
    total_gain += gain
    rows.append({"contract": j + 1, "resid2_before": m_before, "resid2_ea": m_ea,
                "d_resid2": m_ea - m_before, "ll_gain": gain})
    print(f"C{j + 1:2d}{m_before:15.4f}{m_ea:13.4f}{m_ea - m_before:12.4f}{gain:10.2f}")
print(f"\ntotal ll gain (sum over contracts): {total_gain:.2f}  (filter-level: {ll_ea2 - ll_before:.2f})")

front_gain = sum(r["ll_gain"] for r in rows[:4])
back_gain = sum(r["ll_gain"] for r in rows[4:])
print(f"gain from C1-C4: {front_gain:.2f} ({100*front_gain/total_gain:.1f}%)   "
     f"gain from C5-C18: {back_gain:.2f} ({100*back_gain/total_gain:.1f}%)")

# %% [markdown]
# ## Squared-innovation Ljung-Box: does this move the untouched-by-everything-else number?

# %%
print(f"\n{'=' * 90}\nLJUNG-BOX ON SQUARED INNOVATIONS\n{'=' * 90}")
LAGS = [1, 5, 10, 20]


def _lb_sig_count(resid, lag):
    sig = 0
    for j in range(18):
        z = resid[:, j]
        z = z[np.isfinite(z)]
        lb = acorr_ljungbox(z ** 2, lags=[lag], return_df=True)
        if lb.loc[lag, "lb_pvalue"] < 0.01:
            sig += 1
    return sig


for lag in LAGS:
    sig_before = _lb_sig_count(resid_before, lag)
    sig_ea = _lb_sig_count(resid_ea, lag)
    print(f"lag={lag:3d}  before: {sig_before}/18   EA: {sig_ea}/18")

kurt_before = float(_kurtosis(resid_before[np.isfinite(resid_before)].ravel(), fisher=True, bias=False))
kurt_ea = float(_kurtosis(resid_ea[np.isfinite(resid_ea)].ravel(), fisher=True, bias=False))
print(f"\nexcess kurtosis: before={kurt_before:.2f}  EA={kurt_ea:.2f}")

# %%
out = {"per_contract": rows, "front_gain_pct": 100 * front_gain / total_gain,
      "kurt_before": kurt_before, "kurt_ea": kurt_ea,
      "kappa1_phys": p_ea.kappa1_phys, "kappa2_phys": p_ea.kappa2_phys,
      "beta1": p_ea.beta1, "beta2": p_ea.beta2}
out_path = ROOT / "results" / "c1c18_essentially_affine_diagnosis.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
