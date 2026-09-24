# %% [markdown]
# # Rung 2 diagnosis: is the 13,102-nat gain front-concentrated this time?
#
# Rung 1's large gain was diffuse (84% from C5-C18) and in the wrong
# direction -- explained as "chi1 loads everywhere, not just the front."
# This factor is front-EXCLUSIVE by construction (loadings [0.59, 0.21,
# 0.08, 0.03, ...], negligible by C4). If the gain is still diffuse, the
# front-exclusivity fix didn't isolate anything and something more
# generic (volatility clustering, again) is the real explanation. Same
# battery as explore/69: per-contract ll decomposition, Ljung-Box on
# squared innovations, kurtosis.

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
from csr.pricing.fit4_front import fit4_front_gaussian, filter4_front_contributions, pack4_front  # noqa: E402
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
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


def _filter3_with_contributions(p, smooth):
    sys3 = _raw_system3(p, smooth, tau, DT)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
        np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell


ll_before, resid_before, ll_cell_before = _filter3_with_contributions(p_before, smooth_before)

ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]

# re-fit at the converged point to get the FULL param set (multistart only saved a summary)
before_c = before["curve"]
p_start = FourFactorFrontParams(
    kappa2=before["kappa1"], kappa3=before["kappa2"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta0 = pack4_front(p_start, before_c["a"], before_c["b"], before_c["c"], quadratic=True)
p_front, smooth_front, x_filt_front, resid_direct, n_obs_front, ll_front = fit4_front_gaussian(
    log_f, tau, DT, theta0, quadratic=True, kappa_f_q=KAPPA_F_Q_ANCHOR)
print(f"re-fit check: ll={ll_front:.2f}  (multistart reported {best['ll']:.2f})  "
     f"kappa_f_P={p_front.kappa_f_P:.4f}  sigma_f={p_front.sigma_f:.4f}  rho_fx={p_front.rho_fx:.4f}")

ll_front2, xf, Pf, xp, Pp, resid_front, ll_cell_front, n_obs = filter4_front_contributions(
    p_front, smooth_front, log_f, tau, DT)
print(f"cross-check (contributions): ll={ll_front2:.2f}")

# %% [markdown]
# ## Per-contract residual/likelihood comparison

# %%
print(f"\n{'=' * 100}\nPER-CONTRACT: mean resid_std^2, before (S3) vs rung-2 front factor\n{'=' * 100}")
print(f"{'':4s}{'resid2 before':>15s}{'resid2 front':>14s}{'d(resid2)':>12s}{'ll gain':>10s}")
rows = []
total_gain = 0.0
for j in range(18):
    m_before = float(np.nanmean(resid_before[:, j] ** 2))
    m_front = float(np.nanmean(resid_front[:, j] ** 2))
    ll_j_before = float(np.nansum(ll_cell_before[:, j]))
    ll_j_front = float(np.nansum(ll_cell_front[:, j]))
    gain = ll_j_front - ll_j_before
    total_gain += gain
    rows.append({"contract": j + 1, "resid2_before": m_before, "resid2_front": m_front,
                "d_resid2": m_front - m_before, "ll_gain": gain})
    print(f"C{j + 1:2d}{m_before:15.4f}{m_front:14.4f}{m_front - m_before:12.4f}{gain:10.2f}")
print(f"\ntotal ll gain: {total_gain:.2f}  (filter-level: {ll_front2 - ll_before:.2f})")

front4_gain = sum(r["ll_gain"] for r in rows[:4])
back_gain = sum(r["ll_gain"] for r in rows[4:])
c1_only_gain = rows[0]["ll_gain"]
print(f"gain from C1-C4: {front4_gain:.2f} ({100*front4_gain/total_gain:.1f}%)   "
     f"gain from C5-C18: {back_gain:.2f} ({100*back_gain/total_gain:.1f}%)")
print(f"gain from C1 alone: {c1_only_gain:.2f} ({100*c1_only_gain/total_gain:.1f}%)")

# %% [markdown]
# ## Ljung-Box on squared innovations, and kurtosis

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
    sig_front = _lb_sig_count(resid_front, lag)
    print(f"lag={lag:3d}  before: {sig_before}/18   rung-2 front: {sig_front}/18")

kurt_before = float(_kurtosis(resid_before[np.isfinite(resid_before)].ravel(), fisher=True, bias=False))
kurt_front = float(_kurtosis(resid_front[np.isfinite(resid_front)].ravel(), fisher=True, bias=False))
print(f"\nexcess kurtosis: before={kurt_before:.2f}  rung-2 front={kurt_front:.2f}")

# %%
out = {"per_contract": rows, "front4_gain_pct": 100 * front4_gain / total_gain,
      "c1_gain_pct": 100 * c1_only_gain / total_gain,
      "kurt_before": kurt_before, "kurt_front": kurt_front,
      "kappa_f_P": p_front.kappa_f_P, "kappa_f_Q": p_front.kappa_f_Q, "sigma_f": p_front.sigma_f,
      "rho_fx": p_front.rho_fx, "ll_front": ll_front2, "ll_before": ll_before}
out_path = ROOT / "results" / "c1c18_fourfactor_front_diagnosis.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
