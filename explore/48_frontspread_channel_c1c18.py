# %% [markdown]
# # The H channel, redriven: |front spread| replaces lagged RV
#
# explore/47's horse race: lagged RV explains none of the front-contract
# variance-mismatch target once |front spread| (C1-C2, lagged one day) is
# controlled for -- RV's own coefficient collapses to p=0.48 alongside it,
# while front spread stays at p=3.6e-7 and triples RV's standalone R^2
# (0.173 vs 0.058). Days-to-expiry and the roll dummy were both dead on
# arrival (R^2 ~ 0.0001-0.0006, gone once anything else is controlled
# for). Volume/open interest remain untested -- not available on this
# machine (see explore/47's header) -- flagged as a real, still-open gap.
#
# fit3.py now takes independent drivers for the two RV-shaped channels:
# rv_lag for gamma_v (Q-side, kept as-is -- a separately identified,
# genuine effect per the user's explicit instruction to keep it) and
# h_driver for eta_v (H-side, now driven by |front spread| instead of RV,
# since RV does not survive alongside it).
#
# Same discipline as explore/45: causal/standardised driver, gamma_v held
# at EXACTLY 0 while eta_v is fit alone (1 df), then gamma_v freed on top
# (1 more df), bounded by construction (tanh), Gaussian first.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2, kurtosis as _kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    ETA_V_MAX,
    GAMMA_V_MAX,
    STUDENT_T_DF,
    _neg_loglik3,
    _raw_system3,
    fit3_gaussian_smooth,
    fit3_student_t_smooth,
    pack3_smooth,
    unpack3_smooth,
)
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402
from csr.study.recovery3 import realised_vol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
LOG_2PI = float(np.log(2.0 * np.pi))
LAGS = [1, 5, 10, 20]
HORIZONS = (22, 66, 132, 252)
LAMBDA_V_FIXED = 2.0
GAMMA_V_IDX = 11
MIN_HISTORY = 60

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

# %% [markdown]
# ## Front-spread driver: |C1-C2|, lagged one day, causally standardised

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
        # Carried forward through NaN days (e.g. 2020-04-20's undefined
        # log price under a negative settlement) rather than zeroed --
        # explore/49 found the driver reading exactly 0.000 (inert) on
        # 2020-04-21, the single most extreme day in the sample, purely
        # because the prior day's spread was undefined. Zeroing treats a
        # data gap as "calm"; carrying forward the last genuine reading
        # treats it as "state unchanged," which is the correct causal
        # default for a stress indicator across a one-day gap.
        fs_driver[t] = last_valid
    if np.isfinite(abs_spread_lag[t]):
        running_sum += abs_spread_lag[t]
        running_sumsq += abs_spread_lag[t] ** 2
        count += 1

valid_days = int((fs_driver != 0).sum())
print(f"causal standardised |front spread|: {valid_days}/{n} days active")
print(f"driver percentiles (active days): "
     f"p50={np.percentile(fs_driver[fs_driver != 0], 50):.3f}  "
     f"p95={np.percentile(fs_driver[fs_driver != 0], 95):.3f}  "
     f"p99={np.percentile(fs_driver[fs_driver != 0], 99):.3f}")

# reuse the existing RV driver (unchanged -- still drives gamma_v/Q-side)
rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")

# %% [markdown]
# ## S3 baseline (reuse explore/45's Gaussian fit, no refitting)

# %%
diag = json.loads((ROOT / "results" / "c1c18_rv_channel_diagnostic.json").read_text())
s3p = diag["S3"]["params"]
p_s3 = ThreeFactorParams(
    kappa1=s3p["kappa1"], kappa2=s3p["kappa2"], sigma1=s3p["sigma1"], sigma2=s3p["sigma2"],
    sigma_xi=s3p["sigma_xi"], rho12=0.0, rho1x=s3p["rho1x"], rho2x=s3p["rho2x"],
    mu_xi=s3p["mu_xi"], mu_xi_star=s3p["mu_xi_star"], lambda1=s3p["lambda1"], lambda2=s3p["lambda2"],
    gamma_v=0.0,
)
smooth_s3 = SmoothMeasParams(a=s3p["curve"]["a"], b=s3p["curve"]["b"], c=s3p["curve"]["c"], quadratic=True)
ll_s3 = diag["S3"]["ll"]
print(f"\nS3 (Gaussian, no channels): ll={ll_s3:.2f}  k1={p_s3.kappa1:.4f}  k2={p_s3.kappa2:.4f}")

# %% [markdown]
# ## S3 + eta_v(front spread) alone, gamma_v held at EXACTLY 0

# %%
theta_full0 = pack3_smooth(p_s3, smooth_s3.a, smooth_s3.b, smooth_s3.c, quadratic=True,
                           eta_v=0.5, lambda_v=LAMBDA_V_FIXED, rv_extension=True)
reduced_idx = [i for i in range(len(theta_full0)) if i != GAMMA_V_IDX]


def _neg_loglik_eta_only(theta_reduced):
    theta_full = np.zeros(len(theta_full0))
    theta_full[reduced_idx] = theta_reduced
    theta_full[GAMMA_V_IDX] = 0.0
    return _neg_loglik3(theta_full, log_f, tau, DT, True, rv_lag=None, rv_extension=True,
                        h_driver=fs_driver)


theta0_reduced = theta_full0[reduced_idx]
res_eta = minimize(_neg_loglik_eta_only, theta0_reduced, method="BFGS",
                   options={"maxiter": 2000, "gtol": 1e-8})
theta_eta_full = np.zeros(len(theta_full0))
theta_eta_full[reduced_idx] = res_eta.x
theta_eta_full[GAMMA_V_IDX] = 0.0
p_eta, smooth_eta = unpack3_smooth(theta_eta_full, quadratic=True, rv_extension=True)
ll_eta = -res_eta.fun

eta_ratio = abs(smooth_eta.eta_v) / ETA_V_MAX
print(f"\nS3+eta_v(front spread), gamma_v FIXED at 0: ll={ll_eta:.2f}  "
     f"k1={p_eta.kappa1:.4f}  k2={p_eta.kappa2:.4f}")
print(f"eta_v={smooth_eta.eta_v:.4f}  (lambda_v fixed at {LAMBDA_V_FIXED})  "
     f"gamma_v={p_eta.gamma_v:.6f} (must be 0)")
print(f"binding check: |eta_v|/ETA_V_MAX = {eta_ratio:.4f}  "
     f"({'BINDING' if eta_ratio > 0.9 else 'interior'})")

front_weight = np.exp(-smooth_eta.lambda_v * tau[0, 0])
for pct, label in [(50, "p50"), (95, "p95"), (99, "p99")]:
    d = np.percentile(fs_driver[fs_driver != 0], pct)
    multiple = np.exp(smooth_eta.eta_v * front_weight * d)
    print(f"  variance multiple at driver {label}={d:.3f}: {multiple:.3f}x")

LR_eta = 2.0 * (ll_eta - ll_s3)
p_eta_lr = float(chi2.sf(LR_eta, df=1))
print(f"LR(S3+eta_v[frontspread] vs S3) = {LR_eta:.3f}  df=1  p={p_eta_lr:.4g}")

# %% [markdown]
# ## + gamma_v (RV-driven, Q-side, unchanged) free on top

# %%
theta0_both = pack3_smooth(p_eta, smooth_eta.a, smooth_eta.b, smooth_eta.c, quadratic=True,
                           eta_v=smooth_eta.eta_v, lambda_v=LAMBDA_V_FIXED, rv_extension=True)
p_both, smooth_both, _, _, _, ll_both = fit3_gaussian_smooth(
    log_f, tau, DT, theta0_both, quadratic=True, rv_lag=rv_lag, rv_extension=True, h_driver=fs_driver)

eta_ratio_b = abs(smooth_both.eta_v) / ETA_V_MAX
gamma_ratio_b = abs(p_both.gamma_v) / GAMMA_V_MAX
print(f"\nS3+eta_v(frontspread)+gamma_v(RV): ll={ll_both:.2f}  "
     f"k1={p_both.kappa1:.4f}  k2={p_both.kappa2:.4f}")
print(f"eta_v={smooth_both.eta_v:.4f} ({eta_ratio_b * 100:.1f}% of MAX)  "
     f"gamma_v={p_both.gamma_v:.4f} ({gamma_ratio_b * 100:.1f}% of MAX)")

LR_both = 2.0 * (ll_both - ll_eta)
p_both_lr = float(chi2.sf(LR_both, df=1))
print(f"LR(both vs eta_v[frontspread]-only) = {LR_both:.3f}  df=1  p={p_both_lr:.4g}")

# %% [markdown]
# ## Student-t EM (nu=5 fixed) from the Gaussian optimum

# %%
theta0_em = pack3_smooth(p_both, smooth_both.a, smooth_both.b, smooth_both.c, quadratic=True,
                         eta_v=smooth_both.eta_v, lambda_v=smooth_both.lambda_v, rv_extension=True)
p_hat, smooth_hat, x_filt, resid_final, weight, n_iters, ll_hat, _df_fixed = fit3_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF, theta0_em, quadratic=True, rv_lag=rv_lag, rv_extension=True,
    h_driver=fs_driver)

eta_ratio_hat = abs(smooth_hat.eta_v) / ETA_V_MAX
gamma_ratio_hat = abs(p_hat.gamma_v) / GAMMA_V_MAX
print(f"\n{'=' * 90}\nSTUDENT-T EM RESULT (nu={STUDENT_T_DF} fixed)\n{'=' * 90}")
print(f"ll={ll_hat:.2f}  n_em_iters={n_iters}")
print(f"k1={p_hat.kappa1:.4f}  k2={p_hat.kappa2:.4f}")
print(f"eta_v={smooth_hat.eta_v:.4f} ({eta_ratio_hat * 100:.1f}% of MAX) "
     f"-- {'BINDING' if eta_ratio_hat > 0.9 else 'interior'}")
print(f"gamma_v={p_hat.gamma_v:.4f} ({gamma_ratio_hat * 100:.1f}% of MAX) "
     f"-- {'BINDING' if gamma_ratio_hat > 0.9 else 'interior'}")

# %% [markdown]
# ## Acceptance criteria, exactly as explore/46 -- before = c1c18_threefactor.json

# %%
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


def _filter_with_contributions(p, smooth, rv_lag_arg, h_driver_arg):
    sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag_arg, h_driver=h_driver_arg)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell, int(n_obs)


ll_before, resid_before, ll_cell_before, n_obs_before = _filter_with_contributions(p_before, smooth_before, None, None)
ll_after, resid_after, ll_cell_after, n_obs_after = _filter_with_contributions(p_hat, smooth_hat, rv_lag, fs_driver)
print(f"\ncross-check: ll_before={ll_before:.2f} (saved {before['ll']:.2f})  "
     f"ll_after={ll_after:.2f} (EM return {ll_hat:.2f})")

print(f"\n{'=' * 90}\nACCEPTANCE 1: LJUNG-BOX ON SQUARED INNOVATIONS\n{'=' * 90}")


def _ljungbox_squared(resid):
    rows = []
    for j in range(18):
        z = resid[:, j]
        z = z[np.isfinite(z)]
        lb = acorr_ljungbox(z ** 2, lags=LAGS, return_df=True)
        rows.append({"contract": j + 1,
                    **{f"lb_pvalue_{lag}": float(lb.loc[lag, "lb_pvalue"]) for lag in LAGS}})
    return rows


lb_before = _ljungbox_squared(resid_before)
lb_after = _ljungbox_squared(resid_after)
for lag in LAGS:
    sig_before = [r["contract"] for r in lb_before if r[f"lb_pvalue_{lag}"] < 0.01]
    sig_after = [r["contract"] for r in lb_after if r[f"lb_pvalue_{lag}"] < 0.01]
    print(f"lag={lag:3d}  before: {len(sig_before):2d}/18   after: {len(sig_after):2d}/18   {sig_after}")

print(f"\n{'=' * 90}\nACCEPTANCE 2: DOES |FRONT SPREAD| STILL SURVIVE, CONTROLLING FOR RV?\n{'=' * 90}")


def _v_mag_c1(resid, ll_cell):
    logF = -2 * ll_cell - LOG_2PI - resid ** 2
    F = np.exp(logF)
    v_raw = resid * np.sqrt(F)
    return v_raw[:, 0]


z_c1_before = resid_before[:, 0]
z_c1_after = resid_after[:, 0]
target_before = z_c1_before ** 2 - 1.0
target_after = z_c1_after ** 2 - 1.0

rv_raw = np.full(n, np.nan)
ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
for t in range(20, n):
    window = ret[t - 19: t + 1]
    if np.all(np.isfinite(window)):
        rv_raw[t] = np.std(window, ddof=1)
rv_raw_lag = np.concatenate([[np.nan], rv_raw[:-1]])

mech_results = {}
for label, target in (("before", target_before), ("after", target_after)):
    valid = np.isfinite(target) & np.isfinite(rv_raw_lag) & np.isfinite(abs_spread_lag)
    X = sm.add_constant(np.column_stack([rv_raw_lag[valid], abs_spread_lag[valid]]), has_constant="add")
    res = sm.OLS(target[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    mech_results[label] = {"n": int(res.nobs), "coef_spread": float(res.params[2]),
                          "p_spread": float(res.pvalues[2]), "r2": float(res.rsquared)}
    survives = "SURVIVES" if res.pvalues[2] < 0.01 else "gone"
    print(f"{label}: n={int(res.nobs)}  coef_spread={res.params[2]:+.4f}  "
         f"p_spread={res.pvalues[2]:.4g}  R2={res.rsquared:.4f}  ({survives})")

print(f"\n{'=' * 90}\nACCEPTANCE 3: LL DECOMPOSITION AND KURTOSIS\n{'=' * 90}")
sum_resid2_before = float(np.nansum(resid_before ** 2))
sum_resid2_after = float(np.nansum(resid_after ** 2))
logF_before = -2.0 * ll_cell_before - LOG_2PI - resid_before ** 2
logF_after = -2.0 * ll_cell_after - LOG_2PI - resid_after ** 2
sum_logF_before = float(np.nansum(logF_before))
sum_logF_after = float(np.nansum(logF_after))

total_gain = ll_after - ll_before
gain_from_resid2 = -0.5 * (sum_resid2_after - sum_resid2_before)
gain_from_logF = -0.5 * (sum_logF_after - sum_logF_before)
print(f"total ll gain: {total_gain:.2f} nats  "
     f"(resid^2: {gain_from_resid2:+.2f} [{100 * gain_from_resid2 / total_gain:.1f}%]  "
     f"logF: {gain_from_logF:+.2f} [{100 * gain_from_logF / total_gain:.1f}%])")
mean_resid2_before = sum_resid2_before / n_obs_before
mean_resid2_after = sum_resid2_after / n_obs_after
print(f"mean resid_std^2: before={mean_resid2_before:.4f}  after={mean_resid2_after:.4f}  "
     f"(t5 benchmark: 1.667)")
closer = "CLOSER to 1.667" if abs(mean_resid2_after - 1.667) < abs(mean_resid2_before - 1.667) else "NOT closer"
print(f"calibration verdict: {closer}")

excess_kurt_before = float(_kurtosis(resid_before[np.isfinite(resid_before)].ravel(), fisher=True, bias=False))
excess_kurt_after = float(_kurtosis(resid_after[np.isfinite(resid_after)].ravel(), fisher=True, bias=False))
print(f"excess kurtosis (pooled): before={excess_kurt_before:.4f}  after={excess_kurt_after:.4f}  "
     f"({'ROSE -- widening on the wrong days again' if excess_kurt_after > excess_kurt_before else 'FELL -- widening on the right days'})")

# %%
out = {
    "driver": "abs_front_spread_lag1, causally standardised",
    "S3_eta_v_frontspread": {"ll": ll_eta, "eta_v": smooth_eta.eta_v, "LR_vs_S3": LR_eta, "p_vs_S3": p_eta_lr,
                            "binding": eta_ratio > 0.9},
    "S3_eta_v_frontspread_gamma_v": {"ll": ll_both, "eta_v": smooth_both.eta_v, "gamma_v": p_both.gamma_v,
                                    "LR_vs_eta_v_only": LR_both, "p_vs_eta_v_only": p_both_lr},
    "student_t_em": {"ll": ll_hat, "n_em_iters": n_iters, "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2,
                     "eta_v": smooth_hat.eta_v, "eta_v_ratio": eta_ratio_hat,
                     "gamma_v": p_hat.gamma_v, "gamma_v_ratio": gamma_ratio_hat,
                     "params": {"sigma1": p_hat.sigma1, "sigma2": p_hat.sigma2, "sigma_xi": p_hat.sigma_xi,
                               "rho1x": p_hat.rho1x, "rho2x": p_hat.rho2x, "mu_xi": p_hat.mu_xi,
                               "mu_xi_star": p_hat.mu_xi_star, "lambda1": p_hat.lambda1, "lambda2": p_hat.lambda2,
                               "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c,
                                        "lambda_v": smooth_hat.lambda_v}}},
    "acceptance_1_ljung_box_squared": {"before": lb_before, "after": lb_after},
    "acceptance_2_mechanism": mech_results,
    "acceptance_3": {"total_gain": total_gain, "gain_from_resid2": gain_from_resid2, "gain_from_logF": gain_from_logF,
                    "mean_resid2_before": mean_resid2_before, "mean_resid2_after": mean_resid2_after,
                    "excess_kurt_before": excess_kurt_before, "excess_kurt_after": excess_kurt_after},
}
out_path = ROOT / "results" / "c1c18_frontspread_channel.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
