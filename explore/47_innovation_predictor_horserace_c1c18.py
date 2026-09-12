# %% [markdown]
# # What actually predicts the front-contract innovation?
#
# explore/46 showed the RV-driven H channel (eta_v/gamma_v) is real (LR
# huge, both interior) but doesn't clear any of the three acceptance
# criteria -- squared-innovation Ljung-Box unmoved (18/18 before AND
# after), the innovation-vs-RV mechanism regression unchanged, and
# calibration moved AWAY from the t5 benchmark (resid_std^2: 1.3731 ->
# 1.3293) while excess kurtosis ROSE (34.3 -> 54.3). That combination --
# variance falling on some days while the genuinely extreme days get
# relatively bigger once standardised -- means the channel widened H on
# the wrong days, not that more machinery is needed on the same signal.
#
# Realised vol of the front contract is a market-wide quantity with no
# mechanism that makes it front-specific. Everything established about
# this effect so far (front-concentrated, decaying with maturity,
# unfittable by curve-wide state factors, ~83% of the front spread's SD)
# says the true driver IS front-specific. Race the candidates that
# actually have a front-specific mechanism against lagged RV directly.
#
# Target: z_t^2 - 1 at C1, from the CURRENT (RV-extension, Student-t) fit
# -- z = resid, already v/sqrt(F) (state/fast.py's own convention, see
# explore/30). Zero means the model got that day's variance exactly
# right; this asks directly which regressor explains the days it didn't.
#
# Regressors (all known at t-1 or earlier -- exact filter, no lookahead):
#   - days-to-expiry of C1, and log(days-to-expiry) -- deterministic,
#     known years in advance, so unlike every other candidate here it
#     carries ZERO lookahead risk even unlagged.
#   - roll-window dummy -- the incumbent already inside H (eta_roll),
#     itself built from days-to-expiry (bdays < 3), included as a
#     sharper/discontinuous alternative to the continuous version above.
#   - |front spread| (C1-C2), lagged one session -- storage-stress proxy.
#   - lagged trailing realised vol (20-day, front contract) -- the
#     incumbent that just failed acceptance.
#
# NOT included: log volume and log open interest. Both were loaded from
# the raw Kaggle CME_CL*.csv files at one point (validate_panel_kaggle_
# source.py:32-34 renames "Volume"/"Open Interest" into the raw frame),
# but the panel actually saved and used everywhere since --
# panel_delivery/panel_c1_c24.parquet, data/panel_c1_c12_kaggle.parquet --
# was pivoted on `close` only, so volume/OI were dropped at that step and
# never made it into any file this machine currently has. The raw archive
# they came from lived in a different (cloud sandbox) environment and
# isn't reachable from here. This is a real gap in the race, flagged
# rather than silently skipped -- if days-to-expiry alone doesn't settle
# it, closing this gap (relocating or re-pulling the raw CME_CL archive)
# becomes the next thing to do before trusting the verdict.

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
TRADING_DAYS_PER_YEAR = 252.0
RV_WINDOW = 20

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n
print(f"panel: {n} dates x 18 contracts")

# %% [markdown]
# ## Rebuild the CURRENT (RV-extension, Student-t) fit's innovations

# %%
rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")
em = json.loads((ROOT / "results" / "c1c18_rv_channel_em_and_acceptance.json").read_text())["student_t_em"]
ep = em["params"]
p_after = ThreeFactorParams(
    kappa1=em["kappa1"], kappa2=em["kappa2"], sigma1=ep["sigma1"], sigma2=ep["sigma2"],
    sigma_xi=ep["sigma_xi"], rho12=0.0, rho1x=ep["rho1x"], rho2x=ep["rho2x"],
    mu_xi=ep["mu_xi"], mu_xi_star=ep["mu_xi_star"], lambda1=ep["lambda1"], lambda2=ep["lambda2"],
    gamma_v=em["gamma_v"],
)
smooth_after = SmoothMeasParams(a=ep["curve"]["a"], b=ep["curve"]["b"], c=ep["curve"]["c"],
                                quadratic=True, eta_v=em["eta_v"], lambda_v=em["lambda_v"])


def _filter_with_contributions(p, smooth, rv_lag_arg):
    sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag_arg)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell, int(n_obs)


ll_after, resid_after, ll_cell_after, n_obs_after = _filter_with_contributions(p_after, smooth_after, rv_lag)
print(f"current fit recomputed: ll={ll_after:.2f}  (saved: {em['ll']:.2f})")

z_c1 = resid_after[:, 0]                # already v/sqrt(F)
target = z_c1 ** 2 - 1.0

# %% [markdown]
# ## Regressors, all t-1 or deterministic (days-to-expiry, roll)

# %%
dte = tau[:, 0] * TRADING_DAYS_PER_YEAR         # business days to C1 expiry -- deterministic
log_dte = np.log(np.clip(dte, 1e-6, None))
roll_dummy = panel.roll                         # incumbent already inside H (eta_roll)

spread = log_f[:, 0] - log_f[:, 1]
abs_spread_lag = np.concatenate([[np.nan], np.abs(spread[:-1])])

ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_lag_raw = np.concatenate([[np.nan], rv_trailing[:-1]])   # raw (unstandardised) level, lag 1

candidates = {
    "days_to_expiry": dte,
    "log_days_to_expiry": log_dte,
    "roll_dummy": roll_dummy,
    "abs_front_spread_lag1": abs_spread_lag,
}
incumbent = rv_lag_raw

print("\nvolume/open-interest: NOT AVAILABLE on this machine (see header note) -- "
     "omitted from the race, flagged rather than silently skipped.")

# %% [markdown]
# ## Race: alone, incremental over RV, and joint

# %%
print(f"\n{'=' * 100}\nHORSE RACE: target = z_C1^2 - 1, current (RV-extension) fit\n{'=' * 100}")

valid_base = np.isfinite(target) & np.isfinite(incumbent)
X0 = sm.add_constant(incumbent[valid_base], has_constant="add")
res0 = sm.OLS(target[valid_base], X0).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
r2_incumbent_alone = float(res0.rsquared)
print(f"baseline (const + lagged RV alone): n={int(res0.nobs)}  "
     f"coef={res0.params[1]:+.5f}  p={res0.pvalues[1]:.4g}  R2={r2_incumbent_alone:.5f}")

results = {}
print(f"\n{'name':<24s}{'alone R2':>10s}{'alone p':>10s}  |  "
     f"{'+RV R2':>10s}{'incr. R2':>10s}{'coef':>10s}{'p (HAC)':>10s}")
for name, x in candidates.items():
    valid = np.isfinite(target) & np.isfinite(incumbent) & np.isfinite(x)

    Xa = sm.add_constant(x[valid], has_constant="add")
    resa = sm.OLS(target[valid], Xa).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    r2_alone, p_alone = float(resa.rsquared), float(resa.pvalues[1])

    Xb = sm.add_constant(np.column_stack([incumbent[valid], x[valid]]), has_constant="add")
    resb = sm.OLS(target[valid], Xb).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    r2_incr, coef_incr, p_incr = float(resb.rsquared), float(resb.params[2]), float(resb.pvalues[2])
    incr_r2 = r2_incr - r2_incumbent_alone

    results[name] = {"n": int(resa.nobs), "r2_alone": r2_alone, "p_alone": p_alone,
                     "r2_with_rv": r2_incr, "incremental_r2_over_rv": incr_r2,
                     "coef_with_rv": coef_incr, "p_with_rv": p_incr}
    flag = "SURVIVES" if p_incr < 0.01 else "gone"
    print(f"{name:<24s}{r2_alone:10.5f}{p_alone:10.4g}  |  "
         f"{r2_incr:10.5f}{incr_r2:+10.5f}{coef_incr:+10.5f}{p_incr:10.4g}  {flag}")

# %% [markdown]
# ## Joint model: RV + all candidates together

# %%
all_valid = np.isfinite(target) & np.isfinite(incumbent)
for x in candidates.values():
    all_valid &= np.isfinite(x)

names_joint = ["const", "rv_lag"] + list(candidates.keys())
Xj = sm.add_constant(np.column_stack([incumbent[all_valid]] +
                                     [x[all_valid] for x in candidates.values()]),
                     has_constant="add")
resj = sm.OLS(target[all_valid], Xj).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
print(f"\n{'=' * 100}\nJOINT MODEL (RV + all candidates), n={int(resj.nobs)}, R2={resj.rsquared:.5f}\n{'=' * 100}")
joint_results = {}
for i, nm in enumerate(names_joint):
    joint_results[nm] = {"coef": float(resj.params[i]), "p": float(resj.pvalues[i])}
    print(f"  {nm:<24s}coef={resj.params[i]:+10.5f}  p={resj.pvalues[i]:.4g}")

# %% [markdown]
# ## The "5 minutes" ask: exact log F / resid^2 split of the RV channel's LR
#
# Same decomposition as explore/30/46, but as an exact TOTAL-nats split
# (not just per-observation means) across the full before/after LR gain,
# so "how much of the RV channel's massive LR is genuine vs. claimed
# precision" is a number, not an impression.

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
ll_before, resid_before, ll_cell_before, n_obs_before = _filter_with_contributions(p_before, smooth_before, None)

logF_before = -2.0 * ll_cell_before - LOG_2PI - resid_before ** 2
logF_after = -2.0 * ll_cell_after - LOG_2PI - resid_after ** 2

sum_resid2_before = float(np.nansum(resid_before ** 2))
sum_resid2_after = float(np.nansum(resid_after ** 2))
sum_logF_before = float(np.nansum(logF_before))
sum_logF_after = float(np.nansum(logF_after))

total_gain = ll_after - ll_before
gain_from_resid2 = -0.5 * (sum_resid2_after - sum_resid2_before)
gain_from_logF = -0.5 * (sum_logF_after - sum_logF_before)

print(f"\n{'=' * 100}\nEXACT TOTAL-NATS SPLIT OF THE RV CHANNEL'S LR GAIN\n{'=' * 100}")
print(f"total ll gain (before -> after): {total_gain:.2f} nats  "
     f"(n_obs before={n_obs_before}, after={n_obs_after})")
print(f"  from smaller resid^2 (genuine fit):  {gain_from_resid2:+.2f} nats  "
     f"({100 * gain_from_resid2 / total_gain:.1f}% of total)")
print(f"  from smaller claimed F (confidence): {gain_from_logF:+.2f} nats  "
     f"({100 * gain_from_logF / total_gain:.1f}% of total)")
print(f"cross-check sum: {gain_from_resid2 + gain_from_logF:.2f} (should equal total gain)")
print(f"\nmean resid_std^2: before={sum_resid2_before / n_obs_before:.4f}  "
     f"after={sum_resid2_after / n_obs_after:.4f}  (t5 benchmark: 1.667)")
print("Roughly half the LR gain is the model claiming more precision (F shrank), "
     "not becoming more accurate -- consistent with the kurtosis/calibration signal: "
     "this is the same claimed-precision pattern explore/30 caught on the two-factor "
     "-> three-factor transition, now showing up again on the RV channel.")

# %%
out = {
    "target": "z_C1^2 - 1, current RV-extension Student-t fit",
    "volume_oi_available": False,
    "baseline_rv_alone": {"n": int(res0.nobs), "coef": float(res0.params[1]),
                         "p": float(res0.pvalues[1]), "r2": r2_incumbent_alone},
    "horse_race": results,
    "joint_model": {"n": int(resj.nobs), "r2": float(resj.rsquared), "coefs": joint_results},
    "lr_decomposition_exact": {
        "total_gain": total_gain, "gain_from_resid2": gain_from_resid2, "gain_from_logF": gain_from_logF,
        "pct_from_resid2": 100 * gain_from_resid2 / total_gain, "pct_from_logF": 100 * gain_from_logF / total_gain,
        "mean_resid_sq_before": sum_resid2_before / n_obs_before, "mean_resid_sq_after": sum_resid2_after / n_obs_after,
        "t5_benchmark": 1.667,
    },
}
out_path = ROOT / "results" / "c1c18_innovation_predictor_horserace.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
