# %% [markdown]
# # Student-t EM refinement of the eta_v+gamma_v fit, then acceptance
#
# Warm-started from explore/45's clean Gaussian S3+eta_v+gamma_v optimum
# (both channels interior: eta_v at 22.9% of its bound, gamma_v at 17.1%
# -- not the explore/44 plateau). nu fixed at 5 for this EM (Fix 5's
# "Gaussian first" is satisfied by explore/45; this step reintroduces
# Student-t at the EXISTING fixed nu, matching STUDENT_T_DF everywhere
# else in this codebase -- it does not yet jointly estimate nu). A
# lightweight kurtosis-implied-nu diagnostic at the end checks whether the
# data wants nu to move, as an approximation to the full joint (nu, eta_v,
# gamma_v) MLE Fix 5 ultimately calls for.
#
# "Before" comparator throughout: results/c1c18_threefactor.json, the
# Student-t EM fit with NEITHER RV channel (gamma_v=eta_v=0) -- the exact
# model explore/42's Study A3 run used, and the one whose squared-
# innovation Ljung-Box (18/18 contracts significant at every lag, via
# explore/35's four-factor check, byte-for-byte identical to three-factor)
# and ll-decomposition (pooled resid_std^2 = 1.3761, via explore/30) are
# already on record. Recomputing "before" numbers fresh here rather than
# trusting those printouts, since the acceptance criteria below need
# exact, comparable figures under identical code paths.
#
# Acceptance criteria (all three, stated before looking):
#   1. Ljung-Box on squared standardised innovations clears (or at least
#      materially improves from the 18/18-significant baseline).
#   2. The Step-0 mechanism regression (innovation magnitude vs realised
#      vol, controlling for lagged realised vol) -- innovation should stop
#      surviving now that its own predictive content is inside the model.
#   3. ll decomposition: resid_std^2 should sit nearer 1.667 (the correct
#      t5 benchmark, not 1.0) than the pre-extension fit's 1.3761.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    ETA_V_MAX,
    GAMMA_V_MAX,
    STUDENT_T_DF,
    _raw_system3,
    fit3_student_t_smooth,
    pack3_smooth,
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

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")
diag = json.loads((ROOT / "results" / "c1c18_rv_channel_diagnostic.json").read_text())
both = diag["S3_eta_v_gamma_v"]
bp = both["params"]


def _params_from_dump(bp):
    p = ThreeFactorParams(
        kappa1=bp["kappa1"], kappa2=bp["kappa2"], sigma1=bp["sigma1"], sigma2=bp["sigma2"],
        sigma_xi=bp["sigma_xi"], rho12=0.0, rho1x=bp["rho1x"], rho2x=bp["rho2x"],
        mu_xi=bp["mu_xi"], mu_xi_star=bp["mu_xi_star"], lambda1=bp["lambda1"],
        lambda2=bp["lambda2"], gamma_v=bp["gamma_v"],
    )
    c = bp["curve"]
    smooth = SmoothMeasParams(a=c["a"], b=c["b"], c=c["c"], quadratic=True,
                              eta_v=c["eta_v"], lambda_v=c["lambda_v"])
    return p, smooth


p_warm, smooth_warm = _params_from_dump(bp)
print(f"warm start (Gaussian S3+eta_v+gamma_v): ll={both['ll']:.2f}  "
     f"k1={p_warm.kappa1:.4f}  k2={p_warm.kappa2:.4f}  "
     f"eta_v={smooth_warm.eta_v:.4f}  gamma_v={p_warm.gamma_v:.4f}")

# %% [markdown]
# ## Student-t EM from explore/45's Gaussian optimum

# %%
theta0 = pack3_smooth(p_warm, smooth_warm.a, smooth_warm.b, smooth_warm.c, quadratic=True,
                      eta_v=smooth_warm.eta_v, lambda_v=smooth_warm.lambda_v, rv_extension=True)
p_hat, smooth_hat, x_filt, resid_final, weight, n_iters, ll_hat = fit3_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF, theta0, quadratic=True, rv_lag=rv_lag, rv_extension=True)

eta_ratio = abs(smooth_hat.eta_v) / ETA_V_MAX
gamma_ratio = abs(p_hat.gamma_v) / GAMMA_V_MAX
print(f"\n{'=' * 90}\nSTUDENT-T EM RESULT (nu={STUDENT_T_DF} fixed)\n{'=' * 90}")
print(f"ll={ll_hat:.2f}  (Gaussian warm start ll={both['ll']:.2f}; "
     f"pre-extension Student-t ll=261977.28)")
print(f"n_em_iters={n_iters}")
print(f"k1={p_hat.kappa1:.4f}  k2={p_hat.kappa2:.4f}")
print(f"eta_v={smooth_hat.eta_v:.4f}  ({eta_ratio * 100:.1f}% of ETA_V_MAX={ETA_V_MAX}) "
     f"-- {'BINDING' if eta_ratio > 0.9 else 'interior'}")
print(f"gamma_v={p_hat.gamma_v:.4f}  ({gamma_ratio * 100:.1f}% of GAMMA_V_MAX={GAMMA_V_MAX}) "
     f"-- {'BINDING' if gamma_ratio > 0.9 else 'interior'}")
print(f"lambda_v={smooth_hat.lambda_v:.4f}  (fixed at explore/45's LAMBDA_V_FIXED=2.0, not refit here)")

# %% [markdown]
# ## Rebuild both filters via filter_uv_contributions (need ll_cell for the
# ## decomposition and per-contract residuals for Ljung-Box)

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


def _filter_with_contributions(p, smooth, rv_lag_arg):
    sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag_arg)
    x0, P0 = initial_state3(log_f, tau, p)
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
        np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
        np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))
    return ll, resid, ll_cell


ll_before, resid_before, ll_cell_before = _filter_with_contributions(p_before, smooth_before, None)
ll_after, resid_after, ll_cell_after = _filter_with_contributions(p_hat, smooth_hat, rv_lag)
print(f"\ncross-check: ll_before recomputed={ll_before:.2f} (saved: {before['ll']:.2f})  "
     f"ll_after recomputed={ll_after:.2f} (EM return: {ll_hat:.2f})")

# %% [markdown]
# ## Acceptance 1: Ljung-Box on squared standardised innovations

# %%
print(f"\n{'=' * 90}\nACCEPTANCE 1: LJUNG-BOX ON SQUARED INNOVATIONS\n{'=' * 90}")


def _ljungbox_squared(resid):
    rows = []
    for j in range(18):
        z = resid[:, j]
        z = z[np.isfinite(z)]
        lb = acorr_ljungbox(z ** 2, lags=LAGS, return_df=True)
        row = {"contract": j + 1}
        for lag in LAGS:
            row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
        rows.append(row)
    return rows


lb_before = _ljungbox_squared(resid_before)
lb_after = _ljungbox_squared(resid_after)
for lag in LAGS:
    sig_before = [r["contract"] for r in lb_before if r[f"lb_pvalue_{lag}"] < 0.01]
    sig_after = [r["contract"] for r in lb_after if r[f"lb_pvalue_{lag}"] < 0.01]
    print(f"lag={lag:3d}  before: {len(sig_before):2d}/18 significant   "
         f"after: {len(sig_after):2d}/18 significant   {sig_after}")

# %% [markdown]
# ## Acceptance 2: Step-0 mechanism regression, before vs after
#
# Recomputed fresh under identical code on both fits: |innovation| at
# C1/C2, lagged one day, regressed against forward realised_vol at each
# horizon, controlling for lagged trailing realised vol (raw, 20-day,
# front-contract -- the same physical quantity as explore/44's rv_lag
# before its log-standardisation, recomputed here rather than reusing the
# transformed driver so the control variable is in natural units).

# %%
print(f"\n{'=' * 90}\nACCEPTANCE 2: STEP-0 MECHANISM REGRESSION, BEFORE VS AFTER\n{'=' * 90}")

ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
RV_WINDOW = 20
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_control_lag = np.concatenate([[np.nan], rv_trailing[:-1]])


def _v_mag(resid, ll_cell):
    logF = -2 * ll_cell - LOG_2PI - resid ** 2
    F = np.exp(logF)
    v_raw = resid * np.sqrt(F)
    return np.sqrt(v_raw[:, 0] ** 2 + v_raw[:, 1] ** 2)


v_mag_before = _v_mag(resid_before, ll_cell_before)
v_mag_after = _v_mag(resid_after, ll_cell_after)
v_mag_before_lag = np.concatenate([[np.nan], v_mag_before[:-1]])
v_mag_after_lag = np.concatenate([[np.nan], v_mag_after[:-1]])

mech_results = {"before": {}, "after": {}}
for label, v_lag in (("before", v_mag_before_lag), ("after", v_mag_after_lag)):
    for h in HORIZONS:
        y = realised_vol(log_f, h)
        valid = np.isfinite(y) & np.isfinite(v_lag) & np.isfinite(rv_control_lag)
        X = sm.add_constant(np.column_stack([rv_control_lag[valid], v_lag[valid]]),
                            has_constant="add")
        res = sm.OLS(y[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
        mech_results[label][h] = {"n_obs": int(res.nobs), "coef_v_mag": float(res.params[2]),
                                  "p_v_mag": float(res.pvalues[2]), "r_squared": float(res.rsquared)}

print(f"{'h':>5s}  {'before: coef':>13s}{'p':>10s}{'R2':>8s}   {'after: coef':>13s}{'p':>10s}{'R2':>8s}")
for h in HORIZONS:
    b, a = mech_results["before"][h], mech_results["after"][h]
    survives_before = "SURVIVES" if b["p_v_mag"] < 0.01 else "gone"
    survives_after = "SURVIVES" if a["p_v_mag"] < 0.01 else "gone"
    print(f"{h:5d}  {b['coef_v_mag']:+13.4f}{b['p_v_mag']:10.4g}{b['r_squared']:8.4f} ({survives_before:>8s})  "
         f"{a['coef_v_mag']:+13.4f}{a['p_v_mag']:10.4g}{a['r_squared']:8.4f} ({survives_after:>8s})")

# %% [markdown]
# ## Acceptance 3: ll decomposition vs the t5 benchmark (1.667)

# %%
print(f"\n{'=' * 90}\nACCEPTANCE 3: LL DECOMPOSITION\n{'=' * 90}")
logF_before = -2.0 * ll_cell_before - LOG_2PI - resid_before ** 2
logF_after = -2.0 * ll_cell_after - LOG_2PI - resid_after ** 2
pooled_resid_sq_before = float(np.nanmean(resid_before ** 2))
pooled_resid_sq_after = float(np.nanmean(resid_after ** 2))
pooled_logF_before = float(np.nanmean(logF_before))
pooled_logF_after = float(np.nanmean(logF_after))
print(f"pooled resid_std^2:  before={pooled_resid_sq_before:.4f}  after={pooled_resid_sq_after:.4f}  "
     f"(t5 benchmark: 1.667; pre-extension explore/30 figure: 1.3761)")
print(f"pooled mean logF:    before={pooled_logF_before:.4f}  after={pooled_logF_after:.4f}  "
     f"(F ratio after/before = {np.exp(pooled_logF_after - pooled_logF_before):.4f})")
closer = "CLOSER to 1.667" if abs(pooled_resid_sq_after - 1.667) < abs(pooled_resid_sq_before - 1.667) else "NOT closer to 1.667"
print(f"verdict: {closer}")

# %% [markdown]
# ## Lightweight kurtosis-implied nu (approximation, not a joint MLE)
#
# Pearson kurtosis of a standard Student-t(nu) is 3 + 6/(nu-4) for nu>4.
# Solving for nu from the pooled sample's excess kurtosis gives a rough
# read on whether the data wants nu to move now that the RV channel is
# doing some of the tail-thinning work, WITHOUT actually refitting nu
# jointly with (eta_v, gamma_v) -- Fix 5's full request. Pooling across
# all 18 contracts' finite residuals into one sample; each contract's
# resid is already standardised to the model's own claimed scale, so
# pooling treats them as (approximately) exchangeable draws for this
# diagnostic only.

# %%
from scipy.stats import kurtosis as _kurtosis  # noqa: E402

pooled_before = resid_before[np.isfinite(resid_before)].ravel()
pooled_after = resid_after[np.isfinite(resid_after)].ravel()
excess_kurt_before = float(_kurtosis(pooled_before, fisher=True, bias=False))
excess_kurt_after = float(_kurtosis(pooled_after, fisher=True, bias=False))


def _implied_nu(excess_kurt):
    if excess_kurt <= 0:
        return float("inf")
    return 4.0 + 6.0 / excess_kurt


nu_before = _implied_nu(excess_kurt_before)
nu_after = _implied_nu(excess_kurt_after)
print(f"\n{'=' * 90}\nKURTOSIS-IMPLIED NU (APPROXIMATION -- not a joint MLE)\n{'=' * 90}")
print(f"before: excess kurtosis={excess_kurt_before:.4f}  implied nu={nu_before:.2f}")
print(f"after:  excess kurtosis={excess_kurt_after:.4f}  implied nu={nu_after:.2f}")
print(f"fixed nu used in both fits: {STUDENT_T_DF}")
if nu_after > STUDENT_T_DF and nu_after > nu_before:
    print("nu rose above 5 (or moved further above it) once the RV channel is "
         "active -- consistent with the channel doing genuine tail-thinning "
         "work, not just inflating variance.")
elif nu_after <= STUDENT_T_DF:
    print("nu stayed at or below 5 -- the RV channel looks like it is inflating "
         "variance without capturing the tails; a joint (nu, eta_v, gamma_v) "
         "MLE would be needed to settle this properly rather than guessing "
         "from a fixed-nu residual's kurtosis.")

# %%
out = {
    "student_t_em": {
        "ll": ll_hat, "n_em_iters": n_iters,
        "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2,
        "eta_v": smooth_hat.eta_v, "eta_v_ratio": eta_ratio,
        "gamma_v": p_hat.gamma_v, "gamma_v_ratio": gamma_ratio,
        "lambda_v": smooth_hat.lambda_v,
        "params": {"sigma1": p_hat.sigma1, "sigma2": p_hat.sigma2, "sigma_xi": p_hat.sigma_xi,
                  "rho1x": p_hat.rho1x, "rho2x": p_hat.rho2x, "mu_xi": p_hat.mu_xi,
                  "mu_xi_star": p_hat.mu_xi_star, "lambda1": p_hat.lambda1, "lambda2": p_hat.lambda2,
                  "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c}},
    },
    "acceptance_1_ljung_box_squared": {"before": lb_before, "after": lb_after},
    "acceptance_2_mechanism_regression": mech_results,
    "acceptance_3_ll_decomposition": {
        "pooled_resid_sq_before": pooled_resid_sq_before, "pooled_resid_sq_after": pooled_resid_sq_after,
        "pooled_logF_before": pooled_logF_before, "pooled_logF_after": pooled_logF_after,
        "t5_benchmark": 1.667,
    },
    "kurtosis_implied_nu": {"excess_kurt_before": excess_kurt_before, "excess_kurt_after": excess_kurt_after,
                           "nu_before": nu_before, "nu_after": nu_after, "fixed_nu": STUDENT_T_DF},
}
out_path = ROOT / "results" / "c1c18_rv_channel_em_and_acceptance.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
