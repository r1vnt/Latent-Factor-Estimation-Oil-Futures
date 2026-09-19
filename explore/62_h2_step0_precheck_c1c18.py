# %% [markdown]
# # H2 Step 0: two pre-checks before any fitting
#
# News volume rises when the market moves -- a significant gamma_n could
# mean "news intensity forecasts volatility" or "journalists write more
# about volatile markets," and volatility clusters, so lagging I_t by one
# session does not by itself break the second reading. The real test is
# not gamma_n != 0, it's does news intensity add anything beyond lagged
# realised volatility, which already occupies the same channel via
# gamma_v. Two checks, both cheap, both required before spending a fit:
#
#   1. corr(I_{t-1}, RV_{t-1}) -- if intensity and lagged RV move
#      together, gamma_n and gamma_v are fighting over the same variance
#      in the same channel, the exact joint-identification failure that
#      produced the clip plateau (explore/44).
#   2. Incremental content: regress squared standardised innovation on
#      I_{t-1}, controlling for RV_{t-1} -- the same Step-0 logic already
#      used and validated (explore/43, explore/47). If intensity carries
#      nothing beyond lagged RV here, gamma_n will fit to SOMETHING in
#      the eventual MLE but it won't be news.
#
# Driver: data/stage4_daily.parquet's `intensity` column -- log1p(I_t),
# trailing-252-row causal standardisation, already built and tested
# (commit 2e56c34, tests/test_aggregate_stage4.py). NOT n_t (the ~60-doc
# scored count that already feeds R_t/news_variance) -- I_t is the full
# ~1500-doc post-dedup match count, a structurally different quantity.
# Lagged one session here for the pre-checks themselves (the eventual
# fit's own driver construction will re-lag from the raw series).

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
RV_WINDOW = 20

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
dates = panel.dates
n = panel.n
print(f"panel: {n} dates x 18 contracts, {dates.min().date()} -> {dates.max().date()}")

# %% [markdown]
# ## Align news intensity to the panel, lag one session

# %%
news = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet").set_index("date")
news_aligned = news.reindex(dates)
intensity_t = news_aligned["intensity"].to_numpy(dtype=float)
intensity_lag = np.concatenate([[np.nan], intensity_t[:-1]])   # I_{t-1}
n_news_valid = int(np.isfinite(intensity_lag).sum())
print(f"news intensity aligned: {n_news_valid}/{n} dates have a valid I_{{t-1}} "
     f"(news coverage starts {news.index.min().date()}, panel starts {dates.min().date()})")

# %% [markdown]
# ## Lagged trailing realised vol (front contract), same construction used throughout

# %%
ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_lag_raw = np.concatenate([[np.nan], rv_trailing[:-1]])   # RV_{t-1}, raw (unstandardised) level

# %% [markdown]
# ## Check 1: corr(I_{t-1}, RV_{t-1})

# %%
valid = np.isfinite(intensity_lag) & np.isfinite(rv_lag_raw)
corr = float(np.corrcoef(intensity_lag[valid], rv_lag_raw[valid])[0, 1])
print(f"\n{'=' * 90}\nCHECK 1: corr(I_t-1, RV_t-1)\n{'=' * 90}")
print(f"n={int(valid.sum())}  corr = {corr:+.4f}")
if abs(corr) > 0.5:
    print("HIGH collinearity -- gamma_n and gamma_v would be fighting over the same variance. "
         "Proceed with caution, expect a weak/unstable joint fit even if each channel alone looks fine.")
elif abs(corr) > 0.25:
    print("MODERATE collinearity -- not disqualifying, but expect some shared explanatory power; "
         "the incremental-content check (Check 2) matters more than usual here.")
else:
    print("LOW collinearity -- intensity and lagged RV are largely independent signals; the two "
         "channels have room to be separately identified.")

# %% [markdown]
# ## Check 2: incremental content -- squared standardised innovation ~ I_t-1 | RV_t-1

# %%
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
sys3 = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)
ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
print(f"\nS3 baseline recomputed: ll={ll:.2f} (saved {before['ll']:.2f})")

z_c1 = resid[:, 0]   # already standardised, C1 -- same target construction as every prior mechanism check
target = z_c1 ** 2 - 1.0

print(f"\n{'=' * 90}\nCHECK 2: does I_t-1 predict z_C1^2-1, controlling for RV_t-1?\n{'=' * 90}")
valid2 = np.isfinite(target) & np.isfinite(rv_lag_raw) & np.isfinite(intensity_lag)
n2 = int(valid2.sum())

X_rv_only = sm.add_constant(rv_lag_raw[valid2], has_constant="add")
res_rv_only = sm.OLS(target[valid2], X_rv_only).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
r2_rv_only = float(res_rv_only.rsquared)
print(f"baseline (const + RV_t-1 alone): n={n2}  R2={r2_rv_only:.5f}")

X_both = sm.add_constant(np.column_stack([rv_lag_raw[valid2], intensity_lag[valid2]]), has_constant="add")
res_both = sm.OLS(target[valid2], X_both).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
r2_both = float(res_both.rsquared)
coef_intensity = float(res_both.params[2])
p_intensity = float(res_both.pvalues[2])
incr_r2 = r2_both - r2_rv_only
print(f"const + RV_t-1 + I_t-1: n={n2}  R2={r2_both:.5f}  incremental R2={incr_r2:+.5f}")
print(f"I_t-1 coefficient: {coef_intensity:+.5f}  p={p_intensity:.4g}")
survives = p_intensity < 0.01
print(f"\nI_t-1 {'SURVIVES' if survives else 'does NOT survive'} controlling for RV_t-1 (p<0.01 threshold)")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nSTEP 0 VERDICT\n{'=' * 90}")
if survives and abs(corr) < 0.5:
    print("PROCEED -- intensity carries incremental content beyond lagged RV, and collinearity is "
         "low enough that gamma_n and gamma_v should be separately identifiable. Move to Steps 1-3.")
elif survives and abs(corr) >= 0.5:
    print("PROCEED WITH CAUTION -- intensity carries incremental content, but is also fairly "
         "collinear with RV. Expect the joint (gamma_n, gamma_v) fit to be less stable than each "
         "channel alone; watch for one absorbing the other's role during optimisation.")
else:
    print("DO NOT PROCEED to the full fit -- intensity shows no incremental content over lagged RV "
         "at this stage. gamma_n would fit to something in an unconstrained MLE, but Step 0 already "
         "says it would not be news. H2 is answerable now, negatively, without spending a fit on it.")

# %%
out = {
    "n_news_valid": n_news_valid, "n_panel": n,
    "check1_corr_intensity_rv": corr,
    "check2": {"n": n2, "r2_rv_only": r2_rv_only, "r2_both": r2_both, "incremental_r2": incr_r2,
              "coef_intensity": coef_intensity, "p_intensity": p_intensity, "survives": bool(survives)},
}
out_path = ROOT / "results" / "c1c18_h2_step0_precheck.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
