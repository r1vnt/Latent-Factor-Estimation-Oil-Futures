# %% [markdown]
# # C1's excess residual behaviour, split by regime not magnitude
#
# explore/47 found |front spread|_{t-1} explains R^2=0.173 of z_C1^2-1 (the
# squared standardised innovation, current RV-extension fit) -- a
# magnitude relationship. The storage-bound/convenience-yield story makes
# a SIGN prediction instead: backwardation (spread_{t-1}>0, front premium,
# physical tightness) should show materially worse excess behaviour than
# contango (spread_{t-1}<0, storage abundant, convenience yield floored),
# not just "bigger spread, bigger residual" symmetrically in both
# directions. No refitting -- same target/model as explore/47, split by
# sign(spread_{t-1}) instead of |spread_{t-1}|.

# %%
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

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

import json  # noqa: E402
em = json.loads((ROOT / "results" / "c1c18_rv_channel_em_and_acceptance.json").read_text())["student_t_em"]
ep = em["params"]
p = ThreeFactorParams(
    kappa1=em["kappa1"], kappa2=em["kappa2"], sigma1=ep["sigma1"], sigma2=ep["sigma2"],
    sigma_xi=ep["sigma_xi"], rho12=0.0, rho1x=ep["rho1x"], rho2x=ep["rho2x"],
    mu_xi=ep["mu_xi"], mu_xi_star=ep["mu_xi_star"], lambda1=ep["lambda1"], lambda2=ep["lambda2"],
    gamma_v=em["gamma_v"],
)
smooth = SmoothMeasParams(a=ep["curve"]["a"], b=ep["curve"]["b"], c=ep["curve"]["c"], quadratic=True,
                          eta_v=em["eta_v"], lambda_v=em["lambda_v"])
rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")

sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag)
x0, P0 = initial_state3(log_f, tau, p)
ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
print(f"cross-check: ll={ll:.2f}  (saved {em['ll']:.2f})")

z_c1 = resid[:, 0]
target = z_c1 ** 2 - 1.0

spread = log_f[:, 0] - log_f[:, 1]
spread_lag = np.concatenate([[np.nan], spread[:-1]])
abs_spread_lag = np.abs(spread_lag)

# %% [markdown]
# ## Split by regime: backwardation (spread_t-1 > 0) vs contango (< 0)

# %%
backwardation = spread_lag > 0
contango = spread_lag < 0
valid = np.isfinite(target) & np.isfinite(spread_lag)

print(f"\n{'=' * 90}\nC1 EXCESS BEHAVIOUR BY REGIME (target = z_C1^2 - 1)\n{'=' * 90}")
for label, mask in [("backwardation (spread_t-1>0)", backwardation), ("contango (spread_t-1<0)", contango)]:
    m = mask & valid
    n = int(m.sum())
    mean_target = float(np.mean(target[m]))
    mean_abs_spread = float(np.mean(abs_spread_lag[m]))
    print(f"{label:32s}  n={n:5d}  mean(z^2-1)={mean_target:+.4f}  mean|spread|={mean_abs_spread * 1e4:.2f}bp")

# %% [markdown]
# ## Formal test: interaction of regime with |spread| magnitude

# %%
regime_dummy = backwardation.astype(float)   # 1 = backwardation, 0 = contango
interact = abs_spread_lag * regime_dummy
X = sm.add_constant(np.column_stack([abs_spread_lag[valid], regime_dummy[valid], interact[valid]]),
                    has_constant="add")
res = sm.OLS(target[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
print(f"\n{'=' * 90}\nINTERACTION REGRESSION: target ~ |spread| + backwardation + |spread|*backwardation\n{'=' * 90}")
print(f"n={int(res.nobs)}  R2={res.rsquared:.4f}")
print(f"|spread| coef (contango slope):        {res.params[1]:+.4f}  p={res.pvalues[1]:.4g}")
print(f"backwardation dummy (level shift):     {res.params[2]:+.4f}  p={res.pvalues[2]:.4g}")
print(f"|spread| x backwardation (slope diff): {res.params[3]:+.4f}  p={res.pvalues[3]:.4g}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
interact_coef, interact_p = float(res.params[3]), float(res.pvalues[3])
if interact_p < 0.01 and interact_coef > 0:
    print(f"CONFIRMED: the |spread|-residual relationship is materially STEEPER in backwardation "
         f"(interaction coef={interact_coef:+.4f}, p={interact_p:.4g}) -- the storage-bound "
         f"signature. Front misspecification is worse specifically when physical tightness binds, "
         f"not symmetrically in both directions.")
else:
    print(f"NOT CONFIRMED: interaction coef={interact_coef:+.4f}, p={interact_p:.4g} -- the "
         f"relationship does not differ significantly by regime at this threshold. The excess "
         f"residual behaviour appears roughly symmetric in |spread|, not regime-specific.")

# %% [markdown]
# ## Robustness: the top |target| days are all April 2020 (the storage-glut
# ## negative-price episode) and every one of them falls in the contango
# ## bucket by construction (front collapsed relative to back). Re-run
# ## excluding that window, and add a rank-based (Spearman) check that is
# ## insensitive to the magnitude of a handful of extreme days, matching
# ## the project's established discipline (explore/56's absolute-loss/sign
# ## tests existed for exactly this reason).

# %%
from scipy.stats import spearmanr  # noqa: E402

CRASH_START, CRASH_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30")
dates = pd.to_datetime(pd.Series(panel.dates if hasattr(panel, "dates") else raw18.index[-log_f.shape[0]:]))
crash_mask = (dates >= CRASH_START) & (dates <= CRASH_END)
ex_crash = valid & ~crash_mask.to_numpy()

print(f"\n{'=' * 90}\nROBUSTNESS: EXCLUDING 2020-03-01..2020-06-30 (n_dropped={int((valid & crash_mask.to_numpy()).sum())})\n{'=' * 90}")
for label, mask in [("backwardation", backwardation), ("contango", contango)]:
    m = mask & ex_crash
    n = int(m.sum())
    print(f"{label:16s}  n={n:5d}  mean(z^2-1)={float(np.mean(target[m])):+.4f}")

X_ex = sm.add_constant(np.column_stack([abs_spread_lag[ex_crash], regime_dummy[ex_crash], interact[ex_crash]]),
                       has_constant="add")
res_ex = sm.OLS(target[ex_crash], X_ex).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
print(f"\nex-crash interaction regression: n={int(res_ex.nobs)}  R2={res_ex.rsquared:.4f}")
print(f"|spread| coef (contango slope):        {res_ex.params[1]:+.4f}  p={res_ex.pvalues[1]:.4g}")
print(f"backwardation dummy (level shift):     {res_ex.params[2]:+.4f}  p={res_ex.pvalues[2]:.4g}")
print(f"|spread| x backwardation (slope diff): {res_ex.params[3]:+.4f}  p={res_ex.pvalues[3]:.4g}")

rho_back, p_back = spearmanr(abs_spread_lag[backwardation & ex_crash], target[backwardation & ex_crash])
rho_cont, p_cont = spearmanr(abs_spread_lag[contango & ex_crash], target[contango & ex_crash])
print(f"\nSpearman rank correlation (|spread| vs target), ex-crash:")
print(f"  backwardation: rho={rho_back:+.4f}  p={p_back:.4g}  n={int((backwardation & ex_crash).sum())}")
print(f"  contango:      rho={rho_cont:+.4f}  p={p_cont:.4g}  n={int((contango & ex_crash).sum())}")

interact_ex_coef, interact_ex_p = float(res_ex.params[3]), float(res_ex.pvalues[3])
print(f"\n{'=' * 90}\nROBUSTNESS VERDICT\n{'=' * 90}")
if np.sign(interact_ex_coef) != np.sign(interact_coef) or interact_ex_p > 0.10:
    print(f"The full-sample interaction (coef={interact_coef:+.4f}, p={interact_p:.4g}) does NOT survive "
         f"excluding the April 2020 crash window (ex-crash coef={interact_ex_coef:+.4f}, p={interact_ex_p:.4g}) "
         f"-- it was an artefact of that episode dominating the contango bucket, not a genuine regime effect. "
         f"This matches the project's other caught 'large effect driven entirely by extreme days' patterns.")
else:
    print(f"The interaction survives excluding the crash window (ex-crash coef={interact_ex_coef:+.4f}, "
         f"p={interact_ex_p:.4g}, full-sample coef={interact_coef:+.4f}) -- not solely a crash-week artefact.")

# %%
out = {
    "n_backwardation": int((backwardation & valid).sum()), "n_contango": int((contango & valid).sum()),
    "mean_target_backwardation": float(np.mean(target[backwardation & valid])),
    "mean_target_contango": float(np.mean(target[contango & valid])),
    "interaction_coef": interact_coef, "interaction_p": interact_p,
    "contango_slope": float(res.params[1]), "contango_slope_p": float(res.pvalues[1]),
    "backwardation_dummy": float(res.params[2]), "backwardation_dummy_p": float(res.pvalues[2]),
    "r_squared": float(res.rsquared), "n_obs": int(res.nobs),
    "ex_crash_window": {
        "crash_start": str(CRASH_START.date()), "crash_end": str(CRASH_END.date()),
        "n_dropped": int((valid & crash_mask.to_numpy()).sum()),
        "interaction_coef": interact_ex_coef, "interaction_p": interact_ex_p,
        "contango_slope": float(res_ex.params[1]), "contango_slope_p": float(res_ex.pvalues[1]),
        "backwardation_dummy": float(res_ex.params[2]), "backwardation_dummy_p": float(res_ex.pvalues[2]),
        "r_squared": float(res_ex.rsquared), "n_obs": int(res_ex.nobs),
        "spearman_backwardation": {"rho": float(rho_back), "p": float(p_back)},
        "spearman_contango": {"rho": float(rho_cont), "p": float(p_cont)},
    },
}
out_path = ROOT / "results" / "c1c18_c1_backwardation_contango_split.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
