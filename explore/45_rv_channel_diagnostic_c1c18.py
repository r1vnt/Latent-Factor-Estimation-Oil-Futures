# %% [markdown]
# # RV channel, done right: log-RV driver, bounded eta_v, H alone first
#
# Fixes 1-5 from the explore/44 post-mortem (identical convergence across
# every multi-start start was a clipped-plateau signature, not an
# identification success):
#
# 1. H alone first -- gamma_v held at EXACTLY 0 (not just initialised
#    there) while eta_v is fit. The evidence (innovation is an
#    observation-side object, front-concentrated, unfittable by more
#    state factors) names the measurement channel, not the state one.
# 2. One channel at a time, nested, 1 df per step: S3 -> S3+eta_v ->
#    S3+eta_v+gamma_v. lambda_v held fixed at 2.0 for the single-df
#    eta_v step (freeing it too would make that step 2 df, not 1) --
#    revisited only after eta_v itself is shown to behave.
# 3. log(RV), causally standardised (expanding window, not a fixed
#    trailing window that would need its own lookahead-free
#    justification) -- raw RV's right tail is exactly what drove the
#    unbounded version to its plateau.
# 4. eta_v = ETA_V_MAX*tanh(z), gamma_v = GAMMA_V_MAX*tanh(z) -- bounded
#    by construction (fit3.py), not by an exponent clip. Explicitly
#    checked below for whether the estimate sits at its bound.
# 5. Gaussian only here (nu off) -- Student-t and a conditional-variance
#    channel compete to explain the same fat tails, and fitting both at
#    once double-counts. A clean interior Gaussian estimate first; nu
#    only reconsidered afterward, in a later step, once this one is
#    trusted.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    ETA_V_MAX,
    GAMMA_V_MAX,
    _neg_loglik3,
    fit3_gaussian_smooth,
    pack3_smooth,
    unpack3_smooth,
)
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
RV_WINDOW = 20
LAMBDA_V_FIXED = 2.0   # held fixed for the 1-df eta_v-alone step
GAMMA_V_IDX = 11       # position of gamma_v's packed (arctanh-space) value

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

# %% [markdown]
# ## Fix 3: log(RV), causally (expanding-window) standardised

# %%
ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_causal = np.concatenate([[np.nan], rv_trailing[:-1]])   # lag 1 -- never contemporaneous
log_rv = np.where(rv_causal > 0, np.log(rv_causal), np.nan)

MIN_HISTORY = 60
rv_lag = np.zeros(n)   # 0 = inert (exp(eta_v*0)=1), not fabricated
running_sum, running_sumsq, count = 0.0, 0.0, 0
for t in range(n):
    if count >= MIN_HISTORY:
        mean = running_sum / count
        var = running_sumsq / count - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        if np.isfinite(log_rv[t]):
            rv_lag[t] = (log_rv[t] - mean) / std   # standardised using ONLY prior days
    if np.isfinite(log_rv[t]):
        running_sum += log_rv[t]
        running_sumsq += log_rv[t] ** 2
        count += 1

valid_days = int((rv_lag != 0).sum())
print(f"causal standardised log-RV: {valid_days}/{n} days active "
     f"(first {MIN_HISTORY}+{RV_WINDOW} days inert by construction)")
print(f"driver percentiles (active days): "
     f"p50={np.percentile(rv_lag[rv_lag != 0], 50):.3f}  "
     f"p95={np.percentile(rv_lag[rv_lag != 0], 95):.3f}  "
     f"p99={np.percentile(rv_lag[rv_lag != 0], 99):.3f}")

# %% [markdown]
# ## S3 baseline (Gaussian only, no RV extension)

# %%
baseline = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_init = ThreeFactorParams(
    kappa1=baseline["kappa1"], kappa2=baseline["kappa2"], sigma1=baseline["sigma1"],
    sigma2=baseline["sigma2"], sigma_xi=baseline["sigma_xi"], rho12=0.0,
    rho1x=baseline["rho1x"], rho2x=baseline["rho2x"], mu_xi=baseline["mu_xi"],
    mu_xi_star=baseline["mu_xi_star"], lambda1=baseline["lambda1"], lambda2=baseline["lambda2"],
    gamma_v=0.0,
)
c0 = baseline["curve"]
theta0_s3 = pack3_smooth(p_init, c0["a"], c0["b"], c0["c"], quadratic=True)
p_s3, smooth_s3, _, _, _, ll_s3 = fit3_gaussian_smooth(log_f, tau, DT, theta0_s3, quadratic=True)
print(f"\nS3 (Gaussian, no RV): ll={ll_s3:.2f}  k1={p_s3.kappa1:.4f}  k2={p_s3.kappa2:.4f}")

# %% [markdown]
# ## S3 + eta_v alone (gamma_v held at EXACTLY 0 -- Fix 1)

# %%
theta_full0 = pack3_smooth(p_s3, smooth_s3.a, smooth_s3.b, smooth_s3.c, quadratic=True,
                           eta_v=0.5, lambda_v=LAMBDA_V_FIXED, rv_extension=True)
reduced_idx = [i for i in range(len(theta_full0)) if i != GAMMA_V_IDX]


def _neg_loglik_eta_only(theta_reduced):
    theta_full = np.zeros(len(theta_full0))
    theta_full[reduced_idx] = theta_reduced
    theta_full[GAMMA_V_IDX] = 0.0   # gamma_v forced to exactly 0, every call
    return _neg_loglik3(theta_full, log_f, tau, DT, True, rv_lag=rv_lag, rv_extension=True)


theta0_reduced = theta_full0[reduced_idx]
res_eta = minimize(_neg_loglik_eta_only, theta0_reduced, method="BFGS",
                   options={"maxiter": 2000, "gtol": 1e-8})
theta_eta_full = np.zeros(len(theta_full0))
theta_eta_full[reduced_idx] = res_eta.x
theta_eta_full[GAMMA_V_IDX] = 0.0
p_eta, smooth_eta = unpack3_smooth(theta_eta_full, quadratic=True, rv_extension=True)
ll_eta = -res_eta.fun

print(f"\nS3+eta_v (gamma_v FIXED at 0): ll={ll_eta:.2f}  "
     f"k1={p_eta.kappa1:.4f}  k2={p_eta.kappa2:.4f}")
print(f"eta_v={smooth_eta.eta_v:.4f}  (lambda_v fixed at {LAMBDA_V_FIXED})  "
     f"gamma_v={p_eta.gamma_v:.6f} (must be 0)")
print(f"binding check: |eta_v|/ETA_V_MAX = {abs(smooth_eta.eta_v) / ETA_V_MAX:.4f}  "
     f"({'BINDING -- report as a constraint, not an estimate' if abs(smooth_eta.eta_v) / ETA_V_MAX > 0.9 else 'interior'})")

# Interpretable units: variance multiple at percentiles of the driver.
front_weight = np.exp(-smooth_eta.lambda_v * tau[0, 0])   # front contract's own tau
for pct, label in [(50, "p50"), (95, "p95"), (99, "p99")]:
    d = np.percentile(rv_lag[rv_lag != 0], pct)
    multiple = np.exp(smooth_eta.eta_v * front_weight * d)
    print(f"  variance multiple at driver {label}={d:.3f}: {multiple:.3f}x")

LR_eta = 2.0 * (ll_eta - ll_s3)
p_eta_lr = float(chi2.sf(LR_eta, df=1))
print(f"LR(S3+eta_v vs S3) = {LR_eta:.3f}  df=1  p={p_eta_lr:.4g}")

# %% [markdown]
# ## S3 + eta_v + gamma_v (both free -- only meaningful if eta_v alone was clean)

# %%
theta0_both = pack3_smooth(p_eta, smooth_eta.a, smooth_eta.b, smooth_eta.c, quadratic=True,
                           eta_v=smooth_eta.eta_v, lambda_v=LAMBDA_V_FIXED, rv_extension=True)
p_both, smooth_both, _, _, _, ll_both = fit3_gaussian_smooth(
    log_f, tau, DT, theta0_both, quadratic=True, rv_lag=rv_lag, rv_extension=True)

print(f"\nS3+eta_v+gamma_v (both free): ll={ll_both:.2f}  "
     f"k1={p_both.kappa1:.4f}  k2={p_both.kappa2:.4f}")
print(f"eta_v={smooth_both.eta_v:.4f}  gamma_v={p_both.gamma_v:.4f}")
print(f"binding check: |eta_v|/MAX={abs(smooth_both.eta_v) / ETA_V_MAX:.4f}  "
     f"|gamma_v|/MAX={abs(p_both.gamma_v) / GAMMA_V_MAX:.4f}")

LR_both = 2.0 * (ll_both - ll_eta)
p_both_lr = float(chi2.sf(LR_both, df=1))
print(f"LR(S3+eta_v+gamma_v vs S3+eta_v) = {LR_both:.3f}  df=1  p={p_both_lr:.4g}")

# %%
def _dump_params(p, smooth):
    return {"kappa1": p.kappa1, "kappa2": p.kappa2, "sigma1": p.sigma1, "sigma2": p.sigma2,
           "sigma_xi": p.sigma_xi, "rho1x": p.rho1x, "rho2x": p.rho2x, "mu_xi": p.mu_xi,
           "mu_xi_star": p.mu_xi_star, "lambda1": p.lambda1, "lambda2": p.lambda2,
           "gamma_v": p.gamma_v,
           "curve": {"a": smooth.a, "b": smooth.b, "c": smooth.c,
                    "eta_v": smooth.eta_v, "lambda_v": smooth.lambda_v}}


out = {
    "rv_window": RV_WINDOW, "min_history": MIN_HISTORY, "lambda_v_fixed": LAMBDA_V_FIXED,
    "S3": {"ll": ll_s3, "kappa1": p_s3.kappa1, "kappa2": p_s3.kappa2,
          "params": _dump_params(p_s3, smooth_s3)},
    "S3_eta_v": {"ll": ll_eta, "eta_v": smooth_eta.eta_v, "kappa1": p_eta.kappa1,
                "kappa2": p_eta.kappa2, "LR_vs_S3": LR_eta, "p_vs_S3": p_eta_lr,
                "binding": abs(smooth_eta.eta_v) / ETA_V_MAX > 0.9,
                "params": _dump_params(p_eta, smooth_eta)},
    "S3_eta_v_gamma_v": {"ll": ll_both, "eta_v": smooth_both.eta_v, "gamma_v": p_both.gamma_v,
                         "LR_vs_eta_v_only": LR_both, "p_vs_eta_v_only": p_both_lr,
                         "eta_binding": abs(smooth_both.eta_v) / ETA_V_MAX > 0.9,
                         "gamma_binding": abs(p_both.gamma_v) / GAMMA_V_MAX > 0.9,
                         "params": _dump_params(p_both, smooth_both)},
}
out_path = ROOT / "results" / "c1c18_rv_channel_diagnostic.json"
out_path.write_text(json.dumps(out, indent=2))
np.save(ROOT / "data" / "c1c18_log_rv_causal.npy", rv_lag)
print(f"\nsaved to {out_path}")
