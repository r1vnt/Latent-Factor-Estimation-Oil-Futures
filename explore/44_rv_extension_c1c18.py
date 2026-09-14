# %% [markdown]
# # Step 1: RV-driven Q (gamma_v) + H (eta_v, lambda_v) extension on C1-C18
#
# RV series: front contract, trailing 20-day window, lagged 1 session --
# never contemporaneous (same discipline already enforced for rv_lag
# elsewhere: a contemporaneous RV makes the state prior a function of the
# data being explained). Multi-start over eta_v in {0,1,3} x lambda_v in
# {0.5,2,6} (9 Gaussian-stage starts), gamma_v free throughout (already
# part of theta regardless of the RV extension), best start carried into
# the Student-t EM stage.

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import (  # noqa: E402
    STUDENT_T_DF,
    fit3_gaussian_smooth,
    fit3_student_t_smooth,
    pack3_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
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
# ## Trailing, lagged RV series (front contract, 20-day window)

# %%
ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_lag = np.concatenate([[np.nan], rv_trailing[:-1]])
rv_lag = np.nan_to_num(rv_lag, nan=0.0)   # warm-up days: inert (exp(x*0)=1), not fabricated
print(f"rv_lag: mean={np.mean(rv_lag[rv_lag > 0]):.4f}  "
     f"valid days={int((rv_lag > 0).sum())}/{n}")

# %% [markdown]
# ## Warm start from the current three-factor baseline; multi-start eta_v/lambda_v

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
print(f"baseline: k1={p_init.kappa1:.4f}  k2={p_init.kappa2:.4f}  ll={baseline['ll']:.2f}")

ETA_V_STARTS = (0.0, 1.0, 3.0)
LAMBDA_V_STARTS = (0.5, 2.0, 6.0)

print(f"\n{'=' * 90}\nMULTI-START (Gaussian stage), eta_v x lambda_v\n{'=' * 90}")
starts = []
for eta_v0 in ETA_V_STARTS:
    for lambda_v0 in LAMBDA_V_STARTS:
        t0 = time.time()
        theta0 = pack3_smooth(p_init, c0["a"], c0["b"], c0["c"], quadratic=True,
                              eta_v=eta_v0, lambda_v=lambda_v0, rv_extension=True)
        p_g, smooth_g, _, _, _, ll_g = fit3_gaussian_smooth(
            log_f, tau, DT, theta0, quadratic=True, rv_lag=rv_lag, rv_extension=True)
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] eta_v0={eta_v0}  lambda_v0={lambda_v0}  "
             f"ll={ll_g:12.2f}  k1={p_g.kappa1:.4f}  k2={p_g.kappa2:.4f}  "
             f"gamma_v={p_g.gamma_v:.4f}  eta_v={smooth_g.eta_v:.4f}  "
             f"lambda_v={smooth_g.lambda_v:.4f}  ({dt:.1f}s)")
        starts.append({"eta_v0": eta_v0, "lambda_v0": lambda_v0, "ll": ll_g,
                       "p": p_g, "smooth": smooth_g})

best = max(starts, key=lambda s: s["ll"])
print(f"\nbest start: eta_v0={best['eta_v0']}  lambda_v0={best['lambda_v0']}  ll={best['ll']:.2f}")
ll_spread = max(s["ll"] for s in starts) - min(s["ll"] for s in starts)
print(f"ll spread across starts: {ll_spread:.4f}")

# %% [markdown]
# ## Student-t EM from the best start

# %%
print(f"\n{'=' * 90}\nSTUDENT-T EM\n{'=' * 90}")
abc_g = fit_smooth_to_vector(sigma_from_tau(best["smooth"], tau[0]), tau[0], quadratic=True)
theta0_em = pack3_smooth(best["p"], abc_g.a, abc_g.b, abc_g.c, quadratic=True,
                         eta_v=best["smooth"].eta_v, lambda_v=best["smooth"].lambda_v,
                         rv_extension=True)
t0 = time.time()
p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll, _ = fit3_student_t_smooth(
    log_f, tau, DT, STUDENT_T_DF, theta0_em, n_outer=200, quadratic=True,
    rv_lag=rv_lag, rv_extension=True)
dt = time.time() - t0
converged = n_iters < 200

print(f"[{time.strftime('%H:%M:%S')}] done ({dt:.1f}s, {n_iters}/200 EM iters, "
     f"{'CONVERGED' if converged else 'HIT CAP'})")
print(f"kappa1={p_hat.kappa1:.4f}  kappa2={p_hat.kappa2:.4f}")
print(f"gamma_v={p_hat.gamma_v:.4f}  eta_v={smooth_hat.eta_v:.4f}  lambda_v={smooth_hat.lambda_v:.4f}")
print(f"curve: a={smooth_hat.a:.4f}  b={smooth_hat.b:.4f}  c={smooth_hat.c:.4f}")
print(f"ll = {ll:.2f}  (baseline three-factor ll was {baseline['ll']:.2f}, "
     f"gain={ll - baseline['ll']:.2f})")

# %%
out = {
    "rv_window": RV_WINDOW,
    "multi_start": [{"eta_v0": s["eta_v0"], "lambda_v0": s["lambda_v0"], "ll": s["ll"]}
                    for s in starts],
    "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2, "gamma_v": p_hat.gamma_v,
    "eta_v": smooth_hat.eta_v, "lambda_v": smooth_hat.lambda_v,
    "sigma1": p_hat.sigma1, "sigma2": p_hat.sigma2, "sigma_xi": p_hat.sigma_xi,
    "rho1x": p_hat.rho1x, "rho2x": p_hat.rho2x,
    "mu_xi": p_hat.mu_xi, "mu_xi_star": p_hat.mu_xi_star,
    "lambda1": p_hat.lambda1, "lambda2": p_hat.lambda2,
    "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
    "ll": ll, "ll_baseline": baseline["ll"], "n_em_iters": n_iters,
    "converged": converged, "seconds": dt,
}
out_path = ROOT / "results" / "c1c18_rv_extension.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")

np.save(ROOT / "data" / "c1c18_rv_lag20.npy", rv_lag)
print(f"saved rv_lag series to data/c1c18_rv_lag20.npy for downstream acceptance checks")
