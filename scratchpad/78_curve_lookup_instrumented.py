# %% [markdown]
# # Instrumented variant: same as 78_curve_lookup_fast_warmstart, but with
# # a per-outer-iteration print inside the Student-t EM loop (reimplemented
# # inline, since fit4_front_student_t has no callback/verbose hook), so we
# # can actually see convergence instead of waiting blind for up to 200
# # outer iterations with zero visibility.

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(r"C:\Users\bradl\Documents\GitHub\Latent-Factor-Estimation-Oil-Futures")
sys.path.insert(0, str(ROOT / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF, student_t_weights3  # noqa: E402
from csr.pricing.fit4_front import (_neg_loglik4_front, _neg_loglik4_front_weighted,  # noqa: E402
                                    pack4_front, unpack4_front, weighted_filter4_front)
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402

DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]
c0 = before["curve"]
print(f"S3 baseline curve: a={c0['a']:.4f}  b={c0['b']:.4f}  c={c0['c']:.4f}", flush=True)
print(f"multistart record ll={best['ll']:.2f}", flush=True)

p_init = FourFactorFrontParams(
    kappa2=best["kappa2_hat"], kappa3=best["kappa3_hat"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta0 = pack4_front(p_init, c0["a"], c0["b"], c0["c"], quadratic=True)

t_start = time.time()
print("Gaussian warm start: BFGS only, skipping the Nelder-Mead fallback", flush=True)
res = minimize(_neg_loglik4_front, theta0, args=(log_f, tau, DT, True, KAPPA_F_Q_ANCHOR),
              method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
print(f"BFGS success={res.success}  ll={-res.fun:.2f}  nit={res.nit}  "
     f"elapsed={time.time()-t_start:.0f}s", flush=True)
p_g, smooth_g = unpack4_front(res.x, True, KAPPA_F_Q_ANCHOR)

abc_g = fit_smooth_to_vector(sigma_from_tau(smooth_g, tau[0]), tau[0], quadratic=True)
theta0_em = pack4_front(p_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True)

# %% [markdown]
# ## Student-t EM, reimplemented inline with a per-iteration print

# %%
print("\nStarting instrumented Student-t EM", flush=True)
weight = np.ones_like(log_f)
theta = np.asarray(theta0_em, float).copy()
p, smooth = unpack4_front(theta, True, KAPPA_F_Q_ANCHOR)
N_OUTER = 200
TOL = 1e-6
it = 0
t_em_start = time.time()
for it in range(N_OUTER):
    t_iter = time.time()
    res = minimize(_neg_loglik4_front_weighted, theta,
                   args=(log_f, tau, DT, weight, True, KAPPA_F_Q_ANCHOR),
                   method="BFGS", options={"maxiter": 300})
    theta = res.x
    p, smooth = unpack4_front(theta, True, KAPPA_F_Q_ANCHOR)
    ll_now, _, resid_raw, _ = weighted_filter4_front(p, smooth, log_f, tau, DT, np.ones_like(log_f))
    new_weight = student_t_weights3(resid_raw, STUDENT_T_DF)
    shift = float(np.nanmax(np.abs(new_weight - weight)))
    weight = new_weight
    print(f"  outer {it+1:3d}:  inner_nit={res.nit:4d}  ll={ll_now:12.2f}  "
         f"weight_shift={shift:.3e}  iter_time={time.time()-t_iter:6.1f}s  "
         f"total_em_time={time.time()-t_em_start:7.1f}s", flush=True)
    if shift < TOL:
        print(f"  CONVERGED at outer iteration {it+1} (shift {shift:.3e} < tol {TOL})", flush=True)
        break
else:
    print(f"  did NOT converge within {N_OUTER} outer iterations (last shift {shift:.3e})", flush=True)

ll_fit, x_filt, resid_raw, n_obs = weighted_filter4_front(p, smooth, log_f, tau, DT, np.ones_like(log_f))
n_iters = it + 1
smooth_hat = smooth
p_hat = p
df_final = STUDENT_T_DF

print(f"\nconverged in {n_iters} EM iterations  ll={ll_fit:.2f}  (multistart record: {best['ll']:.2f})", flush=True)
print(f"kappa2={p_hat.kappa2:.4f} (record {best['kappa2_hat']:.4f})  "
     f"kappa3={p_hat.kappa3:.4f} (record {best['kappa3_hat']:.4f})  "
     f"kappa_f_P={p_hat.kappa_f_P:.4f} (record {best['kappa_f_P_hat']:.4f})  "
     f"sigma_f={p_hat.sigma_f:.4f} (record {best['sigma_f_hat']:.4f})", flush=True)
print(f"\nunrestricted 4F-front curve: a={smooth_hat.a:.4f}  b={smooth_hat.b:.4f}  c={smooth_hat.c:.4f}", flush=True)

# %% [markdown]
# ## Guard: refuse to report curve numbers if this didn't reconverge to the
# ## established optimum

# %%
LL_TOL = 1.0
if abs(ll_fit - best["ll"]) > LL_TOL:
    raise RuntimeError(
        f"Reconverged ll={ll_fit:.2f} does not match the multistart record "
        f"{best['ll']:.2f} (diff={ll_fit - best['ll']:+.2f}, tol={LL_TOL}) -- this did NOT land "
        f"on the established rung-2 optimum. Refusing to report curve numbers from the wrong point.")
print(f"ll cross-check OK: reconverged ll={ll_fit:.2f} matches the record to within {LL_TOL} nat.", flush=True)


# %% [markdown]
# ## Compare s(tau) = exp(a + b*tau + c*tau^2) implied by each curve, at C1 and C18

# %%
def _s_bp(a, b, c, t):
    return float(np.exp(a + b * t + c * t * t)) * 1e4


tau_c1 = float(np.median(tau[np.isfinite(tau[:, 0]), 0]))
tau_c18 = float(np.median(tau[np.isfinite(tau[:, 17]), 17]))
print(f"\n{'=' * 90}\ns(tau) IMPLIED BY EACH CURVE (bp)\n{'=' * 90}", flush=True)
print(f"{'':12s}{'a':>10s}{'b':>10s}{'c':>10s}{'s(C1)':>10s}{'s(C18)':>10s}", flush=True)
print(f"{'S3':12s}{c0['a']:10.4f}{c0['b']:10.4f}{c0['c']:10.4f}"
     f"{_s_bp(c0['a'], c0['b'], c0['c'], tau_c1):10.2f}{_s_bp(c0['a'], c0['b'], c0['c'], tau_c18):10.2f}", flush=True)
print(f"{'4F-front':12s}{smooth_hat.a:10.4f}{smooth_hat.b:10.4f}{smooth_hat.c:10.4f}"
     f"{_s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c1):10.2f}"
     f"{_s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c18):10.2f}", flush=True)

delta_a, delta_b, delta_c = smooth_hat.a - c0["a"], smooth_hat.b - c0["b"], smooth_hat.c - c0["c"]
print(f"\ndelta: a={delta_a:+.4f}  b={delta_b:+.4f}  c={delta_c:+.4f}", flush=True)

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}", flush=True)
s_c1_before, s_c1_after = _s_bp(c0["a"], c0["b"], c0["c"], tau_c1), _s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c1)
s_c18_before, s_c18_after = _s_bp(c0["a"], c0["b"], c0["c"], tau_c18), _s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c18)
pct_c1 = 100 * (s_c1_after / s_c1_before - 1)
pct_c18 = 100 * (s_c18_after / s_c18_before - 1)
print(f"s(C1) moved {pct_c1:+.1f}%  ({s_c1_before:.2f}bp -> {s_c1_after:.2f}bp)", flush=True)
print(f"s(C18) moved {pct_c18:+.1f}%  ({s_c18_before:.2f}bp -> {s_c18_after:.2f}bp)", flush=True)
if abs(pct_c1) > 15 and abs(pct_c18) > 15:
    print("The curve moved materially at BOTH ends -- consistent with the proposed mechanism: "
         "relieving s(tau) at the front, through the shared 3-parameter curve, changes it at the "
         "back too, with a loading-free pathway for the back-end gain.", flush=True)
elif abs(pct_c1) < 5 and abs(pct_c18) < 5:
    print("The curve barely moved at either end -- the proposed s(tau)-relief mechanism is NOT "
         "supported. 'Generic flexibility' is the honest residual description of the back-end gain.", flush=True)
else:
    print("Partial/asymmetric movement -- inspect delta_a/b/c directly before drawing a conclusion.", flush=True)

# %%
out = {
    "s3_curve": {"a": c0["a"], "b": c0["b"], "c": c0["c"]},
    "fourfactor_front_curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
    "delta": {"a": delta_a, "b": delta_b, "c": delta_c},
    "s_bp_c1": {"before": s_c1_before, "after": s_c1_after, "pct_change": pct_c1},
    "s_bp_c18": {"before": s_c18_before, "after": s_c18_after, "pct_change": pct_c18},
    "ll_reconverged": ll_fit, "ll_record": best["ll"],
}
out_path = ROOT / "results" / "c1c18_fourfactor_front_curve_lookup.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}", flush=True)
