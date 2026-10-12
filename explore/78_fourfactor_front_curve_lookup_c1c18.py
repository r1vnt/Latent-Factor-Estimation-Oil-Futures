# %% [markdown]
# # Lookup (not a fit): does the smooth measurement curve a,b,c move between
# # S3 and the unrestricted front-4-factor fit?
#
# Candidate mechanism for the 75.8%-surviving back-end gain in the
# kappa2/kappa3-pinned refit: the 4th factor relieves s(tau) at the
# front (explore/67's Roll diagnostic already showed s(tau) was
# absorbing front misspecification, not noise). s(tau) is ONE 3-
# parameter curve spanning all 18 maturities, so if the 4th factor lets
# it stop contorting for the front, the back end changes too, through a
# channel with literally no factor loading there. The curve's optimum
# was never saved in c1c18_fourfactor_front_multistart.json -- this
# re-derives it by warm-starting the EM from the ALREADY-ESTABLISHED
# optimal non-curve parameters (kappa2_hat, kappa3_hat, sigma_f_hat,
# kappa_f_P_hat, all fixed facts from explore/70, stable across 5
# multi-starts) and letting only the curve (plus everything else) settle
# -- a re-materialisation of an already-unique optimum, not a new
# search.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF  # noqa: E402
from csr.pricing.fit4_front import fit4_front_gaussian, fit4_front_student_t, pack4_front  # noqa: E402
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402

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
ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]
c0 = before["curve"]
print(f"S3 baseline curve: a={c0['a']:.4f}  b={c0['b']:.4f}  c={c0['c']:.4f}")

p_init = FourFactorFrontParams(
    kappa2=best["kappa2_hat"], kappa3=best["kappa3_hat"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta0 = pack4_front(p_init, c0["a"], c0["b"], c0["c"], quadratic=True)

# Gaussian warm start first (matches explore/70's/74's/75's established
# procedure exactly) -- skipping straight to the Student-t EM from the
# S3 curve as initial guess landed 901 nats below the multistart record
# on the first attempt (kappa2/kappa3/kappa_f_P/sigma_f all drifted from
# their recorded values too) -- a clear sign of the wrong local point,
# not the established optimum. Caught via the ll cross-check before
# reporting any curve numbers from it.
p_g, smooth_g, _, _, _, ll_g = fit4_front_gaussian(log_f, tau, DT, theta0, quadratic=True)
abc_g = fit_smooth_to_vector(sigma_from_tau(smooth_g, tau[0]), tau[0], quadratic=True)
theta0_em = pack4_front(p_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True)
p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll_fit, df_final = fit4_front_student_t(
    log_f, tau, DT, STUDENT_T_DF, theta0_em, quadratic=True)

print(f"\nGaussian warm start: ll={ll_g:.2f}")
print(f"converged in {n_iters} EM iterations  ll={ll_fit:.2f}  (multistart record: {best['ll']:.2f})")
print(f"kappa2={p_hat.kappa2:.4f} (record {best['kappa2_hat']:.4f})  "
     f"kappa3={p_hat.kappa3:.4f} (record {best['kappa3_hat']:.4f})  "
     f"kappa_f_P={p_hat.kappa_f_P:.4f} (record {best['kappa_f_P_hat']:.4f})  "
     f"sigma_f={p_hat.sigma_f:.4f} (record {best['sigma_f_hat']:.4f})")
print(f"\nunrestricted 4F-front curve: a={smooth_hat.a:.4f}  b={smooth_hat.b:.4f}  c={smooth_hat.c:.4f}")

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
print(f"ll cross-check OK: reconverged ll={ll_fit:.2f} matches the record to within {LL_TOL} nat.")


# %% [markdown]
# ## Compare s(tau) = exp(a + b*tau + c*tau^2) implied by each curve, at C1 and C18

# %%
def _s_bp(a, b, c, t):
    return float(np.exp(a + b * t + c * t * t)) * 1e4


tau_c1 = float(np.median(tau[np.isfinite(tau[:, 0]), 0]))
tau_c18 = float(np.median(tau[np.isfinite(tau[:, 17]), 17]))
print(f"\n{'=' * 90}\ns(tau) IMPLIED BY EACH CURVE (bp)\n{'=' * 90}")
print(f"{'':12s}{'a':>10s}{'b':>10s}{'c':>10s}{'s(C1)':>10s}{'s(C18)':>10s}")
print(f"{'S3':12s}{c0['a']:10.4f}{c0['b']:10.4f}{c0['c']:10.4f}"
     f"{_s_bp(c0['a'], c0['b'], c0['c'], tau_c1):10.2f}{_s_bp(c0['a'], c0['b'], c0['c'], tau_c18):10.2f}")
print(f"{'4F-front':12s}{smooth_hat.a:10.4f}{smooth_hat.b:10.4f}{smooth_hat.c:10.4f}"
     f"{_s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c1):10.2f}"
     f"{_s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c18):10.2f}")

delta_a, delta_b, delta_c = smooth_hat.a - c0["a"], smooth_hat.b - c0["b"], smooth_hat.c - c0["c"]
print(f"\ndelta: a={delta_a:+.4f}  b={delta_b:+.4f}  c={delta_c:+.4f}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
s_c1_before, s_c1_after = _s_bp(c0["a"], c0["b"], c0["c"], tau_c1), _s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c1)
s_c18_before, s_c18_after = _s_bp(c0["a"], c0["b"], c0["c"], tau_c18), _s_bp(smooth_hat.a, smooth_hat.b, smooth_hat.c, tau_c18)
pct_c1 = 100 * (s_c1_after / s_c1_before - 1)
pct_c18 = 100 * (s_c18_after / s_c18_before - 1)
print(f"s(C1) moved {pct_c1:+.1f}%  ({s_c1_before:.2f}bp -> {s_c1_after:.2f}bp)")
print(f"s(C18) moved {pct_c18:+.1f}%  ({s_c18_before:.2f}bp -> {s_c18_after:.2f}bp)")
if abs(pct_c1) > 15 and abs(pct_c18) > 15:
    print("The curve moved materially at BOTH ends -- consistent with the proposed mechanism: "
         "relieving s(tau) at the front, through the shared 3-parameter curve, changes it at the "
         "back too, with a loading-free pathway for the back-end gain.")
elif abs(pct_c1) < 5 and abs(pct_c18) < 5:
    print("The curve barely moved at either end -- the proposed s(tau)-relief mechanism is NOT "
         "supported. 'Generic flexibility' is the honest residual description of the back-end gain.")
else:
    print("Partial/asymmetric movement -- inspect delta_a/b/c directly before drawing a conclusion.")

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
print(f"\nsaved to {out_path}")
