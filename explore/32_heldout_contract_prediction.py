# %% [markdown]
# # Held-out-contract prediction: fit C1-C12, predict C13-C18
#
# The test immune to both open issues (calibration bookkeeping, Davies
# problem): fit each model on C1-C12 only, use the filtered state at each
# date to predict C13-C18's price via that model's own A(tau)/loadings at
# those unseen maturities, compare to the actual (held-out) price. In-
# sample likelihood can be inflated by claimed precision; out-of-sample
# prediction on maturities the fit never saw cannot.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit import (  # noqa: E402
    STUDENT_T_DF,
    fit_s0_gaussian_smooth,
    fit_s0_student_t_smooth,
    pack_smooth,
)
from csr.pricing.fit3 import (  # noqa: E402
    fit3_gaussian_smooth,
    fit3_student_t_smooth,
    pack3_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import A_tau3, ThreeFactorParams  # noqa: E402
from csr.pricing.twofactor import A_tau, TwoFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x 18 contracts")

log_f_fit = panel.log_f[:, :12]
tau_fit = panel.tau[:, :12]
log_f_held = panel.log_f[:, 12:18]
tau_held = panel.tau[:, 12:18]

# %% [markdown]
# ## Fit two-factor smooth quadratic on C1-C12

# %%
two_factor = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s2 = two_factor["structural"]
p2_init = TwoFactorParams(kappa=s2["kappa"], sigma_chi=s2["sigma_chi"],
                          sigma_xi=s2["sigma_xi"], rho=s2["rho"], mu_xi=s2["mu_xi"],
                          mu_xi_star=s2["mu_xi_star"], lambda_chi=s2["lambda_chi"],
                          sigma_meas=np.array([0.005]))
c2 = two_factor["curve"]
theta0_2 = pack_smooth(p2_init, c2["a"], c2["b"], c2["c"], quadratic=True)
p2_g, _, _, _, _, _ = fit_s0_gaussian_smooth(log_f_fit, tau_fit, DT, theta0_2, quadratic=True)
abc2_g = fit_smooth_to_vector(p2_g.sigma_meas[0], tau_fit[0], quadratic=True)
p2_hat, _, _, _, n_iters2 = fit_s0_student_t_smooth(
    log_f_fit, tau_fit, DT, STUDENT_T_DF,
    theta0=pack_smooth(p2_g, abc2_g.a, abc2_g.b, abc2_g.c, quadratic=True),
    quadratic=True, n_outer=200)
print(f"two-factor on C1-C12: kappa={p2_hat.kappa:.4f}  n_iters={n_iters2}")

# %% [markdown]
# ## Fit three-factor smooth quadratic on C1-C12

# %%
three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p3_init = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
)
c3 = three_factor["curve"]
theta0_3 = pack3_smooth(p3_init, c3["a"], c3["b"], c3["c"], quadratic=True)
p3_g, smooth3_g, _, _, _, _ = fit3_gaussian_smooth(log_f_fit, tau_fit, DT, theta0_3, quadratic=True)
abc3_g = fit_smooth_to_vector(sigma_from_tau(smooth3_g, tau_fit[0]), tau_fit[0], quadratic=True)
p3_hat, smooth3_hat, x_filt3, _, _, n_iters3, ll3_fit, _ = fit3_student_t_smooth(
    log_f_fit, tau_fit, DT, STUDENT_T_DF,
    theta0=pack3_smooth(p3_g, abc3_g.a, abc3_g.b, abc3_g.c, quadratic=True),
    n_outer=200, quadratic=True)
print(f"three-factor on C1-C12: k1={p3_hat.kappa1:.4f}  k2={p3_hat.kappa2:.4f}  "
     f"n_iters={n_iters3}")

# %% [markdown]
# ## Get two-factor's filtered state too (need x_filt, not just p_hat)

# %%
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
ll2_fit, x_filt2, P_filt2, resid2_fit, n_obs2 = run_spec(
    REGISTRY["S0"], p2_hat, log_f_fit, tau_fit, DT)

# %% [markdown]
# ## Predict C13-C18 from each model's filtered state, compare to actual

# %%
print(f"\n{'=' * 90}\nHELD-OUT PREDICTION: C13-C18, from C1-C12 filtered state\n{'=' * 90}")

# Two-factor: chi = x_filt2[:,0], xi = x_filt2[:,1]
chi2 = x_filt2[:, 0]
xi2 = x_filt2[:, 1]
pred2 = np.empty_like(log_f_held)
for j in range(6):
    pred2[:, j] = (A_tau(p2_hat, tau_held[:, j])
                  + np.exp(-p2_hat.kappa * tau_held[:, j]) * chi2 + xi2)
err2 = log_f_held - pred2

# Three-factor: chi1 = x_filt3[:,0], chi2f = x_filt3[:,1], xi = x_filt3[:,2]
chi1_3 = x_filt3[:, 0]
chi2_3 = x_filt3[:, 1]
xi3 = x_filt3[:, 2]
pred3 = np.empty_like(log_f_held)
for j in range(6):
    pred3[:, j] = (A_tau3(p3_hat, tau_held[:, j])
                  + np.exp(-p3_hat.kappa1 * tau_held[:, j]) * chi1_3
                  + np.exp(-p3_hat.kappa2 * tau_held[:, j]) * chi2_3 + xi3)
err3 = log_f_held - pred3

print(f"{'contract':10s}{'2F RMSE(bp)':>14s}{'3F RMSE(bp)':>14s}{'ratio(3F/2F)':>14s}")
for j in range(6):
    rmse2_j = np.sqrt(np.nanmean(err2[:, j] ** 2)) * 1e4
    rmse3_j = np.sqrt(np.nanmean(err3[:, j] ** 2)) * 1e4
    print(f"C{13 + j:<9d}{rmse2_j:14.3f}{rmse3_j:14.3f}{rmse3_j / rmse2_j:14.4f}")

rmse2_pooled = np.sqrt(np.nanmean(err2 ** 2)) * 1e4
rmse3_pooled = np.sqrt(np.nanmean(err3 ** 2)) * 1e4
print(f"\n{'pooled':10s}{rmse2_pooled:14.3f}{rmse3_pooled:14.3f}"
     f"{rmse3_pooled / rmse2_pooled:14.4f}")

print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if rmse3_pooled < rmse2_pooled:
    print(f"THREE-FACTOR PREDICTS HELD-OUT MATURITIES BETTER "
         f"({rmse3_pooled:.2f}bp vs {rmse2_pooled:.2f}bp, "
         f"{100 * (1 - rmse3_pooled / rmse2_pooled):.1f}% lower RMSE) -- "
         f"the second mean-reverting factor captures real curve structure, "
         f"not just in-sample noise.")
else:
    print(f"TWO-FACTOR PREDICTS HELD-OUT MATURITIES BETTER OR EQUAL "
         f"({rmse2_pooled:.2f}bp vs {rmse3_pooled:.2f}bp) -- the second "
         f"factor's in-sample gain does not transfer to unseen maturities; "
         f"consistent with it partly absorbing noise rather than structure.")

# %%
out = {
    "two_factor_c1c12": {"kappa": p2_hat.kappa},
    "three_factor_c1c12": {"kappa1": p3_hat.kappa1, "kappa2": p3_hat.kappa2},
    "rmse_bp": {"two_factor": rmse2_pooled, "three_factor": rmse3_pooled,
               "per_contract_2F": (np.sqrt(np.nanmean(err2 ** 2, axis=0)) * 1e4).tolist(),
               "per_contract_3F": (np.sqrt(np.nanmean(err3 ** 2, axis=0)) * 1e4).tolist()},
}
(ROOT / "results" / "c1c18_heldout_contract_prediction.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_heldout_contract_prediction.json")
