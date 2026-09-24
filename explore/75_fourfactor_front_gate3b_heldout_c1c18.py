# %% [markdown]
# # Gate 3b: the direct test -- fit C5-C18 (blind to the front), predict C1-C4
#
# The mirror of Gate 3a, and per the user, "the one I'd actually report":
# this factor's claim is specifically about the front, so the honest test
# fits it on data where its own loading is negligible everywhere
# (kappa_f_Q=12.3 gives ~0.01 loading by C5) and asks whether it still
# helps predict the four contracts it never saw. Both 3F and 4F-front
# fit fresh on C5-C18 -- not reusing the C1-C12 fits, since this is a
# different split entirely.
#
# Note going in: chi_f is only weakly identified from C5-C18 data alone
# (its own likelihood contribution there is small by construction) --
# if sigma_f gets driven toward zero, that is itself an informative
# result (this factor's apparent value requires seeing the front to
# calibrate against), not a bug.

# %%
import json
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
from csr.pricing.fit4_front import (  # noqa: E402
    fit4_front_gaussian,
    fit4_front_student_t,
    pack4_front,
)
from csr.pricing.fourfactor_front import A_tau4_front, FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.threefactor import A_tau3, ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f_fit, tau_fit = panel.log_f[:, 4:], panel.tau[:, 4:]      # C5-C18, 14 contracts
log_f_held, tau_held = panel.log_f[:, :4], panel.tau[:, :4]    # C1-C4, held out

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]
c0 = before["curve"]

# %% [markdown]
# ## Three-factor, fresh fit on C5-C18

# %%
p3_init = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
)
theta0_3 = pack3_smooth(p3_init, c0["a"], c0["b"], c0["c"], quadratic=True)
p3_g, smooth3_g, _, _, _, ll3_g = fit3_gaussian_smooth(log_f_fit, tau_fit, DT, theta0_3, quadratic=True)
abc3_g = fit_smooth_to_vector(sigma_from_tau(smooth3_g, tau_fit[0]), tau_fit[0], quadratic=True)
p3_hat, smooth3_hat, x_filt3, _, _, n_iters3, ll3_fit, _ = fit3_student_t_smooth(
    log_f_fit, tau_fit, DT, STUDENT_T_DF,
    theta0=pack3_smooth(p3_g, abc3_g.a, abc3_g.b, abc3_g.c, quadratic=True), quadratic=True)
print(f"3F on C5-C18: ll={ll3_fit:.2f}  k1={p3_hat.kappa1:.4f}  k2={p3_hat.kappa2:.4f}")

# %% [markdown]
# ## Front-4-factor, fresh fit on C5-C18

# %%
p4_init = FourFactorFrontParams(
    kappa2=best["kappa2_hat"], kappa3=best["kappa3_hat"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
theta0_4 = pack4_front(p4_init, c0["a"], c0["b"], c0["c"], quadratic=True)
p4_g, smooth4_g, _, _, _, ll4_g = fit4_front_gaussian(log_f_fit, tau_fit, DT, theta0_4, quadratic=True)
print(f"Gaussian 4F-front on C5-C18: ll={ll4_g:.2f}  kappa_f_P={p4_g.kappa_f_P:.4f}  sigma_f={p4_g.sigma_f:.4f}")

abc4_g = fit_smooth_to_vector(sigma_from_tau(smooth4_g, tau_fit[0]), tau_fit[0], quadratic=True)
theta0_4em = pack4_front(p4_g, abc4_g.a, abc4_g.b, abc4_g.c, quadratic=True)
p4_hat, smooth4_hat, x_filt4, _, _, n_iters4, ll4_fit, _ = fit4_front_student_t(
    log_f_fit, tau_fit, DT, STUDENT_T_DF, theta0_4em, quadratic=True)
print(f"Student-t 4F-front on C5-C18: ll={ll4_fit:.2f}  n_iters={n_iters4}  "
     f"kappa_f_P={p4_hat.kappa_f_P:.4f}  sigma_f={p4_hat.sigma_f:.4f}")
print(f"NOTE: if sigma_f collapsed toward ~0 relative to its full-panel value "
     f"({best['sigma_f_hat']:.4f}), chi_f is weakly identified from C5-C18 alone -- "
     f"expected, per the module docstring, not a bug.")

# %% [markdown]
# ## Predict C1-C4 from each model's filtered state

# %%
chi1_3, chi2_3, xi3 = x_filt3[:, 0], x_filt3[:, 1], x_filt3[:, 2]
pred3 = np.empty_like(log_f_held)
for j in range(4):
    pred3[:, j] = (A_tau3(p3_hat, tau_held[:, j])
                  + np.exp(-p3_hat.kappa1 * tau_held[:, j]) * chi1_3
                  + np.exp(-p3_hat.kappa2 * tau_held[:, j]) * chi2_3 + xi3)
err3 = log_f_held - pred3

chi_f4, chi2_4, chi3_4, xi4 = x_filt4[:, 0], x_filt4[:, 1], x_filt4[:, 2], x_filt4[:, 3]
pred4 = np.empty_like(log_f_held)
for j in range(4):
    pred4[:, j] = (A_tau4_front(p4_hat, tau_held[:, j])
                  + np.exp(-p4_hat.kappa_f_Q * tau_held[:, j]) * chi_f4
                  + np.exp(-p4_hat.kappa2 * tau_held[:, j]) * chi2_4
                  + np.exp(-p4_hat.kappa3 * tau_held[:, j]) * chi3_4 + xi4)
err4 = log_f_held - pred4

print(f"\n{'=' * 90}\nGATE 3b: HELD-OUT PREDICTION, C1-C4 FROM C5-C18 (BLIND TO THE FRONT)\n{'=' * 90}")
print(f"{'contract':10s}{'3F RMSE(bp)':>14s}{'4F-front RMSE(bp)':>18s}")
rmse3_per, rmse4_per = [], []
for j in range(4):
    r3 = float(np.sqrt(np.nanmean(err3[:, j] ** 2)) * 1e4)
    r4 = float(np.sqrt(np.nanmean(err4[:, j] ** 2)) * 1e4)
    rmse3_per.append(r3); rmse4_per.append(r4)
    print(f"C{j + 1:<9d}{r3:14.3f}{r4:18.3f}")

rmse3_pooled = float(np.sqrt(np.nanmean(err3 ** 2)) * 1e4)
rmse4_pooled = float(np.sqrt(np.nanmean(err4 ** 2)) * 1e4)
n_wins = sum(1 for j in range(4) if rmse4_per[j] < rmse3_per[j])
print(f"\npooled: 3F={rmse3_pooled:.3f}bp  4F-front={rmse4_pooled:.3f}bp  "
     f"gain={100 * (1 - rmse4_pooled / rmse3_pooled):.2f}%   4F-front wins {n_wins}/4")

print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if rmse4_pooled < rmse3_pooled and n_wins >= 3:
    print("4F-front, calibrated blind to the front, still predicts the front better -- the "
         "front-concentrated factor's benefit is not just an in-sample artefact of seeing the "
         "data it's meant to explain.")
else:
    print("4F-front does NOT predict the withheld front better when calibrated blind to it -- "
         "the in-sample gain does not survive this held-out test, the strongest evidence yet "
         "that rung 2's front-specific story does not hold up.")

# %%
out = {"rmse_3F_pooled": rmse3_pooled, "rmse_4Ffront_pooled": rmse4_pooled,
      "per_contract_3F": rmse3_per, "per_contract_4Ffront": rmse4_per, "n_wins": n_wins,
      "sigma_f_c5c18": p4_hat.sigma_f, "sigma_f_full_panel": best["sigma_f_hat"],
      "kappa_f_P_c5c18": p4_hat.kappa_f_P}
out_path = ROOT / "results" / "c1c18_fourfactor_front_gate3b_heldout.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
