# %% [markdown]
# # Gate 3a: mirror of the original three-factor test -- fit C1-C12, predict C13-C18
#
# Same design explore/32 used to justify the third factor (48.4bp vs
# 87.2bp for two factors) -- now applied to the front-concentrated fourth
# factor. Reuses that exact 3F/2F benchmark rather than refitting it.
# This is NOT the direct test of the front hypothesis (Gate 3b is); it's
# the standard this paper already holds every other factor addition to,
# applied evenly to its own preferred result.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF  # noqa: E402
from csr.pricing.fit4_front import (  # noqa: E402
    fit4_front_gaussian,
    fit4_front_student_t,
    pack4_front,
)
from csr.pricing.fourfactor_front import A_tau4_front, FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f_fit, tau_fit = panel.log_f[:, :12], panel.tau[:, :12]
log_f_held, tau_held = panel.log_f[:, 12:18], panel.tau[:, 12:18]

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
heldout_benchmark = json.loads((ROOT / "results" / "c1c18_heldout_contract_prediction.json").read_text())
ms = json.loads((ROOT / "results" / "c1c18_fourfactor_front_multistart.json").read_text())
best = ms["starts"][ms["best_start_idx"]]

# %% [markdown]
# ## Warm-start from the full-panel front-4-factor fit, refit on C1-C12

# %%
p_init = FourFactorFrontParams(
    kappa2=best["kappa2_hat"], kappa3=best["kappa3_hat"],
    sigma_f=best["sigma_f_hat"], sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
    kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=best["kappa_f_P_hat"],
    rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
    mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
    lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
)
c0 = before["curve"]
theta0 = pack4_front(p_init, c0["a"], c0["b"], c0["c"], quadratic=True)
p_g, smooth_g, _, _, _, ll_g = fit4_front_gaussian(log_f_fit, tau_fit, DT, theta0, quadratic=True)
print(f"Gaussian on C1-C12: ll={ll_g:.2f}  kappa_f_P={p_g.kappa_f_P:.4f}  sigma_f={p_g.sigma_f:.4f}")

abc_g = fit_smooth_to_vector(sigma_from_tau(smooth_g, tau_fit[0]), tau_fit[0], quadratic=True)
theta0_em = pack4_front(p_g, abc_g.a, abc_g.b, abc_g.c, quadratic=True)
p_hat, smooth_hat, x_filt, resid, weight, n_iters, ll_fit, df_final = fit4_front_student_t(
    log_f_fit, tau_fit, DT, STUDENT_T_DF, theta0_em, quadratic=True)
print(f"Student-t on C1-C12: ll={ll_fit:.2f}  n_iters={n_iters}  "
     f"kappa_f_P={p_hat.kappa_f_P:.4f}  sigma_f={p_hat.sigma_f:.4f}")

# %% [markdown]
# ## Predict C13-C18 from the filtered state

# %%
chi_f, chi2, chi3, xi = x_filt[:, 0], x_filt[:, 1], x_filt[:, 2], x_filt[:, 3]
pred4 = np.empty_like(log_f_held)
for j in range(6):
    pred4[:, j] = (A_tau4_front(p_hat, tau_held[:, j])
                  + np.exp(-p_hat.kappa_f_Q * tau_held[:, j]) * chi_f
                  + np.exp(-p_hat.kappa2 * tau_held[:, j]) * chi2
                  + np.exp(-p_hat.kappa3 * tau_held[:, j]) * chi3 + xi)
err4 = log_f_held - pred4

print(f"\n{'=' * 90}\nGATE 3a: HELD-OUT PREDICTION, C13-C18 FROM C1-C12\n{'=' * 90}")
print(f"{'contract':10s}{'3F RMSE(bp)':>14s}{'4F-front RMSE(bp)':>18s}")
rmse3_per = heldout_benchmark["rmse_bp"]["per_contract_3F"]
rmse4_per = []
for j in range(6):
    rmse4_j = float(np.sqrt(np.nanmean(err4[:, j] ** 2)) * 1e4)
    rmse4_per.append(rmse4_j)
    print(f"C{13 + j:<9d}{rmse3_per[j]:14.3f}{rmse4_j:18.3f}")

rmse3_pooled = heldout_benchmark["rmse_bp"]["three_factor"]
rmse4_pooled = float(np.sqrt(np.nanmean(err4 ** 2)) * 1e4)
print(f"\npooled: 3F={rmse3_pooled:.3f}bp  4F-front={rmse4_pooled:.3f}bp  "
     f"gain={100 * (1 - rmse4_pooled / rmse3_pooled):.2f}%")

n_wins = sum(1 for j in range(6) if rmse4_per[j] < rmse3_per[j])
print(f"4F-front beats 3F on {n_wins}/6 held-out maturities")

# %%
out = {"rmse_3F_pooled": rmse3_pooled, "rmse_4Ffront_pooled": rmse4_pooled,
      "per_contract_4Ffront": rmse4_per, "n_wins": n_wins, "kappa_f_P": p_hat.kappa_f_P}
out_path = ROOT / "results" / "c1c18_fourfactor_front_gate3a_heldout.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
