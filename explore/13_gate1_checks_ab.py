# %% [markdown]
# # Gate 1 checks A and B: is the LR gain real, or a C4 boundary artefact?
#
# Check A: decompose the S0->S0P LR statistic by contract. If C4 (which
# landed on the measurement-error floor, 1.43bp, in the original fit)
# accounts for more than half the gain, the fit is a boundary artefact
# wearing a pricing-error costume, not evidence the mechanism is working.
#
# Check B (decisive): refit S0P with C4 dropped entirely. If s_1 still
# collapses to ~10bp on a C1-C3 panel, the diagnosis is confirmed
# independent of whatever is happening to C4. If s_1 stays near 50bp, the
# "pricing error" in the 4-contract fit was exploiting the C1/C4 pair
# specifically, and the front-curve story is wrong.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                        # noqa: E402
from csr.pricing.fit import (STUDENT_T_DF, fit_s0_gaussian, fit_s0_student_t,  # noqa: E402
                             pack, per_contract_loglik)
from csr.pricing.pricing_error import (fit_s0p_student_t, multi_start_fit_s0p,  # noqa: E402
                                       pack_s0p, per_contract_loglik_s0p)
from csr.simulate import SS_OIL                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel = Panel.from_prices(prices, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia",
                          verbose=False)
m = panel.m
print(f"panel: {panel.n} trading days x {m} contracts")

# %% [markdown]
# ## Check A: LR-by-contract on the original 4-contract fit

# %%
print("\nrefitting S0 (4-contract baseline)...")
p_gauss, _, _, _, _, _ = fit_s0_gaussian(panel.log_f, panel.tau, DT, m,
                                        theta0=pack(SS_OIL))
p_s0, x_filt_s0, resid_s0, weight_s0, n_iters_s0 = fit_s0_student_t(
    panel.log_f, panel.tau, DT, m, STUDENT_T_DF, theta0=pack(p_gauss))
ll_s0_by_contract = per_contract_loglik(p_s0, panel.log_f, panel.tau, DT)
print(f"S0 s_i(bp): {(p_s0.sigma_meas * 1e4).round(2).tolist()}")
print(f"S0 per-contract ll: {ll_s0_by_contract.round(2).tolist()}  "
     f"(sum={ll_s0_by_contract.sum():.2f})")

print("\nrefitting S0P (known-best start: phi0=0.95, sigma_p0=5e-4)...")
theta0 = pack_s0p(p_gauss, phi=0.95, sigma_p=5e-4)
fit_s0p = fit_s0p_student_t(panel.log_f, panel.tau, DT, m, STUDENT_T_DF, theta0)
ll_s0p_by_contract = per_contract_loglik_s0p(
    fit_s0p.p, fit_s0p.phi, fit_s0p.sigma_p, panel.log_f, panel.tau, DT)
print(f"S0P: phi={fit_s0p.phi:.4f}  sigma_p={fit_s0p.sigma_p:.5f}  ll={fit_s0p.ll:.2f}")
print(f"S0P s_i(bp): {(fit_s0p.p.sigma_meas * 1e4).round(2).tolist()}")
print(f"S0P per-contract ll: {ll_s0p_by_contract.round(2).tolist()}  "
     f"(sum={ll_s0p_by_contract.sum():.2f})")

lr_by_contract = 2.0 * (ll_s0p_by_contract - ll_s0_by_contract)
total_lr = lr_by_contract.sum()
print("\nLR contribution by contract (2*(S0P - S0)):")
for i, lr_i in enumerate(lr_by_contract):
    print(f"  C{i + 1}: {lr_i:+10.1f}  ({100 * lr_i / total_lr:6.1f}% of total)")
print(f"  TOTAL:  {total_lr:+10.1f}")
print(f"\nCheck A verdict: C4 share of total LR gain = "
     f"{100 * lr_by_contract[3] / total_lr:.1f}%  "
     f"({'BOUNDARY ARTEFACT -- C4 dominates' if lr_by_contract[3] > 0.5 * total_lr else 'not C4-dominated'})")

# %% [markdown]
# ## Check B (decisive): refit S0P with C4 dropped entirely

# %%
log_f_3 = panel.log_f[:, :3]
tau_3 = panel.tau[:, :3]
m3 = 3
p_init_3 = SS_OIL.replace(sigma_meas=SS_OIL.sigma_meas[:3])

print("\nrefitting S0 on C1-C3 only...")
p_gauss_3, _, _, _, _, _ = fit_s0_gaussian(log_f_3, tau_3, DT, m3, theta0=pack(p_init_3))
p_s0_3, x_filt_s0_3, resid_s0_3, weight_s0_3, n_iters_s0_3 = fit_s0_student_t(
    log_f_3, tau_3, DT, m3, STUDENT_T_DF, theta0=pack(p_gauss_3))
print(f"S0 (C1-C3) s_i(bp): {(p_s0_3.sigma_meas * 1e4).round(2).tolist()}")

print("\nmulti-start S0P on C1-C3 only (same 8-point grid as the 4-contract run)...")
ms_3 = multi_start_fit_s0p(
    log_f_3, tau_3, DT, m3, STUDENT_T_DF, p_gauss_3,
    phi_grid=(0.5, 0.7, 0.85, 0.95), sigma_p_grid=(5e-4, 3e-3))
print(f"ll spread across starts: {ms_3['ll_spread']:.4f}")
for s in ms_3["starts"]:
    print(f"  start phi0={s['phi0']:.2f} sigma_p0={s['sigma_p0']:.1e}  ->  "
         f"phi_hat={s['fit'].phi:+.4f}  sigma_p_hat={s['fit'].sigma_p:.5f}  "
         f"ll={s['fit'].ll:.2f}")

best_3 = ms_3["best"]
s1_3 = best_3.p.sigma_meas[0] * 1e4
print(f"\nS0P (C1-C3, best): phi={best_3.phi:+.4f}  sigma_p={best_3.sigma_p:.5f}  "
     f"ll={best_3.ll:.2f}")
print(f"S0P (C1-C3) s_i(bp): {(best_3.p.sigma_meas * 1e4).round(2).tolist()}")

verdict_b = ("COLLAPSES -- diagnosis confirmed independent of C4"
            if s1_3 < 15.0 else
            "STAYS HIGH -- the C1/C4 pair was being exploited, front-curve story is wrong")
print(f"\nCheck B verdict: s_1 (C1-C3 panel) = {s1_3:.2f}bp -> {verdict_b}")

print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
print(f"Check A: C4 share of LR gain = {100 * lr_by_contract[3] / total_lr:.1f}%")
print(f"Check B: s_1 with C4 dropped = {s1_3:.2f}bp (vs {fit_s0p.p.sigma_meas[0]*1e4:.2f}bp with C4 present)")
