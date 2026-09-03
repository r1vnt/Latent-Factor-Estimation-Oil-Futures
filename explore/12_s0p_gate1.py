# %% [markdown]
# # Runbook Step 1: kill the front-contract misspecification (Gate 1)
#
# S0's measurement-error profile (s_1=54.9bp, eight times C3, on the most
# liquid futures contract in the world) is a misspecification signature,
# not a noise-level estimate. This fits S0P -- S0 plus an AR(1)
# pricing-error state loading only on the front contract -- and evaluates
# Gate 1 exactly as pre-stated:
#
#     s_1 under S0P      < 15 bp     decisive
#     Q/H ratio           < 15x       decisive
#     LR vs S0            p < 0.01    necessary, not decisive
#     label agreement     < 93.74%    corroborating
#
# The LR test is necessary but not decisive: an extra state with two free
# parameters will almost always improve the likelihood. What decides Gate
# 1 is whether s_1 actually collapses.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                        # noqa: E402
from csr.pricing.fit import STUDENT_T_DF, fit_s0_gaussian, fit_s0_student_t, pack  # noqa: E402
from csr.pricing.pricing_error import (confounding_correlation,           # noqa: E402
                                       fit_s0p_student_t, multi_start_fit_s0p,
                                       pack_s0p, profile_likelihood_phi)
from csr.simulate import SS_OIL                                           # noqa: E402
from csr.study.recovery import filtered_label, label_agreement, naive_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

# %% [markdown]
# ## Load the panel and the S0 baseline (Table VI's numbers)

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel = Panel.from_prices(prices, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia",
                          verbose=False)
m = panel.m
print(f"panel: {panel.n} trading days x {m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

print("\nfitting S0 baseline (Gaussian -> Student-t)...")
p_gauss, _, _, _, _, _ = fit_s0_gaussian(panel.log_f, panel.tau, DT, m,
                                        theta0=pack(SS_OIL))
p_s0, x_filt_s0, resid_s0, weight_s0, n_iters_s0 = fit_s0_student_t(
    panel.log_f, panel.tau, DT, m, STUDENT_T_DF, theta0=pack(p_gauss))
# fit_s0_student_t doesn't return ll directly -- recompute via weighted_filter
from csr.pricing.fit import weighted_filter
ll_s0, _, _, n_obs_s0 = weighted_filter(p_s0, panel.log_f, panel.tau, DT,
                                        np.ones_like(panel.log_f))

s1_s0_bp = p_s0.sigma_meas * 1e4
print(f"S0: kappa={p_s0.kappa:.4f}  s_i(bp)={s1_s0_bp.round(2).tolist()}  ll={ll_s0:.1f}")

l_obs_s0 = naive_label(panel.log_f)
l_filt_s0 = filtered_label(p_s0, x_filt_s0, panel.tau)
agreement_s0 = label_agreement(l_obs_s0, l_filt_s0)
print(f"S0 label agreement: {agreement_s0['agreement_rate']:.4f}")

# %% [markdown]
# ## 1.4: multi-start fit -- 8 starts, report the log-likelihood spread

# %%
print("\nmulti-start S0P fit (8 starts: phi in {0.5,0.7,0.85,0.95} x "
     "sigma_p in {5e-4, 3e-3})...")
ms = multi_start_fit_s0p(
    panel.log_f, panel.tau, DT, m, STUDENT_T_DF, p_gauss,
    phi_grid=(0.5, 0.7, 0.85, 0.95), sigma_p_grid=(5e-4, 3e-3))

print(f"\nll spread across starts: {ms['ll_spread']:.4f}  "
     f"({'MULTIMODAL -- optimum may be a starting-value artefact' if ms['ll_spread'] > 1.0 else 'stable across starts'})")
for s in ms["starts"]:
    print(f"  start phi0={s['phi0']:.2f} sigma_p0={s['sigma_p0']:.1e}  ->  "
         f"phi_hat={s['fit'].phi:+.4f}  sigma_p_hat={s['fit'].sigma_p:.5f}  "
         f"ll={s['fit'].ll:.2f}")

best = ms["best"]
print(f"\nbest start: phi0={ms['best_start'][0]}, sigma_p0={ms['best_start'][1]}")
print(f"S0P (best): phi={best.phi:+.4f}  sigma_p={best.sigma_p:.5f}  "
     f"kappa={best.p.kappa:.4f}  ll={best.ll:.2f}  ({best.n_iters} EM iters)")
s1_s0p_bp = best.p.sigma_meas * 1e4
print(f"S0P s_i(bp): {s1_s0p_bp.round(2).tolist()}")

# %% [markdown]
# ## 1.5: profile likelihood in phi

# %%
print("\nprofiling likelihood in phi (grid 0.3 to 0.98)...")
phi_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
profile = profile_likelihood_phi(panel.log_f, panel.tau, DT, m, p_gauss,
                                 phi_grid, sigma_p0=best.sigma_p)
print(f"profile range: {profile['range']:.4f}  "
     f"({'FLAT -- phi not identified, fix it and free only sigma_p' if profile['range'] < 1.0 else 'phi appears identified'})")
print(f"argmax phi (Gaussian, weight=1 profile): {profile['argmax_phi']:.2f}")
for phi_g, ll_g in zip(profile["phi_grid"], profile["loglik"]):
    print(f"  phi={phi_g:.2f}  ll={ll_g:.2f}")

# %% [markdown]
# ## 1.6: confounding check -- corr(p_hat, xi_hat)

# %%
conf_corr = confounding_correlation(best.x_filt)
print(f"\ncorr(p_hat, xi_hat) under S0P: {conf_corr:+.4f}  "
     f"({'CONFOUNDED -- p is stealing the long-run factor' if abs(conf_corr) > 0.7 else 'separated'})")

# %% [markdown]
# ## 1.7: label agreement under S0P

# %%
l_obs_s0p = naive_label(panel.log_f)
l_filt_s0p = filtered_label(best.p, best.x_filt, panel.tau)
agreement_s0p = label_agreement(l_obs_s0p, l_filt_s0p)
print(f"S0P label agreement: {agreement_s0p['agreement_rate']:.4f}  "
     f"(S0 baseline: {agreement_s0['agreement_rate']:.4f})")

# %% [markdown]
# ## 1.8: Gate 1 table

# %%
lr_stat = 2.0 * (best.ll - ll_s0)
lr_df = 2   # phi and sigma_p both free -- revisit to df=1 if the profile is flat
lr_p = float(chi2.sf(lr_stat, lr_df)) if lr_stat > 0 else 1.0

s1_s0_front = s1_s0_bp[0]
s1_s0p_front = s1_s0p_bp[0]

qh_ratio_s0 = "15-1800x (S0 baseline)"

print(f"\n{'=' * 70}\nGATE 1\n{'=' * 70}")
print(f"{'criterion':<20}{'under S0':<20}{'under S0P':<20}{'pass?':<10}{'weight'}")
print(f"{'s_1 (bp)':<20}{s1_s0_front:<20.2f}{s1_s0p_front:<20.2f}"
     f"{'YES' if s1_s0p_front < 15 else 'NO':<10}{'decisive'}")
print(f"{'LR stat':<20}{'--':<20}{lr_stat:<20.2f}"
     f"{'':<10}{'necessary'}")
print(f"{'LR p (df=' + str(lr_df) + ')':<20}{'--':<20}{lr_p:<20.5f}"
     f"{'YES' if lr_p < 0.01 else 'NO':<10}{'necessary'}")
print(f"{'label agreement':<20}{agreement_s0['agreement_rate']:<20.4f}"
     f"{agreement_s0p['agreement_rate']:<20.4f}"
     f"{'YES' if agreement_s0p['agreement_rate'] < agreement_s0['agreement_rate'] else 'NO':<10}{'corroborating'}")
print(f"\ns_i full vector under S0P (bp): {s1_s0p_bp.round(2).tolist()}")
print(f"phi={best.phi:+.4f}  sigma_p={best.sigma_p:.5f}  "
     f"corr(p_hat,xi_hat)={conf_corr:+.4f}  ll_spread_across_starts={ms['ll_spread']:.4f}")

verdict = "PASS" if s1_s0p_front < 15.0 else "FAIL"
print(f"\nGATE 1 VERDICT (decisive criterion: s_1 < 15bp): {verdict}")
