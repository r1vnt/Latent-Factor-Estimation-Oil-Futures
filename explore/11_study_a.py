# %% [markdown]
# # Study A: state recovery and predictive validity (Gate 4)
#
# Price-only -- does not touch the news pipeline. Fits S0 on the real EIA
# panel (Student-t observation noise, Sec. XI-F -- not the Gaussian fit,
# which sits at a boundary and whose kappa is unstable across sample
# windows), then runs the full Sec. 5 reporting battery from
# csr.study.recovery: both required horizons (5, 22 trading days) x three
# targets, with/without April 2020, the primary target split by realised-
# vol tercile, and the continuous-form robustness check.
#
# Filtered states are never smoothed here -- x_filt comes straight out of
# fit_s0_student_t, which returns weighted_filter's raw filtered pass, not
# anything routed through state.fast.smooth_rts. Using the smoother would
# manufacture exactly the look-ahead this study exists to rule out.
#
# Parameters are estimated on the full sample and the regression run on the
# same data -- in-sample, noted per Sec. 5, not hidden. If Gate 4 passes,
# re-run with expanding-window re-estimation before this goes in the paper.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                        # noqa: E402
from csr.pricing.fit import (STUDENT_T_DF, filtered_covariance,           # noqa: E402
                             fit_s0_gaussian, fit_s0_student_t, pack)
from csr.simulate import SS_OIL                                           # noqa: E402
from csr.study.recovery import (EXTENDED_HORIZONS, circular_block_bootstrap_monotonicity,  # noqa: E402
                                circular_block_bootstrap_restricted_null,
                                confidence_split_regression, disagreement_by_year,
                                disagreement_clustering, disagreement_event_study,
                                disagreement_event_study_block_bootstrap,
                                encompassing_regression, filtered_label,
                                label_agreement, naive_label, print_corrected_report,
                                print_horizon_table, print_report, realised_vol,
                                run_corrected_battery, run_horizon_robustness,
                                run_study_a, xi_encompasses_log_price,
                                xi_monopoly_bootstrap_null, xi_monopoly_test)

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

# %% [markdown]
# ## Load the panel -- basis="business" to match DT's business-day clock

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel = Panel.from_prices(prices, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia",
                          verbose=False)
m = panel.m
print(f"panel: {panel.n} trading days x {m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

# %% [markdown]
# ## Fit S0: Gaussian first (initialises the Student-t EM loop, matching
# ## how this was validated in explore/02_fit_s0.py), then Student-t

# %%
print("\nfitting Gaussian S0 (initialisation for the Student-t fit)...")
p_gauss, hess_inv, xf_gauss, resid_gauss, n_obs_gauss, ll_gauss = fit_s0_gaussian(
    panel.log_f, panel.tau, DT, m, theta0=pack(SS_OIL))
print(f"  Gaussian: kappa={p_gauss.kappa:.4f}  half-life={p_gauss.half_life_days:.1f}d  "
     f"ll={ll_gauss:.1f}  n_obs={n_obs_gauss}")

print(f"\nfitting Student-t S0 (df={STUDENT_T_DF:.0f}, EM/scale-mixture)...")
p_hat, x_filt, resid_t, weight_t, n_iters = fit_s0_student_t(
    panel.log_f, panel.tau, DT, m, STUDENT_T_DF, theta0=pack(p_gauss))
print(f"  Student-t: kappa={p_hat.kappa:.4f}  half-life={p_hat.half_life_days:.1f}d  "
     f"({n_iters} EM iterations)")
print(f"  sigma_chi={p_hat.sigma_chi:.4f}  sigma_xi={p_hat.sigma_xi:.4f}  rho={p_hat.rho:+.4f}")

covid = (panel.dates >= "2020-04-01") & (panel.dates <= "2020-04-30")
covid_weight = weight_t[covid]
covid_weight = covid_weight[np.isfinite(covid_weight)]
print(f"  April 2020 EM weights: mean={covid_weight.mean():.3f}  "
     f"min={covid_weight.min():.3f}  (vs overall mean {np.nanmean(weight_t):.3f})")

# %% [markdown]
# ## Run Study A

# %%
result = run_study_a(p_hat, x_filt, panel.log_f, panel.tau, panel.dates)
print_report(result)

# %% [markdown]
# ## Sec. 5: this run is in-sample (parameters fit on the same data the
# ## regression runs on) -- noted here explicitly, not buried.

# %%
print("\nNOTE: parameters were estimated on the full sample and the "
     "regression above runs on that same sample -- in-sample, per Sec. 5. "
     "If Gate 4 passes, re-run with expanding-window re-estimation "
     "(the backtest layer) before this goes in the paper.")

# %% [markdown]
# ## Shared-measurement-error diagnosis: the corrected battery
#
# The primary target above, spread_change, is s_{t+h} - s_t; the naive
# label is sign(s_t). The same s_t enters both, negated -- noise in s_t
# mechanically produces a negative naive-label coefficient with no genuine
# predictive content behind it, and the filtered label, having denoised
# exactly that s_t, structurally cannot exploit the same artefact. Three
# specifications below break that mechanism without touching the battery
# above, which stands as the naive-specification panel that motivated this
# section (csr/study/recovery.py's Sec. 6 docstring has the full argument,
# and test_shared_noise_artefact_vanishes_under_cross_maturity_with_a_
# perfect_label in tests/test_recovery.py reproduces it directly on
# simulated data with a known true state before this was trusted on WTI).
#
# Cross-maturity (labels from C1/C2, target from C3/C4) is the decisive
# one: if the filtered label still loses there, the original fail is real,
# not an artefact.

# %%
corrected = run_corrected_battery(p_hat, x_filt, panel.log_f, panel.tau)
print_corrected_report(corrected)

# %% [markdown]
# ## Label agreement: run before anything else
#
# The degenerate non-overlap rows at h=66/132 are a symptom, not a
# sampling curiosity -- Study A's spec assumed ~85% naive/filtered
# agreement, and the encompassing regression identifies b1/b2 only from
# DISAGREEMENT days. The true rate determines how much information exists
# to separate the two labels with at every horizon, not just the
# sparsely-sampled ones.

# %%
l_obs_full = naive_label(panel.log_f)
l_filt_full = filtered_label(p_hat, x_filt, panel.tau)

agreement = label_agreement(l_obs_full, l_filt_full)
print(f"\nlabel agreement: {agreement['agreement_rate']:.4f} "
     f"({agreement['n_disagree']} disagreements / {agreement['n_valid']} valid days)")

by_year = disagreement_by_year(l_obs_full, l_filt_full, panel.dates)
print("\ndisagreements by year:")
print(by_year.to_string())

clustering = disagreement_clustering(l_obs_full, l_filt_full, panel.log_f)
print(f"\nclustering: median|s_t| disagree={clustering['median_abs_s_disagree']:.6f}  "
     f"agree={clustering['median_abs_s_agree']:.6f}  "
     f"(mean disagree={clustering['mean_abs_s_disagree']:.6f}  "
     f"agree={clustering['mean_abs_s_agree']:.6f})")
print(f"disagreement runs: n={clustering['n_runs']}  "
     f"median length={clustering['run_length_median']:.1f}  "
     f"max length={clustering['run_length_max']}")

# %% [markdown]
# ## Horizon robustness: the third spec flaw
#
# h in {5, 22} were chosen by convention, never checked against the fitted
# kappa above (~1.31, half-life ~193 trading days) -- those horizons span
# only 1.8%/7.6% of chi's reversion, while the shared-measurement-error
# bias fixed above is O(1) in h. Cross-maturity inherited the same
# horizons and the same power problem, which is why it came back
# inconclusive rather than resolving. Reruns the decisive cross-maturity
# spec at h in {5, 22, 66, 132, 252}: the theory's falsifiable prediction
# is that filtered performance relative to naive improves with horizon,
# since signal accumulates and the bias doesn't. Now includes the joint
# (b1=b2=0) and discrimination (b1=b2) Wald tests, and label
# correlation/VIF, at every horizon.

# %%
horizon_rows = run_horizon_robustness(p_hat, x_filt, panel.log_f, panel.tau)
print_horizon_table(horizon_rows)

# %% [markdown]
# ## Circular block bootstrap: is the p-value monotonicity real?
#
# Null distribution for "|t(b2)| increases monotonically across all five
# horizons" -- a pre-stated prediction, so this is confirmatory, not a
# fishing expedition. Block-bootstraps the label pair (block length ~252,
# preserving their own serial and cross dependence) against the real
# target series, breaking any true temporal alignment between labels and
# y.

# %%
boot = circular_block_bootstrap_monotonicity(
    p_hat, x_filt, panel.log_f, panel.tau, block_length=252, n_boot=2000, seed=0)
print(f"\ncircular block bootstrap (block=252, n_boot={boot['n_boot']}, "
     f"{boot['n_skipped']} skipped as degenerate):")
print(f"  P(|t(b2)| monotone increasing across all 5 horizons | null) = "
     f"{boot['fraction_monotone']:.4f}  ({boot['n_monotone']}/{boot['n_used']})")

# %% [markdown]
# ## Correctly-specified null: H0: b1 = b2, not "labels carry no information"
#
# The joint Wald test (below, horizon table) already rejects "no
# information" hard at h>=66 -- so the bootstrap above tested a null
# already independently disproven, which says nothing about b2
# specifically. The diff-Wald test fails to reject b1=b2 at every
# horizon; this resamples residuals from that restriction instead of
# resampling the labels.

# %%
boot_restricted = circular_block_bootstrap_restricted_null(
    p_hat, x_filt, panel.log_f, panel.tau, block_length=252, n_boot=2000, seed=0)
print(f"\nrestricted-null (b1=b2) bootstrap (block=252, "
     f"n_boot={boot_restricted['n_boot']}, "
     f"{boot_restricted['n_skipped']} skipped as degenerate):")
print(f"  P(|t(b2)| monotone increasing across all 5 horizons | b1=b2) = "
     f"{boot_restricted['fraction_monotone']:.4f}  "
     f"({boot_restricted['n_monotone']}/{boot_restricted['n_used']})")

# %% [markdown]
# ## Gating diagnostic: is there anything left for the filter to remove?
#
# s_i in basis points of log price, and the (approximate, post-transient)
# steady-state Kalman gain onto chi per contract.

# %%
print(f"\ns_i (measurement error, bp): {(p_hat.sigma_meas * 1e4).round(2).tolist()}")
print(f"kappa: {p_hat.kappa:.4f}  sigma_chi: {p_hat.sigma_chi:.4f}  "
     f"sigma_xi: {p_hat.sigma_xi:.4f}")

tau_mean = panel.tau.mean(axis=0)
z_chi_mean = np.exp(-p_hat.kappa * tau_mean)
Z_static = np.column_stack([z_chi_mean, np.ones(panel.m)])
eigvals = np.linalg.eigvalsh(Z_static.T @ Z_static)
print(f"\ncurrent panel (C1-C4) conditioning: lambda_min={eigvals.min():.4f}  "
     f"lambda_min/contract={eigvals.min() / panel.m:.4f}  "
     f"cond(Z)={np.linalg.cond(Z_static):.2f}")

# %% [markdown]
# ## Sec 3: testing the filter as more than a classifier
#
# (a) Disagreement event study: on the 225 days where the labels differ,
# which one correctly signed the subsequent cross-maturity spread change?
# (b) Confidence split: does the filtered label win on high-confidence
# (low P_filt) days specifically?
# (c) xi monopoly test: does xi_hat predict long-horizon level changes,
# something the spread cannot address at all?

# %%
P_filt = filtered_covariance(p_hat, panel.log_f, panel.tau, DT)

print("\n(a) disagreement event study (McNemar, cross-maturity spread change):")
for h in EXTENDED_HORIZONS:
    ev = disagreement_event_study(l_obs_full, l_filt_full, panel.log_f, h)
    print(f"  h={h:4d}  n_disagree={ev['n_disagree']}  "
         f"obs_win_rate={ev['obs_win_rate']:.3f}  filt_win_rate={ev['filt_win_rate']:.3f}  "
         f"(only_obs={ev['n_only_obs_correct']} only_filt={ev['n_only_filt_correct']} "
         f"both_right={ev['n_both_correct']} both_wrong={ev['n_both_wrong']})  "
         f"McNemar p={ev['mcnemar_p']:.4f}")

print("\n(b) confidence split (median filtered chi variance):")
for h in EXTENDED_HORIZONS:
    r_low, r_high = confidence_split_regression(p_hat, x_filt, P_filt,
                                                panel.log_f, panel.tau, h)
    print(f"  h={h:4d}  {r_low}")
    print(f"           {r_high}")

print("\n(c) xi monopoly test (front-contract level change, no naive comparator):")
for h in EXTENDED_HORIZONS:
    r_xi = xi_monopoly_test(x_filt, panel.log_f, h)
    print(f"  h={h:4d}  n={r_xi.n_obs}  b(xi_hat)={r_xi.b2:+.5f} (p={r_xi.p_b2:.4f})  "
         f"R2={r_xi.r_squared:.4f}")

# %% [markdown]
# ## Realised-vol decomposition, restricted to the 225 disagreement days
#
# The one target with no shared-noise path to the labels (Sec. 6/8), and
# the one where the filtered label won cleanly full-sample. Run as fix
# 3(a): on the 225 disagreement days specifically, not full-sample.

# %%
valid_full = np.isfinite(l_obs_full) & np.isfinite(l_filt_full)
disagree_full = (l_obs_full != l_filt_full) & valid_full
print(f"\nrealised-vol on disagreement days only (n={int(disagree_full.sum())}):")
for h in EXTENDED_HORIZONS:
    y_vol = realised_vol(panel.log_f, h)
    r_vol = encompassing_regression(y_vol, l_obs_full, l_filt_full, h,
                                    target="realised_vol", subset="disagree_only",
                                    extra_mask=disagree_full, maxlags=max(h - 1, 0))
    print(f"  {r_vol}")

# %% [markdown]
# ## Three checks before scoping any fix
#
# The s-hat profile ([54.9, 6.5, 5.1, 16.6] bp) reframed the batch: front-
# contract misspecification, not a noise-level estimate. Two of the
# downstream results built on that reframing can still be overturned by
# their own weaknesses -- checked here before either goes in the paper.

# %%
print("\n(1) McNemar block-bootstrap correction (block_length = max(h, 20)):")
for h in EXTENDED_HORIZONS:
    bb = disagreement_event_study_block_bootstrap(
        l_obs_full, l_filt_full, panel.log_f, h, n_boot=2000, seed=0)
    print(f"  h={h:4d}  n_disagree={bb['n_disagree']}  block={bb['block_length']}  "
         f"observed_diff={bb['observed_diff']:+.4f}  "
         f"block_bootstrap_se={bb['block_bootstrap_se']:.4f}  "
         f"z={bb['z']:.3f}  p={bb['p']:.4f}  (n_boot_used={bb['n_boot_used']})")

# %%
print("\n(2) xi encompassing test: front_return ~ 1 + log_f1 + xi_hat:")
for h in EXTENDED_HORIZONS:
    r_enc = xi_encompasses_log_price(x_filt, panel.log_f, h)
    print(f"  h={h:4d}  n={r_enc.n_obs}  "
         f"c1(log_f1)={r_enc.b1:+.5f}(p={r_enc.p_b1:.4f})  "
         f"c2(xi_hat)={r_enc.b2:+.5f}(p={r_enc.p_b2:.4f})  "
         f"R2={r_enc.r_squared:.4f}  [{r_enc.pattern()}]")

# %%
print("\n(3) xi monopoly R^2: adjusted, and block-bootstrap null distribution:")
for h in EXTENDED_HORIZONS:
    r_xi = xi_monopoly_test(x_filt, panel.log_f, h)
    boot_r2 = xi_monopoly_bootstrap_null(x_filt, panel.log_f, h, n_boot=2000, seed=0)
    print(f"  h={h:4d}  R2={r_xi.r_squared:.4f}  R2_adj={r_xi.r_squared_adj:.4f}  "
         f"null R2 mean={boot_r2['null_r2_mean']:.4f}  "
         f"null R2 95th pct={boot_r2['null_r2_p95']:.4f}  "
         f"P(null R2 >= observed)={boot_r2['fraction_null_r2_exceeds_observed']:.4f}")
