# Pre-registration: C1-C4 H3a robustness reporting rule

Written before running the check below. Git commit timestamp on this file is
the evidence of pre-registration, same discipline as
`2026-09-19_h1_h2_h3_revisions.md`.

## Question

Table h3a (`results/c1c18_h3a_leaveoneout.json`) reports the filter losing to
a dynamics-free Nelson-Siegel fit at C1-C4, by 24-90%. Is that RMSE gap a
pervasive property of the filter's same-day prediction at the front, or is
it concentrated in a handful of extreme sessions (COVID crash, pre-expiry
noise) that dominate a squared-error statistic the way front-spread errors
already did in Section sec:denoisetest?

## Construction — no fitting

Re-derive the exact per-day, per-contract leave-one-out prediction errors
already computed once in `explore/65_h3a_cross_sectional_leaveoneout_c1c18.py`
(same fitted S3 params from `results/c1c18_threefactor.json`, same `filter_uv`
leave-one-out construction, same per-day Nelson-Siegel refit) for C1-C4 only.
No parameters are re-estimated. For each of C1-C4, on the filter-side error
series `err_f[t] = log_f[t,i] - filter_pred[t,i]`:

- RMSE (already reported) and MAE, in bp.
- RMSE excluding 2020-03-01 to 2020-05-31 inclusive (the WTI negative-price
  episode, already identified as an artefact elsewhere in this project —
  `explore/76_c1_backwardation_contango_split_c1c18.py`).
- RMSE excluding the final five NYMEX business sessions before each
  observation's own contract's expiry (`roll_indicator`-style window, applied
  per column rather than only to the front contract).

## Falsifier, stated in advance

MAE/RMSE at C1 is the diagnostic because it does not depend on an arbitrary
exclusion window: a ratio that low is possible only if a small number of
sessions carry squared error far above the rest.

- **MAE/RMSE at C1 < 0.2** -> the RMSE figure in Table h3a is outlier-dominated.
  The exclusion-adjusted RMSE figures (ex-COVID, ex-near-expiry) become the
  primary characterization of C1-C4 performance in the text, and the original
  pooled RMSE is reported alongside as the unadjusted figure.
- **MAE/RMSE at C1 >= 0.2** -> the gap is pervasive, not a handful of
  sessions. The original RMSE in Table h3a stands as primary; the exclusion
  figures are reported only as a supplementary check that does not change
  the conclusion, and the H3a front-end paragraph is not rewritten.

0.2 is roughly a quarter of the Gaussian MAE/RMSE ratio (~0.8), chosen in
advance as the threshold at which "a few sessions dominate" is the only
remaining explanation.
