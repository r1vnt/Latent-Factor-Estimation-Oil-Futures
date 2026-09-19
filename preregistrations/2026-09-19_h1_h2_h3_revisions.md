# Pre-registration: H1'/H2'/H3a revisions

Written before running any of the tests below. Git commit timestamp on
this file is the evidence of pre-registration, per the discipline this
document itself argues for. Original H1/H2/H3 results stay in the
record unchanged; these are additions, not replacements.

## H2' — reconstructed intensity, contemporaneous, scheduled-release subsample

**Diagnostic first (one line):** corr(I_t, RV_{t-1}) vs the already-reported
corr(I_{t-1}, RV_{t-1})=0.108. Materially above 0.108 (>0.3) confirms the
reverse-causality/mediation concern before doing anything else.

**Construction:** deseasonalize I_t (weekday dummies + holiday flag +
linear/loess trend for secular digital-coverage growth), then surprise =
ratio to trailing 22-day median (or AR(1) residual) of the deseasonalized
series. Identify scheduled-release days: EIA weekly petroleum status
report (Wednesdays, with Thursday shift on some holidays), OPEC meeting
dates, contract expiry (already computed via cl_expiry).

**Falsifier, stated in advance:** contemporaneous corr(surprise intensity,
RV) on scheduled-release days > 0.25 supports H2'. < 0.15 means the null
is real and survives a proper construction.

**Corpus coverage:** report exactly why the original check used 1,914 of
3,254 panel dates (news coverage starts 2015-02-19; panel starts 2010) --
already understood, not a bug, restated here for the record.

## H1' — news content discriminates transient from permanent shocks

**Reduced form, no refitting:** for each day t, regress h-ahead cumulative
return (h in {5, 22, 66}) on day-t's own return, interacted with each news
feature (sent_mean, theme shares). A positive, significant interaction
means news-heavy moves persist more than news-quiet moves of the same size.

**Falsifier:** no interaction term significant at p<0.01 across all h and
features -> the reduced form shows nothing, the structural version (news
loadings splitting chi/xi attribution) is not worth building.

## H3a — cross-sectional leave-one-out, no dynamics

**Test:** for each contract i and each date, filter on the OTHER 17
contracts only (contract i's own observation withheld that date), predict
contract i's SAME-DAY price from the filtered state. Compare RMSE against
a flexible cross-sectional fit (Nelson-Siegel-style 3-parameter shape,
no dynamics, no filter) to the other 17 contracts on the SAME day,
predicting contract i out of sample.

**Falsifier:** filter beats the flexible cross-sectional fit -> measurement
error is real and the filter removes it cross-sectionally (H3a supported),
independent of H3b's already-established temporal rejection. Filter loses
or ties -> the cross-sectional claim doesn't survive either, a stronger
and more complete negative than H3b alone.

**Explicitly distinct from** the held-out-maturity result (44.6%, C13-C18)
-- that compares 2F vs 3F with dynamics; this compares filter vs.
no-dynamics curve, isolating exactly what's at issue.

## Priority order (per the size/cost argument)
1. H2 one-line diagnostic (immediate)
2. H2' full reconstruction (cheap, reduced-form)
3. H3a leave-one-out (nearly free, machinery exists)
4. H1' reduced form (one afternoon)
5. H1' structural -- only if (4) shows something
