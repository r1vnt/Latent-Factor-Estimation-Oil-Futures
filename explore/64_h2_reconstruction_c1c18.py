# %% [markdown]
# # H2' reconstruction: deseasonalized surprise intensity, scheduled-release subsample
#
# Pre-registered in preregistrations/2026-09-19_h1_h2_h3_revisions.md.
# Step 1 (explore/63) ruled out reverse-causality-through-the-control
# (corr(I_t, RV_t-1)=0.089, corr(I_t, RV_t)=0.108 -- neither elevated).
# This is the remaining candidate: raw article counts carry a strong
# day-of-week cycle that volatility doesn't, attenuating (or flipping)
# any real relationship. Deseasonalize, measure surprise relative to a
# trailing baseline, then test on the subsample where news arrival is
# genuinely exogenous to that day's trading (scheduled releases).
#
# Note on scope: `intensity` (stage4_daily.parquet) already trailing-252
# -day standardizes log1p(I_t), which should absorb most SECULAR trend
# (slow drift as digital coverage expanded) -- the piece that
# standardisation does NOT remove is day-of-week seasonality, since a
# 252-day window mixes all weekdays together. That's the specific gap
# this reconstruction targets. Built from RAW I_t, not the pre-processed
# `intensity` column, for full control over the pipeline.
#
# Scheduled-release days: all Wednesdays (EIA's weekly petroleum status
# report schedule -- a known ~weekly cadence, not adjusted here for the
# Monday-holiday Thursday shift, a handful of weeks/year) and CL contract
# expiry dates (exact, via csr.curve.expiry). OPEC meeting dates are NOT
# included -- no verified calendar of historical dates is available in
# this repo's data, and fabricating one would be worse than omitting it.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.curve.expiry import nearby_expiries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RV_WINDOW = 20
SURPRISE_WINDOW = 22

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau, dates = panel.log_f, panel.tau, panel.dates
n = panel.n

news = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet").set_index("date")
news_aligned = news.reindex(dates)
I_t = news_aligned["I_t"].to_numpy(dtype=float)

# %% [markdown]
# ## Deseasonalize: weekday-of-week OLS residual on log1p(I_t)

# %%
log_I = np.log1p(I_t)
valid_log = np.isfinite(log_I)
dow = dates.dayofweek.to_numpy()   # 0=Mon .. 4=Fri (panel is business days only)
dow_dummies = pd.get_dummies(dow, prefix="dow", drop_first=True).to_numpy(dtype=float)
X_season = sm.add_constant(dow_dummies, has_constant="add")

res_season = sm.OLS(log_I[valid_log], X_season[valid_log]).fit()
deseason = np.full(n, np.nan)
deseason[valid_log] = res_season.resid
print(f"day-of-week seasonality: R2={res_season.rsquared:.4f}  "
     f"(fraction of log(1+I_t) variance explained by weekday alone)")
print(f"dow coefficients (Tue..Fri vs Mon): {np.round(res_season.params[1:], 4)}")

# %% [markdown]
# ## Surprise: deseasonalized value relative to its own trailing 22-day median

# %%
surprise = np.full(n, np.nan)
for t in range(SURPRISE_WINDOW, n):
    window = deseason[t - SURPRISE_WINDOW:t]   # causal: PRIOR 22 days, not including today
    if np.all(np.isfinite(window)):
        surprise[t] = deseason[t] - np.median(window)

# %% [markdown]
# ## Realised vol (contemporaneous, ending today -- this test is about same-day arrival)

# %%
ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_today = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_today[t] = np.std(window, ddof=1)

# %% [markdown]
# ## Scheduled-release subsample: Wednesdays and CL expiry dates

# %%
is_wednesday = (dow == 2)
expiries = nearby_expiries(dates, 1, "CL")[:, 0]
is_expiry_day = np.array([pd.Timestamp(dates[t]).normalize() == pd.Timestamp(expiries[t]).normalize()
                          for t in range(n)])
scheduled = is_wednesday | is_expiry_day
print(f"\nscheduled-release days in sample: {scheduled.sum()}/{n}  "
     f"(Wednesdays={is_wednesday.sum()}, expiry days={is_expiry_day.sum()}, overlap={int((is_wednesday & is_expiry_day).sum())})")
print("NOTE: OPEC meeting dates NOT included -- no verified historical calendar available in this repo.")

# %% [markdown]
# ## The test: contemporaneous corr(surprise, RV) -- full sample vs. scheduled subsample

# %%
def _corr(a, b, mask=None):
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid = valid & mask
    return float(np.corrcoef(a[valid], b[valid])[0, 1]), int(valid.sum())

c_full, n_full = _corr(surprise, rv_today)
c_sched, n_sched = _corr(surprise, rv_today, scheduled)
c_unsched, n_unsched = _corr(surprise, rv_today, ~scheduled)

print(f"\n{'=' * 90}\nH2' RESULT: corr(deseasonalised surprise intensity, contemporaneous RV)\n{'=' * 90}")
print(f"full sample:          corr={c_full:+.4f}  n={n_full}")
print(f"scheduled-release:    corr={c_sched:+.4f}  n={n_sched}")
print(f"non-scheduled days:   corr={c_unsched:+.4f}  n={n_unsched}")

# %% [markdown]
# ## Verdict, against the pre-registered falsifier

# %%
print(f"\n{'=' * 90}\nVERDICT (pre-registered: >0.25 supports H2', <0.15 confirms the null)\n{'=' * 90}")
if c_sched > 0.25:
    print(f"H2' SUPPORTED: corr={c_sched:.3f} on scheduled-release days exceeds the 0.25 bar. "
         f"The original null was a construction failure, not an absent relationship.")
elif c_sched < 0.15:
    print(f"NULL CONFIRMED: corr={c_sched:.3f} stays below 0.15 even after deseasonalising, "
         f"measuring surprise, and restricting to exogenous-arrival days. This is now a "
         f"stronger null than the original -- it survives a proper construction.")
else:
    print(f"AMBIGUOUS: corr={c_sched:.3f} sits between the two pre-registered bounds -- neither "
         f"falsifier fires cleanly.")

# %%
import json  # noqa: E402
out = {
    "dow_seasonality_r2": float(res_season.rsquared),
    "corr_full": c_full, "n_full": n_full,
    "corr_scheduled": c_sched, "n_scheduled": n_sched,
    "corr_unscheduled": c_unsched, "n_unscheduled": n_unsched,
    "n_scheduled_days_in_sample": int(scheduled.sum()),
}
out_path = ROOT / "results" / "c1c18_h2_reconstruction.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
