# %% [markdown]
# # Dry run: Stage 4 news aggregate -> NewsData -> run_spec on S0/SD/SM
#
# Not for results -- Stage 2 has only reached roughly half the window, and
# the specs aren't tuned (SS_OIL is a plausible starting point, not a fit).
# This is integration testing on real data, ahead of Study A, so that a join
# bug surfaces now rather than after the remaining ~1,600 days finish:
#
# - Does the date alignment between the news table (every calendar day) and
#   the price panel (trading days only) work?
# - Does R_t (the news measurement variance) come out finite everywhere it's
#   actually used?
# - Do NaN coverage days propagate correctly through the filter's exact
#   missing-observation handling, rather than crashing or silently zeroing?
# - Does SM still nest exactly against S0 with real news data, not just the
#   synthetic fixture in tests/test_nesting.py?
#
# Already found and fixed via this exercise, before this script even ran
# cleanly: append_news_block() was not lag-shifting the news measurement
# variance the same way it shifts the news observation, which silently
# poisoned the log-likelihood on any real gap day adjacent to covered days.
# See the commit on src/csr/state/augment.py and src/csr/pricing/specs.py.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                          # noqa: E402
from csr.pricing.specs import NewsData, REGISTRY, run_spec  # noqa: E402
from csr.simulate import SS_OIL                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0   # one NYMEX business day, matching explore/02_fit_s0.py

# %% [markdown]
# ## Load both sides independently first

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel_full = Panel.from_prices(prices, date_col="date", contract_col="contract",
                               price_col="settle", basis="business", source="eia",
                               verbose=False)

news_df = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet")
news_df = news_df.set_index("date").sort_index()

# %% [markdown]
# ## Window: slice both to where Stage 2 has actually produced real rows
#
# Not the full 2015-2024 span -- most of it is still NaN because Stage 2 is
# still working through it. Slicing to the completed stretch exercises real
# joined news+price data on ~1,200 trading days rather than mostly-NaN rows,
# which is a weaker test of the actual join logic.

# %%
has_news = news_df["n_t"].notna()
# Longest CONTIGUOUS run, not naive first/last non-null: Stage 2 processes
# days sequentially, but a handful of one-off validation runs from earlier
# in development landed isolated rows far ahead of the sequential frontier
# (e.g. a single already-scored 2024-01-15 sitting alone past an otherwise
# solid Feb-2015-to-late-2019 block). Naive first/last would silently pull
# in the entire mostly-NaN gap between them.
run_id = (~has_news).cumsum()
run_lengths = has_news.groupby(run_id).transform("size").where(has_news, 0)
best_run = run_id[run_lengths == run_lengths.max()].iloc[0]
in_best_run = has_news & (run_id == best_run)
first_complete = news_df.index[in_best_run][0]
last_complete = news_df.index[in_best_run][-1]
print(f"stage 4's longest contiguous completed stretch: "
     f"{first_complete.date()} .. {last_complete.date()} "
     f"({in_best_run.sum():,} days; {has_news.sum():,} total non-null days "
     f"exist across the full {len(news_df):,}-day window, including "
     f"isolated one-off rows outside this stretch)")

panel = panel_full.slice(first_complete, last_complete)
print(f"panel sliced to {panel.n} trading days, {panel.m} contracts")

# %% [markdown]
# ## Join: reindex the news table onto the panel's trading-day calendar
#
# The news table has one row per CALENDAR day, including weekends (GDELT
# collects around the clock); the panel only has trading days. Reindexing
# onto panel.dates keeps exactly the trading-day rows and drops weekend-only
# news -- a real design choice, not a null default: weekend sentiment is
# currently not folded into the adjacent trading day, it is simply absent
# from the aligned series. Worth revisiting when the specs are actually
# tuned, not decided silently here.

# %%
aligned = news_df.reindex(panel.dates)
n_matched = aligned.index.isin(news_df.index).sum()
print(f"date alignment: {n_matched}/{panel.n} trading days found a stage-4 "
     f"row (should be 100% -- stage 4 is reindexed onto every calendar day "
     f"in its own window, so every trading day within range must have one, "
     f"NaN-valued or not)")
assert n_matched == panel.n, "a trading day fell outside stage 4's own reindexed range"

coverage_rate = aligned["n_t"].notna().mean()
print(f"of those, {coverage_rate:.1%} have real (non-NaN) news on that exact "
     f"trading day; the rest are genuine coverage gaps within the window")

# %% [markdown]
# ## Build NewsData
#
# features = standardised sent_mean, the one channel B/G actually load on
# for the preregistered S0/SD/SM/SDM lattice; theme shares and tone dims are
# the Section XIII cross-check, not part of this lattice. counts/dispersion
# stay RAW (news_variance's own contract). roll comes straight from the
# Panel -- curve/expiry.py already computes it, no need to rebuild it here.
# release/rv_lag are left out for this dry run: gamma_r/gamma_v are 0 in
# SS_OIL, so they have no effect either way, and building a real EIA release
# calendar is its own task, not a blocker for testing the news join itself.

# %%
sent = aligned["sent_mean"].to_numpy(dtype=float)
valid = ~np.isnan(sent)
sent_z = np.full_like(sent, np.nan)
sent_z[valid] = (sent[valid] - sent[valid].mean()) / sent[valid].std()

news = NewsData(
    features=sent_z.reshape(-1, 1),
    counts=aligned["n_t"].to_numpy(dtype=float),
    dispersion=aligned["disp"].to_numpy(dtype=float),
    intensity=aligned["intensity"].to_numpy(dtype=float),
    roll=panel.roll,
)

# The invariant the R-shift fix depends on: a day's feature reading and its
# own dispersion/counts must be missing together. aggregate_stage4.py
# enforces this by construction (one has_data mask applied to every value
# column), but checking it here, on the actual joined array, is what the
# guard added to run_spec() would catch anyway -- doing it explicitly first
# gives a clearer diagnostic if it ever doesn't hold.
feat_missing = np.isnan(news.features).any(axis=1)
other_missing = np.isnan(news.dispersion) | np.isnan(news.counts)
assert np.array_equal(feat_missing, other_missing), \
    "features/dispersion/counts NaN patterns disagree after the join"
print("NaN-alignment invariant holds across the joined series")

# %% [markdown]
# ## Run S0 (no news) -- must simply produce a finite likelihood

# %%
ll0, xf0, Pf0, resid0, nobs0 = run_spec(REGISTRY["S0"], SS_OIL, panel.log_f,
                                        panel.tau, DT)
print(f"S0:  loglik={ll0:.2f}  n_obs={nobs0}  finite={np.isfinite(ll0)}")
assert np.isfinite(ll0)
assert np.all(np.isfinite(xf0))

# %% [markdown]
# ## Nesting checks on REAL data (test_nesting.py does this on synthetic
# ## data only -- this is the same assertion, on the actual join)

# %%
llD0, xfD0, _, _, nobsD0 = run_spec(REGISTRY["SD"], SS_OIL, panel.log_f,
                                    panel.tau, DT, news, B=np.zeros((2, news.p)))
print(f"SD (B=0):  |ll - ll0| = {abs(llD0 - ll0):.2e}  "
     f"max|state diff| = {np.max(np.abs(xfD0 - xf0)):.2e}")
assert abs(llD0 - ll0) < 1e-6
assert np.max(np.abs(xfD0 - xf0)) < 1e-8

llM0, xfM0, _, _, nobsM0 = run_spec(REGISTRY["SM"], SS_OIL, panel.log_f,
                                    panel.tau, DT, news, G=np.zeros((news.p, 2)))
print(f"SM (G=0):  max|state diff| = {np.max(np.abs(xfM0 - xf0)):.2e}  "
     f"n_obs {nobs0} -> {nobsM0} (+{nobsM0 - nobs0} news rows)")
assert np.max(np.abs(xfM0 - xf0)) < 1e-8
assert nobsM0 > nobs0

# %% [markdown]
# ## Non-trivial G and B -- the actual scenario the R-shift bug would have
# ## corrupted: real gap days, real loadings, must still be finite

# %%
llD, xfD, _, _, nobsD = run_spec(REGISTRY["SD"], SS_OIL, panel.log_f, panel.tau,
                                 DT, news, B=np.array([[0.01], [0.005]]))
print(f"SD (B!=0):  loglik={llD:.2f}  finite={np.isfinite(llD)}")
assert np.isfinite(llD)
assert np.all(np.isfinite(xfD))

llM, xfM, _, resM, nobsM = run_spec(REGISTRY["SM"], SS_OIL, panel.log_f, panel.tau,
                                    DT, news, G=np.array([[0.3, 0.2]]))
print(f"SM (G!=0):  loglik={llM:.2f}  finite={np.isfinite(llM)}  n_obs={nobsM}")
assert np.isfinite(llM)
assert np.all(np.isfinite(xfM))
news_resid = resM[:, panel.m:]
finite_news_resid = news_resid[~np.isnan(news_resid)]
print(f"news residuals: {len(finite_news_resid):,} finite values, "
     f"range [{finite_news_resid.min():.2f}, {finite_news_resid.max():.2f}]")
assert np.all(np.isfinite(finite_news_resid))

llDM, xfDM, _, _, nobsDM = run_spec(REGISTRY["SDM"], SS_OIL, panel.log_f, panel.tau,
                                    DT, news, B=np.array([[0.01], [0.005]]),
                                    G=np.array([[0.3, 0.2]]))
print(f"SDM:  loglik={llDM:.2f}  finite={np.isfinite(llDM)}")
assert np.isfinite(llDM)

print("\nall integration checks passed on real joined data")
