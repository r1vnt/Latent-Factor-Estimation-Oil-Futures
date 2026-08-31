# %% [markdown]
# # Gate 2, full history
#
# `03_gate2_diagnostics.py`'s precision-first filter was decisive but small
# (n=15, one week). The BigQuery extract (`scripts/pull_gdelt_bigquery.py`)
# runs the identical conjunctive predicate server-side across 2015-2024, so
# this turns that one-week anecdote into a nine-year distribution at
# essentially no extra cost once the query exists.
#
# Also checks something the batch-timestamp finding raises but doesn't
# settle on its own: crawl latency is heterogeneous across sources, so the
# lag this measures is a mixture, not a single delay. If the median/IQR
# drift across years that's still fine for a daily model (absorbed into
# R_t) but it means Delta_ell in Sec. XI-J should be read as "typical
# delay," not a structural constant -- worth confirming rather than
# asserting.
#
# NOTE: written ahead of the extract landing (blocked on GCP credentials
# setup, which needs the user's own action -- see pull_gdelt_bigquery.py's
# docstring). Structurally mirrors 03_gate2_diagnostics.py's approach but
# not yet run against real data; re-check the release-time math below once
# it is, since this version needs the release DATE per row rather than a
# fixed single week.

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1] / "data" / "gdelt_bigquery_extract.parquet"
FIG = Path(__file__).resolve().parents[1] / "figures"
ET = "America/New_York"
RELEASE_HOUR, RELEASE_MIN = 10, 30

# %% [markdown]
# ## Load and compute lag relative to each row's own Wednesday release
#
# The BigQuery query already applied the conjunctive oil-term AND
# inventory-term predicate server-side, so this is release-candidate
# documents only -- no local re-filtering needed, just the day-of-week
# restriction the timestamp check itself needs.

# %%
df = pd.read_parquet(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
df["year"] = df.timestamp.dt.year
rel = df[df.timestamp.dt.dayofweek == 2].copy()
print(f"{len(df):,} total extracted documents, {len(rel):,} on a Wednesday")

release_time = rel.timestamp.dt.normalize() + pd.Timedelta(
    hours=RELEASE_HOUR, minutes=RELEASE_MIN)
rel["lag_min"] = (rel.timestamp - release_time).dt.total_seconds() / 60.0

# %% [markdown]
# ## The three numbers, full history

# %%
lag = rel["lag_min"].to_numpy()
neg_share = float((lag < -5).mean())
median = float(np.median(lag))
iqr = float(np.percentile(lag, 75) - np.percentile(lag, 25))
print(f"n = {len(lag):,} across {rel.year.nunique()} years "
     f"({rel.year.min()}-{rel.year.max()})")
print(f"share published >5min before release : {neg_share:6.2%}")
print(f"median lag                            : {median:6.1f} min")
print(f"inter-quartile range                  : {iqr:6.1f} min")

# %% [markdown]
# ## Stability across years -- is Delta_ell one number or a moving target?

# %%
by_year = rel.groupby("year")["lag_min"].agg(
    n="size", median="median",
    share_early=lambda s: float((s < -5).mean()),
    iqr=lambda s: np.percentile(s, 75) - np.percentile(s, 25),
)
print(by_year.round(1))
print("\nIf median/IQR drift materially across years, Delta_ell (Sec. XI-J) "
     "should be reported as a typical delay averaged over a mixture of "
     "source-level crawl latencies, not a single structural lag.")

# %% [markdown]
# ## Histogram, full history

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(np.clip(lag, -600, 600), bins=100, color="#2f6076")
ax.axvline(0, color="#94402f", lw=1.4, ls="--", label="scheduled release")
ax.set_xlabel("minutes relative to scheduled release (10:30 ET Wed)")
ax.set_ylabel(f"documents (n={len(lag):,}, {rel.year.min()}-{rel.year.max()})")
ax.set_title("Publication lag, precision-first filter, full GKG history")
ax.legend()
fig.tight_layout()
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "04_gate2_full_history_histogram.png", dpi=130)
print(f"saved {FIG / '04_gate2_full_history_histogram.png'}")
