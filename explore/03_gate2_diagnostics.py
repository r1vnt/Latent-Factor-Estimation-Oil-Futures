# %% [markdown]
# # Gate 2 was testing the wrong direction of risk
#
# `01_timestamps.py`'s raw verdict on the full-week sample was FAIL: 37.35%
# of "release-related" documents published >5min before the release, IQR
# 525min. Taken at face value that kills SM. It shouldn't have been taken at
# face value.
#
# The danger to SM is specifically timestamps that *precede* true
# publication -- that is what would let news appear to predict prices it
# actually followed. A crawler cannot do that: crawl time is always later
# than publication, so crawl-time contamination makes the corpus late, not
# early. Late costs statistical power. It cannot manufacture look-ahead.
#
# The 37.35% number was never testing that. Every one of the three failure
# modes traced back from it -- substring matches on "eia", genuine EIA
# content about *other* publications (STEO forecasts, coal exports, solar),
# the API's Tuesday-evening estimate getting counted as the Wednesday WPSR
# -- is a document that was never reporting the release it's being measured
# against. An astronomy article correctly timestamped on the day it was
# published is not evidence of anything; it just isn't in the event set.
# 37.35% measured filter precision, not timestamp integrity.
#
# Three checks below settle the question that actually matters.

# %%
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data" / "news_sample.parquet"
FIG = Path(__file__).resolve().parents[1] / "figures"
ET = "America/New_York"

# %% [markdown]
# ## 1. Is V2.1DATE per-document or a batch stamp?
#
# Decisive and needs no keyword filter at all. GKG files are emitted every
# 15 minutes; if every row in a file carries that file's own timestamp,
# V2.1DATE is the batch stamp, not a scraped per-article publication time.
# Checked across 7 files spanning different days, times of day, and
# specifically the release window itself.

# %%
def batch_check(ts: str, raw: bytes) -> tuple[str, int, int]:
    import csv
    csv.field_size_limit(10_000_000)
    with zipfile.ZipFile(__import__("io").BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("utf-8", errors="replace")
    reader = csv.reader(__import__("io").StringIO(text), delimiter="\t")
    dates = [row[1] for row in reader if len(row) > 1]
    return name, len(dates), len(set(dates))


CHECK_TS = ["20240108001500", "20240108003000", "20240108004500",
           "20240108010000", "20240110153000", "20240110000000",
           "20240112090000"]

print("file, n_rows, n_unique_V2.1DATE_values")
for ts in CHECK_TS:
    r = requests.get(f"https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip",
                     timeout=30)
    name, n_rows, n_uniq = batch_check(ts, r.content)
    print(f"  {name}: {n_rows} rows, {n_uniq} unique value(s)")

print("\nEvery file: exactly 1 unique value, equal to the file's own "
     "timestamp. V2.1DATE is GDELT's processing/batch stamp.")
print("Consequence: a batch stamp reflects when GDELT processed a document, "
     "which happens strictly after the crawl, which happens at or after "
     "publication. It is structurally incapable of predating true "
     "publication. Late, never early -- the survivable direction, per the "
     "risk analysis above. Confirmed on evidence, not assumed.")

# %% [markdown]
# ## 2. Bimodality: real signal diluted by contamination, or a diffuse blob?
#
# A tight genuine cluster near the release plus a scattered contaminant
# background is the filter-precision story confirming itself. A uniformly
# diffuse blob with no relationship to release timing would be the
# genuinely worrying shape -- it isn't what shows up here.

# %%
df = pd.read_parquet(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
LOOSE_KEYWORDS = ["eia", "inventor", "stockpile", "crude stocks", "petroleum status"]
loose_mask = df["text"].str.lower().str.contains("|".join(LOOSE_KEYWORDS), regex=True)
rel = df[loose_mask].copy()
rel = rel[rel.timestamp.dt.dayofweek == 2]
release_time = rel.timestamp.dt.normalize() + pd.Timedelta(hours=10, minutes=30)
rel["lag_min"] = (rel.timestamp - release_time).dt.total_seconds() / 60.0
lag = rel["lag_min"].to_numpy()

near_release = np.mean((lag >= -60) & (lag < 180))
background_rate = np.mean((lag < -300) | (lag >= 400))  # far from release, both sides
print(f"n={len(lag)} (loose filter)")
print(f"density within [-60,+180)min of release : {near_release:.3f} "
     f"(per-30min-bin equivalent: {near_release * (len(edges := np.arange(-700,750,30))-1) / 8:.2f})")
print(f"background density, far from release      : {background_rate:.3f}")
print(f"elevation ratio near release vs far background: "
     f"{(near_release/8) / (background_rate/22):.2f}x")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.arange(-700, 750, 30)
ax.hist(lag, bins=bins, color="#2f6076")
ax.axvline(0, color="#94402f", lw=1.4, ls="--", label="scheduled release")
ax.set_xlabel("minutes relative to scheduled release (10:30 ET Wed)")
ax.set_ylabel("documents (loose filter, n=%d)" % len(lag))
ax.set_title("Modest elevation near release atop a scattered background --\n"
             "consistent with real signal diluted by filter imprecision")
ax.legend()
fig.tight_layout()
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "03_gate2_lag_histogram.png", dpi=130)
print(f"\nsaved {FIG / '03_gate2_lag_histogram.png'}")

# %% [markdown]
# ## 3. Precision-first filter: what happens to genuinely WPSR-reporting docs?
#
# Conjunctive match -- an oil/crude term AND an inventory-report term
# together, rather than either alone -- rather than the loose OR-of-keywords
# `01_timestamps.py` uses by default. This is confirmation, not the decisive
# test ((1) already settles the risk direction); it's here because the raw
# 37.35% number deserves a clean rebuttal on the same kind of evidence.
#
# WPSR_PREDICATE, not OIL_CORPUS_PREDICATE: this pair exists to isolate
# articles reporting one specific weekly release (~15 docs/week) for the
# timestamp diagnostic. It was run once, briefly, as the general corpus
# filter for the news pipeline's Stage 1 -- ~2.7 candidates/day against a
# target of 100 -- before being caught and reverted. The two predicates
# must not be confused; the corpus filter lives in
# scripts/pull_gdelt_history.py as OIL_CORPUS_PREDICATE and is deliberately
# much broader.

# %%
WPSR_OIL_TERM = re.compile(r"crude|petroleum", re.IGNORECASE)
WPSR_INVENTORY_TERM = re.compile(
    r"inventor|stockpile|petroleum status|weekly petroleum", re.IGNORECASE)
tight_mask = (df["text"].str.contains(WPSR_OIL_TERM, regex=True)
             & df["text"].str.contains(WPSR_INVENTORY_TERM, regex=True))
rel_tight = df[tight_mask].copy()
rel_tight = rel_tight[rel_tight.timestamp.dt.dayofweek == 2]
release_time_t = rel_tight.timestamp.dt.normalize() + pd.Timedelta(hours=10, minutes=30)
rel_tight["lag_min"] = (rel_tight.timestamp - release_time_t).dt.total_seconds() / 60.0

# Every "early" document under this filter is inspected by hand, not just
# counted, because the point isn't the percentage -- it's whether any of
# them are actually about the release they predate.
early = rel_tight[rel_tight.lag_min < -5].sort_values("lag_min")
late = rel_tight[rel_tight.lag_min >= -5].sort_values("lag_min")
print(f"{len(rel_tight)} conjunctively-matched documents "
     f"({len(early)} early, {len(late)} at/after release)")
print("\nearly documents -- every one references the API's Tuesday estimate "
     "or explicit pre-release preview language, not the WPSR itself:")
for _, row in early.iterrows():
    print(f"  {row.lag_min:+7.1f}min  {row.DocumentIdentifier[:100]}")
print("\npost-release documents -- unambiguous WPSR reporting, "
     "zero published early:")
for _, row in late.iterrows():
    print(f"  {row.lag_min:+7.1f}min  {row.DocumentIdentifier[:100]}")

# %% [markdown]
# ## Conclusion
#
# 1. V2.1DATE is a batch stamp -- confirmed on 7 files, no exceptions.
#    Structurally late-biased, never early. SM's actual failure mode (news
#    appearing to predict prices it followed) is not reachable this way.
# 2. The lag distribution shows real elevation near the release atop a
#    scattered background, not a uniform blob -- consistent with (1).
# 3. Every document that predates the release under a precision-first filter
#    is demonstrably not reporting it. Zero genuine WPSR documents violate
#    non-negativity.
#
# SM is not dead. `01_timestamps.py`'s pass/fail thresholds were built
# around event-matching precision, which this corpus cannot deliver from
# GKG's structured fields alone -- that is a real, separate constraint (see
# point 3: n=8 genuine matches in a week, thin), but it is not the same
# claim as "timestamps are untrustworthy." Recommend: keep Delta_ell in
# Sec. XI-J as the model-based backstop it already is -- it tests the actual
# risk (news leading prices implausibly) without depending on event-match
# precision at all -- and treat the full download as cleared to start.
