# %% [markdown]
# # Timestamp verification
#
# **Run this on day one, on a one-week sample, before the full download finishes.**
#
# This is the single highest-severity risk in the project. If recorded news
# timestamps are *crawl* times rather than *publication* times, the resulting
# look-ahead is indistinguishable from genuine predictive content — it will look
# exactly like alpha, and every result in Study B will be corrupted.
#
# The test uses events whose release time is known to the minute from outside
# the corpus. The EIA Weekly Petroleum Status Report is published Wednesdays at
# 10:30 US Eastern. Articles reporting it cannot legitimately predate that.
#
# Three things must hold:
#
# 1. The lag distribution is **non-negative** — essentially no mass before the
#    release. Negative mass means crawl times, or timezone corruption.
# 2. It is **tight** — a sharp mode within an hour or two. A diffuse
#    distribution means timestamps are being assigned at index time.
# 3. It is **stable across the sample** — no regime shift where the archive
#    changed its convention mid-history.
#
# Exploration only. Nothing here computes a number that appears in the paper;
# if it needs to, it moves into `csr/news/` with a test.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ET = "America/New_York"
RELEASE_HOUR, RELEASE_MIN = 10, 30      # EIA WPSR, Wednesdays
KEYWORDS = ["eia", "inventor", "stockpile", "crude stocks", "petroleum status"]

# %% [markdown]
# ## Load a sample
#
# Expected columns: `timestamp` (parseable, ideally tz-aware) and `text`.
# Point this at whatever the first slice of the download produced.

# %%
SAMPLE = Path("../data/news_sample.parquet")   # <- edit

if SAMPLE.exists():
    df = pd.read_parquet(SAMPLE)
else:
    # Synthetic stand-in so the notebook runs before data lands.
    # Deliberately includes a contaminated subset, so you can see what
    # failure looks like before you meet it for real.
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-03 10:30", tz=ET)
    weeks = 40
    rows = []
    for w in range(weeks):
        rel = base + pd.Timedelta(weeks=w)
        n_ok = rng.integers(8, 25)
        for _ in range(n_ok):                      # well-behaved: minutes after
            rows.append((rel + pd.Timedelta(minutes=float(rng.gamma(2, 12))),
                         "EIA crude inventories fell"))
        for _ in range(rng.integers(0, 4)):        # contaminated: crawl times
            rows.append((rel + pd.Timedelta(hours=float(rng.normal(-3, 6))),
                         "EIA inventory report preview"))
    df = pd.DataFrame(rows, columns=["timestamp", "text"])
    print(f"no sample found at {SAMPLE} — using synthetic data with "
          "deliberate contamination")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
print(f"{len(df):,} documents   {df.timestamp.min()} .. {df.timestamp.max()}")

# %% [markdown]
# ## Isolate documents that report the release

# %%
mask = df["text"].str.lower().str.contains("|".join(KEYWORDS), regex=True)
rel_docs = df[mask].copy()
rel_docs = rel_docs[rel_docs.timestamp.dt.dayofweek == 2]      # Wednesday
print(f"{len(rel_docs):,} Wednesday release-related documents")

release_time = rel_docs.timestamp.dt.normalize() + pd.Timedelta(
    hours=RELEASE_HOUR, minutes=RELEASE_MIN)
rel_docs["lag_min"] = (rel_docs.timestamp - release_time).dt.total_seconds() / 60.0

# %% [markdown]
# ## The three checks

# %%
lag = rel_docs["lag_min"].to_numpy()
neg_share = float((lag < -5).mean())
median = float(np.median(lag))
iqr = float(np.percentile(lag, 75) - np.percentile(lag, 25))

print(f"  share published >5 min BEFORE release : {neg_share:6.2%}   (want <1%)")
print(f"  median lag                            : {median:6.1f} min")
print(f"  inter-quartile range                  : {iqr:6.1f} min   (want <120)")

verdict = "PASS" if (neg_share < 0.01 and 0 <= median < 240 and iqr < 120) else "FAIL"
print(f"\n  VERDICT: {verdict}")
if verdict == "FAIL":
    print("  -> Do not proceed with the measurement specification (SM).")
    print("     Either fix the timestamp source or restrict to the driver")
    print("     specification with a conservative lag. See Sec. XIII.")

# %% [markdown]
# ## Look at the distribution
#
# A clean corpus shows a sharp right-skewed mode a few minutes after release.
# Mass to the left of zero is the failure mode.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

axes[0].hist(np.clip(lag, -600, 600), bins=80, color="#2f6076")
axes[0].axvline(0, color="#94402f", lw=1.4, ls="--")
axes[0].set_xlabel("minutes after scheduled release")
axes[0].set_ylabel("documents")
axes[0].set_title("Publication lag distribution")

rel_docs["week"] = rel_docs.timestamp.dt.tz_localize(None).dt.to_period("W").dt.start_time
by_week = rel_docs.groupby("week")["lag_min"].median()
axes[1].plot(by_week.index, by_week.to_numpy(), lw=1.0, color="#2f6076")
axes[1].axhline(0, color="#94402f", lw=1.0, ls="--")
axes[1].set_ylabel("median lag (min)")
axes[1].set_title("Stability through the sample")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## If it fails
#
# Ranked by what to try first:
#
# 1. **Timezone.** A whole-hours offset (±60, ±300, ±480) is almost always a
#    tz bug, not contamination. Check whether the source stamps UTC and you are
#    reading it as local, or vice versa. Cheapest fix, most common cause.
# 2. **Field choice.** Many archives carry several date fields — `seendate`,
#    `publishdate`, `crawldate`. GDELT's `seendate` is when GDELT saw it, not
#    when it was published. Look for a publisher-supplied field first.
# 3. **Aggregator republication.** Syndicated copy is re-stamped on
#    republication. Dedup before drawing conclusions — this also matters for
#    `I_t`, since inflated counts shrink `R_t` exactly where the filter is most
#    sensitive to it.
# 4. **If none of these fix it:** the corpus cannot support the measurement
#    reading. Fall back to the driver specification with a conservative lag and
#    say so in the write-up. Do not proceed and hope.
#
# The model-based cross-check is the free lead–lag parameter of §XI-J: estimate
# `Δ_ℓ` over a grid that includes negative values. A likelihood preferring
# `Δ_ℓ < 0` is independent evidence of the same problem.
