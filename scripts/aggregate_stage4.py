"""Stage 4 of the locked news-pipeline spec: the daily aggregate table
that is the model's actual input.

Joins stage 3's per-document log (date, url_hash, outcome, finbert_score,
...) back to stage 1's output (DocumentIdentifier, Themes, tone dims) via
url_hash -- stage 3 doesn't carry the URL or themes itself, so this
recomputes the same hash stage 2 used and joins on it. I_t comes directly
from stage 2's manifest, where it was already computed from the day's full
match set (not the 100-candidate draw) with (source_domain, normalised
slug) dedup -- not recomputed here.

Per day:
    I_t          all matching docs that day, deduped -- from stage 2's
                 manifest, coverage-normalised (see below). Feeds the
                 process-noise intensity scaling.
    files_present / coverage  how many of the 24 scheduled quarter-sample
                 hours actually fetched that day, out of 24. GDELT's
                 archive has genuine gaps (mostly clustered right after
                 the Feb 2015 launch); a day with 20/24 files understates
                 I_t by construction, indistinguishably from a genuinely
                 quiet news day, unless corrected. I_t above is already
                 coverage-normalised (I_t_raw * 24 / files_present);
                 files_present/coverage are kept as explicit columns too,
                 not just folded into the correction, so a downstream user
                 can see which days rest on thin coverage rather than
                 trusting the normalisation blindly.
    n_t          count of outcome == "scored". Never the candidate count,
                 never rescaled -- a different quantity from I_t, and R_t
                 widening on a thin-n_t day is correct behaviour, not bias.
    sent_mean    mean FinBERT score over outcome == "scored" rows
    disp         within-day dispersion sigma_t (std of FinBERT scores) --
                 feeds R_t = sigma_t^2 / n_t (eq. 23)
    theme shares supply disruption / OPEC policy / geopolitics / demand --
                 fraction of that day's scored documents whose V1Themes
                 match each category. The mapping isn't specified verbatim
                 anywhere upstream; it's a keyword-substring classifier
                 against real GDELT taxonomy codes, documented below, not
                 asserted as authoritative. WB_507_ENERGY_AND_EXTRACTIVES,
                 ENV_OIL, and WB_539/WB_548 are deliberately absent here,
                 same as from the corpus predicate itself -- all four
                 empirically identified as too broad to carry signal about
                 anything, including which theme bucket a document
                 belongs in (see pull_gdelt_history.py for each finding).
    tone dims    the six V1.5Tone components, averaged over the same
                 scored subset as sent_mean, for an apples-to-apples free
                 lexicon cross-check (Sec. XIII)
    intensity    log(1 + I_t), I_t already coverage-normalised, standardised
                 against a TRAILING 252-ROW ROLLING mean/std (min_periods
                 below), not globally and not within calendar year -- I_t,
                 not n_t: n_t saturates near 60 by design once a day has
                 enough candidates, so it stops carrying information about
                 true volume past that point; I_t doesn't.

                 A within-year version of this was tried first, motivated
                 by explore/08_retention_rate_by_year.py's finding of real
                 non-trending year-to-year variance in the theme-code
                 predicate's retention rate (47% to 81%, corr(year,
                 retention) = -0.044 -- no drift, but real per-year steps,
                 which a single global standardisation would have let leak
                 into gamma in eq. (24) as spurious year-level jumps). But
                 calendar-year grouping has two problems of its own, both
                 fatal for this specific column: (1) both ends of the
                 window are partial years -- 2015 starts 19 February
                 (~223 trading days) and 2024 ends 5 April (~66 trading
                 days) -- so 2024's mean/std would be estimated off a
                 quarter of a normal year, and if Q1 news volume differs
                 systematically from the annual average (it does, most
                 years), 2024 sits on a different footing from every other
                 year purely from sample-size noise, not the effect the
                 within-year switch was meant to remove; (2) it is
                 look-ahead -- day 40 of a year gets standardised against
                 day 300's value too, which is fine for a single fixed
                 historical table but is exactly what a Study C expanding-
                 window backtest cannot tolerate once this pipeline feeds
                 one. A trailing rolling window fixes both: no year
                 boundary, so no partial-year distortion, and by
                 construction only ever uses rows up to and including t.

A day with files_present == 0 has no observation at all, not a confirmed
zero, and gets NaN across every field -- never zero, which would be the
same error as flooring a negative price into a small positive one instead
of marking it missing, just in the news channel instead of the price
channel. The output is reindexed onto the full window (stage 1's
START..END) so every day gets a row, present or not; a day with no
scored documents (whether from zero coverage or zero candidates
surviving the predicate) must be a NaN row in this table, not an absent
one, since an absent row risks silently vanishing in a later join against
the price panel rather than being carried through as an explicit missing
observation for the filter's exact missing-data handling to use.

Dedup happens before any of this: stage 1 already dedups by
DocumentIdentifier at flush time, stage 2's I_t computation dedups on
(source_domain, normalised slug), and stage 2's MinHash pass dedups the
scored set on near-duplicate content. This re-dedups on url_hash
defensively rather than trusting those upstream invariants blindly.

Usage: python scripts/aggregate_stage4.py
Output: data/stage4_daily.parquet
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_gdelt_history import OUT_PATH as STAGE1_OUTPUT   # noqa: E402
from pull_gdelt_history import START as WINDOW_START         # noqa: E402
from pull_gdelt_history import END as WINDOW_END              # noqa: E402
from pull_gdelt_history import TONE_COLS                      # noqa: E402
from pull_gdelt_history import coverage_by_day                # noqa: E402
from fetch_score_stage2 import STAGE2_MANIFEST, STAGE3_LOG     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "stage4_daily.parquet"
HOURS_SCHEDULED = 24   # quarter-sample: 1 file/hour
INTENSITY_ROLLING_WINDOW = 252   # trailing rows for intensity's rolling
                                 # mean/std -- see the module docstring
INTENSITY_MIN_PERIODS = 30   # below this many valid trailing readings
                             # there's not enough history for a meaningful
                             # "normal" to compare against; NaN rather than
                             # a noisy estimate off a handful of days. This
                             # is the price of a trailing window: the first
                             # ~30-252 days of the whole series (not of
                             # every year, unlike the calendar-year version)
                             # get a thinner-than-ideal baseline, a one-time
                             # startup transient rather than a recurring
                             # per-year distortion.

# A keyword-substring mapping from GDELT's real V1Themes taxonomy codes to
# four coarse categories. Not specified upstream -- built from taxonomy
# codes actually observed in this project's own GKG samples (ENV_OIL,
# ECON_OILPRICE, WB_539_OIL_AND_GAS_POLICY_..., etc.) plus GDELT's general
# conflict/economy vocabulary. A document can match more than one category;
# shares are per-category incidence rates, not a partition.
THEME_CATEGORIES = {
    "supply_disruption": re.compile(
        r"SANCTION|EMBARGO|PIPELINE|REFINERY|STRIKE|BLOCKADE|DISRUPT|"
        r"OUTAGE|SHUTDOWN|ARMEDCONFLICT|MARITIME_INCIDENT|PIRACY",
        re.IGNORECASE),
    "opec_policy": re.compile(r"OPEC|PRODUCTION_CUT|QUOTA|WB_539_OIL_AND_GAS_POLICY",
                              re.IGNORECASE),
    "geopolitics": re.compile(
        r"MILITARY|WAR|CONFLICT|TERROR|CRISISLEX|UNREST|PROTEST|SANCTIONS|"
        r"DIPLOMATIC|GEOPOLIT",
        re.IGNORECASE),
    "demand": re.compile(
        r"ECON_GROWTH|ECON_GDP|ECON_CONSUMER|MANUFACTURING|ECON_INTEREST_RATE|"
        r"ECON_STOCKMARKET|TRADE_DISPUTE",
        re.IGNORECASE),
}


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()[:16]


def main():
    if not STAGE1_OUTPUT.exists() or not STAGE3_LOG.exists():
        sys.exit(f"missing input: need both {STAGE1_OUTPUT} and {STAGE3_LOG}")
    if not STAGE2_MANIFEST.exists():
        sys.exit(f"missing input: need {STAGE2_MANIFEST} (has I_t per day)")

    s1 = pd.read_parquet(STAGE1_OUTPUT)
    s1 = s1.drop_duplicates(subset=["DocumentIdentifier"])
    s1["url_hash"] = s1["DocumentIdentifier"].map(url_hash)

    s3 = pd.read_parquet(STAGE3_LOG)
    s3 = s3.drop_duplicates(subset=["url_hash"])

    joined = s3.merge(s1[["url_hash", "Themes", *TONE_COLS]], on="url_hash",
                      how="left")
    scored = joined[joined["outcome"] == "scored"].copy()

    if not scored.empty:
        for cat, pattern in THEME_CATEGORIES.items():
            scored[f"theme_{cat}"] = scored["Themes"].fillna("").str.contains(
                pattern, regex=True)

        agg = scored.groupby("date").agg(
            sent_mean=("finbert_score", "mean"),
            disp=("finbert_score", "std"),
            n_t=("finbert_score", "size"),
            **{f"theme_{cat}": (f"theme_{cat}", "mean") for cat in THEME_CATEGORIES},
            **{c: (c, "mean") for c in TONE_COLS},
        ).reset_index()
        agg["disp"] = agg["disp"].fillna(0.0)   # a single-document day has no spread
    else:
        agg = pd.DataFrame(columns=["date", "sent_mean", "disp", "n_t",
                                   *[f"theme_{c}" for c in THEME_CATEGORIES],
                                   *TONE_COLS])

    it = pd.read_csv(STAGE2_MANIFEST, dtype={"day": str})[["day", "I_t"]]
    agg = agg.merge(it, left_on="date", right_on="day", how="outer")
    # left_on/right_on differ, so pandas keeps both "date" and "day" rather
    # than deduping them -- every row that came only from `it` (I_t exists,
    # but stage 3 has no scored rows for that day) has date == NaN and
    # day == the real key, and the reverse for a scored-only row.
    agg["date"] = agg["date"].fillna(agg["day"])
    agg = agg.drop(columns="day")

    # Coverage: files actually fetched vs the 24 scheduled. Missing entirely
    # from the manifest (files_present=0) is NOT the same as I_t=0 with
    # attempted-but-empty coverage -- both end up coverage=0 here, and both
    # correctly produce NaN below, but the reasoning differs: the first is
    # "we never got the data", the second is "we got the data and it says
    # zero," and this table intentionally cannot tell them apart because
    # eq. (24) shouldn't either -- both are "no reliable volume reading."
    cov = coverage_by_day()
    agg["files_present"] = agg["date"].map(lambda d: cov.get(d, 0))
    agg["coverage"] = agg["files_present"] / HOURS_SCHEDULED

    # I_t normalised by coverage where any data exists; NaN (not 0) where
    # none does. Every other field: NaN wherever there's no I_t, since a
    # missing news observation must never masquerade as a confirmed zero --
    # the same rule Panel.build applies to non-positive prices, just in the
    # news channel.
    has_data = agg["files_present"] > 0
    agg["I_t"] = np.where(has_data, agg["I_t"] / agg["coverage"].replace(0, np.nan),
                          np.nan)

    value_cols = ["sent_mean", "disp", "n_t",
                 *[f"theme_{c}" for c in THEME_CATEGORIES], *TONE_COLS]
    for c in value_cols:
        if c in agg.columns:
            agg[c] = np.where(has_data, agg[c], np.nan)

    # log1p(I_t) only for now -- the trailing rolling standardisation below
    # needs correctly sorted, gap-complete rows, which don't exist until
    # after the reindex further down.
    agg["intensity"] = np.where(has_data, np.log1p(agg["I_t"]), np.nan)

    # Reindex onto the full window so every day is a row, present or not --
    # an absent date risks silently vanishing in a later join against the
    # price panel; a NaN row survives it and reaches the filter's exact
    # missing-data handling instead.
    agg["date"] = pd.to_datetime(agg["date"], format="%Y%m%d")
    # WINDOW_END is exclusive in pull_gdelt_history.py (the last fetched day
    # is WINDOW_END - 1 day) -- date_range is inclusive of both ends, so the
    # upper bound here is shifted back one day to match, rather than adding
    # a day that stage 1 will never have data for.
    full_index = pd.date_range(WINDOW_START.date(),
                               (WINDOW_END - pd.Timedelta(days=1)).date(), freq="D")
    agg = agg.set_index("date").reindex(full_index).rename_axis("date").reset_index()
    agg["files_present"] = agg["files_present"].fillna(0).astype(int)
    agg["coverage"] = agg["files_present"] / HOURS_SCHEDULED

    # Trailing rolling standardisation, now that rows are sorted by date
    # with every day present (a rolling window over unsorted or gappy rows
    # would silently use the wrong neighbours) -- see the module docstring
    # for why this replaced a within-calendar-year version. Right-aligned
    # (pandas' default): the window at row t is [t-251, t], never t+1
    # onward, so this never uses information from a day after the one
    # it's standardising -- the property a within-year version couldn't
    # have, since day 40 of a year is standardised using day 300's value
    # too.
    log_int = agg["intensity"]
    roll_mean = log_int.rolling(INTENSITY_ROLLING_WINDOW,
                                min_periods=INTENSITY_MIN_PERIODS).mean()
    roll_std = log_int.rolling(INTENSITY_ROLLING_WINDOW,
                               min_periods=INTENSITY_MIN_PERIODS).std()
    agg["intensity"] = (log_int - roll_mean) / roll_std.replace(0, np.nan)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_PATH, index=False)

    n_total = len(agg)
    n_missing = agg["I_t"].isna().sum()
    n_partial = ((agg["files_present"] > 0) & (agg["files_present"] < HOURS_SCHEDULED)).sum()
    print(f"wrote {n_total:,} days -> {OUT_PATH}")
    print(f"  span: {agg.date.min().date()} .. {agg.date.max().date()}")
    print(f"  {n_missing:,} days with no observation (NaN, not zero)")
    print(f"  {n_partial:,} days with partial coverage (I_t normalised)")
    if agg["I_t"].notna().any():
        print(f"  I_t: mean={agg.I_t.mean():.1f} median={agg.I_t.median():.0f}")
    if agg["n_t"].notna().any():
        print(f"  n_t: mean={agg.n_t.mean():.1f} median={agg.n_t.median():.0f} "
             f"min={agg.n_t.min():.0f} max={agg.n_t.max():.0f}")
    for cat in THEME_CATEGORIES:
        col = f"theme_{cat}"
        if agg[col].notna().any():
            print(f"  {col}: mean share {agg[col].mean():.2%}")


if __name__ == "__main__":
    main()
