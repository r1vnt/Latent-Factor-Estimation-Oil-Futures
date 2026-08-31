"""Stage 4 of the locked news-pipeline spec: the daily aggregate table
that is the model's actual input.

Joins stage 3's per-document log (date, url_hash, fetch_ok, finbert_ok,
finbert_score, ...) back to stage 1's output (DocumentIdentifier, Themes,
tone dims) via url_hash -- stage 3 doesn't carry the URL or themes itself,
so this recomputes the same hash stage 2 used and joins on it.

Per day:
    sent_mean    mean FinBERT score, successful scores only
    disp         within-day dispersion sigma_t (std of FinBERT scores) --
                 feeds R_t = sigma_t^2 / n_t (eq. 23)
    n_t          count of successful scores. Successful scores only --
                 never the candidate count, never rescaled. This is a
                 different quantity from the quarter-sample undercount
                 pull_gdelt_history.py's docstring used to warn about: that
                 was about using stage 1's raw match count as a volume
                 proxy, which this pipeline doesn't do. n_t here is
                 genuinely how many documents were averaged, and R_t
                 widening on a thin day is correct behaviour, not bias.
    theme shares supply disruption / OPEC policy / geopolitics / demand --
                 fraction of that day's *successfully scored* documents
                 whose V1Themes match each category. The mapping itself
                 isn't specified verbatim anywhere upstream; it's a
                 keyword-substring classifier against real GDELT taxonomy
                 codes, documented below, not asserted as authoritative.
    tone dims    the six V1.5Tone components, averaged over the same
                 successfully-scored subset as sent_mean, for an
                 apples-to-apples free lexicon cross-check (Sec. XIII)
    intensity    standardised log(1 + n_t)

Dedup happens before any of this: stage 1 already dedups by
DocumentIdentifier at flush time and stage 2 samples from `.unique()`
URLs, so this re-dedups defensively rather than trusting that upstream
invariant blindly.

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
from pull_gdelt_history import TONE_COLS                    # noqa: E402
from fetch_score_stage2 import STAGE3_LOG                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "stage4_daily.parquet"

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
        r"ECON_STOCKMARKET|TRADE_DISPUTE|WB_507_ENERGY_AND_EXTRACTIVES",
        re.IGNORECASE),
}


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()[:16]


def main():
    if not STAGE1_OUTPUT.exists() or not STAGE3_LOG.exists():
        sys.exit(f"missing input: need both {STAGE1_OUTPUT} and {STAGE3_LOG}")

    s1 = pd.read_parquet(STAGE1_OUTPUT)
    s1 = s1.drop_duplicates(subset=["DocumentIdentifier"])
    s1["url_hash"] = s1["DocumentIdentifier"].map(url_hash)

    s3 = pd.read_parquet(STAGE3_LOG)
    s3 = s3.drop_duplicates(subset=["url_hash"])

    joined = s3.merge(s1[["url_hash", "Themes", *TONE_COLS]], on="url_hash",
                      how="left")
    scored = joined[joined["finbert_ok"]].copy()
    if scored.empty:
        sys.exit("no successfully-scored documents in stage 3 log yet")

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
    agg["intensity"] = np.log1p(agg["n_t"])
    agg["intensity"] = (agg["intensity"] - agg["intensity"].mean()) / agg["intensity"].std()

    agg["date"] = pd.to_datetime(agg["date"], format="%Y%m%d")
    agg = agg.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(agg):,} days -> {OUT_PATH}")
    print(f"  span: {agg.date.min().date()} .. {agg.date.max().date()}")
    print(f"  n_t: mean={agg.n_t.mean():.1f} median={agg.n_t.median():.0f} "
         f"min={agg.n_t.min()} max={agg.n_t.max()}")
    for cat in THEME_CATEGORIES:
        print(f"  theme_{cat}: mean share {agg[f'theme_{cat}'].mean():.2%}")


if __name__ == "__main__":
    main()
