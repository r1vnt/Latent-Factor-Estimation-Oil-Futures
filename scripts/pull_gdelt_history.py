"""Stage 1 of the locked news-pipeline spec: resumable, checkpointed GKG
history pull, quarter-sample resolution.

Window: 2015-02-19 (GDELT v2 start) -> 2024-04-05 (EIA series end), ~3,330
days. Quarter-sample (1 file/hour, not all 4): 22 Mbps measured on this
connection, confirmed a real ceiling (24 workers moved nothing over 12), so
full resolution is ~9 days against this one's ~2.3.

Five fields kept, everything else discarded at parse time -- V2GCAM's
2,300+ dimensions, the enhanced per-mention-offset fields, and
V2RelatedImages are never written:
    V2.1DATE            date (batch stamp; 15-min granularity is immaterial
                         at daily frequency)
    DocumentIdentifier   URL, fetched in stage 2
    SourceCommonName     attrition diagnostic (stage 5)
    V1Themes             topic shares -- V1, not V2EnhancedThemes, which
                         repeats each theme once per mention with offsets
    V1.5Tone              all six dims, kept as the free lexicon cross-check
                         for Sec. XIII, not just the headline tone value

OIL_CORPUS_PREDICATE (OR of empirically-derived theme codes and URL token
matches) -- NOT WPSR_PREDICATE, the conjunctive oil-AND-inventory filter
in pull_gdelt_bigquery.py and explore/03_gate2_diagnostics.py step 3.
WPSR_PREDICATE was built and validated for one job: isolating articles
reporting the Weekly Petroleum Status Report, for the Gate-2 timestamp
test -- it gave 15 documents/week, exactly right for that, and was run
here for one session (44min, since discarded) before being caught as
catastrophically wrong for a daily sentiment corpus: ~2.7 candidates/day
against a target of 100, most days with n_t < 2 where R_t = sigma_t^2/n_t
isn't even defined. The two predicates serve different purposes, are
named differently so they can't be confused again, and must stay
different -- WPSR_PREDICATE remains correct and unchanged for the Gate-2
scripts.

OIL_THEME_CODES is not a guessed keyword list: derived by downloading one
full-resolution day (2024-01-09, a non-release Tuesday, 96 files,
explore/06_derive_oil_theme_codes.py), counting every V1Themes code across
167,487 rows, and selecting the energy/oil-relevant ones from the ~9,800
observed by inspection -- screening out substring false positives a naive
regex would have caught (POLITICAL_TURMOIL via "turm-OIL", the GASTRO*
disease family via "gas", ALOPECIA via "al-OPEC-ia", SOIL_MANAGEMENT via
"s-OIL", OLIVE_OIL/COCONUT_OIL, TEAR_GAS) and excluding non-oil energy
sources (solar/wind/nuclear/geothermal/biomass) that would dilute the
corpus with off-topic renewable-energy stories. Frequencies from that
sample are recorded next to each code below.

A checkpoint manifest (data/gdelt_history_manifest.csv) records every
timestamp already processed, so a restart after any interruption resumes
rather than re-downloading from the start; output is flushed periodically
too, so a crash costs at most one checkpoint interval.

Usage:
    python scripts/pull_gdelt_history.py --dry-run
    python scripts/pull_gdelt_history.py --run       # resumes automatically
                                                       # if rerun
    python scripts/pull_gdelt_history.py --sample full --dry-run  # for
                                                       # comparison only;
                                                       # quarter is the
                                                       # locked decision
"""

import argparse
import csv
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parents[1]
# A directory of small per-checkpoint part files, not one growing file.
# flush_output() used to read the entire accumulated parquet back into
# memory, concat the new batch, drop_duplicates, and rewrite the whole
# thing -- on every checkpoint (every CHECKPOINT_EVERY files). At ~1.29M
# rows/714MB that read-modify-write cycle needed enough contiguous memory
# that a checkpoint's dedup step (`combined.take(...)` inside
# drop_duplicates, internally) raised `pyarrow.lib.ArrowMemoryError: malloc
# of size 1066829760 failed` with the system down to 0.6GB free -- and
# because that exception had no handler between it and main()'s top level,
# it unwound straight through the `finally` block's own retry of the same
# oversized flush and past the `ex.shutdown()` call, leaving the
# ThreadPoolExecutor's already-queued fetch workers running unsupervised:
# every subsequent fetch they completed kept landing in the `pending` set
# main() still held (nothing left to drain it), so the process kept
# growing in memory *after* it had already crashed, never exiting on its
# own. A directory of small parts bounds each write to one checkpoint
# batch's worth of rows, forever, instead of scaling with total corpus
# size -- dedup on DocumentIdentifier moves entirely to read time, which
# both downstream readers already do or tolerate (aggregate_stage4.py
# calls drop_duplicates after reading; fetch_score_stage2.py's
# candidates_for_day() already narrows to .unique() URLs regardless of
# upstream duplication). pd.read_parquet() treats a directory of parquet
# files as one dataset transparently, so every existing call site that
# reads OUT_PATH needed no change.
OUT_PATH = ROOT / "data" / "gdelt_history_extract"
MANIFEST_PATH = ROOT / "data" / "gdelt_history_manifest.csv"

START = datetime(2015, 2, 19, 0, 0, tzinfo=timezone.utc)   # GDELT v2 start
END = datetime(2024, 4, 6, 0, 0, tzinfo=timezone.utc)       # exclusive; covers
                                                             # through 2024-04-05,
                                                             # the EIA series end

BASE_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
MAX_WORKERS = 12    # more workers didn't raise throughput at this connection's
                    # measured ceiling; not worth raising further
RETRIES = 3
TIMEOUT = (10, 15)
HARD_DEADLINE = 60
STALL_LIMIT = 120
CHECKPOINT_EVERY = 25   # flush manifest + accumulated output every N files

# OIL_THEME_CODES: sampled 2024-01-09 (Tuesday, non-release day), 96 files,
# 167,487 rows, ~9,800 distinct codes observed. Frequency = row count in
# that one-day sample; a code's absence from GDELT's globally-top-300 by
# raw frequency does NOT mean it's rare in oil news specifically -- generic
# administrative/crisis/governance tags (TAX_FNCACT_*, CRISISLEX_*, WB_*
# health/education) dominate the overall top 300, so this was derived by
# searching all observed codes, not just the top 300.
#
# WB_507_ENERGY_AND_EXTRACTIVES was in the first cut and removed after step-6
# validation on 2024-01-15 (quarter-sample, ordinary day): it accounted for
# 71.5% of that day's candidates (1933/2703) and 53% matched via it ALONE,
# with nothing else corroborating -- including a ranked list of Frasier
# episodes, an Elvis festival writeup, a cookware sale, and an NHS ward
# occupancy story, none remotely oil-related. GDELT's classifier applies
# this broad World Bank sector code far more liberally than the "energy
# extractives" name suggests. Dropping it took that day's candidate count
# from 2703 to 1270 -- still comfortably inside the 1000-2000 target -- and
# a re-sampled eyeball check came back clean (see explore/06 for both runs).
#
# ENV_OIL, WB_539_OIL_AND_GAS_POLICY_STRATEGY_AND_INSTITUTIONS, and
# WB_548_PPP_IN_OIL_AND_GAS removed on the same evidence pattern, caught by
# a routine check after ~1.4M accumulated rows (throughput was running
# ~3.4x the validation-day rate -- worth checking on its own). A 50-row
# random sample from the live corpus, eyeballed, was >20% clearly
# unrelated -- Hillary Clinton campaign coverage, a Zillow listing, a
# biopharma press release, restaurant reviews, arthritis symptoms, a
# sewage spill, Brazil impeachment coverage, drag racing -- and tracing
# each back showed ENV_OIL present in 73% of all matches in a 5000-row
# sample and the SOLE signal (no corroboration) for 28.8% of them;
# WB_539+WB_548 (which co-occur as a pair) sole-signal for another 6.9%.
# The initially-suspected mechanism (the "brent"/"oil" URL tokens) turned
# out not to be it: URL-only matches with no theme corroboration were only
# 1.8% of the corpus at this sample size. Dropping all three theme codes
# leaves 63.4% of the prior corpus -- ~3200/day estimated, still
# comfortably inside the target band -- re-validated clean, see
# explore/06 and explore/07 (retroactive re-filter + re-validation).
OIL_THEME_CODES = frozenset({
    "ENV_NATURALGAS",                              # 1620
    "ECON_OILPRICE",                               # 1128
    "FUELPRICES",                                  # 596
    "TAX_FNCACT_OILMAN",                           # 558
    "WB_2298_REFINERIES",                          # 297
    "WB_549_OIL_AND_GAS_SYSTEMS",                  # 252
    "ECON_HEATINGOIL",                             # 231
    "MANMADE_DISASTER_GAS_EXPLOSION",              # 214
    "ECON_GASOLINEPRICE",                          # 195
    "WB_544_MID_AND_DOWNSTREAM_OIL_AND_GAS",       # 135
    "ENV_BIOFUEL",                                 # 129 -- oil-substitute transport fuel
    "MANMADE_DISASTER_OIL_SPILL",                  # 122
    "WB_1768_OIL_AND_GAS_PIPELINE",                # 118
    "WB_1751_LIQUEFIED_NATURAL_GAS",               # 91
    "ECON_NATGASPRICE",                            # 78
    "WB_532_BIOFUELS_ENERGY",                      # 78
    "WB_2290_OIL_AND_GAS_EXPORT",                  # 76
    "WB_1815_FUEL_TAXES",                          # 50 -- moves gasoline/diesel demand
    "WB_551_GAS_TRANSPORTATION_STORAGE_AND_DISTRIBUTION",  # 49
    "WB_1627_ENERGY_SUBSIDIES",                    # 47 -- commonly fuel subsidies in practice
    "WB_1628_FUEL_SUBSIDIES",                      # 40
    "WB_2291_OIL_AND_GAS_DISTRIBUTION",            # 38
    "WB_2273_UPSTREAM_OIL_AND_GAS",                # 36
    "WB_512_GAS_FIRED_POWER",                      # 32
    "WB_545_NATIONAL_OIL_COMPANIES",               # 26
    "WB_540_GOVERNANCE_FOR_OIL_AND_GAS",           # 26
    "WB_2289_OIL_AND_GAS_REFINING",                # 13
    "WB_1702_OILFIELD_SERVICES",                   # 12
    "WB_1753_GAS_TO_POWER",                        # 5
    "WB_2304_OFFSHORE_OIL_AND_GAS_PRODUCTION",     # 3
    "WB_2292_OIL_AND_GAS_PRODUCTION_SHARING",      # 1
    "WB_2294_PROFIT_OIL",                          # 1
    "WB_513_OIL_FIRED_POWER",                      # 1
    # Excluded, not oil/gas: WB_525_RENEWABLE_ENERGY, WB_528_SOLAR_ENERGY,
    # WB_509_NUCLEAR_ENERGY, WB_529_WIND_ENERGY, WB_530_GEOTHERMAL_ENERGY,
    # WB_531_BIOMASS_ENERGY, WB_533_ENERGY_EFFICIENCY, WB_2428_ICT_AND_ENERGY,
    # WB_2122_SMART_ENERGY, WB_1767_ENERGY_FINANCE, WB_521_ENERGY_ACCESS,
    # WB_520_PPP_IN_ENERGY_AND_POWER, WB_582_GREENHOUSE_GAS_ACCOUNTING,
    # MANMADE_DISASTER_GAS_MAIN (local utility incident, not market-moving).
})

# URL token match: exact tokens after splitting DocumentIdentifier on
# non-alphanumeric characters, not substrings -- "oil" as a substring
# catches "oilers", "spoiled", "toilet"; as a token bounded by /, -, ., _
# it does not.
URL_OIL_TOKENS = frozenset({
    "crude", "opec", "petroleum", "wti", "brent", "refinery", "refineries",
    "gasoline", "diesel", "barrel", "barrels", "oil", "oilprice",
})
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")


def theme_matches(v1themes: str) -> bool:
    return any(code in OIL_THEME_CODES for code in v1themes.split(";"))


def url_matches(url: str) -> bool:
    tokens = _TOKEN_SPLIT.split(url.lower())
    return any(t in URL_OIL_TOKENS for t in tokens)


_SLUG_DATE = re.compile(r"/\d{4}/\d{1,2}/\d{1,2}/")
_SLUG_ISODATE = re.compile(r"/\d{4}-\d{1,2}-\d{1,2}(?=/|$)")
_SLUG_LONGNUM = re.compile(r"-?\d{4,}")
_SLUG_NUMSEGMENT = re.compile(r"/\d+(?=/|$)")


def normalize_slug(url: str) -> str:
    """URL path, with dates/IDs/trailing numerics stripped -- for I_t dedup
    on (source_domain, normalised slug): catches near-duplicate paths on
    the same site (tracking params, timestamped variants), which is the
    only dedup possible before any article text has been fetched."""
    from urllib.parse import urlparse
    path = urlparse(url).path.lower()
    path = _SLUG_DATE.sub("/", path)
    path = _SLUG_ISODATE.sub("", path)
    path = _SLUG_LONGNUM.sub("", path)
    path = _SLUG_NUMSEGMENT.sub("/", path)
    return path.rstrip("/")


COL_DATE = 1
COL_SOURCE_NAME = 3
COL_DOCID = 4
COL_V1THEMES = 7    # not V2ENHANCEDTHEMES (8) -- that repeats each theme
                    # once per mention with character offsets
COL_TONE = 15       # V1.5TONE: 6 comma-separated dims (see TONE_COLS below)
MIN_COLS = 16

# V1.5Tone field order per the GKG 2.1 codebook.
TONE_COLS = ["tone_avg", "tone_pos", "tone_neg", "tone_polarity",
            "tone_activity_density", "tone_selfref_density"]


def full_timestamps():
    t = START
    while t < END:
        yield t.strftime("%Y%m%d%H%M%S")
        t += timedelta(minutes=15)


def quarter_timestamps():
    """One file per hour, on the hour: 1/4 of full resolution."""
    t = START
    while t < END:
        yield t.strftime("%Y%m%d%H%M%S")
        t += timedelta(hours=1)


def fetch(ts):
    # Identical to pull_gdelt_sample.py's fetch(): per-call session (a
    # shared one stalled completely, twice, with zero errors -- see that
    # script's comments) and a manual hard deadline (requests' read timeout
    # only fires on gaps between chunks, not total transfer time).
    #
    # 404 specifically returns immediately, no retry: GDELT's archive has
    # genuine gaps (files that were never generated, mostly clustered right
    # after the Feb 2015 launch while the system was stabilising), and a
    # missing file will never stop being missing. Every OTHER non-200
    # status stays in the retry loop, since a 5xx/429 could be transient --
    # conflating them with 404 would mean a real but temporary server
    # hiccup gets marked permanently unfetchable on the first bad response.
    url = BASE_URL.format(ts=ts)
    with requests.Session() as session:
        for attempt in range(1, RETRIES + 1):
            t0 = time.monotonic()
            try:
                with session.get(url, timeout=TIMEOUT, stream=True) as r:
                    if r.status_code == 404:
                        return ts, None, "http 404"
                    if r.status_code != 200:
                        if attempt == RETRIES:
                            return ts, None, f"http {r.status_code}"
                        continue
                    chunks = []
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        chunks.append(chunk)
                        if time.monotonic() - t0 > HARD_DEADLINE:
                            raise TimeoutError(f"exceeded {HARD_DEADLINE}s total")
                    content = b"".join(chunks)
                if content:
                    return ts, content, None
                return ts, None, "empty response"
            except (requests.RequestException, TimeoutError) as e:
                if attempt == RETRIES:
                    return ts, None, str(e)
    return ts, None, "unreachable"


def parse_gkg(ts, raw):
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if not names:
            return rows
        text = zf.read(names[0]).decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < MIN_COLS:
            continue
        docid, themes = row[COL_DOCID], row[COL_V1THEMES]
        if theme_matches(themes) or url_matches(docid):
            blob = f"{docid} {themes}".lower()
            tone_vals = row[COL_TONE].split(",") if row[COL_TONE] else []
            tone_vals = (tone_vals + [""] * 6)[:6]   # pad/truncate defensively
            rows.append((row[COL_DATE], docid, blob, row[COL_SOURCE_NAME],
                        themes, *tone_vals))
    return rows


def load_done() -> set[str]:
    """Timestamps that never need attempting again: successes, and
    confirmed-permanent 404s. Anything else ("failed" -- a transient
    error that exhausted its retries within one session) stays eligible
    for a future restart, since the condition that caused it might not
    still hold. Without this distinction, every restart re-fetched every
    timestamp that had ever 404'd, including genuine archive gaps that
    will never stop 404ing -- four restarts had compounded this into a
    growing, permanently-retried backlog concentrated at the chronological
    start of the queue (the part that had been through the most restart
    cycles), which is what produced a startling 90%+ failure rate right
    after a restart and looked like server throttling until traced."""
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH, newline="") as f:
        return {row["ts"] for row in csv.DictReader(f)
               if row["status"] in ("ok", "not_found")}


def coverage_by_day() -> dict[str, int]:
    """Files with status 'ok' per day (YYYYMMDD), out of the 24 scheduled
    quarter-sample hours. Distinct from "complete" (attempted, regardless
    of outcome) -- this is genuinely fetched. A day with fewer than 24
    understates I_t if not corrected for, indistinguishably from a
    genuinely quiet news day; a day with 0 has no observation at all and
    must not silently become a zero downstream (same error as flooring a
    negative price instead of marking it missing, in different clothes)."""
    counts: dict[str, int] = {}
    if not MANIFEST_PATH.exists():
        return counts
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                day = row["ts"][:8]
                counts[day] = counts.get(day, 0) + 1
    return counts


def append_manifest(entries: list[tuple[str, str, int]]):
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ts", "status", "n_rows"])
        w.writerows(entries)


_flush_counter = 0   # process-local; part filenames only need to be unique
                    # within one run, not across restarts (see below)


def flush_output(new_records: list[tuple]) -> int:
    """Write new_records as one new part file under OUT_PATH/ -- never reads
    or rewrites what's already there. Returns len(new_df) (this batch only,
    after dropping any duplicate DocumentIdentifier within the batch itself
    -- not a running total across the whole corpus, since getting an exact
    deduped total would mean reading everything back, the cost this
    function exists to avoid)."""
    global _flush_counter
    cols = ["V2.1DATE", "DocumentIdentifier", "text", "SourceCommonName",
           "Themes", *TONE_COLS]
    new_df = pd.DataFrame(new_records, columns=cols)
    new_df["timestamp"] = pd.to_datetime(new_df["V2.1DATE"], format="%Y%m%d%H%M%S",
                                         utc=True)
    for c in TONE_COLS:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    new_df = new_df[["timestamp", "text", "DocumentIdentifier",
                     "SourceCommonName", "Themes", *TONE_COLS]]
    new_df = new_df.drop_duplicates(subset=["DocumentIdentifier"])

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    _flush_counter += 1
    # PID in the name: two restarts launched close enough together to share
    # a wall-clock second (this session's own Stage 1 relaunch history
    # includes exactly that -- two processes, one second apart, see the
    # OUT_PATH comment above) must still not collide on a part filename.
    part_name = f"part-{int(time.time())}-{os.getpid()}-{_flush_counter:05d}.parquet"
    part_path = OUT_PATH / part_name
    tmp_path = part_path.with_suffix(".tmp")
    # Write-then-rename, not a direct write, same reasoning as before: a
    # concurrent reader (stage 2, mid-scan of the dataset directory) must
    # never see a torn part file, only the complete file or none at all.
    # Every part file is new and nothing else ever writes to it, so unlike
    # the old single-shared-file design this can't collide with a stage-2
    # read of the *same* path -- the PermissionError retry stays here only
    # as cheap insurance against transient AV/indexer file locks, not
    # because it's expected.
    new_df.to_parquet(tmp_path, index=False)
    for attempt in range(5):
        try:
            os.replace(tmp_path, part_path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))
    return len(new_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["full", "quarter"], default="quarter")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the file count and estimated time, don't fetch")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--rate-mbps", type=float, default=22.0,
                    help="measured connection ceiling, for the dry-run ETA")
    args = ap.parse_args()

    ts_all = list(full_timestamps() if args.sample == "full" else quarter_timestamps())
    done = load_done()
    ts_todo = [ts for ts in ts_all if ts not in done]

    avg_mb_per_file = 6.5   # measured earlier this session across 4 eras
    total_gb = len(ts_all) * avg_mb_per_file / 1024
    remaining_gb = len(ts_todo) * avg_mb_per_file / 1024
    # GB -> Gigabits (x8) -> Megabits (x1000) -> seconds (/rate_mbps) -> hours (/3600)
    eta_hours = remaining_gb * 8000 / args.rate_mbps / 3600

    print(f"sample={args.sample}: {len(ts_all):,} files total "
         f"(~{total_gb:.0f} GB), {len(done):,} already done, "
         f"{len(ts_todo):,} remaining (~{remaining_gb:.0f} GB)")
    print(f"at {args.rate_mbps:.0f} Mbps measured: "
         f"~{eta_hours:.1f} hours remaining (~{eta_hours/24:.1f} days)")

    if not args.run:
        print("\n(dry run only -- pass --run to actually execute; safe to "
             "interrupt and rerun, it resumes from the manifest)")
        return
    if not ts_todo:
        print("nothing left to do")
        return

    ok, failed = 0, []
    buffer = []
    manifest_buffer = []
    i = 0
    rows_written = 0   # this session's new rows, pre-cross-batch-dedup --
                       # not the corpus's true size; see flush_output()
    t_start = time.monotonic()

    ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = {ex.submit(fetch, ts) for ts in ts_todo}
    try:
        while pending:
            batch_done, pending = wait(pending, timeout=STALL_LIMIT,
                                       return_when=FIRST_COMPLETED)
            if not batch_done:
                print(f"no completions in {STALL_LIMIT}s -- {len(pending)} files "
                     f"still outstanding this session; checkpoint is saved, "
                     f"rerun the same command to resume", file=sys.stderr)
                break
            for fut in batch_done:
                i += 1
                ts, raw, err = fut.result()
                if raw is None:
                    failed.append(ts)
                    status = "not_found" if err == "http 404" else "failed"
                    manifest_buffer.append((ts, status, 0))
                    continue
                try:
                    rows = parse_gkg(ts, raw)
                except Exception as e:
                    failed.append(ts)
                    manifest_buffer.append((ts, f"parse_error", 0))
                    continue
                ok += 1
                buffer.extend(rows)
                manifest_buffer.append((ts, "ok", len(rows)))

            if i % CHECKPOINT_EVERY == 0 or not pending:
                # Output flushed before the manifest is written, not after:
                # if flush_output raises (even past its own retries), the
                # manifest must NOT yet say these timestamps are done --
                # otherwise a resume would skip them, thinking their rows
                # already made it to disk when they never did. The reverse
                # order risks silent data loss; this order risks only
                # redundant re-fetching, which a duplicate DocumentIdentifier
                # across two part files costs nothing beyond what read-time
                # dedup in every downstream consumer already absorbs.
                batch_rows = flush_output(buffer) if buffer else 0
                rows_written += batch_rows
                buffer = []
                append_manifest(manifest_buffer)
                manifest_buffer = []
                elapsed = time.monotonic() - t_start
                rate = i / elapsed
                eta_h = (len(ts_todo) - i) / rate / 3600 if rate > 0 else float("inf")
                print(f"  {i}/{len(ts_todo)} this session, {len(failed)} failed, "
                     f"[{elapsed/3600:.2f}h elapsed, {rate*3600:.0f} files/h, "
                     f"~{eta_h:.1f}h remaining]"
                     + (f" -- {rows_written:,} new rows written this session"
                        if rows_written else ""),
                     file=sys.stderr)
    finally:
        # Shut the executor down and cancel not-yet-started futures FIRST,
        # before attempting any final flush -- not after. This exact order
        # is what let one flush_output() failure (an OOM inside the old
        # read-modify-write design; see the OUT_PATH comment) turn into a
        # runaway process: the exception unwound past ex.shutdown() with
        # worker threads still pulling queued fetches, and with nothing
        # left to consume their results, the process kept growing in
        # memory *after* it had already crashed, instead of exiting.
        # Shutting down first bounds that regardless of what the final
        # flush does or whether it succeeds.
        ex.shutdown(wait=False, cancel_futures=True)
        if buffer:
            flush_output(buffer)
        if manifest_buffer:
            append_manifest(manifest_buffer)

    print(f"session done: {ok} fetched, {len(failed)} failed this session")
    print(f"checkpoint saved to {MANIFEST_PATH}; rerun to resume remaining files")


if __name__ == "__main__":
    main()
