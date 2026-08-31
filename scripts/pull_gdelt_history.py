"""Resumable, checkpointed full-history GDELT pull -- the streaming-local
alternative to pull_gdelt_bigquery.py.

At this connection's measured ceiling (~22 Mbps -- doubling worker count
12->24 didn't move the aggregate at all, so it's a real cap, not a latency
artifact), full resolution (2015-2024, ~2.1TB) is ~9-10 days of wall clock,
not the 2 days a faster-connection assumption would give. That's long
enough that per-file resumability isn't optional: a checkpoint manifest
(data/gdelt_history_manifest.csv) records every timestamp already
processed, so a restart after any interruption skips straight to where it
left off rather than re-downloading from the start. Output is flushed
periodically too, so a crash costs at most one checkpoint interval, not the
whole run.

Same precision-first predicate as pull_gdelt_bigquery.py and
explore/03_gate2_diagnostics.py step 3 (oil/crude term AND inventory term,
not "eia" alone) -- kept identical across all three routes so results are
comparable regardless of which one actually runs.

Two sampling modes:
  --sample full     all 96 files/day (~2.1TB, ~9-10 days at this connection)
  --sample quarter  1 file/hour, 24/day (~1/4 the volume and time)

The quarter mode is a real estimation tradeoff, not a free lunch: n_t
(document counts feeding R_t = sigma^2/n_t in the news-variance
construction) will be ~1/4 of true daily volume and needs scaling back up
by the sampling factor (4x) before use, or R_t comes out 4x too large.
sigma^2 itself stays unbiased, just noisier from fewer documents per day.

Usage:
    python scripts/pull_gdelt_history.py --sample full --dry-run
    python scripts/pull_gdelt_history.py --sample quarter --run
    python scripts/pull_gdelt_history.py --sample full --run   # resumes
                                                                 # automatically
                                                                 # if rerun
"""

import argparse
import csv
import io
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
OUT_PATH = ROOT / "data" / "gdelt_history_extract.parquet"
MANIFEST_PATH = ROOT / "data" / "gdelt_history_manifest.csv"

START = datetime(2015, 2, 18, 0, 0, tzinfo=timezone.utc)   # GKG 2.0 launch
END = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)       # exclusive; matches
                                                             # pull_gdelt_bigquery.py

BASE_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
MAX_WORKERS = 12    # more workers didn't raise throughput at this connection's
                    # measured ceiling; not worth raising further
RETRIES = 3
TIMEOUT = (10, 15)
HARD_DEADLINE = 60
STALL_LIMIT = 120
CHECKPOINT_EVERY = 25   # flush manifest + accumulated output every N files

# Same conjunctive predicate as pull_gdelt_bigquery.py's QUERY and
# explore/03_gate2_diagnostics.py step 3.
OIL_TERM = re.compile(r"crude|petroleum", re.IGNORECASE)
INV_TERM = re.compile(r"inventor|stockpile|petroleum status|weekly petroleum",
                      re.IGNORECASE)

COL_DATE = 1
COL_SOURCE_NAME = 3
COL_DOCID = 4
COL_V2THEMES = 8
COL_TONE = 15
MIN_COLS = 16


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
    url = BASE_URL.format(ts=ts)
    with requests.Session() as session:
        for attempt in range(1, RETRIES + 1):
            t0 = time.monotonic()
            try:
                with session.get(url, timeout=TIMEOUT, stream=True) as r:
                    if r.status_code != 200:
                        return ts, None, f"http {r.status_code}"
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
        docid, themes = row[COL_DOCID], row[COL_V2THEMES]
        blob = f"{docid} {themes}"
        if OIL_TERM.search(blob) and INV_TERM.search(blob):
            tone_field = row[COL_TONE].split(",")[0] if row[COL_TONE] else ""
            rows.append((row[COL_DATE], docid, blob.lower(), row[COL_SOURCE_NAME],
                        themes, tone_field))
    return rows


def load_done() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH, newline="") as f:
        return {row["ts"] for row in csv.DictReader(f) if row["status"] == "ok"}


def append_manifest(entries: list[tuple[str, str, int]]):
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ts", "status", "n_rows"])
        w.writerows(entries)


def flush_output(new_records: list[tuple]) -> int:
    """Merge new_records into the accumulated parquet; dedup by URL."""
    cols = ["V2.1DATE", "DocumentIdentifier", "text", "SourceCommonName",
           "Themes", "Tone"]
    new_df = pd.DataFrame(new_records, columns=cols)
    new_df["timestamp"] = pd.to_datetime(new_df["V2.1DATE"], format="%Y%m%d%H%M%S",
                                         utc=True)
    new_df["Tone"] = pd.to_numeric(new_df["Tone"], errors="coerce")
    new_df = new_df[["timestamp", "text", "DocumentIdentifier",
                     "SourceCommonName", "Themes", "Tone"]]

    if OUT_PATH.exists():
        combined = pd.concat([pd.read_parquet(OUT_PATH), new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["DocumentIdentifier"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    return len(combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["full", "quarter"], default="full")
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
                    manifest_buffer.append((ts, "failed", 0))
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
                append_manifest(manifest_buffer)
                manifest_buffer = []
                total_rows = flush_output(buffer) if buffer else None
                buffer = []
                elapsed = time.monotonic() - t_start
                rate = i / elapsed
                eta_h = (len(ts_todo) - i) / rate / 3600 if rate > 0 else float("inf")
                print(f"  {i}/{len(ts_todo)} this session, {len(failed)} failed, "
                     f"[{elapsed/3600:.2f}h elapsed, {rate*3600:.0f} files/h, "
                     f"~{eta_h:.1f}h remaining]"
                     + (f" -- {total_rows:,} rows accumulated" if total_rows else ""),
                     file=sys.stderr)
    finally:
        if manifest_buffer:
            append_manifest(manifest_buffer)
        if buffer:
            flush_output(buffer)
        ex.shutdown(wait=False, cancel_futures=True)

    print(f"session done: {ok} fetched, {len(failed)} failed this session")
    print(f"checkpoint saved to {MANIFEST_PATH}; rerun to resume remaining files")


if __name__ == "__main__":
    main()
