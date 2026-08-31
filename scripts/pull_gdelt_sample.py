"""Pull one week of GDELT v2 GKG and filter to oil-related documents.

Gate 2: this is the sample the timestamp check (explore/01_timestamps.py)
runs against, deciding whether the measurement specification (SM) is viable
at all. Bulk GKG files, not the DOC API -- the codebook states V2.1DATE is
publication time, where the API's `seendate` is when GDELT observed it, and
that distinction is the whole gate.

GKG carries no article body -- only structured fields. The searchable
"text" proxy used for both the broad oil filter here and the EIA-release
filter downstream is built from the document URL and GDELT's theme
taxonomy only, not organizations/names, which is where the noisier ad hoc
version of this script picked up false positives. V2SourceCommonName and
V1.5Tone are carried through unfiltered for the dedup/scoring work that
follows a pass, not used for matching here.

Usage: python scripts/pull_gdelt_sample.py
Output: data/news_sample.parquet
  columns: timestamp, text, DocumentIdentifier, SourceCommonName, Themes, Tone
"""

import csv
import io
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Python's csv default field-size limit (128KB) is smaller than some GKG
# rows' longest field (typically V2.1Quotations or V2.1Amounts on a
# document with heavy entity extraction); uncaught, that's a hard crash
# partway through a file, not a per-row skip. 10MB is generous headroom
# without risking the OverflowError csv.field_size_limit(sys.maxsize) hits
# on some Windows/32-bit-C-long builds.
csv.field_size_limit(10_000_000)

# One full week, per Sec. XII-A's "one week, not ten years": the timestamp
# check needs a real sample, not a synthetic stand-in, but a full download
# is a days-long commitment that should wait on this gate passing.
WEEK_START = datetime(2024, 1, 8, 0, 0, tzinfo=timezone.utc)
WEEK_END = datetime(2024, 1, 14, 23, 45, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)

BASE_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "news_sample.parquet"
MAX_WORKERS = 12
RETRIES = 3
TIMEOUT = (10, 15)  # (connect, read-between-chunks) seconds
HARD_DEADLINE = 60   # total seconds per attempt, enforced manually below --
                     # requests' read timeout only fires on a gap between
                     # chunks, so a connection trickling data steadily but
                     # slowly never times out on its own and can hang a
                     # worker indefinitely (observed: 20/672 files in 22min).
STALL_LIMIT = 120   # seconds with zero completions before giving up on the
                    # rest of the run and writing out what's already in hand
                    # -- belt-and-braces on top of HARD_DEADLINE, since a
                    # stall was observed to survive that too (see fetch()).

OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "opec", "wti", "brent", "barrel",
    "gasoline", "refinery", "refiner", "drilling", "oilfield",
    "energy information administration", "eia", "inventor", "stockpile",
]

# GKG 2.1 column indices (0-based).
COL_DATE = 1
COL_SOURCE_NAME = 3
COL_DOCID = 4
COL_V2THEMES = 8
COL_TONE = 15
MIN_COLS = 16


def timestamps():
    t = WEEK_START
    while t <= WEEK_END:
        yield t.strftime("%Y%m%d%H%M%S")
        t += STEP


def fetch(ts):
    # A fresh session per call, not shared across worker threads: a shared
    # session's connection pool was implicated in the stall this hit twice
    # in a row (fresh curl calls outside the pool succeeded while every
    # worker sharing one session sat stuck for 10+ minutes with zero
    # completions and zero raised errors).
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
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                return rows
            text = zf.read(names[0]).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        return rows

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < MIN_COLS:
            continue
        docid, themes = row[COL_DOCID], row[COL_V2THEMES]
        blob = f"{docid} {themes}".lower()
        if any(kw in blob for kw in OIL_KEYWORDS):
            tone_field = row[COL_TONE].split(",")[0] if row[COL_TONE] else ""
            rows.append((row[COL_DATE], docid, blob, row[COL_SOURCE_NAME],
                        themes, tone_field))
    return rows


def main():
    ts_list = list(timestamps())
    print(f"week: {WEEK_START.isoformat()} .. {WEEK_END.isoformat()}  "
          f"({len(ts_list)} files)")

    ok, failed = 0, []
    records = []
    i = 0
    t_start = time.monotonic()
    stalled = False

    ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = {ex.submit(fetch, ts) for ts in ts_list}
    try:
        while pending:
            done, pending = wait(pending, timeout=STALL_LIMIT,
                                 return_when=FIRST_COMPLETED)
            if not done:
                print(f"  no completions in {STALL_LIMIT}s -- {len(pending)} files "
                      f"still outstanding; giving up on the rest and writing "
                      f"what's already in hand", file=sys.stderr)
                stalled = True
                break
            for fut in done:
                i += 1
                ts, raw, err = fut.result()
                if raw is None:
                    failed.append((ts, err))
                    continue
                ok += 1
                try:
                    records.extend(parse_gkg(ts, raw))
                except Exception as e:
                    # A malformed file's parse error shouldn't cost the
                    # other 671 -- record it as a failure and move on.
                    failed.append((ts, f"parse error: {e}"))
            elapsed = time.monotonic() - t_start
            rate = i / elapsed
            eta_min = (len(ts_list) - i) / rate / 60 if rate > 0 else float("inf")
            print(f"  {i}/{len(ts_list)} files, {len(failed)} failed, "
                  f"{len(records)} rows  [{elapsed / 60:.1f}min elapsed, "
                  f"{rate:.2f} files/s, ~{eta_min:.0f}min remaining]",
                  file=sys.stderr)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    print(f"downloaded {ok}/{len(ts_list)} files, {len(failed)} failed"
          f"{', STALLED -- incomplete run' if stalled else ''}")
    if failed:
        print(f"  failed timestamps: {[f[0] for f in failed[:10]]}"
              f"{' ...' if len(failed) > 10 else ''}")

    if not records:
        print("no oil-related documents found — aborting write")
        return

    df = pd.DataFrame(records, columns=[
        "V2.1DATE", "DocumentIdentifier", "text", "SourceCommonName",
        "Themes", "Tone",
    ])
    df["timestamp"] = pd.to_datetime(df["V2.1DATE"], format="%Y%m%d%H%M%S", utc=True)
    df["Tone"] = pd.to_numeric(df["Tone"], errors="coerce")
    df = df[["timestamp", "text", "DocumentIdentifier", "SourceCommonName",
            "Themes", "Tone"]].dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["DocumentIdentifier"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df):,} rows -> {OUT_PATH}")
    print(f"  span: {df.timestamp.min()} .. {df.timestamp.max()}")


if __name__ == "__main__":
    main()
