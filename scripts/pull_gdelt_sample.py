"""Pull a short GDELT v2 GKG sample and filter to oil-related documents.

Day-1 sanity check, not the full download. Scoped to a single release-day
window rather than a full week: the timestamp check in
`explore/01_timestamps.py` only uses documents whose *own* V2.1DATE falls on
a Wednesday (the EIA WPSR release day), so a window that safely covers one
Wednesday in US Eastern time is sufficient for that check. A full week adds
six days of files that get discarded by the check's own day-of-week filter.

GKG has no full article body — only structured fields. We build a
searchable "text" proxy from the document URL, themes, organizations, and
extracted names, and filter broadly to oil-related documents. The finer
EIA-release keyword filter happens downstream in the timestamp script.

Usage: python scripts/pull_gdelt_sample.py
Output: data/news_sample.parquet  (columns: timestamp, text)
"""

import csv
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Window: covers US-Eastern Wednesday 2024-01-10 with buffer on both sides.
# EST = UTC-5 in January, so ET Wednesday spans [Jan10 05:00, Jan11 05:00) UTC.
WINDOW_START = datetime(2024, 1, 10, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2024, 1, 11, 8, 0, tzinfo=timezone.utc)  # exclusive-ish
STEP = timedelta(minutes=15)

BASE_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "news_sample.parquet"
MAX_WORKERS = 8
RETRIES = 3
TIMEOUT = (10, 45)  # (connect, read) seconds

OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "opec", "wti", "brent", "barrel",
    "gasoline", "refinery", "refiner", "drilling", "oilfield",
    "energy information administration", "eia", "inventor", "stockpile",
]

# GKG 2.1 column indices (0-based) we use.
COL_DATE = 1
COL_DOCID = 4
COL_V1THEMES = 7
COL_V2THEMES = 8
COL_V1ORGS = 13
COL_V2ORGS = 14
COL_ALLNAMES = 23
MIN_COLS = 24


def timestamps():
    t = WINDOW_START
    while t <= WINDOW_END:
        yield t.strftime("%Y%m%d%H%M%S")
        t += STEP


def fetch(ts, session):
    url = BASE_URL.format(ts=ts)
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200 and r.content:
                return ts, r.content, None
            return ts, None, f"http {r.status_code} len={len(r.content)}"
        except requests.RequestException as e:
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
        blob = " ".join([
            row[COL_DOCID], row[COL_V1THEMES], row[COL_V2THEMES],
            row[COL_V1ORGS], row[COL_V2ORGS], row[COL_ALLNAMES],
        ]).lower()
        if any(kw in blob for kw in OIL_KEYWORDS):
            rows.append((row[COL_DATE], row[COL_DOCID], blob[:500]))
    return rows


def main():
    ts_list = list(timestamps())
    print(f"window: {WINDOW_START.isoformat()} .. {WINDOW_END.isoformat()}  "
          f"({len(ts_list)} files)")

    session = requests.Session()
    ok, failed = 0, []
    records = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch, ts, session): ts for ts in ts_list}
        for i, fut in enumerate(as_completed(futures), 1):
            ts, raw, err = fut.result()
            if raw is None:
                failed.append((ts, err))
            else:
                ok += 1
                records.extend(parse_gkg(ts, raw))
            if i % 20 == 0 or i == len(ts_list):
                print(f"  {i}/{len(ts_list)} files fetched, "
                      f"{len(failed)} failed, {len(records)} oil-related rows so far",
                      file=sys.stderr)

    print(f"downloaded {ok}/{len(ts_list)} files, {len(failed)} failed")
    if failed:
        print(f"  failed timestamps: {[f[0] for f in failed[:10]]}"
              f"{' ...' if len(failed) > 10 else ''}")

    if not records:
        print("no oil-related documents found — aborting write")
        return

    df = pd.DataFrame(records, columns=["V2.1DATE", "DocumentIdentifier", "text"])
    df["timestamp"] = pd.to_datetime(df["V2.1DATE"], format="%Y%m%d%H%M%S", utc=True)
    df = df[["timestamp", "text", "DocumentIdentifier"]].dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["DocumentIdentifier"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df):,} rows -> {OUT_PATH}")
    print(f"  span: {df.timestamp.min()} .. {df.timestamp.max()}")


if __name__ == "__main__":
    main()
