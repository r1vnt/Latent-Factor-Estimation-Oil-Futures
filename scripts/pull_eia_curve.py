"""Pull daily CL1-CL4 NYMEX futures settlements from the free EIA API.

Gate 1 (curve depth) settled without KCL Bloomberg/LSEG/Datastream access, so
this is the thin path: contracts 1-4 only, from EIA's `petroleum/pri/fut`
route, workable with the Bayesian prior on kappa from Sec. XII-A.

Requires EIA_API_KEY, read from the environment or a local `.env` file at the
repo root (gitignored -- see `.env.example`). Never commit the key or paste
it into chat; put it directly in `.env`:

    EIA_API_KEY=your-key-here

Usage:
    python scripts/pull_eia_curve.py --discover   # confirm series IDs first
    python scripts/pull_eia_curve.py               # pull and write parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "eia_cl_futures.parquet"
BASE = "https://api.eia.gov/v2/petroleum/pri/fut"

# Candidate series IDs for Cushing WTI (CL) futures contracts 1-4, daily.
# Confirmed against the EIA v1 naming convention (PET.RCLC{n}.D); verify with
# --discover before trusting a full pull, since v2 facet codes have shifted
# EIA's schema before.
SERIES = {"RCLC1": 1, "RCLC2": 2, "RCLC3": 3, "RCLC4": 4}

START = "2010-01-01"
PAGE_LENGTH = 5000


def load_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("EIA_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    sys.exit(
        "EIA_API_KEY not found. Put it in a local .env file at the repo "
        "root (EIA_API_KEY=...) or set it as an environment variable. "
        "See .env.example."
    )


def discover(api_key: str):
    """List available series under petroleum/pri/fut so codes can be verified."""
    r = requests.get(
        f"{BASE}/facet/series",
        params={"api_key": api_key},
        timeout=30,
    )
    r.raise_for_status()
    facets = r.json()["response"]["facets"]
    print(f"{len(facets)} series available under petroleum/pri/fut\n")
    hits = [f for f in facets if "crude oil future" in f.get("name", "").lower()
            or "cushing" in f.get("name", "").lower()]
    print("Candidates matching 'crude oil future' / 'cushing':")
    for f in hits:
        flag = "  <-- expected" if f["id"] in SERIES else ""
        print(f"  {f['id']:12s} {f['name']}{flag}")
    missing = [sid for sid in SERIES if sid not in {f["id"] for f in hits}]
    if missing:
        print(f"\nWARNING: expected series not found in candidates: {missing}")
        print("Inspect the full facet list before trusting a pull:")
        for f in facets:
            print(f"  {f['id']:12s} {f['name']}")


def pull_series(api_key: str, series_id: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        r = requests.get(
            f"{BASE}/data",
            params={
                "api_key": api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": series_id,
                "start": START,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": PAGE_LENGTH,
            },
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()["response"]
        batch = payload["data"]
        rows.extend(batch)
        if len(batch) < PAGE_LENGTH:
            break
        offset += PAGE_LENGTH
        time.sleep(0.2)  # be polite to a free public API
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true",
                     help="list available series IDs instead of pulling")
    args = ap.parse_args()

    api_key = load_api_key()

    if args.discover:
        discover(api_key)
        return

    frames = []
    for series_id, contract in SERIES.items():
        print(f"pulling {series_id} (contract {contract})...")
        df = pull_series(api_key, series_id)
        if df.empty:
            print(f"  WARNING: no rows returned for {series_id}")
            continue
        df = df[["period", "value"]].rename(columns={"period": "date", "value": "settle"})
        df["date"] = pd.to_datetime(df["date"])
        df["settle"] = pd.to_numeric(df["settle"], errors="coerce")
        df["contract"] = contract
        frames.append(df)
        print(f"  {len(df):,} rows, {df.date.min().date()} .. {df.date.max().date()}")

    if not frames:
        sys.exit("no data pulled -- check series IDs with --discover")

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["settle"])
    panel = panel.sort_values(["date", "contract"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {len(panel):,} rows -> {OUT_PATH}")
    print(f"  contracts: {sorted(panel.contract.unique())}")
    print(f"  span: {panel.date.min().date()} .. {panel.date.max().date()}")


if __name__ == "__main__":
    main()
