"""DORMANT -- superseded by scripts/pull_gdelt_history.py for the corpus
pull (GCP setup was dropped; local streaming does the same column
projection with no account). Kept for the WPSR_PREDICATE query pattern,
which is still the right tool if a BigQuery route is ever revisited for
the timestamp diagnostic specifically -- not for corpus building.

Extract WPSR-release-related GKG rows via GDELT's public BigQuery table.

Same WPSR_PREDICATE validated locally in explore/03_gate2_diagnostics.py
step 3: an oil/crude term AND an inventory term together (not "eia" alone,
which pulls in EIA's other publications -- STEO forecasts, solar, ethanol
-- unrelated to the WPSR). This isolates ~15 documents/week reporting one
specific release; it is NOT a general oil-news filter and must not be used
as one -- that mistake, made once with this exact predicate, cost 44
minutes of a corpus pull before being caught (see
scripts/pull_gdelt_history.py's OIL_CORPUS_PREDICATE, which is the
broad filter for that job).

Table/schema confirmed from GDELT's own BigQuery documentation and worked
examples (not memory alone -- see blog.gdeltproject.org/announcing-
partitioned-gdelt-bigquery-tables and google-bigquery-gkg-2-0-sample-
queries). `--check-schema` re-verifies live against INFORMATION_SCHEMA
before the real query runs, since a wrong column name here wastes quota
silently rather than erroring loudly.

Setup (one-time, on your side -- I can't do this part):
    1. Create a free GCP project at console.cloud.google.com (no card
       needed for BigQuery's public-dataset sandbox / free tier).
    2. Install the Cloud SDK, then:
           gcloud auth application-default login
       (opens a browser for you to sign in; caches credentials locally.)
    3. Put the project ID in .env:  GCP_PROJECT_ID=your-project-id

Usage:
    python scripts/pull_gdelt_bigquery.py --check-schema   # verify columns
    python scripts/pull_gdelt_bigquery.py --dry-run         # estimate cost
    python scripts/pull_gdelt_bigquery.py --run             # actually pull

Output: data/gdelt_bigquery_extract.parquet
  columns: timestamp, text, DocumentIdentifier, SourceCommonName, Themes, Tone
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "gdelt_bigquery_extract.parquet"
TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

START_DATE = "2015-02-18"   # GKG 2.0 launch
END_DATE = "2025-01-01"     # exclusive; matches the 2015-2024 window

# WPSR_PREDICATE: same conjunctive pattern as explore/03_gate2_diagnostics.py
# step 3. Isolates WPSR-reporting articles specifically -- see the module
# docstring's warning before reusing this as a general oil-news filter.
WPSR_OIL_TERM_SQL = r"crude|petroleum"
WPSR_INVENTORY_TERM_SQL = r"inventor|stockpile|petroleum status|weekly petroleum"
QUERY = f"""
SELECT
  DATE AS v21date,
  DocumentIdentifier,
  SourceCommonName,
  V2Themes,
  V2Tone
FROM `{TABLE}`
WHERE _PARTITIONTIME >= TIMESTAMP("{START_DATE}")
  AND _PARTITIONTIME <  TIMESTAMP("{END_DATE}")
  AND (
    REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(DocumentIdentifier, ''), ' ',
                                 IFNULL(V2Themes, ''))), r'{WPSR_OIL_TERM_SQL}')
  )
  AND (
    REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(DocumentIdentifier, ''), ' ',
                                 IFNULL(V2Themes, ''))), r'{WPSR_INVENTORY_TERM_SQL}')
  )
"""


def load_project_id() -> str:
    pid = os.environ.get("GCP_PROJECT_ID")
    if pid:
        return pid.strip().strip("'\"")
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GCP_PROJECT_ID="):
                pid = line.split("=", 1)[1].strip().strip("'\"")
                if pid:
                    return pid
    sys.exit(
        "GCP_PROJECT_ID not found. Put it in .env (GCP_PROJECT_ID=your-"
        "project-id) or set it as an environment variable. See the "
        "module docstring for one-time setup steps."
    )


def check_schema(client: bigquery.Client):
    """Verify the columns this script depends on actually exist, live."""
    needed = {"DATE", "DocumentIdentifier", "SourceCommonName", "V2Themes",
             "V2Tone"}
    sql = """
        SELECT column_name FROM `gdelt-bq.gdeltv2.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'gkg_partitioned'
    """
    cols = {row.column_name for row in client.query(sql).result()}
    missing = needed - cols
    print(f"table has {len(cols)} columns")
    if missing:
        print(f"MISSING expected columns: {missing}")
        print(f"available columns: {sorted(cols)}")
        sys.exit(1)
    print(f"all {len(needed)} expected columns present: {sorted(needed)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-schema", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate bytes scanned without running the query")
    ap.add_argument("--run", action="store_true",
                    help="actually run the query and write the parquet")
    args = ap.parse_args()

    project = load_project_id()
    client = bigquery.Client(project=project)

    if args.check_schema:
        check_schema(client)
        return

    if args.dry_run or not args.run:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(QUERY, job_config=job_config)
        gb = job.total_bytes_processed / 1e9
        print(f"dry run: {gb:.3f} GB would be scanned "
             f"({gb / 1000 * 100:.2f}% of the 1TB/month free tier)")
        if not args.run:
            print("\n(dry run only -- pass --run to actually execute)")
            return

    print("running query against full partitioned table...")
    job = client.query(QUERY)
    df = job.to_dataframe()
    print(f"{len(df):,} rows returned, "
         f"{job.total_bytes_processed / 1e9:.3f} GB scanned")

    df = df.rename(columns={
        "v21date": "V2.1DATE", "V2Themes": "Themes", "V2Tone": "Tone",
    })
    df["timestamp"] = pd.to_datetime(df["V2.1DATE"].astype(str),
                                     format="%Y%m%d%H%M%S", utc=True)
    df["text"] = (df["DocumentIdentifier"].fillna("") + " "
                 + df["Themes"].fillna("")).str.lower()
    df["Tone"] = pd.to_numeric(df["Tone"].str.split(",").str[0],
                               errors="coerce")
    df = df[["timestamp", "text", "DocumentIdentifier", "SourceCommonName",
            "Themes", "Tone"]].dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["DocumentIdentifier"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df):,} rows -> {OUT_PATH}")
    print(f"  span: {df.timestamp.min()} .. {df.timestamp.max()}")


if __name__ == "__main__":
    main()
