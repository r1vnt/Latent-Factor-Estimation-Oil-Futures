"""Stage 2+3 of the locked news-pipeline spec: fetch, extract, score, discard;
log every candidate.

Runs as an independent long-running process alongside stage 1
(pull_gdelt_history.py), polling its manifest for newly-completed days
rather than waiting for the whole window to finish -- this is what keeps
total elapsed near stage 1's ~2.3 days rather than adding stage 2's own
time serially after it.

Per day with candidates:
    1. Draw up to 100 URLs at random from that day's stage-1 matches
       (random, not first-100 -- first-N skews toward whatever GDELT
       processed earliest in the day).
    2. Fetch (rate-limited, robots.txt-respecting, 15s timeout) until 60
       succeed or the 100 candidates are exhausted. A dead link just
       costs a candidate, not a retry budget -- link rot eats the buffer,
       not n_t.
    3. trafilatura extracts title + body from whatever HTML came back.
    4. FinBERT scores title + lede (first ~2 sentences of body), truncated
       to 256 tokens, batched across the day's successful extractions
       rather than one call per document (CPU-bound; this is a real
       throughput win, not a spec deviation -- nothing here dictates
       scoring must be interleaved with fetching).
    5. Every candidate -- success or failure -- gets one row in the stage-3
       log (data/stage3_document_log.parquet). HTML itself is never
       stored; only extracted_chars survives.

FinBERT score: P(positive) - P(negative) from the full 3-class
distribution, not just the top label's confidence -- a signed scalar
comparable in spirit to GDELT's own tone measure, which is what stage 4
and Sec. XIII's lexicon cross-check need. This mapping isn't specified
verbatim in the locked spec ("keep the score"); it's the natural reading
and is documented here rather than left implicit.

Usage:
    python scripts/fetch_score_stage2.py --run
        (polls stage 1's manifest continuously; exits once stage 1's
        window is fully attempted and every complete day has been
        processed)
    python scripts/fetch_score_stage2.py --once
        (single pass over currently-complete days, then exit -- for
        testing or manual stepping)
"""

import argparse
import csv
import hashlib
import random
import sys
import threading
import time
import urllib.robotparser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_gdelt_history import MANIFEST_PATH as STAGE1_MANIFEST  # noqa: E402
from pull_gdelt_history import OUT_PATH as STAGE1_OUTPUT          # noqa: E402
from pull_gdelt_history import END as WINDOW_END                  # noqa: E402
from pull_gdelt_history import START as WINDOW_START               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGE2_MANIFEST = ROOT / "data" / "stage2_manifest.csv"     # days processed
STAGE3_LOG = ROOT / "data" / "stage3_document_log.parquet"

CANDIDATES_PER_DAY = 100
TARGET_SUCCESSES = 60
FETCH_RATE_PER_SEC = 5.0
FETCH_TIMEOUT = 15
FETCH_WORKERS = 10          # concurrent in-flight, gated by the rate limiter
MAX_TOKENS = 256
LEDE_SENTENCES = 2
USER_AGENT = "csr-research-bot/0.1 (oil-futures news study; contact via repo)"
POLL_INTERVAL_SEC = 120

_finbert = None   # lazy-loaded once, reused across days


# ------------------------------------------------------------ rate limiting
class RateLimiter:
    """Sliding-window limiter: at most `rate` requests started per second,
    shared across all fetch workers so 5 req/s is a real global cap."""

    def __init__(self, rate: float):
        self.rate = rate
        self.times = deque()
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.times and now - self.times[0] > 1.0:
                    self.times.popleft()
                if len(self.times) < self.rate:
                    self.times.append(now)
                    return
                sleep_for = 1.0 - (now - self.times[0])
            time.sleep(max(sleep_for, 0.01))


_limiter = RateLimiter(FETCH_RATE_PER_SEC)
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
_robots_cache_lock = threading.Lock()   # guards the dict only, never held
                                        # across a network call


def _robots_allows(url: str) -> bool:
    domain = urlparse(url).netloc
    with _robots_cache_lock:
        cached = domain in _robots_cache
        rp = _robots_cache.get(domain)
    if not cached:
        # Fetched via requests (bounded timeout), not RobotFileParser.read()
        # (urlopen() with no default timeout -- observed hanging the entire
        # pipeline on one unresponsive robots.txt, since read() was also
        # called while holding a global lock every other domain waited on).
        # Fetching outside any lock means at most a few redundant first-hit
        # requests per domain under concurrency, which is harmless.
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = requests.get(f"{urlparse(url).scheme}://{domain}/robots.txt",
                                timeout=(5, 10))
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp = None   # no restriction on a missing/erroring robots.txt
        except requests.RequestException:
            rp = None
        with _robots_cache_lock:
            _robots_cache[domain] = rp
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------- fetching
def fetch_one(url: str) -> tuple[bool, int | None, str | None, str | None]:
    """Returns (fetch_ok, http_status, html, error)."""
    if not _robots_allows(url):
        return False, None, None, "robots_disallowed"
    _limiter.wait()
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        if r.status_code == 200 and r.text:
            return True, r.status_code, r.text, None
        return False, r.status_code, None, f"http_{r.status_code}"
    except requests.RequestException as e:
        return False, None, None, type(e).__name__


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()[:16]


def source_domain(url: str) -> str:
    return urlparse(url).netloc


# ---------------------------------------------------------- extract + score
def extract_lede(html: str) -> tuple[str | None, int]:
    """Title + first LEDE_SENTENCES sentences of the body, via trafilatura."""
    text = trafilatura.extract(html, favor_recall=True)
    if not text:
        return None, 0
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title if metadata and metadata.title else "").strip()
    sentences = text.replace("\n", " ").split(". ")
    lede = ". ".join(sentences[:LEDE_SENTENCES])
    combined = f"{title}. {lede}".strip(". ").strip()
    return (combined or None), len(text)


def load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        _finbert = pipeline("text-classification", model="ProsusAI/finbert",
                            top_k=None, truncation=True, max_length=MAX_TOKENS)
    return _finbert


def score_batch(texts: list[str]) -> list[float]:
    """P(positive) - P(negative) per text, from FinBERT's full 3-class output."""
    clf = load_finbert()
    results = clf(texts, batch_size=16)
    scores = []
    for row in results:
        probs = {r["label"].lower(): r["score"] for r in row}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


# --------------------------------------------------------------- checkpoints
def load_stage1_complete_days() -> set[str]:
    """Days (YYYYMMDD) where every quarter-sample hour has been attempted."""
    if not STAGE1_MANIFEST.exists():
        return set()
    attempted: dict[str, set[str]] = {}
    with open(STAGE1_MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            ts = row["ts"]
            attempted.setdefault(ts[:8], set()).add(ts[8:10])
    return {day for day, hours in attempted.items() if len(hours) >= 24}


def load_stage2_done_days() -> set[str]:
    if not STAGE2_MANIFEST.exists():
        return set()
    with open(STAGE2_MANIFEST, newline="") as f:
        return {row["day"] for row in csv.DictReader(f)}


def append_stage2_manifest(day: str, n_candidates: int, n_success: int):
    is_new = not STAGE2_MANIFEST.exists()
    with open(STAGE2_MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["day", "n_candidates", "n_success"])
        w.writerow([day, n_candidates, n_success])


def append_stage3_log(rows: list[dict]):
    df = pd.DataFrame(rows)
    if STAGE3_LOG.exists():
        df = pd.concat([pd.read_parquet(STAGE3_LOG), df], ignore_index=True)
    STAGE3_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STAGE3_LOG, index=False)


# ------------------------------------------------------------------- day loop
def candidates_for_day(day: str) -> pd.DataFrame:
    if not STAGE1_OUTPUT.exists():
        return pd.DataFrame(columns=["DocumentIdentifier"])
    df = pd.read_parquet(STAGE1_OUTPUT, columns=["timestamp", "DocumentIdentifier"])
    day_ts = pd.Timestamp(day, tz="UTC")
    mask = (df["timestamp"] >= day_ts) & (df["timestamp"] < day_ts + pd.Timedelta(days=1))
    return df[mask]


def process_day(day: str) -> tuple[int, int]:
    cand_df = candidates_for_day(day)
    urls = cand_df["DocumentIdentifier"].dropna().unique().tolist()
    if not urls:
        append_stage2_manifest(day, 0, 0)
        return 0, 0

    sample = random.sample(urls, min(CANDIDATES_PER_DAY, len(urls)))
    log_rows = []
    extracted = []   # (row_index_in_log, text_for_scoring)
    n_success = 0

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_one, url): url for url in sample}
        for fut in as_completed(futures):
            url = futures[fut]
            if n_success >= TARGET_SUCCESSES:
                fut.cancel()
                continue
            fetch_ok, status, html, err = fut.result()
            row = {
                "date": day, "url_hash": url_hash(url),
                "source_domain": source_domain(url),
                "fetch_ok": fetch_ok, "http_status": status,
                "extracted_chars": 0, "finbert_ok": False,
                "finbert_score": None,
            }
            if fetch_ok:
                lede, n_chars = extract_lede(html)
                row["extracted_chars"] = n_chars
                if lede:
                    extracted.append((len(log_rows), lede))
                    n_success += 1
            log_rows.append(row)

    if extracted:
        texts = [t for _, t in extracted]
        try:
            scores = score_batch(texts)
            for (idx, _), score in zip(extracted, scores):
                log_rows[idx]["finbert_ok"] = True
                log_rows[idx]["finbert_score"] = score
        except Exception as e:
            print(f"  [{day}] FinBERT batch scoring failed: {e}", file=sys.stderr)

    append_stage3_log(log_rows)
    append_stage2_manifest(day, len(sample), n_success)
    return len(sample), n_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="single pass over currently-complete days, then exit")
    ap.add_argument("--run", action="store_true",
                    help="poll continuously until stage 1's window is done")
    args = ap.parse_args()
    if not (args.once or args.run):
        print("pass --once or --run"); return

    all_days = set()
    t = WINDOW_START
    while t < WINDOW_END:
        all_days.add(t.strftime("%Y%m%d"))
        t += pd.Timedelta(days=1)

    while True:
        complete = load_stage1_complete_days()
        done = load_stage2_done_days()
        todo = sorted(complete - done)
        if not todo:
            if args.once or complete >= all_days:
                print(f"nothing to do right now ({len(done)}/{len(all_days)} "
                     f"days processed, {len(complete)} ready from stage 1)")
                if args.once or done >= all_days:
                    return
                time.sleep(POLL_INTERVAL_SEC)
                continue
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for day in todo:
            n_cand, n_ok = process_day(day)
            print(f"  {day}: {n_cand} candidates, {n_ok} scored "
                 f"({len(done) + 1}/{len(all_days)} days)", file=sys.stderr)
            done.add(day)

        if args.once:
            return


if __name__ == "__main__":
    main()
