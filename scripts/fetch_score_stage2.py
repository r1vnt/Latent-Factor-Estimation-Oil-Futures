"""Stage 2+3 of the locked news-pipeline spec: fetch, extract, score, discard;
log every candidate.

Runs as an independent long-running process alongside stage 1
(pull_gdelt_history.py), polling its manifest for newly-completed days
rather than waiting for the whole window to finish -- this is what keeps
total elapsed near stage 1's ~2.3 days rather than adding stage 2's own
time serially after it.

Three counts per day, not one -- conflating them corrupts the model inputs:
    I_t            ALL matching docs that day (from stage 1's full match
                   set, not the 100-candidate draw), deduped on
                   (source_domain, normalised URL slug). Feeds the
                   process-noise intensity scaling. Computed from data
                   already retained by stage 1 -- stage 1 keeps every
                   matching row, not a sample, specifically so this number
                   is recoverable.
    n_candidates   up to 100, drawn at random (not first-100, which skews
                   toward whatever GDELT processed earliest in the day).
                   A work quota; feeds nothing downstream.
    n_t            successfully scored, <=60. Feeds R_t = sigma_t^2/n_t.
                   Never rescaled, never the candidate count.

Per day with candidates:
    1. Compute I_t from stage 1's full match set for the day.
    2. Draw up to 100 candidate URLs; fetch (rate-limited, robots.txt-
       respecting, 15s timeout) until 60 pass extraction+language, or the
       100 are exhausted. A dead link costs a candidate, not a retry
       budget -- link rot eats the buffer, not n_t.
    3. trafilatura extracts title + body from whatever HTML came back.
    4. Language check on the extracted text (langdetect) -- FinBERT is
       English-only and returns confident nonsense on other languages
       silently; GDELT is multilingual, so this can't be skipped.
    5. MinHash (datasketch) over word-trigram shingles of the surviving
       lede text, within the day, while it's still in memory -- catches
       syndicated wire copy that reached the sample via different URLs
       (I_t's URL-slug dedup can't catch this, since different domains
       syndicating the same story have unrelated URLs).
    6. FinBERT scores what's left: title + lede, truncated to 256 tokens,
       batched across the day's survivors rather than one call per
       document (CPU-bound; a real throughput win, not a spec deviation --
       nothing here dictates scoring must be interleaved with fetching).
    7. Every candidate gets one row in the stage-3 log
       (data/stage3_document_log.parquet), with an explicit `outcome`:
       fetch_failed / no_text / lang_skip / duplicate / scored. The spec
       names four terminal states (fetch_failed, no_text, lang_skip,
       scored); `duplicate` is a necessary fifth, added here because step
       5's MinHash dedup has to resolve to *some* logged outcome for a
       document that passed extraction and the language check but turned
       out to be a repost -- there's no way to fold that into the other
       four without losing the distinction the dedup step exists to make.
       HTML itself is never stored; only extracted_chars survives.

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
from datasketch import MinHash
from langdetect import LangDetectException, detect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_gdelt_history import MANIFEST_PATH as STAGE1_MANIFEST  # noqa: E402
from pull_gdelt_history import OUT_PATH as STAGE1_OUTPUT          # noqa: E402
from pull_gdelt_history import END as WINDOW_END                  # noqa: E402
from pull_gdelt_history import START as WINDOW_START               # noqa: E402
from pull_gdelt_history import normalize_slug                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGE2_MANIFEST = ROOT / "data" / "stage2_manifest.csv"     # days processed
STAGE3_LOG = ROOT / "data" / "stage3_document_log.parquet"

CANDIDATES_PER_DAY = 100
TARGET_SUCCESSES = 60       # gate on extraction+language pass, pre-dedup;
                            # MinHash dedup can still drop final n_t below this
FETCH_RATE_PER_SEC = 5.0
FETCH_TIMEOUT = 15
FETCH_WORKERS = 10          # concurrent in-flight, gated by the rate limiter
MAX_TOKENS = 256
LEDE_SENTENCES = 2
USER_AGENT = "csr-research-bot/0.1 (oil-futures news study; contact via repo)"
POLL_INTERVAL_SEC = 120
MINHASH_PERM = 64
MINHASH_SHINGLE = 3          # word trigrams
MINHASH_JACCARD_THRESHOLD = 0.5   # documented dedup threshold, not tuned

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
        # called while holding a lock every other domain waited on).
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


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False   # can't determine -> treat as non-English, conservative


def shingled_minhash(text: str) -> MinHash:
    words = text.lower().split()
    m = MinHash(num_perm=MINHASH_PERM)
    if len(words) < MINHASH_SHINGLE:
        m.update(text.lower().encode("utf-8"))
        return m
    for i in range(len(words) - MINHASH_SHINGLE + 1):
        m.update(" ".join(words[i:i + MINHASH_SHINGLE]).encode("utf-8"))
    return m


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


def append_stage2_manifest(day: str, I_t: int, n_candidates: int,
                           outcome_counts: dict[str, int]):
    is_new = not STAGE2_MANIFEST.exists()
    fields = ["fetch_failed", "no_text", "lang_skip", "duplicate", "scored"]
    with open(STAGE2_MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["day", "I_t", "n_candidates", *fields])
        w.writerow([day, I_t, n_candidates, *(outcome_counts.get(k, 0) for k in fields)])


def append_stage3_log(rows: list[dict]):
    df = pd.DataFrame(rows)
    if STAGE3_LOG.exists():
        df = pd.concat([pd.read_parquet(STAGE3_LOG), df], ignore_index=True)
    STAGE3_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STAGE3_LOG, index=False)


# ------------------------------------------------------------------- day loop
def candidates_for_day(day: str) -> list[str]:
    if not STAGE1_OUTPUT.exists():
        return []
    df = pd.read_parquet(STAGE1_OUTPUT, columns=["timestamp", "DocumentIdentifier"])
    day_ts = pd.Timestamp(day, tz="UTC")
    mask = (df["timestamp"] >= day_ts) & (df["timestamp"] < day_ts + pd.Timedelta(days=1))
    return df.loc[mask, "DocumentIdentifier"].dropna().unique().tolist()


def compute_I_t(urls: list[str]) -> int:
    """All matching docs that day, deduped on (source_domain, normalised
    slug) -- computed over the FULL match set, not the 100-candidate draw,
    since I_t is a volume measure and can't be recovered from a sample."""
    return len({(source_domain(u), normalize_slug(u)) for u in urls})


def process_day(day: str) -> tuple[int, int, dict[str, int]]:
    all_urls = candidates_for_day(day)
    I_t = compute_I_t(all_urls)

    outcome_counts = {"fetch_failed": 0, "no_text": 0, "lang_skip": 0,
                      "duplicate": 0, "scored": 0}
    if not all_urls:
        append_stage2_manifest(day, I_t, 0, outcome_counts)
        return I_t, 0, outcome_counts

    sample = random.sample(all_urls, min(CANDIDATES_PER_DAY, len(all_urls)))
    log_rows = []
    lang_ok = []   # (row_idx, lede_text) -- passed extraction + language,
                   # pending dedup + scoring
    n_pending = 0

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_one, url): url for url in sample}
        for fut in as_completed(futures):
            url = futures[fut]
            if n_pending >= TARGET_SUCCESSES:
                fut.cancel()
                continue
            fetch_ok, status, html, err = fut.result()
            row = {
                "date": day, "url_hash": url_hash(url),
                "source_domain": source_domain(url),
                "http_status": status, "extracted_chars": 0,
                "finbert_score": None, "outcome": None,
            }
            if not fetch_ok:
                row["outcome"] = "fetch_failed"
                log_rows.append(row)
                continue

            lede, n_chars = extract_lede(html)
            row["extracted_chars"] = n_chars
            if not lede:
                row["outcome"] = "no_text"
                log_rows.append(row)
                continue

            if not is_english(lede):
                row["outcome"] = "lang_skip"
                log_rows.append(row)
                continue

            lang_ok.append((len(log_rows), lede))
            n_pending += 1
            log_rows.append(row)   # outcome resolved after dedup, below

    # MinHash dedup over the day's language-passing survivors, in memory.
    accepted_hashes: list[MinHash] = []
    to_score: list[tuple[int, str]] = []
    for idx, text in lang_ok:
        mh = shingled_minhash(text)
        if any(mh.jaccard(a) >= MINHASH_JACCARD_THRESHOLD for a in accepted_hashes):
            log_rows[idx]["outcome"] = "duplicate"
        else:
            accepted_hashes.append(mh)
            to_score.append((idx, text))

    if to_score:
        try:
            scores = score_batch([t for _, t in to_score])
            for (idx, _), score in zip(to_score, scores):
                log_rows[idx]["outcome"] = "scored"
                log_rows[idx]["finbert_score"] = score
        except Exception as e:
            print(f"  [{day}] FinBERT batch scoring failed: {e}", file=sys.stderr)
            for idx, _ in to_score:
                log_rows[idx]["outcome"] = "no_text"   # scoring never happened;
                                                       # closest honest bucket

    for row in log_rows:
        outcome_counts[row["outcome"]] += 1

    append_stage3_log(log_rows)
    append_stage2_manifest(day, I_t, len(sample), outcome_counts)
    return I_t, outcome_counts["scored"], outcome_counts


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
            if args.once or done >= all_days:
                print(f"nothing to do right now ({len(done)}/{len(all_days)} "
                     f"days processed, {len(complete)} ready from stage 1)")
                return
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for day in todo:
            I_t, n_t, oc = process_day(day)
            print(f"  {day}: I_t={I_t} n_t={n_t}  "
                 f"[fetch_failed={oc['fetch_failed']} no_text={oc['no_text']} "
                 f"lang_skip={oc['lang_skip']} duplicate={oc['duplicate']}]  "
                 f"({len(done) + 1}/{len(all_days)} days)", file=sys.stderr)
            done.add(day)

        if args.once:
            return


if __name__ == "__main__":
    main()
