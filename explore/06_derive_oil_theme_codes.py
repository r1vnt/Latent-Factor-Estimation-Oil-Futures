# %% [markdown]
# # Deriving OIL_THEME_CODES empirically
#
# Not from memory -- from the data. Downloads one full day of GKG at full
# resolution (96 files), parses every row unfiltered, splits V1Themes on
# `;`, and counts code frequencies across the whole day. The 300 most
# frequent codes get printed; energy/oil-relevant ones are selected by
# inspection and committed to pull_gdelt_history.py as an explicit constant
# with this sampling date and each code's observed frequency recorded.
#
# Sample day: 2024-01-09 (Tuesday) -- deliberately NOT the Wednesday EIA
# release day, so the derived vocabulary reflects general oil/energy news,
# not release-specific coverage.

# %%
import csv
import io
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pull_gdelt_history import fetch, COL_V1THEMES, MIN_COLS   # noqa: E402

SAMPLE_DAY = "20240109"
ts_list = [f"{SAMPLE_DAY}{h:02d}{m:02d}00" for h in range(24) for m in (0, 15, 30, 45)]
print(f"{len(ts_list)} full-resolution files for {SAMPLE_DAY}")

# %% [markdown]
# ## Download and count, unfiltered

# %%
theme_counts = Counter()
total_rows = 0
failed = 0

with ThreadPoolExecutor(max_workers=12) as ex:
    for ts, raw, err in ex.map(fetch, ts_list):
        if raw is None:
            failed += 1
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        for row in reader:
            if len(row) < MIN_COLS:
                continue
            total_rows += 1
            for code in row[COL_V1THEMES].split(";"):
                code = code.strip()
                if code:
                    theme_counts[code] += 1

print(f"{failed}/{len(ts_list)} files failed, {total_rows:,} total rows parsed, "
     f"{len(theme_counts):,} distinct theme codes observed")

# Persist full counts so re-analysis never needs a re-download.
COUNTS_PATH = Path(__file__).resolve().parents[1] / "data" / f"theme_counts_{SAMPLE_DAY}.csv"
with open(COUNTS_PATH, "w", newline="", encoding="utf-8") as f:
    w = __import__("csv").writer(f)
    w.writerow(["code", "count"])
    for code, n in theme_counts.most_common():
        w.writerow([code, n])
print(f"saved full counts -> {COUNTS_PATH}")

# %% [markdown]
# ## Top 300 codes by frequency

# %%
top300 = theme_counts.most_common(300)
for code, n in top300:
    print(f"{n:8d}  {code}")

# %% [markdown]
# ## Energy/oil-relevant codes, selected by inspection
#
# The top-300-by-raw-frequency cutoff turned out too narrow on its own:
# GDELT's globally most frequent codes are dominated by generic
# administrative/crisis/governance tags (TAX_FNCACT_*, CRISISLEX_*, WB_*
# health/education codes), which crowd out topic-specific vocabulary like
# oil even though it's a real, recurring topic. Only 2 codes matching the
# oil/energy substring pattern land in the top 300. Widened to search all
# ~9,800 observed codes instead, sorted by frequency, for the actual
# selection -- otherwise this misses real codes below an arbitrary cutoff.

# %%
import re
candidate_pattern = re.compile(r"OIL|ENERGY|GAS|PETROL|OPEC|FUEL|REFIN", re.IGNORECASE)
candidates = [(code, n) for code, n in theme_counts.most_common() if candidate_pattern.search(code)]
print(f"\n{len(candidates)} candidate energy/oil codes across all "
     f"{len(theme_counts):,} observed codes:")
for code, n in candidates:
    print(f"{n:8d}  {code}")
