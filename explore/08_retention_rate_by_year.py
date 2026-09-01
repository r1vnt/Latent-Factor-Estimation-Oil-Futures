# %% [markdown]
# # Retention rate by year: did dropping ENV_OIL / WB_539 / WB_548 land
# # uniformly across 2015-2024, or does it look like GDELT's tagging
# # conventions for those three codes drifted over the window?
#
# The corpus this would ideally be measured against (matches under the old,
# wider theme-code set) no longer exists on disk -- `07_retroactive_refilter.py`
# overwrote it in place when the set was tightened, and stage 1 only ever
# retains rows that pass the *current* predicate, not a history of what used
# to pass. So this re-fetches a small, independent sample -- one day per
# year, 8 files spread across the day (not all 24) -- parses every row in
# each file regardless of whether it matches anything, and classifies each
# one under both the old and the new theme-code set directly, rather than
# reasoning about what was already dropped.
#
# If retention (new_total / old_total) is roughly flat across years, the
# removal was uniform and I_t/gamma in eq. (24) aren't carrying a drift
# artifact from this change. If it trends with year, GDELT's usage of these
# three codes changed over the window and the removal itself introduced a
# trend into I_t that isn't real news volume.
#
# Picks March 10 each year (a Tuesday-or-nearest weekday, never the EIA's
# Wednesday release day, consistent with the original OIL_THEME_CODES
# derivation's non-release-day discipline) -- 2015-03-10 postdates GDELT's
# 2015-02-19 launch and 2024-03-10 predates the window's 2024-04-05 end, so
# every year in scope has a valid sample date.

# %%
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_gdelt_history as m1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "retention_by_year.csv"

DROPPED_CODES = frozenset({
    "ENV_OIL",
    "WB_539_OIL_AND_GAS_POLICY_STRATEGY_AND_INSTITUTIONS",
    "WB_548_PPP_IN_OIL_AND_GAS",
})
OLD_THEME_CODES = m1.OIL_THEME_CODES | DROPPED_CODES
NEW_THEME_CODES = m1.OIL_THEME_CODES   # current, post-removal set

SAMPLE_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]   # 8 of 24 -- enough rows/year
                                              # to estimate a rate, small
                                              # enough not to compete with
                                              # the live Stage 1/2 pulls


def sample_date_for_year(year: int) -> datetime:
    d = datetime(year, 3, 10, tzinfo=timezone.utc)
    while d.weekday() == 2:   # 2 == Wednesday; nudge off the EIA release day
        d += timedelta(days=1)
    return d


def classify(themes: str, docid: str) -> tuple[bool, bool]:
    codes = themes.split(";")
    url_hit = m1.url_matches(docid)
    old_hit = any(c in OLD_THEME_CODES for c in codes) or url_hit
    new_hit = any(c in NEW_THEME_CODES for c in codes) or url_hit
    return old_hit, new_hit


# %% [markdown]
# ## Fetch 8 files/day for one day per year, classify every row

# %%
import csv
import io
import zipfile

records = []
for year in range(2015, 2025):
    day = sample_date_for_year(year)
    print(f"{year}: sampling {day.date()}")
    old_total = new_total = 0
    dropped_code_hits = {c: 0 for c in DROPPED_CODES}
    n_files_ok = 0
    for h in SAMPLE_HOURS:
        ts = (day + timedelta(hours=h)).strftime("%Y%m%d%H%M%S")
        ts_, raw, err = m1.fetch(ts)
        if raw is None:
            print(f"  {ts}: skip ({err})")
            continue
        n_files_ok += 1
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                continue
            text = zf.read(names[0]).decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        for row in reader:
            if len(row) < m1.MIN_COLS:
                continue
            docid, themes = row[m1.COL_DOCID], row[m1.COL_V1THEMES]
            old_hit, new_hit = classify(themes, docid)
            if old_hit:
                old_total += 1
                if new_hit:
                    new_total += 1
                else:
                    for c in DROPPED_CODES:
                        if c in themes.split(";"):
                            dropped_code_hits[c] += 1
    retention = new_total / old_total if old_total else float("nan")
    records.append({
        "year": year, "date": day.date().isoformat(),
        "files_ok": n_files_ok, "old_total": old_total, "new_total": new_total,
        "dropped": old_total - new_total, "retention_rate": retention,
        **{f"dropped_by_{c}": dropped_code_hits[c] for c in DROPPED_CODES},
    })
    print(f"  old={old_total} new={new_total} retention={retention:.1%}"
         if old_total else "  no matches this sample")

# %% [markdown]
# ## Summary table + trend check

# %%
df = pd.DataFrame(records)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}")
print(df[["year", "date", "files_ok", "old_total", "new_total", "dropped",
        "retention_rate"]].to_string(index=False))

valid = df.dropna(subset=["retention_rate"])
if len(valid) >= 3:
    corr = valid["year"].corr(valid["retention_rate"])
    print(f"\ncorr(year, retention_rate) = {corr:.3f}")
    print("(near 0: removal looks uniform across years -- no drift artifact "
         "in I_t from this change. Large |corr|: GDELT's usage of these "
         "three codes trended over the window and the removal introduced "
         "a spurious trend into I_t.)")
