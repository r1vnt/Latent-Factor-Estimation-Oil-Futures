# %% [markdown]
# # Retroactive re-filter after dropping ENV_OIL / WB_539 / WB_548
#
# Stage 1 retains every field needed to re-evaluate the predicate (URL,
# V1Themes) for every row it's already kept -- so tightening the theme-code
# set is a filter over the parquet already on disk, not a re-download.
# `pull_gdelt_history.py`'s OIL_THEME_CODES has already been updated; this
# re-applies theme_matches()/url_matches() to the accumulated corpus with
# the new set and drops whatever no longer qualifies.
#
# Stage 2's progress (510 days, scored against the old/diluted candidate
# pools) does NOT survive this: those pools included documents that no
# longer match, so re-drawing is required. Its manifest and log are wiped
# at the end here, not carried forward selectively -- a day's candidate
# pool composition changed, so there's no way to salvage a specific day's
# already-scored set without re-drawing it.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_gdelt_history as m1   # noqa: E402
from fetch_score_stage2 import STAGE2_MANIFEST, STAGE3_LOG  # noqa: E402

# %% [markdown]
# ## Re-filter the accumulated corpus

# %%
df = pd.read_parquet(m1.OUT_PATH)
before = len(df)
print(f"before: {before:,} rows")

keep = df.apply(
    lambda r: m1.theme_matches(r["Themes"]) or m1.url_matches(r["DocumentIdentifier"]),
    axis=1,
)
filtered = df[keep].reset_index(drop=True)
after = len(filtered)
print(f"after:  {after:,} rows ({after / before:.1%} retained)")

tmp_path = m1.OUT_PATH.with_suffix(".tmp")
filtered.to_parquet(tmp_path, index=False)
import os
os.replace(tmp_path, m1.OUT_PATH)
print(f"wrote filtered corpus -> {m1.OUT_PATH}")

# %% [markdown]
# ## Wipe stage 2/3 -- their progress was drawn from the diluted pool

# %%
for p in [STAGE2_MANIFEST, STAGE3_LOG]:
    if p.exists():
        p.unlink()
        print(f"removed {p}")

# %% [markdown]
# ## Fresh eyeball validation on the re-filtered corpus

# %%
sample = filtered["DocumentIdentifier"].sample(50, random_state=2024)
for u in sample:
    print(u[:110])
