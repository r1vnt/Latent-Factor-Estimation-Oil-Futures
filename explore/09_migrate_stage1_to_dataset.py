# %% [markdown]
# # One-time migration: single-file Stage 1 output -> directory-of-parts
#
# flush_output()'s old design (read the whole accumulated parquet, concat
# the new batch, dedup, rewrite the whole thing) OOM'd the live run at
# ~1.29M rows / 714MB, and the resulting crash left an unsupervised
# ThreadPoolExecutor growing in memory with nothing to stop it (see the
# OUT_PATH comment in pull_gdelt_history.py for the full mechanism). The
# fix moves OUT_PATH from a single file to a directory of small,
# never-reread part files. This migrates the one big file that already
# exists into that format as a single seed part, verifies the migration
# read-for-read before touching anything irreversibly, and only then
# removes the old file.

# %%
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_gdelt_history as m1   # noqa: E402

OLD_FILE = m1.ROOT / "data" / "gdelt_history_extract.parquet"
NEW_DIR = m1.OUT_PATH   # data/gdelt_history_extract/, post-fix

assert NEW_DIR.name != OLD_FILE.name, "OUT_PATH wasn't actually repointed"

# %% [markdown]
# ## Read the old file once (last time anything reads it whole like this)

# %%
old = pd.read_parquet(OLD_FILE)
print(f"old file: {len(old):,} rows, columns: {list(old.columns)}")

# %% [markdown]
# ## Write it as a single seed part in the new directory

# %%
NEW_DIR.mkdir(parents=True, exist_ok=True)
seed_path = NEW_DIR / "part-00000-migrated.parquet"
tmp_path = seed_path.with_suffix(".tmp")
old.to_parquet(tmp_path, index=False)
import os
os.replace(tmp_path, seed_path)
print(f"wrote seed part -> {seed_path}")

# %% [markdown]
# ## Verify before deleting anything: read the new directory back, compare

# %%
new = pd.read_parquet(NEW_DIR)
assert len(new) == len(old), f"row count mismatch: old={len(old)} new={len(new)}"
assert set(new["DocumentIdentifier"]) == set(old["DocumentIdentifier"]), \
    "DocumentIdentifier set mismatch after migration"
assert set(new.columns) == set(old.columns)
print(f"verified: {len(new):,} rows read back identically from {NEW_DIR}")

# Column-projected read too -- this is the path candidates_for_day() uses.
proj = pd.read_parquet(NEW_DIR, columns=["timestamp", "DocumentIdentifier"])
assert len(proj) == len(old)
print(f"verified: column-projected read also returns {len(proj):,} rows")

# %% [markdown]
# ## Only now remove the old single file

# %%
OLD_FILE.unlink()
print(f"removed {OLD_FILE} -- {NEW_DIR} is now the sole Stage 1 output")
