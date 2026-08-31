# %% [markdown]
# # Stage 5: attrition diagnostic, after not before
#
# No upfront source-panel decision was needed because this answers it
# retrospectively, from data already held -- a groupby over stage 3's
# per-candidate log, not another fetch.
#
# Two outcomes:
#   - Composition stable across years -> use everything.
#   - Composition drifted -> restrict to sources present across the whole
#     window and recompute stage 4 from data already held.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_score_stage2 import STAGE3_LOG   # noqa: E402

# %% [markdown]
# ## Load and group by year x source

# %%
log = pd.read_parquet(STAGE3_LOG)
log["year"] = log["date"].str[:4].astype(int)
success = log[log["finbert_ok"]].copy()

print(f"{len(log):,} candidates logged, {len(success):,} successfully scored "
     f"({len(success) / max(len(log), 1):.1%} success rate)")

by_year_source = (success.groupby(["year", "source_domain"]).size()
                  .rename("n").reset_index())
year_totals = success.groupby("year").size().rename("year_total")
by_year_source = by_year_source.merge(year_totals, on="year")
by_year_source["share"] = by_year_source["n"] / by_year_source["year_total"]

# %% [markdown]
# ## Composition stability
#
# For each source, the range of its yearly share across the window. A
# source whose share swings wildly (present at 20% some years, absent
# others) is a composition-drift signal; a source with a roughly constant
# share throughout is not.

# %%
pivot = by_year_source.pivot(index="source_domain", columns="year", values="share").fillna(0.0)
pivot["min_share"] = pivot.min(axis=1)
pivot["max_share"] = pivot.max(axis=1)
pivot["range"] = pivot["max_share"] - pivot["min_share"]
pivot["present_every_year"] = (pivot.drop(columns=["min_share", "max_share", "range"]) > 0).all(axis=1)

top_by_volume = success["source_domain"].value_counts().head(20).index
print("\nTop 20 sources by volume -- share per year, and whether present every year:")
print(pivot.loc[pivot.index.intersection(top_by_volume),
               ["min_share", "max_share", "range", "present_every_year"]]
     .sort_values("range", ascending=False).round(4))

n_years = success["year"].nunique()
stable_sources = pivot[pivot["present_every_year"]].index.tolist()
print(f"\n{len(stable_sources)} of {len(pivot)} sources present in every one "
     f"of {n_years} years")

# %% [markdown]
# ## Verdict

# %%
# A simple, legible bar: composition is "stable" if the sources present in
# every year account for most of the successfully-scored volume. Drifted
# otherwise -- worth a by-eye look at the printed table above regardless of
# which side of the bar this lands on, since the bar is a summary, not a
# substitute for reading the table.
stable_volume_share = success[success["source_domain"].isin(stable_sources)].shape[0] / max(len(success), 1)
print(f"volume share from sources present every year: {stable_volume_share:.1%}")

if stable_volume_share >= 0.8:
    print("VERDICT: composition stable -- use everything.")
else:
    print("VERDICT: composition drifted -- restrict stage 4 to "
         f"{len(stable_sources)} sources present across the whole window "
         "and recompute (groupby on data already held, not another fetch).")
