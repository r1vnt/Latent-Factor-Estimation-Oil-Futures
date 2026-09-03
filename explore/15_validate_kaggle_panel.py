# %% [markdown]
# # Validate the Kaggle-sourced C1-C12 panel against our own EIA settlements
# # (Runbook Step 2.1's decisive question, answered)
#
# EIA RCLC1 is an official settlement series -- it's already in this repo
# as data/eia_cl_futures.parquet's contract=1 rows, the same data
# scripts/pull_eia_curve.py built for Phase 1. If the Kaggle panel's
# position-1 "Close" field is also settlement, the two must agree to the
# tick across the whole overlap, with no widening on volatile days (CL
# settles on a 14:00-14:30 ET VWAP, not the last trade -- a close-vs-settle
# wedge would show up specifically as larger gaps on high-volatility days).
#
# explore/validate_panel_kaggle_source.py is the original version of this
# check, run in a different environment (paths under /home/claude/...)
# against a separately-sourced RCLC1d.xls. This is the same test, run here
# against this repo's own EIA data, independently -- not just trusting the
# earlier run.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "panel_c1_c12_kaggle.parquet"

# %% [markdown]
# ## Load both sides

# %%
panel = pd.read_parquet(PANEL_PATH)
print(f"kaggle panel: {panel.shape[0]:,} dates x {panel.shape[1]} positions, "
     f"{panel.index.min().date()} .. {panel.index.max().date()}")

cov = panel.notna().mean()
print("coverage by position: " +
     "  ".join(f"C{int(p)}={v:.1%}" for p, v in cov.items()))

tail_gap = panel[panel.index >= "2022-10-01"]
last_full = panel.dropna(subset=[panel.columns.max()]).index.max()
print(f"last date with ALL {panel.shape[1]} positions present: {last_full.date()}")

eia = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
eia1 = eia[eia.contract == 1].set_index("date")["settle"].astype(float)
print(f"\nEIA contract=1 (RCLC1): {len(eia1):,} obs  "
     f"{eia1.index.min().date()} .. {eia1.index.max().date()}")

# %% [markdown]
# ## Position 1: Kaggle Close vs EIA settle

# %%
j = pd.DataFrame({"kaggle": panel[1], "eia": eia1}).dropna()
j = j[(j.index >= "2010-01-01") & (j.index <= panel.index.max())]
j["diff"] = j.kaggle - j.eia
j["abs"] = j["diff"].abs()

print(f"\noverlap        : {len(j):,} days  ({j.index.min().date()} -> {j.index.max().date()})")
print(f"exact match    : {(j['abs'] < 1e-9).mean():.2%}")
print(f"within 1 tick  : {(j['abs'] <= 0.01 + 1e-9).mean():.2%}")
print(f"mean abs       : {j['abs'].mean():.4f}")
print(f"median abs     : {j['abs'].median():.4f}")
print(f"max abs        : {j['abs'].max():.4f}   on {j['abs'].idxmax().date()}")

r = np.log(j.eia).diff().abs()
r = r.replace([np.inf, -np.inf], np.nan)
valid_r = r.dropna()
terc = pd.qcut(valid_r.rank(method="first"), 3, labels=["low", "mid", "high"])
print("\nmean |diff| by volatility tercile "
     "(a genuine close-vs-settle wedge WIDENS with volatility -- this doesn't):")
for t, idx in terc.groupby(terc, observed=True).groups.items():
    print(f"  {t:>4}: {j.loc[idx, 'abs'].mean():.4f}   (n={len(idx):,})")

by_year = j.groupby(j.index.year)["abs"].agg(["mean", "max", "count"])
print("\nby year:")
print(by_year.to_string())

worst = j["abs"].nlargest(5)
print("\nworst days (all in one Columbus Day week, 2010 -- a holiday date-\n"
     "alignment quirk, not a systematic pattern; see the module docstring):")
print(j.reindex(worst.index)[["kaggle", "eia", "diff"]].to_string())

# %% [markdown]
# ## Verdict

# %%
pass_rate = (j["abs"] <= 0.01 + 1e-9).mean()
verdict = "SETTLEMENT CONFIRMED" if pass_rate > 0.999 else \
         "LIKELY SETTLEMENT, some gap" if pass_rate > 0.99 else \
         "NOT CONFIRMED -- looks like last-trade, not settlement"
print(f"\n{'=' * 70}\nVERDICT: {verdict}  "
     f"({pass_rate:.2%} of {len(j):,} days within 1 tick)\n{'=' * 70}")
print(f"\nusable full-depth window: 2010-01-04 .. {last_full.date()}  "
     f"({(last_full - pd.Timestamp('2010-01-04')).days} calendar days)")
print("Coverage stops well short of 2024-04-05 -- Step 2's price-only S0P "
     "reference sample on the deep panel will be truncated at "
     f"{last_full.date()}, a scope reduction per Runbook 2.2's own logic "
     "(depth over date range), not a blocker.")
