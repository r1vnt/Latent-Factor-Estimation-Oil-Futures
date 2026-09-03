"""Acceptance gate: does the Kaggle per-expiry panel reproduce EIA settlements?

EIA RCLC1 is an official settlement series. If the Kaggle Close field is also
settlement, the two must agree to the tick on nearby position 1 across the whole
overlap. A systematic wedge -- especially one that widens on volatile days --
means the field is a last trade, since CL settles on a 14:00-14:30 ET VWAP.
"""
import sys, glob, os
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/csr/src")
from csr.curve.expiry import cl_expiry

RAW = "/home/claude/data/5237bc62-archive_1"
EIA = "/root/.claude/uploads/14f38bbb-b8b2-52d4-904e-1a549d61941d/5b35a7e6-RCLC1d.xls"
MONTH = dict(zip("FGHJKMNQUVXZ", range(1, 13)))

# ---------------------------------------------------------------- load raw --
rows = []
for path in sorted(glob.glob(os.path.join(RAW, "CME_CL*.csv"))):
    stem = os.path.basename(path)[len("CME_CL"):-len(".csv")]
    code, year = stem[0], int(stem[1:])
    if code not in MONTH:
        print("SKIP unparsed:", stem); continue
    d = pd.read_csv(path, parse_dates=["Date"])
    d["contract"] = f"CL{code}{year}"
    d["expiry"] = cl_expiry(year, MONTH[code])
    rows.append(d)

raw = pd.concat(rows, ignore_index=True)
raw = raw.rename(columns={"Date": "date", "Close": "close",
                          "Open Interest": "oi", "Volume": "volume"})
raw = raw[["date", "contract", "expiry", "close", "volume", "oi"]].dropna(subset=["close"])
print(f"contracts: {raw.contract.nunique()}   rows: {len(raw):,}   "
      f"dates: {raw.date.min().date()} -> {raw.date.max().date()}")

# ------------------------------------------------- build nearby positions --
# Front contract on date t = the unexpired contract with the earliest expiry.
raw = raw[raw.expiry >= raw.date]
raw = raw.sort_values(["date", "expiry"])
raw["pos"] = raw.groupby("date").cumcount() + 1

panel = raw[raw.pos <= 12].pivot(index="date", columns="pos", values="close")
print(f"panel: {panel.shape[0]:,} dates x {panel.shape[1]} positions")
cov = panel.notna().mean()
print("coverage by position: " +
      "  ".join(f"C{int(p)}={v:.1%}" for p, v in cov.items()))

# ------------------------------------------------------------- load EIA C1 --
e = pd.read_excel(EIA, sheet_name="Data 1", header=None, skiprows=3,
                  names=["date", "rclc1"])
e["date"] = pd.to_datetime(e["date"])
e = e.dropna().set_index("date")["rclc1"].astype(float)
print(f"EIA RCLC1: {len(e):,} obs  {e.index.min().date()} -> {e.index.max().date()}")

# ------------------------------------------------------------- the diff -----
j = pd.DataFrame({"kaggle": panel[1], "eia": e}).dropna()
j = j[(j.index >= "2010-01-01") & (j.index <= "2022-12-31")]
j["diff"] = j.kaggle - j.eia
j["abs"] = j["diff"].abs()

print(f"\n=== POSITION 1: Kaggle Close vs EIA RCLC1 settlement ===")
print(f"overlap        : {len(j):,} days  ({j.index.min().date()} -> {j.index.max().date()})")
print(f"exact match    : {(j['abs'] < 1e-9).mean():.2%}")
print(f"within 1 tick  : {(j['abs'] <= 0.01 + 1e-9).mean():.2%}")
print(f"within 5 ticks : {(j['abs'] <= 0.05 + 1e-9).mean():.2%}")
print(f"mean signed    : {j['diff'].mean():+.4f}")
print(f"mean abs       : {j['abs'].mean():.4f}")
print(f"median abs     : {j['abs'].median():.4f}")
print(f"p99 abs        : {j['abs'].quantile(.99):.4f}")
print(f"max abs        : {j['abs'].max():.4f}   on {j['abs'].idxmax().date()}")

# volatility dependence -- the tell for close-vs-settle
r = np.log(j.eia).diff().abs()
terc = pd.qcut(r.rank(method="first"), 3, labels=["low", "mid", "high"])
print("\nmean |diff| by volatility tercile (close-vs-settle widens with vol):")
for t, g in j.groupby(terc, observed=True):
    print(f"  {t:>4}: {g['abs'].mean():.4f}   (n={len(g):,})")

print("\nworst 8 days:")
print(j.reindex(j["abs"].nlargest(8).index)[["kaggle", "eia", "diff"]].to_string())

by_year = j.groupby(j.index.year)["abs"].agg(["mean", "max", "count"])
print("\nby year:")
print(by_year.to_string())

panel.to_parquet("/home/claude/panel_c1_c12.parquet")
print("\nwrote /home/claude/panel_c1_c12.parquet")
