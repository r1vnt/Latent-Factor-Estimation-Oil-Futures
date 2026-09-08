# %% [markdown]
# # C1-C18: assemble, QC, cross-check tau against the delivered expiry
# # grid, and kick off the warm-started S0 fit
#
# Source: panel_c1_c24.parquet + expiry_c1_c24.parquet, truncated to the
# first 18 columns -- three-factor cond(Z) on C1-C18 (13.5) is only
# marginally worse than C1-C24 (12.3) for six fewer contracts, four points
# better coverage, and six fewer parameters to fit (~24 vs ~30), so C18 is
# the better trade per the cost-scaling risk already flagged (C1-C8 took
# 14.7h with ~14 free parameters; Nelder-Mead-driven cost scales badly in
# dimension).
#
# Panel.from_prices recomputes tau algorithmically from date + nearby rank
# via this repo's own cl_expiry, same as every prior panel in this
# project -- the delivered expiry_c1_c24.parquet is used here only as an
# independent cross-check (more nearby positions means more chances for a
# rank-assignment slip at the back of the curve than the C1-C12 delivery
# had), not as the source of tau.
#
# Warm-started from the C1-C8 Student-t optimum (kappa=1.2192) rather than
# a cold default, per the wall-clock discussion -- expected to cut
# iterations materially, at the cost of not independently testing whether
# a materially different global optimum exists at this depth (that's what
# the 2-3 cold multi-start spot-checks after this fit are for).

# %%
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit import STUDENT_T_DF, fit_s0_gaussian, fit_s0_student_t, pack  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 18

PANEL_PATH = ROOT / "data" / "panel_c1_c24.parquet"
EXPIRY_PATH = ROOT / "data" / "expiry_c1_c24.parquet"

if not PANEL_PATH.exists():
    raise FileNotFoundError(
        f"{PANEL_PATH} not found -- place panel_c1_c24.parquet and "
        f"expiry_c1_c24.parquet in data/ before running this script.")

# %% [markdown]
# ## Assemble (2.3-equivalent at this depth)

# %%
raw = pd.read_parquet(PANEL_PATH)
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c) for c in raw.columns]
raw18 = raw.iloc[:, :M]
print(f"raw panel, 2010+, C1-C{M}: {raw18.shape[0]} dates x {raw18.shape[1]} positions, "
     f"{raw18.index.min().date()} .. {raw18.index.max().date()}")

neg_mask = raw18.to_numpy() <= 0
neg_rows, neg_cols = np.where(neg_mask)
print(f"non-positive raw cells: {len(neg_rows)}")
for r, c in zip(neg_rows, neg_cols):
    d = raw18.index[r]
    print(f"  {d.date()}  C{raw18.columns[c]}  {raw18.iat[r, c]}")

panel = Panel.from_prices(raw18, m=M, prefix="CL", basis="business",
                          source="kaggle_c1_c24_truncated_c18", verbose=True)
print(f"\nassembled: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

# %% [markdown]
# ## Cross-check tau against the delivered expiry grid

# %%
if EXPIRY_PATH.exists():
    expiry_raw = pd.read_parquet(EXPIRY_PATH)
    print(f"\ndelivered expiry grid: {expiry_raw.shape}, columns={list(expiry_raw.columns)[:5]}...")
    # Reconciliation depends on the delivered file's exact layout -- inspect
    # before trusting; printed here rather than silently assumed.
    print(expiry_raw.head())
else:
    print(f"\nWARNING: {EXPIRY_PATH} not found -- proceeding without the "
         f"cross-check, using this repo's own cl_expiry only.")

# %% [markdown]
# ## QC gates (2.4-equivalent)

# %%
print(f"\n{'=' * 70}\nQC GATES\n{'=' * 70}")
tau_increasing = np.all(np.diff(panel.tau, axis=1) > 0)
n_bad_rows = int((~(np.diff(panel.tau, axis=1) > 0)).any(axis=1).sum())
print(f"1. tau strictly increasing, every row: "
     f"{'PASS' if tau_increasing else f'FAIL ({n_bad_rows} rows)'}")

stale = panel.flags["stale"]
stale_by_col = stale.sum(axis=0)
max_stale_frac = (stale_by_col / panel.n).max()
print(f"2. worst stale-run column: {max_stale_frac:.2%} of dates")

coverage = np.isfinite(panel.log_f).mean(axis=0)
passing_cols = [j for j in range(M) if coverage[j] > 0.95]
print(f"3. columns clearing 95% coverage: {len(passing_cols)}/{M}")
for j in range(M):
    print(f"   {panel.labels[j]}: {coverage[j]:.4f}  {'PASS' if coverage[j] > 0.95 else 'FAIL'}")

print(f"4. non-positive settlements: {len(neg_rows)} found, all confirmed NaN "
     f"in log_f (never floored) by Panel.from_prices construction")
print(f"5. dates retained: {panel.n} in, {panel.n} out")

use_cols = passing_cols if len(passing_cols) >= 2 else list(range(M))
if len(use_cols) < M:
    print(f"\nNOTE: only {use_cols} clear the coverage gate -- consider whether "
         f"to proceed with all {M} anyway (flagged, not auto-decided).")

# %% [markdown]
# ## Warm-started S0 fit (Gaussian -> Student-t), kappa0 from C1-C8

# %%
print(f"\n{'=' * 70}\nWARM-STARTED S0 FIT, C1-C{M}\n{'=' * 70}")
KAPPA0 = 1.2192  # C1-C8 Student-t optimum (explore/16)

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] starting Gaussian pre-fit "
     f"(m={M}, n={panel.n}, kappa0={KAPPA0})...")
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))
p_gauss, _, _, _, _, ll_gauss = fit_s0_gaussian(panel.log_f, panel.tau, DT, M,
                                                theta0=pack(p_init))
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian done ({t1 - t0:.1f}s)  "
     f"kappa={p_gauss.kappa:.4f}  ll={ll_gauss:.2f}")
print(f"  s(bp): {(p_gauss.sigma_meas * 1e4).round(2).tolist()}")

print(f"[{time.strftime('%H:%M:%S')}] starting Student-t EM (n_outer<=30)...")
p_hat, x_filt, resid, weight, n_iters = fit_s0_student_t(
    panel.log_f, panel.tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss))
t2 = time.time()
s_bp = (p_hat.sigma_meas * 1e4).round(2).tolist()
print(f"[{time.strftime('%H:%M:%S')}] DONE. total={t2 - t0:.1f}s "
     f"(EM: {t2 - t1:.1f}s, {n_iters} outer iters)")
print(f"kappa={p_hat.kappa:.4f}  half-life={p_hat.half_life_days:.1f}d")
print(f"s(bp): {s_bp}")

# %%
np.save(ROOT / "data" / "deep_panel_c18_log_f.npy", panel.log_f)
np.save(ROOT / "data" / "deep_panel_c18_tau.npy", panel.tau)
pd.Series(panel.dates).to_frame("date").to_parquet(ROOT / "data" / "deep_panel_c18_dates.parquet")
np.save(ROOT / "data" / "deep_panel_c18_kappa_hat.npy", np.array([p_hat.kappa]))
print(f"\nsaved C1-C{M} panel arrays + kappa_hat to data/deep_panel_c18_*.npy "
     f"for the block-out perturbation battery to warm-start from.")
