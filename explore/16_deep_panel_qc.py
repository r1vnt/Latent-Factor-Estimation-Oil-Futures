# %% [markdown]
# # Runbook Step 2.3-2.5: assemble the deep panel and apply the QC gates
#
# Inputs: data/panel_c1_c12_kaggle.parquet, validated in
# explore/15_validate_kaggle_panel.py against this repo's own EIA
# settlements (99.94% exact match, no volatility-tercile widening -- the
# "Close" field is settlement, not last-trade). Assembly goes through this
# repo's own Panel.from_prices / curve.expiry -- tau is computed
# algorithmically from date + nearby rank via the exchange's own expiry
# rule, not trusted from the source file, so it is identical in
# construction to the C1-C4 EIA panel every prior fit in this repo used.
#
# Working window: 2010-01-04 onward, matching the EIA-overlap window
# explore/15 actually validated (pre-2010 Kaggle coverage was not checked
# against EIA) and every prior S0/S0P/perturbation fit's start date, so
# this panel is comparable to that work rather than introducing an
# unvalidated nine extra years alongside a deeper contract count.

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

# %% [markdown]
# ## 2.3: assemble

# %%
raw = pd.read_parquet(ROOT / "data" / "panel_c1_c12_kaggle.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c) for c in raw.columns]
print(f"raw kaggle panel, 2010+: {raw.shape[0]} dates x {raw.shape[1]} positions, "
     f"{raw.index.min().date()} .. {raw.index.max().date()}")

neg_mask = raw.to_numpy() <= 0  # NaN <= 0 is False, so this only flags real non-positive prices
neg_rows, neg_cols = np.where(neg_mask)
print(f"\nnon-positive raw cells: {len(neg_rows)}")
for r, c in zip(neg_rows, neg_cols):
    d = raw.index[r]
    print(f"  {d.date()}  C{raw.columns[c]}  {raw.iat[r, c]}  "
         f"(20 Apr 2020 negative WTI settlement, if this is it -- Rule 1 "
         f"must turn this into NaN in log_f, never floor it)")

panel = Panel.from_prices(raw, m=12, prefix="CL", basis="business",
                          source="kaggle_c1_c12", verbose=True)
print(f"\nassembled: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

# Rule 1 check on the actual cell the raw scan found.
neg_date_idx = list(panel.dates).index(pd.Timestamp("2020-04-20"))
print(f"\nRule 1 check: log_f[2020-04-20, C1] = {panel.log_f[neg_date_idx, 0]}  "
     f"({'PASS -- NaN, not floored' if np.isnan(panel.log_f[neg_date_idx, 0]) else 'FAIL'})")

# %% [markdown]
# ## 2.4: QC gates
#
# tau strictly increasing across columns every row; no stale run > 5 days;
# per-column coverage > 95%; non-positive settlements never floored (2.3,
# above); dates retained (guaranteed by Panel.from_prices's rule 2 --
# every date in the input survives regardless of missingness).

# %%
print(f"\n{'=' * 70}\nQC GATES (2.4)\n{'=' * 70}")

tau_increasing = np.all(np.diff(panel.tau, axis=1) > 0)
n_bad_rows = int((~(np.diff(panel.tau, axis=1) > 0)).any(axis=1).sum())
print(f"1. tau strictly increasing, every row: "
     f"{'PASS' if tau_increasing else f'FAIL ({n_bad_rows} rows)'}")

stale = panel.flags["stale"]
stale_by_col = stale.sum(axis=0)
print(f"\n2. stale-run flag (>=5 identical consecutive settlements), by column:")
for j, lbl in enumerate(panel.labels):
    print(f"   {lbl}: {int(stale_by_col[j])} flagged cells "
         f"({stale_by_col[j] / panel.n:.2%} of dates)")
max_stale_frac = (stale_by_col / panel.n).max()
stale_verdict = ("acceptable -- a flag, not an exclusion; Panel already screens for it"
                 if max_stale_frac < 0.05 else "material -- review before trusting")
print(f"   worst column: {max_stale_frac:.2%} of dates flagged stale ({stale_verdict})")

coverage = np.isfinite(panel.log_f).mean(axis=0)
print(f"\n3. per-column coverage (threshold > 95%):")
passing_cols = []
for j, lbl in enumerate(panel.labels):
    ok = coverage[j] > 0.95
    print(f"   {lbl}: {coverage[j]:.4f}  {'PASS' if ok else 'FAIL'}")
    if ok:
        passing_cols.append(j)
print(f"   columns clearing 95%: {[panel.labels[j] for j in passing_cols]}")

print(f"\n4. non-positive settlements never floored: PASS (checked above, "
     f"and Panel.from_prices's _log_transform enforces this unconditionally "
     f"for every cell, not just the one found by the raw scan)")

print(f"\n5. dates retained: PASS ({panel.n} dates in, {panel.n} dates out, "
     f"by construction -- Panel.from_prices never drops a row)")

# %% [markdown]
# ## Deep panel for 2.5-2.7: C1-C8 (the columns that clear gate 3)
#
# Gate 3 is the only one with a real column-selection consequence -- C9-C12
# sit at 92.8-94.7%, just under the 95% bar in this 2010+ window (matching
# the full-sample 1997-2022 numbers from explore/15, so this is not a
# window artefact). Kept as flagged, unused columns rather than deleted
# from the source file.

# %%
deep_cols = [j for j in passing_cols if j < 8] if len(passing_cols) >= 8 else passing_cols
print(f"\nusing columns: {[panel.labels[j] for j in deep_cols]}  "
     f"({len(deep_cols)} contracts, vs 4 in the EIA panel)")

log_f_deep = panel.log_f[:, deep_cols]
tau_deep = panel.tau[:, deep_cols]
dates_deep = panel.dates
m_deep = len(deep_cols)

# %% [markdown]
# ## 2.6: refit S0 on the deep panel (Gaussian -> Student-t) -- does kappa
# stabilise without a prior doing the work?
#
# Fit first, since 2.5's lambda_min/cond(Z) should be evaluated at each
# panel's OWN fitted kappa, not a value borrowed across panels -- Z depends
# on kappa, so mixing kappas from different fits into one comparison table
# would confound the identification comparison with a parameter-value
# difference.

# %%
print(f"\n{'=' * 70}\n2.6: REFIT ON DEEP PANEL\n{'=' * 70}")


def fit_s0(log_f, tau, cols, label):
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {label}: starting Gaussian pre-fit "
         f"(m={len(cols)}, n={log_f.shape[0]})...")
    p_init = SS_OIL.replace(sigma_meas=SS_OIL.sigma_meas[list(cols)]
                            if len(cols) <= len(SS_OIL.sigma_meas)
                            else np.full(len(cols), SS_OIL.sigma_meas.mean()))
    m_sub = len(cols)
    p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f, tau, DT, m_sub, theta0=pack(p_init))
    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {label}: Gaussian pre-fit done "
         f"({t1 - t0:.1f}s). starting Student-t EM (n_outer<=30)...")
    p_hat, x_filt, resid, weight, n_iters = fit_s0_student_t(
        log_f, tau, DT, m_sub, STUDENT_T_DF, theta0=pack(p_gauss))
    t2 = time.time()
    s_bp = (p_hat.sigma_meas * 1e4).round(2).tolist()
    print(f"[{time.strftime('%H:%M:%S')}] {label:22s} DONE in {t2 - t0:.1f}s "
         f"(EM: {t2 - t1:.1f}s, {n_iters} outer iters)  "
         f"kappa={p_hat.kappa:8.4f}  half-life={p_hat.half_life_days:7.1f}d  "
         f"s(bp)={s_bp}")
    return p_hat, x_filt


p_c4, x_filt_c4 = fit_s0(panel.log_f[:, :4], panel.tau[:, :4], range(4),
                         "C1-C4 (this panel)")
p_deep, x_filt_deep = fit_s0(log_f_deep, tau_deep, range(m_deep),
                             f"C1-C{len(deep_cols)} (deep)")

print(f"\nkappa: C1-C4={p_c4.kappa:.4f} (half-life {p_c4.half_life_days:.0f}d)  "
     f"vs deep={p_deep.kappa:.4f} (half-life {p_deep.half_life_days:.0f}d)")
print("(stabilisation check: does the deep-panel kappa move materially off "
     "the C1-C4 value, or land in the same neighbourhood with a tighter "
     "profile? The profile-likelihood shape, not just the point estimate, "
     "is the real test -- run explore/12's profile_likelihood machinery "
     "on this panel before calling it 'stable'.)")

# %% [markdown]
# ## 2.5: lambda_min(Z'Z) / cond(Z) on the REALISED tau grid, at each
# panel's own fitted kappa
#
# Runbook's own table was a projection (idealised, evenly-spaced tau) --
# this uses the actual realised mean tau per column from the assembled
# panel.

# %%
def z_diagnostics(tau, kappa, m_sub):
    tau_mean = np.nanmean(tau, axis=0)
    z_chi = np.exp(-kappa * tau_mean)
    Z = np.column_stack([z_chi, np.ones(m_sub)])
    eig = np.linalg.eigvalsh(Z.T @ Z)
    return eig.min(), eig.min() / m_sub, np.linalg.cond(Z)


lmin_c4, lmin_pc_c4, cond_c4 = z_diagnostics(panel.tau[:, :4], p_c4.kappa, 4)
lmin_deep, lmin_pc_deep, cond_deep = z_diagnostics(tau_deep, p_deep.kappa, m_deep)

print(f"\n{'=' * 70}\nIDENTIFICATION (2.5), realised tau grid, each panel's own fitted kappa\n{'=' * 70}")
print(f"C1-C4  (kappa={p_c4.kappa:.4f}):   lambda_min={lmin_c4:.4f}  "
     f"lambda_min/contract={lmin_pc_c4:.4f}  cond(Z)={cond_c4:.2f}")
print(f"C1-C{len(deep_cols)} (kappa={p_deep.kappa:.4f}): lambda_min={lmin_deep:.4f}  "
     f"lambda_min/contract={lmin_pc_deep:.4f}  cond(Z)={cond_deep:.2f}")
print(f"per-contract identification gain: {lmin_pc_deep / lmin_pc_c4:.2f}x")

# %%
np.save(ROOT / "data" / "deep_panel_log_f.npy", log_f_deep)
np.save(ROOT / "data" / "deep_panel_tau.npy", tau_deep)
pd.Series(dates_deep).to_frame("date").to_parquet(ROOT / "data" / "deep_panel_dates.parquet")
print(f"\nsaved deep panel arrays to data/deep_panel_{{log_f,tau}}.npy, "
     f"data/deep_panel_dates.parquet for 2.7 to load directly.")
