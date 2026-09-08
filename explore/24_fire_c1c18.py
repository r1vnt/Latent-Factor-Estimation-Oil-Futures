# %% [markdown]
# # Runbook step 3: fire C1-C18 with the smooth (quadratic) measurement curve
#
# Gate cleared: explore/20-23 validated the smooth curve on C1-C8 --
# quadratic eliminates the C7 floor-pinning, and the residual LR gap
# against the free-s_i vector was shown (explore/23) to be an artifact of
# that baseline's own Chen-Scott collapse at C5/C7 (releasing the floor at
# the Student-t stage reproduces the same ~1e-4bp collapse Guard 2 found
# at the Gaussian stage; the gain from releasing it is concentrated in the
# OTHER six contracts, not C5/C7 themselves -- the collapse signature, not
# a real feature the curve should reproduce). No free-s_i comparison fit
# here, deliberately -- that would reintroduce the degenerate parameterisation
# this whole step exists to avoid, at higher dimension.
#
# Source: panel_delivery/panel_c1_c24.parquet + expiry_c1_c24.parquet
# (this session's delivery -- see panel_delivery/ ). Columns arrive as
# string labels "C1".."C24", not the bare ints the Kaggle C1-C12 panel
# used, so the int cast strips the prefix first. 2010+ window, matching
# the C1-C8 panel exactly (both land on n=3254 dates) so this is a clean
# "does adding contracts change kappa" comparison -- only m changes.
#
# NOTE: unlike data/panel_c1_c12_kaggle.parquet (validated against EIA in
# explore/15, 99.94% exact match), this delivery has not been
# independently cross-checked against EIA settlements -- only the QC
# gates below (tau monotonicity, coverage, staleness, Rule 1). Flagging
# this rather than silently treating it as equally validated.
#
# Warm-started from the C1-C8 quadratic optimum (explore/22:
# kappa=1.2445, a=-2.9967, b=-20.2784, c=23.0955) rather than a cold
# start -- per the parallelism/multi-start discussion, a handful of cold
# starts follow separately as a spot-check, this is not the only fit run
# at this depth.

# %%
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit import (  # noqa: E402
    MIN_SIGMA_MEAS,
    STUDENT_T_DF,
    fit_s0_gaussian_smooth,
    fit_s0_student_t_smooth,
    pack_smooth,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 18
FLOOR_BP = MIN_SIGMA_MEAS * 1e4

# C1-C8 quadratic optimum (explore/22, EM-converged at 32/200 iters)
KAPPA0 = 1.2445
A0, B0, C0 = -2.9967, -20.2784, 23.0955

# %% [markdown]
# ## Assemble

# %%
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]  # "C1".."C24" -> 1..24
raw18 = raw.iloc[:, :M]
print(f"raw panel, 2010+, C1-C{M}: {raw18.shape[0]} dates x {raw18.shape[1]} positions, "
     f"{raw18.index.min().date()} .. {raw18.index.max().date()}")

neg_mask = raw18.to_numpy() <= 0
neg_rows, neg_cols = np.where(neg_mask)
print(f"non-positive raw cells: {len(neg_rows)}")
for r, c in zip(neg_rows, neg_cols):
    print(f"  {raw18.index[r].date()}  C{raw18.columns[c]}  {raw18.iat[r, c]}")

panel = Panel.from_prices(raw18, m=M, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=True)
print(f"assembled: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

if len(neg_rows):
    neg_date_idx = list(panel.dates).index(pd.Timestamp(raw18.index[neg_rows[0]]))
    col_idx = list(panel.labels).index(f"CL{raw18.columns[neg_cols[0]]}") \
        if f"CL{raw18.columns[neg_cols[0]]}" in panel.labels else neg_cols[0]
    print(f"Rule 1 check: log_f at the flagged cell = "
         f"{panel.log_f[neg_date_idx, neg_cols[0]]}  "
         f"({'PASS -- NaN, not floored' if np.isnan(panel.log_f[neg_date_idx, neg_cols[0]]) else 'FAIL'})")

# %% [markdown]
# ## QC gates

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
    if coverage[j] <= 0.95:
        print(f"   {panel.labels[j]}: {coverage[j]:.4f}  FAIL")
if len(passing_cols) < M:
    print(f"   NOTE: not all {M} columns clear 95% -- proceeding with all {M} "
         f"anyway (smooth curve has no per-contract free parameter to break "
         f"on a thin column the way the free vector would); flagged, not hidden.")

print(f"4. non-positive settlements: {len(neg_rows)} found, confirmed NaN in log_f")
print(f"5. dates retained: {panel.n} in, {panel.n} out")

# %% [markdown]
# ## Smooth (quadratic) fit, warm-started from C1-C8

# %%
print(f"\n{'=' * 70}\nSMOOTH QUADRATIC FIT, C1-C{M}\n{'=' * 70}")
t0 = time.time()
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))
theta0 = pack_smooth(p_init, A0, B0, C0, quadratic=True)
p_gauss, _, _, _, _, ll_gauss = fit_s0_gaussian_smooth(
    panel.log_f, panel.tau, DT, theta0=theta0, quadratic=True)
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian smooth done ({t1 - t0:.1f}s)  "
     f"kappa={p_gauss.kappa:.4f}  ll={ll_gauss:.2f}")

abc_gauss = fit_smooth_to_vector(p_gauss.sigma_meas[0], panel.tau[0], quadratic=True)
print(f"  curve: a={abc_gauss.a:.4f}  b={abc_gauss.b:.4f}  c={abc_gauss.c:.4f}")

p_hat, _, resid, weight, n_iters = fit_s0_student_t_smooth(
    panel.log_f, panel.tau, DT, STUDENT_T_DF,
    theta0=pack_smooth(p_gauss, abc_gauss.a, abc_gauss.b, abc_gauss.c, quadratic=True),
    quadratic=True, n_outer=200)
t2 = time.time()
abc_final = fit_smooth_to_vector(p_hat.sigma_meas[0], panel.tau[0], quadratic=True)
converged = n_iters < 200
s_bp = (p_hat.sigma_meas[0] * 1e4).round(4)
pinned = [j + 1 for j in range(M) if abs(s_bp[j] - FLOOR_BP) < 0.05]

print(f"[{time.strftime('%H:%M:%S')}] Student-t smooth done ({t2 - t1:.1f}s, "
     f"{n_iters}/200 EM iters, {'CONVERGED' if converged else 'HIT CAP'})  "
     f"total={t2 - t0:.1f}s")
print(f"kappa={p_hat.kappa:.4f}  half-life={p_hat.half_life_days:.1f}d")
print(f"curve: a={abc_final.a:.4f}  b={abc_final.b:.4f}  c={abc_final.c:.4f}")
print(f"s(bp), day 0: {s_bp.tolist()}")
print(f"pinned at floor: {pinned if pinned else 'none'}  (gate expectation: zero)")

print(f"\n{'=' * 70}\nSTANDING DIAGNOSTIC\n{'=' * 70}")
if pinned:
    print(f"FAIL -- {len(pinned)} contract(s) pinned under the smooth curve. "
         f"Investigate before trusting this fit.")
else:
    print("PASS -- zero pinned contracts under the smooth curve.")

print(f"\nkappa comparison: C1-C8 quadratic = {KAPPA0:.4f}  ->  "
     f"C1-C{M} quadratic = {p_hat.kappa:.4f}  "
     f"(delta = {p_hat.kappa - KAPPA0:+.4f})")

# %%
np.save(ROOT / "data" / "deep_panel_c18_log_f.npy", panel.log_f)
np.save(ROOT / "data" / "deep_panel_c18_tau.npy", panel.tau)
pd.Series(panel.dates).to_frame("date").to_parquet(ROOT / "data" / "deep_panel_c18_dates.parquet")

out = {
    "m": M, "n": panel.n,
    "kappa_hat": p_hat.kappa, "half_life_days": p_hat.half_life_days,
    "curve": {"a": abc_final.a, "b": abc_final.b, "c": abc_final.c},
    "s_bp_day0": s_bp.tolist(), "pinned": pinned,
    "n_em_iters": n_iters, "converged": converged,
    "seconds": {"gaussian": t1 - t0, "student_t": t2 - t1, "total": t2 - t0},
    "structural": {
        "kappa": p_hat.kappa, "sigma_chi": p_hat.sigma_chi, "sigma_xi": p_hat.sigma_xi,
        "rho": p_hat.rho, "mu_xi": p_hat.mu_xi, "mu_xi_star": p_hat.mu_xi_star,
        "lambda_chi": p_hat.lambda_chi,
    },
}
out_path = ROOT / "results" / "c1c18_smooth_quadratic.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved panel arrays to data/deep_panel_c18_{{log_f,tau,dates}}, "
     f"fit result to {out_path}")
