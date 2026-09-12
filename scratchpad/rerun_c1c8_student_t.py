# Ad hoc, not committed (matches the convention explore/17's docstring
# refers to for scratchpad/diagnose_slow_fit.py).
#
# Purpose: recover the actual Student-t EM sigma_meas vector for C1-C8,
# which explore/16 printed but never saved to disk. explore/17's
# multistart established kappa=1.5015 as the Gaussian-stage optimum,
# confirmed stable across 7 starting points spanning a 10x range -- use
# that as the Gaussian warm start (matching explore/18's choice) rather
# than explore/16's original, less-informed start.
#
# Gate question this answers: after the Student-t EM re-weighting, are
# C5's and C7's measurement-error sigmas still pinned at
# MIN_SIGMA_MEAS = 1.4286bp (the floor), or has the EM moved them off it?
#   off the floor  -> EM handles the degeneracy at depth; skip runbook
#                     step 2, go straight to step 3 (fire C1-C18)
#   still pinned   -> do step 2 (smooth measurement curve) first

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import (  # noqa: E402
    MIN_SIGMA_MEAS,
    STUDENT_T_DF,
    fit_s0_gaussian,
    fit_s0_student_t,
    pack,
)
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
M = 8
KAPPA0 = 1.5015  # explore/17's confirmed C1-C8 Gaussian-stage optimum

log_f = np.load(ROOT / "data" / "deep_panel_log_f.npy")
tau = np.load(ROOT / "data" / "deep_panel_tau.npy")
print(f"[{time.strftime('%H:%M:%S')}] deep panel (C1-C8): {log_f.shape}, "
     f"floor={MIN_SIGMA_MEAS * 1e4:.4f}bp")

t0 = time.time()
p_init = SS_OIL.replace(kappa=KAPPA0, sigma_meas=np.full(M, 0.005))
print(f"[{time.strftime('%H:%M:%S')}] starting Gaussian pre-fit (warm start "
     f"kappa0={KAPPA0}, default floor)...")
p_gauss, _, _, _, _, ll_gauss = fit_s0_gaussian(log_f, tau, DT, M, theta0=pack(p_init))
t1 = time.time()
s_gauss_bp = (p_gauss.sigma_meas * 1e4).round(4).tolist()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian done ({t1 - t0:.1f}s)  "
     f"kappa={p_gauss.kappa:.4f}  ll={ll_gauss:.2f}")
print(f"  s_gauss(bp): {s_gauss_bp}")

print(f"[{time.strftime('%H:%M:%S')}] starting Student-t EM (n_outer<=30)...")
p_hat, x_filt, resid, weight, n_iters = fit_s0_student_t(
    log_f, tau, DT, M, STUDENT_T_DF, theta0=pack(p_gauss))
t2 = time.time()
s_hat_bp = (p_hat.sigma_meas * 1e4).round(4).tolist()
print(f"[{time.strftime('%H:%M:%S')}] Student-t EM done ({t2 - t1:.1f}s, "
     f"{n_iters} outer iters). total={t2 - t0:.1f}s")
print(f"kappa={p_hat.kappa:.4f}  half-life={p_hat.half_life_days:.1f}d")
print(f"s_student_t(bp): {s_hat_bp}")

floor_bp = MIN_SIGMA_MEAS * 1e4
pinned = [i + 1 for i, s in enumerate(s_hat_bp) if abs(s - floor_bp) < 0.05]
print(f"\n{'=' * 70}\nGATE VERDICT\n{'=' * 70}")
print(f"floor = {floor_bp:.4f}bp")
print(f"positions still pinned at floor after Student-t EM: "
     f"{[f'C{i}' for i in pinned] if pinned else 'NONE'}")
if pinned:
    print("STILL PINNED -> do runbook step 2 (smooth measurement curve) first.")
else:
    print("OFF THE FLOOR -> EM handles the degeneracy at depth; skip step 2, "
         "go straight to step 3 (fire C1-C18).")

out = {
    "kappa0_warm_start": KAPPA0,
    "gaussian": {"kappa": p_gauss.kappa, "ll": ll_gauss, "s_bp": s_gauss_bp,
                "seconds": t1 - t0},
    "student_t": {"kappa": p_hat.kappa, "half_life_days": p_hat.half_life_days,
                 "s_bp": s_hat_bp, "n_outer_iters": n_iters, "seconds": t2 - t1},
    "floor_bp": floor_bp,
    "pinned_positions": pinned,
    "total_seconds": t2 - t0,
}
out_path = ROOT / "results" / "c1c8_student_t_svector.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
