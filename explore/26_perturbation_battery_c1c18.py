# %% [markdown]
# # Runbook step 4: full perturbation battery, C1-C18, smooth quadratic curve
#
# explore/25's kappa profile confirmed the C1-C8 -> C1-C18 kappa shift
# (1.2445 -> 0.7574) is a real, sharply-identified disagreement between
# panels, not flat-profile noise. This battery maps where it comes from:
# every leave-one-contract-out subset (18 fits), the three block-outs
# (drop C1-C4 / C5-C12 / C13-C18), and two sub-period splits -- the full
# 23-fit version, not the 5-fit reduced one, since more perturbations is
# strictly better evidence for (or against) the stability claim. Run
# sequentially (no parallelism needed at this wall-clock), warm-started
# from the C1-C18 baseline, with 3 configs re-run cold (baseline itself,
# one block-out, one leave-one-out) as a basin-trapping check per the
# warm-start caution: a warm start can only ever find the optimum nearest
# its starting point, not confirm it's the global one.
#
# Sub-period boundaries match explore/14's own convention (2010-2016 /
# 2017+), truncated to this panel's actual end date (2022-11-21).

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
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.pricing.twofactor import TwoFactorParams  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
FLOOR_BP = MIN_SIGMA_MEAS * 1e4

# %% [markdown]
# ## Reassemble the full C1-C18 panel (for .slice() on sub-periods)

# %%
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

baseline_fit = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s0, c0 = baseline_fit["structural"], baseline_fit["curve"]
p_warm = TwoFactorParams(
    kappa=s0["kappa"], sigma_chi=s0["sigma_chi"], sigma_xi=s0["sigma_xi"],
    rho=s0["rho"], mu_xi=s0["mu_xi"], mu_xi_star=s0["mu_xi_star"],
    lambda_chi=s0["lambda_chi"], sigma_meas=np.array([0.005]),
)
A_WARM, B_WARM, C_WARM = c0["a"], c0["b"], c0["c"]
p_warm_full = p_warm.replace(sigma_meas=np.exp(A_WARM + B_WARM * panel.tau + C_WARM * panel.tau ** 2))
BASELINE_LL, *_ = run_spec(REGISTRY["S0"], p_warm_full, panel.log_f, panel.tau, DT)
print(f"baseline ll (recomputed from saved fit): {BASELINE_LL:.2f}")

# Generic cold start -- deliberately NOT derived from any fitted result.
p_cold = SS_OIL.replace(kappa=1.49, sigma_meas=np.array([0.005]))
A_COLD, B_COLD, C_COLD = np.log(0.005), 0.0, 0.0


def fit_smooth_quadratic(log_f, tau, label, cold=False):
    t0 = time.time()
    m = log_f.shape[1]
    p_start = p_cold if cold else p_warm
    a0, b0, c0_ = (A_COLD, B_COLD, C_COLD) if cold else (A_WARM, B_WARM, C_WARM)
    theta0 = pack_smooth(p_start, a0, b0, c0_, quadratic=True)
    p_gauss, _, _, _, _, _ = fit_s0_gaussian_smooth(log_f, tau, DT, theta0=theta0, quadratic=True)
    abc_g = fit_smooth_to_vector(p_gauss.sigma_meas[0], tau[0], quadratic=True)
    p_hat, _, _, _, n_iters = fit_s0_student_t_smooth(
        log_f, tau, DT, STUDENT_T_DF,
        theta0=pack_smooth(p_gauss, abc_g.a, abc_g.b, abc_g.c, quadratic=True),
        quadratic=True, n_outer=200)
    ll, *_ = run_spec(REGISTRY["S0"], p_hat, log_f, tau, DT)
    abc = fit_smooth_to_vector(p_hat.sigma_meas[0], tau[0], quadratic=True)
    s_bp = (p_hat.sigma_meas[0] * 1e4).round(4)
    pinned = [j + 1 for j in range(m) if abs(s_bp[j] - FLOOR_BP) < 0.05]
    dt = time.time() - t0
    tag = " [COLD]" if cold else ""
    print(f"[{time.strftime('%H:%M:%S')}] {label:26s}{tag:7s} m={m:2d}  "
         f"kappa={p_hat.kappa:8.4f}  half-life={p_hat.half_life_days:7.1f}d  "
         f"ll={ll:12.2f}  pinned={pinned if pinned else 'none':10}  "
         f"n_iters={n_iters:3d}  ({dt:.1f}s)")
    return {"label": label, "cold": cold, "m": m, "kappa": p_hat.kappa,
           "half_life": p_hat.half_life_days, "ll": ll,
           "curve": {"a": abc.a, "b": abc.b, "c": abc.c},
           "pinned": pinned, "n_iters": n_iters, "seconds": dt}


results = []
t_start = time.time()

# %% [markdown]
# ## Leave-one-contract-out (18 fits)

# %%
print(f"\n{'=' * 90}\nLEAVE-ONE-OUT (18 fits)\n{'=' * 90}")
for drop in range(18):
    cols = [j for j in range(18) if j != drop]
    r = fit_smooth_quadratic(panel.log_f[:, cols], panel.tau[:, cols],
                             f"drop C{drop + 1}")
    r["cols"] = cols
    results.append(r)

# %% [markdown]
# ## Block-outs (3 fits)

# %%
print(f"\n{'=' * 90}\nBLOCK-OUTS (3 fits)\n{'=' * 90}")
blocks = {
    "drop C1-C4": [j for j in range(18) if j not in range(0, 4)],
    "drop C5-C12": [j for j in range(18) if j not in range(4, 12)],
    "drop C13-C18": [j for j in range(18) if j not in range(12, 18)],
}
for label, cols in blocks.items():
    r = fit_smooth_quadratic(panel.log_f[:, cols], panel.tau[:, cols], label)
    r["cols"] = cols
    results.append(r)

# %% [markdown]
# ## Sub-period splits (2 fits, full C1-C18)

# %%
print(f"\n{'=' * 90}\nSUB-PERIOD SPLITS (2 fits)\n{'=' * 90}")
panel_early = panel.slice("2010-01-01", "2016-12-31")
panel_late = panel.slice("2017-01-01", "2022-11-21")
print(f"early: {panel_early.n} days ({panel_early.dates.min().date()} .. "
     f"{panel_early.dates.max().date()})   "
     f"late: {panel_late.n} days ({panel_late.dates.min().date()} .. "
     f"{panel_late.dates.max().date()})")
for label, sub in [("C1-C18, 2010-2016", panel_early), ("C1-C18, 2017-2022", panel_late)]:
    r = fit_smooth_quadratic(sub.log_f, sub.tau, label)
    r["cols"] = list(range(18))
    results.append(r)

# %% [markdown]
# ## Cold-start basin-trapping check (3 fits: baseline, one block-out, one LOO)

# %%
print(f"\n{'=' * 90}\nCOLD-START SPOT CHECK (3 fits)\n{'=' * 90}")
cold_configs = [
    ("C1-C18 baseline", list(range(18)), panel.log_f, panel.tau),
    ("drop C5-C12", blocks["drop C5-C12"], panel.log_f[:, blocks["drop C5-C12"]],
    panel.tau[:, blocks["drop C5-C12"]]),
    ("drop C1", [j for j in range(18) if j != 0],
    panel.log_f[:, [j for j in range(18) if j != 0]],
    panel.tau[:, [j for j in range(18) if j != 0]]),
]
for label, cols, log_f_sub, tau_sub in cold_configs:
    r = fit_smooth_quadratic(log_f_sub, tau_sub, label, cold=True)
    r["cols"] = cols
    results.append(r)

t_total = time.time() - t_start
print(f"\ntotal battery wall-clock: {t_total:.1f}s ({t_total / 60:.1f} min)")

# %% [markdown]
# ## Summary

# %%
print(f"\n{'=' * 90}\nBATTERY SUMMARY\n{'=' * 90}")
loo = [r for r in results if r["label"].startswith("drop C") and r["m"] == 17 and not r["cold"]]
block = [r for r in results if r["label"].startswith("drop C") and r["m"] != 17 and not r["cold"]]
subperiod = [r for r in results if "201" in r["label"] and not r["cold"]]
cold = [r for r in results if r["cold"]]

kappas_loo = [r["kappa"] for r in loo]
print(f"baseline (C1-C18, full):        kappa={s0['kappa']:.4f}")
print(f"leave-one-out spread (18 fits): {max(kappas_loo) - min(kappas_loo):.4f}  "
     f"range [{min(kappas_loo):.4f}, {max(kappas_loo):.4f}]")
print(f"block-out kappas: " + "  ".join(f"{r['label']}={r['kappa']:.4f}" for r in block))
print(f"sub-period kappas: " + "  ".join(f"{r['label']}={r['kappa']:.4f}" for r in subperiod))

print(f"\nlargest kappa movers under leave-one-out (top 5 by |delta| from baseline):")
deltas = sorted(loo, key=lambda r: abs(r["kappa"] - s0["kappa"]), reverse=True)[:5]
for r in deltas:
    print(f"  {r['label']:14s} kappa={r['kappa']:8.4f}  "
         f"delta={r['kappa'] - s0['kappa']:+8.4f}")

print(f"\ncold vs. warm basin-trapping check:")
for c in cold:
    warm_match = next((r for r in results if r["label"] == c["label"] and not r["cold"]), None)
    if warm_match is None and c["label"] == "C1-C18 baseline":
        warm_kappa, warm_ll = s0["kappa"], BASELINE_LL
    elif warm_match is not None:
        warm_kappa, warm_ll = warm_match["kappa"], warm_match["ll"]
    else:
        warm_kappa, warm_ll = float("nan"), float("nan")
    print(f"  {c['label']:20s}  cold: kappa={c['kappa']:8.4f} ll={c['ll']:12.2f}   "
         f"warm: kappa={warm_kappa:8.4f} ll={warm_ll:12.2f}   "
         f"delta_kappa={c['kappa'] - warm_kappa:+.4f}  delta_ll={c['ll'] - warm_ll:+.3f}")

all_pinned = [r for r in results if r["pinned"]]
print(f"\npinned contracts across all {len(results)} fits: "
     f"{'NONE -- gate holds throughout' if not all_pinned else all_pinned}")

# %%
out_path = ROOT / "results" / "c1c18_perturbation_battery.json"
out_path.write_text(json.dumps({"results": results, "baseline_kappa": s0["kappa"],
                               "total_seconds": t_total}, indent=2, default=str))
print(f"\nsaved to {out_path}")
