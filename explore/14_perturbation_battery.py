# %% [markdown]
# # Perturbation battery: is the measurement-error vector identified at all?
#
# Runbook 1.9-1.10. Plain S0 (Student-t, no pricing-error state) refit on
# every leave-one-contract-out subset of C1-C4, plus C1-C4 split into two
# non-overlapping time periods. Reports kappa and the full s vector for
# each -- the rank ordering of s across subsets is the statistic that
# matters, not any single fit's pattern. If it reorders, no pattern in any
# one fit's s vector may be interpreted, which is the finding Step 1's
# three fits already pointed at.
#
# Prediction under the depth story (1.10): kappa should be LESS stable on
# three contracts than four, not more. If kappa instead stabilises when a
# contract is dropped, that specific contract is doing something the
# "depth" story alone doesn't explain.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                        # noqa: E402
from csr.pricing.fit import STUDENT_T_DF, fit_s0_gaussian, fit_s0_student_t, pack  # noqa: E402
from csr.simulate import SS_OIL                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel = Panel.from_prices(prices, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia",
                          verbose=False)
print(f"panel: {panel.n} trading days x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")


def fit_plain_s0(log_f, tau, cols, label):
    p_init = SS_OIL.replace(sigma_meas=SS_OIL.sigma_meas[list(cols)])
    m_sub = len(cols)
    log_f_sub = log_f[:, cols]
    tau_sub = tau[:, cols]
    p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f_sub, tau_sub, DT, m_sub,
                                             theta0=pack(p_init))
    p_hat, x_filt, resid, weight, n_iters = fit_s0_student_t(
        log_f_sub, tau_sub, DT, m_sub, STUDENT_T_DF, theta0=pack(p_gauss))
    s_bp = (p_hat.sigma_meas * 1e4).round(2).tolist()
    print(f"{label:20s}  kappa={p_hat.kappa:7.4f}  half-life={p_hat.half_life_days:7.1f}d  "
         f"s(bp)={s_bp}")
    return {"label": label, "cols": cols, "kappa": p_hat.kappa,
           "half_life": p_hat.half_life_days, "s_bp": s_bp, "p": p_hat}


# %% [markdown]
# ## Leave-one-contract-out subsets (plus the full C1-C4 baseline)

# %%
results = []
results.append(fit_plain_s0(panel.log_f, panel.tau, (0, 1, 2, 3), "C1-C4 (baseline)"))
results.append(fit_plain_s0(panel.log_f, panel.tau, (0, 1, 2), "C1-C3 (drop C4)"))
results.append(fit_plain_s0(panel.log_f, panel.tau, (0, 1, 3), "C1,C2,C4 (drop C3)"))
results.append(fit_plain_s0(panel.log_f, panel.tau, (0, 2, 3), "C1,C3,C4 (drop C2)"))
results.append(fit_plain_s0(panel.log_f, panel.tau, (1, 2, 3), "C2-C4 (drop C1)"))

# %% [markdown]
# ## Time-period splits (all four contracts)

# %%
panel_early = panel.slice("2010-01-01", "2016-12-31")
panel_late = panel.slice("2017-01-01", "2024-04-05")
print(f"\nearly: {panel_early.n} days ({panel_early.dates.min().date()} "
     f".. {panel_early.dates.max().date()})")
print(f"late:  {panel_late.n} days ({panel_late.dates.min().date()} "
     f".. {panel_late.dates.max().date()})")

results.append(fit_plain_s0(panel_early.log_f, panel_early.tau, (0, 1, 2, 3),
                            "C1-C4, 2010-2016"))
results.append(fit_plain_s0(panel_late.log_f, panel_late.tau, (0, 1, 2, 3),
                            "C1-C4, 2017-2024"))

# %% [markdown]
# ## Summary: kappa spread, and where the inflated s_i lands each time

# %%
print(f"\n{'=' * 78}\nPERTURBATION BATTERY SUMMARY\n{'=' * 78}")
print(f"{'fit':22s}{'kappa':>10s}{'half-life(d)':>14s}  s vector (bp)")
for r in results:
    print(f"{r['label']:22s}{r['kappa']:10.4f}{r['half_life']:14.1f}  {r['s_bp']}")

kappas_4contract = [r["kappa"] for r in results if len(r["cols"]) == 4]
kappas_3contract = [r["kappa"] for r in results
                    if len(r["cols"]) == 3 and "20" not in r["label"]]
print(f"\nkappa spread, 4-contract fits (baseline + 2 time splits): "
     f"{max(kappas_4contract) - min(kappas_4contract):.4f}  "
     f"(values: {[round(k, 4) for k in kappas_4contract]})")
print(f"kappa spread, 3-contract leave-one-out fits: "
     f"{max(kappas_3contract) - min(kappas_3contract):.4f}  "
     f"(values: {[round(k, 4) for k in kappas_3contract]})")
print(f"\nprediction check (1.10): 3-contract spread "
     f"{'>' if max(kappas_3contract)-min(kappas_3contract) > max(kappas_4contract)-min(kappas_4contract) else '<='} "
     f"4-contract spread  "
     f"({'consistent with pure depth story' if max(kappas_3contract)-min(kappas_3contract) > max(kappas_4contract)-min(kappas_4contract) else 'depth story incomplete -- something contract-specific is happening'})")

print("\nwhich contract carries the largest s_i in each fit (rank instability check):")
for r in results:
    if len(r["cols"]) < 4:
        continue
    worst_idx = int(np.argmax(r["s_bp"]))
    print(f"  {r['label']:22s}  largest s_i: C{r['cols'][worst_idx]+1} = {r['s_bp'][worst_idx]}bp")
