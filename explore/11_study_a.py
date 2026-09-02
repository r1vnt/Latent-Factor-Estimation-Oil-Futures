# %% [markdown]
# # Study A: state recovery and predictive validity (Gate 4)
#
# Price-only -- does not touch the news pipeline. Fits S0 on the real EIA
# panel (Student-t observation noise, Sec. XI-F -- not the Gaussian fit,
# which sits at a boundary and whose kappa is unstable across sample
# windows), then runs the full Sec. 5 reporting battery from
# csr.study.recovery: both required horizons (5, 22 trading days) x three
# targets, with/without April 2020, the primary target split by realised-
# vol tercile, and the continuous-form robustness check.
#
# Filtered states are never smoothed here -- x_filt comes straight out of
# fit_s0_student_t, which returns weighted_filter's raw filtered pass, not
# anything routed through state.fast.smooth_rts. Using the smoother would
# manufacture exactly the look-ahead this study exists to rule out.
#
# Parameters are estimated on the full sample and the regression run on the
# same data -- in-sample, noted per Sec. 5, not hidden. If Gate 4 passes,
# re-run with expanding-window re-estimation before this goes in the paper.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel                                        # noqa: E402
from csr.pricing.fit import STUDENT_T_DF, fit_s0_gaussian, fit_s0_student_t, pack  # noqa: E402
from csr.simulate import SS_OIL                                           # noqa: E402
from csr.study.recovery import print_report, run_study_a                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

# %% [markdown]
# ## Load the panel -- basis="business" to match DT's business-day clock

# %%
prices = pd.read_parquet(ROOT / "data" / "eia_cl_futures.parquet")
panel = Panel.from_prices(prices, date_col="date", contract_col="contract",
                          price_col="settle", basis="business", source="eia",
                          verbose=False)
m = panel.m
print(f"panel: {panel.n} trading days x {m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

# %% [markdown]
# ## Fit S0: Gaussian first (initialises the Student-t EM loop, matching
# ## how this was validated in explore/02_fit_s0.py), then Student-t

# %%
print("\nfitting Gaussian S0 (initialisation for the Student-t fit)...")
p_gauss, hess_inv, xf_gauss, resid_gauss, n_obs_gauss, ll_gauss = fit_s0_gaussian(
    panel.log_f, panel.tau, DT, m, theta0=pack(SS_OIL))
print(f"  Gaussian: kappa={p_gauss.kappa:.4f}  half-life={p_gauss.half_life_days:.1f}d  "
     f"ll={ll_gauss:.1f}  n_obs={n_obs_gauss}")

print(f"\nfitting Student-t S0 (df={STUDENT_T_DF:.0f}, EM/scale-mixture)...")
p_hat, x_filt, resid_t, weight_t, n_iters = fit_s0_student_t(
    panel.log_f, panel.tau, DT, m, STUDENT_T_DF, theta0=pack(p_gauss))
print(f"  Student-t: kappa={p_hat.kappa:.4f}  half-life={p_hat.half_life_days:.1f}d  "
     f"({n_iters} EM iterations)")
print(f"  sigma_chi={p_hat.sigma_chi:.4f}  sigma_xi={p_hat.sigma_xi:.4f}  rho={p_hat.rho:+.4f}")

covid = (panel.dates >= "2020-04-01") & (panel.dates <= "2020-04-30")
covid_weight = weight_t[covid]
covid_weight = covid_weight[np.isfinite(covid_weight)]
print(f"  April 2020 EM weights: mean={covid_weight.mean():.3f}  "
     f"min={covid_weight.min():.3f}  (vs overall mean {np.nanmean(weight_t):.3f})")

# %% [markdown]
# ## Run Study A

# %%
result = run_study_a(p_hat, x_filt, panel.log_f, panel.tau, panel.dates)
print_report(result)

# %% [markdown]
# ## Sec. 5: this run is in-sample (parameters fit on the same data the
# ## regression runs on) -- noted here explicitly, not buried.

# %%
print("\nNOTE: parameters were estimated on the full sample and the "
     "regression above runs on that same sample -- in-sample, per Sec. 5. "
     "If Gate 4 passes, re-run with expanding-window re-estimation "
     "(the backtest layer) before this goes in the paper.")
