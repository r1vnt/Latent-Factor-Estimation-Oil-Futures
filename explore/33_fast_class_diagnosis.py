# %% [markdown]
# # Characterising the missing fast component before building it
#
# Ljung-Box already established there's a dynamic component missing from
# the three-factor model (all 18 contracts, p=0.0000 through lag 20). Before
# adding a fourth state, get an informed starting rate rather than a guess:
# fit an exponential decay to the empirical ACF of the standardised
# innovations, averaged across contracts, and back out the implied kappa.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)

three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p3 = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"],
)
smooth3 = SmoothMeasParams(a=three_factor["curve"]["a"], b=three_factor["curve"]["b"],
                           c=three_factor["curve"]["c"], quadratic=True)
sys3 = _raw_system3(p3, smooth3, panel.tau, DT)
x03, P03 = initial_state3(panel.log_f, panel.tau, p3)
n3 = sys3["n"]
ll3, xf3, Pf3, xp3, Pp3, resid3, ll_cell3, n_obs3 = filter_uv_contributions(
    np.ascontiguousarray(panel.log_f), np.ascontiguousarray(sys3["Z"]),
    np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
    np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((n3, 3)),
    np.ascontiguousarray(x03), np.ascontiguousarray(P03),
)

# %% [markdown]
# ## Average ACF across contracts, fit exp(-kappa*dt*lag)

# %%
MAX_LAG = 30
acfs = np.full((18, MAX_LAG + 1), np.nan)
for j in range(18):
    z = resid3[:, j]
    z = z[np.isfinite(z)]
    acfs[j] = acf(z, nlags=MAX_LAG, fft=True)

mean_acf = np.nanmean(acfs, axis=0)
lags = np.arange(MAX_LAG + 1)
print(f"{'lag':>4s}{'mean ACF':>10s}")
for lag in range(MAX_LAG + 1):
    print(f"{lag:4d}{mean_acf[lag]:10.4f}")


def decay(lag, kappa, a0):
    return a0 * np.exp(-kappa * DT * lag)


# Fit on lags 1..MAX_LAG (skip lag 0, which is 1.0 by definition).
popt, pcov = curve_fit(decay, lags[1:], mean_acf[1:], p0=[50.0, 0.3],
                       bounds=([0.1, 0.0], [2000.0, 1.0]))
kappa_fast, a0_fast = popt
half_life_fast = np.log(2.0) / kappa_fast * 365.25

print(f"\n{'=' * 78}\nFITTED DECAY\n{'=' * 78}")
print(f"implied kappa_fast = {kappa_fast:.4f}  "
     f"(half-life = {half_life_fast:.2f} trading days = {half_life_fast * 365.25 / 252:.2f} calendar days)")
print(f"a0 (lag-0-ish amplitude) = {a0_fast:.4f}")
print(f"\ncontext: current three-factor kappa1 (fast)=1.887 (half-life 134d), "
     f"kappa2 (slow)=0.389 (half-life 651d)")

# %%
out = {"mean_acf": mean_acf.tolist(), "fitted_kappa_fast": kappa_fast,
      "fitted_half_life_days": half_life_fast, "fitted_a0": a0_fast}
(ROOT / "results" / "c1c18_fast_class_diagnosis.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_fast_class_diagnosis.json")
