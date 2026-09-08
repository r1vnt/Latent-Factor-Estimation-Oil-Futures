# %% [markdown]
# # Is the missing dynamic in levels or in volatility?
#
# The four-factor model's fast class moved substantively (46.7d half-life,
# not stuck at its warm start) and gained +12,556 nats, but left Ljung-Box
# on the standardised innovations byte-for-byte unchanged from the three-
# factor model -- 18/18 contracts, p=0.0000 at every lag. A sum-of-OU-
# processes model can only ever add more autocorrelation in the MEAN; if
# the missing structure is actually in the VARIANCE (volatility clustering
# -- a big move today makes a big move tomorrow more likely, regardless of
# sign), no amount of additional mean-reverting factors would ever fix the
# Ljung-Box-on-levels test, and a fifth factor would predictably repeat
# explore/34's non-result. Check squared standardised innovations directly.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit4 import _raw_system4  # noqa: E402
from csr.pricing.fourfactor import FourFactorParams, initial_state4  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)

four_factor = json.loads((ROOT / "results" / "c1c18_fourfactor.json").read_text())
p4 = FourFactorParams(
    kappa1=four_factor["kappa1"], kappa2=four_factor["kappa2"], kappa3=four_factor["kappa3"],
    sigma1=four_factor["sigma1"], sigma2=four_factor["sigma2"], sigma3=four_factor["sigma3"],
    sigma_xi=four_factor["sigma_xi"],
    rho1x=four_factor["rho1x"], rho2x=four_factor["rho2x"], rho3x=four_factor["rho3x"],
    mu_xi=four_factor["mu_xi"], mu_xi_star=four_factor["mu_xi_star"],
    lambda1=four_factor["lambda1"], lambda2=four_factor["lambda2"], lambda3=four_factor["lambda3"],
)
smooth4 = SmoothMeasParams(a=four_factor["curve"]["a"], b=four_factor["curve"]["b"],
                           c=four_factor["curve"]["c"], quadratic=True)
sys4 = _raw_system4(p4, smooth4, panel.tau, DT)
x04, P04 = initial_state4(panel.log_f, panel.tau, p4)
n4 = sys4["n"]
ll4, xf4, Pf4, xp4, Pp4, resid4, ll_cell4, n_obs4 = filter_uv_contributions(
    np.ascontiguousarray(panel.log_f), np.ascontiguousarray(sys4["Z"]),
    np.ascontiguousarray(sys4["d"]), np.ascontiguousarray(sys4["T"]),
    np.ascontiguousarray(sys4["c"]), np.ascontiguousarray(sys4["Q"]),
    np.ascontiguousarray(sys4["Hdiag"]), np.zeros((n4, 4)),
    np.ascontiguousarray(x04), np.ascontiguousarray(P04),
)
print(f"ll4 recomputed = {ll4:.2f}  (saved: {four_factor['ll']:.2f})")

# %% [markdown]
# ## Ljung-Box on squared standardised innovations, per contract

# %%
print(f"\n{'=' * 90}\nLJUNG-BOX ON SQUARED INNOVATIONS (volatility clustering)\n{'=' * 90}")
LAGS = [1, 5, 10, 20]
sq_results = []
for j in range(18):
    z = resid4[:, j]
    z = z[np.isfinite(z)]
    z_sq = z ** 2
    lb = acorr_ljungbox(z_sq, lags=LAGS, return_df=True)
    row = {"contract": j + 1}
    for lag in LAGS:
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    sq_results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"  C{j + 1:2d}: {flags}")

sig1_sq = [r["contract"] for r in sq_results if r["lb_pvalue_1"] < 0.01]
sig5_sq = [r["contract"] for r in sq_results if r["lb_pvalue_5"] < 0.01]
print(f"\nsignificant (p<0.01) at lag 1: {sig1_sq}")
print(f"significant (p<0.01) at lag 5: {sig5_sq}")

# %% [markdown]
# ## Mean ACF of squared innovations, and of levels for comparison

# %%
MAX_LAG = 20
acf_levels = np.full((18, MAX_LAG + 1), np.nan)
acf_sq = np.full((18, MAX_LAG + 1), np.nan)
for j in range(18):
    z = resid4[:, j]
    z = z[np.isfinite(z)]
    acf_levels[j] = acf(z, nlags=MAX_LAG, fft=True)
    acf_sq[j] = acf(z ** 2, nlags=MAX_LAG, fft=True)

mean_acf_levels = np.nanmean(acf_levels, axis=0)
mean_acf_sq = np.nanmean(acf_sq, axis=0)
print(f"\n{'=' * 90}\nMEAN ACF: levels vs. squared, four-factor innovations\n{'=' * 90}")
print(f"{'lag':>4s}{'levels':>10s}{'squared':>10s}")
for lag in range(MAX_LAG + 1):
    print(f"{lag:4d}{mean_acf_levels[lag]:10.4f}{mean_acf_sq[lag]:10.4f}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"levels ACF at lag 1: {mean_acf_levels[1]:.4f}  (this is what 4 factors couldn't fix)")
print(f"squared ACF at lag 1: {mean_acf_sq[1]:.4f}")
if mean_acf_sq[1] > 0.05 and len(sig1_sq) >= 15:
    print("VOLATILITY CLUSTERING PRESENT -- squared innovations are strongly "
         "autocorrelated too. A GARCH-type variance process (or the existing "
         "gamma/intensity scale_variances channel, driven by something with "
         "its own short-lag persistence) is a live candidate independent of "
         "adding more mean-reverting levels factors. This does not by itself "
         "prove the LEVEL autocorrelation is caused by volatility clustering, "
         "but a level effect that is actually driven by clustering (fat-tailed, "
         "clustered shocks correlate in levels via their magnitude even with "
         "zero true mean persistence) is a well-known confound of Ljung-Box.")
else:
    print("Squared innovations look cleaner than levels -- volatility "
         "clustering is not the obvious explanation. The level autocorrelation "
         "is more likely a genuine missed mean-dynamic (still probably not "
         "a 5th OU factor, given the 4th one's complete non-effect) or a "
         "microstructure/calendar artefact (roll mechanics, day-of-week).")

# %%
out = {"ljung_box_squared": sq_results, "mean_acf_levels": mean_acf_levels.tolist(),
      "mean_acf_squared": mean_acf_sq.tolist()}
(ROOT / "results" / "c1c18_garch_check.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_garch_check.json")
