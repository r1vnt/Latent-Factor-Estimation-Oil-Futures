# %% [markdown]
# # Pre-whitened GARCH check: is the squared-residual autocorrelation
# # independent of the level autocorrelation, or just an echo of it?
#
# explore/35 found squared innovations autocorrelated (ACF(1)=0.68) nearly
# as strongly as levels (0.77) -- ambiguous, because ANY level persistence
# mechanically induces some correlation in squares even with zero true
# variance clustering. Isolate the two: fit a short AR(p) to the levels
# per contract, then Ljung-Box the AR RESIDUALS' squares. If that's still
# strongly autocorrelated, there's a genuine variance-clustering (GARCH)
# effect independent of the mean persistence a fifth OU factor would
# target. If it goes roughly white, explore/35's squared-ACF finding was
# just downstream of the same still-unexplained level effect.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit4 import _raw_system4  # noqa: E402
from csr.pricing.fourfactor import FourFactorParams, initial_state4  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
AR_ORDER = 10  # covers most of the observed decay (ACF still ~0.45 at lag 10)

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
_, _, _, _, _, resid4, _, _ = filter_uv_contributions(
    np.ascontiguousarray(panel.log_f), np.ascontiguousarray(sys4["Z"]),
    np.ascontiguousarray(sys4["d"]), np.ascontiguousarray(sys4["T"]),
    np.ascontiguousarray(sys4["c"]), np.ascontiguousarray(sys4["Q"]),
    np.ascontiguousarray(sys4["Hdiag"]), np.zeros((n4, 4)),
    np.ascontiguousarray(x04), np.ascontiguousarray(P04),
)

# %% [markdown]
# ## AR(10) pre-whiten each contract's innovations, then Ljung-Box the
# ## squared AR residuals

# %%
LAGS = [1, 5, 10, 20]
print(f"{'':4s}{'AR(10) removes':>16s}{'  sq-resid Ljung-Box p-values (post-whiten)':>50s}")
results = []
for j in range(18):
    z = resid4[:, j]
    z = z[np.isfinite(z)]
    ar_fit = AutoReg(z, lags=AR_ORDER).fit()
    ar_resid = ar_fit.resid
    r2_ar = 1.0 - np.var(ar_resid) / np.var(z)

    ar_resid_sq = ar_resid ** 2
    lb = acorr_ljungbox(ar_resid_sq, lags=LAGS, return_df=True)
    row = {"contract": j + 1, "ar_r2": float(r2_ar)}
    for lag in LAGS:
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"C{j + 1:2d}  R^2={r2_ar:.4f}   {flags}")

sig1 = [r["contract"] for r in results if r["lb_pvalue_1"] < 0.01]
sig5 = [r["contract"] for r in results if r["lb_pvalue_5"] < 0.01]
mean_r2 = np.mean([r["ar_r2"] for r in results])

print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"mean AR(10) R^2 removing level persistence: {mean_r2:.4f}")
print(f"post-whitening, squared-residual sig (p<0.01) at lag 1: {sig1}")
print(f"post-whitening, squared-residual sig (p<0.01) at lag 5: {sig5}")
if len(sig1) >= 12:
    print("GENUINE VOLATILITY CLUSTERING -- squared AR residuals are STILL "
         "strongly autocorrelated after removing the level persistence. "
         "This is independent of the mean-dynamics question: a variance "
         "process (GARCH, or the existing gamma/intensity channel driven by "
         "something with short-lag persistence) is a real, separate finding.")
elif len(sig1) <= 3:
    print("NOT independent volatility clustering -- once level persistence is "
         "removed, the squares go roughly white. explore/35's squared-ACF "
         "finding was downstream of the same still-unexplained level effect, "
         "not a separate variance-clustering phenomenon.")
else:
    print("MIXED -- some contracts retain squared-residual autocorrelation "
         "post-whitening, others don't. Not a clean story either way; report "
         "as a partial/contract-specific finding rather than a general one.")

# %%
out = {"ar_order": AR_ORDER, "mean_ar_r2": mean_r2, "per_contract": results}
(ROOT / "results" / "c1c18_prewhitened_garch_check.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_prewhitened_garch_check.json")
