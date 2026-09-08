# %% [markdown]
# # Sanity check the +32,334 nat gain, and test for a missing fast class
#
# Two independent checks against the C1-C18 baselines (two-factor:
# results/c1c18_smooth_quadratic.json; three-factor: explore/27's
# results/c1c18_threefactor.json), both re-evaluated fresh on the full
# panel rather than trusted from earlier printouts.
#
# 1. Residual SD: 0.56 nats/obs implies a Gaussian variance ratio of
#    e^1.12 ~ 3.07, i.e. residual SD down ~43%. Verify directly rather
#    than trusting the implication -- if the actual residual scale didn't
#    fall by roughly that much, the ll gain is coming from somewhere
#    other than better pricing (state near-collapse, degenerate variance,
#    etc.) and needs explaining before anything is built on it.
#
# 2. Ljung-Box on the three-factor model's standardised innovations, per
#    contract, short lags. Standardised by sqrt(Hdiag) -- the curve's own
#    measurement-variance, not the full one-step prediction variance
#    Z P_pred Z' + H, which filter_uv's return signature doesn't expose
#    per-cell (only P_pred at the start of each day, before that day's m
#    sequential univariate updates). A practical, clearly-flagged
#    approximation, not the textbook-exact standardisation -- adequate for
#    "is there short-lag structure left" as a diagnostic, not a formal test.
#    White -> two classes suffice. Autocorrelated at short lags -> a fast
#    component (e.g. weekly inventory surprises) has no state to load onto
#    and is leaking into measurement error.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF, weighted_filter3  # noqa: E402
from csr.pricing.smoothmeas import hdiag_from_tau, sigma_from_tau, SmoothMeasParams  # noqa: E402
from csr.pricing.specs import REGISTRY, run_spec  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402
from csr.pricing.twofactor import TwoFactorParams  # noqa: E402
from csr.simulate import SS_OIL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts")

# %% [markdown]
# ## 1. Residual SD check

# %%
two_factor = json.loads((ROOT / "results" / "c1c18_smooth_quadratic.json").read_text())
s2, c2 = two_factor["structural"], two_factor["curve"]
p2 = TwoFactorParams(
    kappa=s2["kappa"], sigma_chi=s2["sigma_chi"], sigma_xi=s2["sigma_xi"],
    rho=s2["rho"], mu_xi=s2["mu_xi"], mu_xi_star=s2["mu_xi_star"],
    lambda_chi=s2["lambda_chi"],
    sigma_meas=np.exp(c2["a"] + c2["b"] * panel.tau + c2["c"] * panel.tau ** 2),
)
ll2, x_filt2, P_filt2, resid2, n_obs2 = run_spec(REGISTRY["S0"], p2, panel.log_f, panel.tau, DT)

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
ll3, x_filt3, resid3, n_obs3 = weighted_filter3(p3, smooth3, panel.log_f, panel.tau, DT,
                                                np.ones_like(panel.log_f))

print(f"\n{'=' * 90}\nRESIDUAL SD CHECK\n{'=' * 90}")
print(f"ll2={ll2:.2f} (recomputed)  ll3={ll3:.2f} (recomputed)  "
     f"gain={ll3 - ll2:.2f} nats over {np.isfinite(resid2).sum()} obs")

sd2 = np.nanstd(resid2)
sd3 = np.nanstd(resid3)
rmse2 = np.sqrt(np.nanmean(resid2 ** 2))
rmse3 = np.sqrt(np.nanmean(resid3 ** 2))
print(f"\nraw residual SD:   two-factor={sd2:.6f}  three-factor={sd3:.6f}  "
     f"ratio={sd3 / sd2:.4f}  (SD reduction: {100 * (1 - sd3 / sd2):.1f}%)")
print(f"raw residual RMSE: two-factor={rmse2:.6f}  three-factor={rmse3:.6f}  "
     f"ratio={rmse3 / rmse2:.4f}  (RMSE reduction: {100 * (1 - rmse3 / rmse2):.1f}%)")

n_valid = np.isfinite(resid2).sum()
predicted_ratio = np.exp((ll2 - ll3) / n_valid)  # sigma3/sigma2 implied by the ll gain
print(f"\nimplied SD ratio from ll gain (per-obs): {predicted_ratio:.4f}  "
     f"(quoted expectation: ~0.57, i.e. ~43% reduction)")
print(f"actual SD ratio (raw, unstandardised):    {sd3 / sd2:.4f}")
print(f"NOTE: raw SD mixes all 18 contracts' very different scales (front "
     f"~160bp vs deep ~4-8bp under either curve) -- a per-contract or "
     f"Hdiag-standardised comparison is the fairer read; see below.")

# Per-contract residual SD ratio, since raw pooled SD is dominated by
# whichever contracts carry the most residual variance under each model.
print(f"\nper-contract residual SD (bp), two-factor vs three-factor:")
for j in range(18):
    sd2_j = np.nanstd(resid2[:, j]) * 1e4
    sd3_j = np.nanstd(resid3[:, j]) * 1e4
    print(f"  C{j + 1:2d}: 2F={sd2_j:8.3f}bp  3F={sd3_j:8.3f}bp  "
         f"ratio={sd3_j / sd2_j:.3f}")

# %% [markdown]
# ## 2. Ljung-Box on three-factor standardised innovations, per contract

# %%
print(f"\n{'=' * 90}\nLJUNG-BOX, three-factor standardised innovations, per contract\n{'=' * 90}")
Hdiag3 = hdiag_from_tau(smooth3, panel.tau)  # (n, 18), variance
std_innov = resid3 / np.sqrt(Hdiag3)

LAGS = [1, 5, 10, 20]
lb_results = []
for j in range(18):
    z = std_innov[:, j]
    z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z, lags=LAGS, return_df=True)
    row = {"contract": j + 1, "n": len(z)}
    for lag in LAGS:
        row[f"lb_stat_{lag}"] = float(lb.loc[lag, "lb_stat"])
        row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    lb_results.append(row)
    flags = "  ".join(f"lag{lag}:p={row[f'lb_pvalue_{lag}']:.4f}"
                      f"{'*' if row[f'lb_pvalue_{lag}'] < 0.01 else ''}" for lag in LAGS)
    print(f"  C{j + 1:2d} (n={row['n']:4d}):  {flags}")

sig_at_lag1 = [r for r in lb_results if r["lb_pvalue_1"] < 0.01]
sig_at_lag5 = [r for r in lb_results if r["lb_pvalue_5"] < 0.01]
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"contracts significant (p<0.01) at lag 1:  {[r['contract'] for r in sig_at_lag1]}")
print(f"contracts significant (p<0.01) at lag 5:  {[r['contract'] for r in sig_at_lag5]}")
if len(sig_at_lag1) == 0 and len(sig_at_lag5) <= 2:
    print("WHITE (roughly) -- little short-lag structure left. Two classes "
         "look sufficient; no strong evidence of a missing fast component.")
else:
    print(f"AUTOCORRELATED at short lags in {len(sig_at_lag5)}/18 contracts -- "
         f"consistent with a fast-decaying component (e.g. weekly inventory "
         f"surprises) that has no state to load onto in the current two-class "
         f"model. Consider a third, fast rate before finalising Phase 2's "
         f"pre-registration.")

# %%
out = {
    "residual_check": {
        "ll2": ll2, "ll3": ll3, "n_obs": int(n_valid),
        "sd2": sd2, "sd3": sd3, "sd_ratio": sd3 / sd2,
        "rmse2": rmse2, "rmse3": rmse3, "rmse_ratio": rmse3 / rmse2,
        "predicted_ratio_from_ll": predicted_ratio,
    },
    "ljung_box": lb_results,
}
out_path = ROOT / "results" / "c1c18_residual_and_ljungbox.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
