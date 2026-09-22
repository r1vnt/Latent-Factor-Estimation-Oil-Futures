# %% [markdown]
# # Roll's estimator: is fitted s(tau) absorbing noise, or misspecification?
#
# s(tau) is fit to make the likelihood work, which means it absorbs
# whatever the model can't otherwise explain -- genuine microstructure
# noise AND model misspecification both land in the same bucket, and at
# the front they pull in opposite directions: C1 is the most liquid
# contract on the curve, so its TRUE measurement noise should be the
# SMALLEST, while its fitted s is the LARGEST. A single smooth curve
# can't represent both facts at once.
#
# Roll (1984): under bid-ask bounce, the effective spread is recoverable
# from the first-order autocovariance of returns, with no model and no
# filter -- s_Roll = 2*sqrt(-Cov(r_t, r_{t+1})) when that covariance is
# negative (as bid-ask bounce predicts); undefined (reported as NaN, not
# forced to zero) when it isn't, since a non-negative autocovariance
# means the bounce assumption itself doesn't hold that day/contract, not
# that the spread is exactly zero.
#
# If fitted s(C1) >> Roll-implied noise at C1, the smooth curve is
# absorbing misspecification, not (only) measurement error, at the
# front -- confirms the diagnosis and justifies moving a component into
# the state (the essentially-affine extension) rather than continuing
# to ask H to do this job alone.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams, sigma_from_tau  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
smooth = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                          c=before["curve"]["c"], quadratic=True)
s_fitted_bp = sigma_from_tau(smooth, tau[0]) * 1e4   # bp, day-0 curve value per contract

# %% [markdown]
# ## Roll's estimator, per contract

# %%
print(f"{'=' * 90}\nROLL'S ESTIMATOR vs. FITTED s(tau), PER CONTRACT\n{'=' * 90}")
print(f"{'':4s}{'cov(r_t,r_t+1)':>16s}{'Roll s (bp)':>13s}{'fitted s (bp)':>15s}{'fitted/Roll':>13s}")
rows = []
for j in range(18):
    r = np.diff(log_f[:, j])
    r = r[np.isfinite(r)]
    cov1 = float(np.cov(r[:-1], r[1:])[0, 1])
    if cov1 < 0:
        roll_s = float(2.0 * np.sqrt(-cov1)) * 1e4   # bp, full round-trip spread
        # Roll's own s is a ROUND-TRIP (bid-ask) spread; the measurement-noise
        # SD comparable to s(tau) is roughly half that (one-way deviation
        # from the efficient price), so also report the half-spread reading.
        roll_sd_bp = roll_s / 2.0
    else:
        roll_s, roll_sd_bp = float("nan"), float("nan")
    ratio = s_fitted_bp[j] / roll_sd_bp if np.isfinite(roll_sd_bp) and roll_sd_bp > 0 else float("nan")
    rows.append({"contract": j + 1, "cov1": cov1, "roll_spread_bp": roll_s,
                "roll_halfspread_bp": roll_sd_bp, "fitted_s_bp": float(s_fitted_bp[j]),
                "ratio_fitted_over_roll": ratio})
    print(f"C{j + 1:2d}{cov1 * 1e8:16.4f}{roll_s:13.3f}{s_fitted_bp[j]:15.3f}{ratio:13.2f}")
print("(cov1 column x1e8 for readability; Roll s = round-trip spread, fitted s is a one-sided SD --"
     " ratio uses Roll's implied HALF-spread as the comparable one-sided quantity)")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
c1_ratio = rows[0]["ratio_fitted_over_roll"]
if np.isfinite(c1_ratio) and c1_ratio > 3:
    print(f"fitted s(C1) is {c1_ratio:.1f}x Roll's implied noise at C1 -- the smooth curve is "
         f"absorbing far more than microstructure noise at the front. Confirms the diagnosis: "
         f"s(tau) is fitting a compromise between genuine noise and misspecification, and the "
         f"front's share of that compromise is mostly misspecification.")
elif np.isfinite(c1_ratio):
    print(f"fitted s(C1) is only {c1_ratio:.1f}x Roll's implied noise -- less separation than "
         f"expected; the curve's front-loaded value may be closer to genuine microstructure "
         f"noise than the diagnosis assumed.")
else:
    print("Roll's estimator is undefined at C1 (non-negative first-order autocovariance -- the "
         "bid-ask-bounce assumption doesn't hold there, which is itself informative).")

# %%
out_path = ROOT / "results" / "c1c18_roll_estimator_diagnostic.json"
out_path.write_text(json.dumps({"per_contract": rows}, indent=2, default=float))
print(f"\nsaved to {out_path}")
