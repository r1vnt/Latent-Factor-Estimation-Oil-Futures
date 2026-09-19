# %% [markdown]
# # Absolute-loss DM and the sign test, block-bootstrapped -- the test built
# # for the mechanism actually documented (helps on typical days, fails on
# # extremes at the front, where the residual is persistent structure, not
# # noise). Squared loss is dominated by exactly those extreme days; a sign
# # test on daily win rate is immune to them by construction.
#
# Decision rule, pre-stated before running:
#   - Sign test significant, filter winning, at C1     -> real third leg,
#     the mechanism confirmed by the test designed for it.
#   - Sign test significant, filter losing, at C1       -> also a result,
#     a cleaner negative than anything so far.
#   - Null under both losses, MDE still above the predicted effect -> the
#     existing wording stands verbatim: consistent with, does not
#     strongly confirm.
#
# Both tests use the same honest, shared-x_pred one-step error e_honest
# from explore/54/55 (no same-day leakage), against the random walk
# error, block-bootstrapped (not HAC/OLS) throughout -- reusing
# csr.study.recovery's _block_length_ok/_circular_block_indices, the
# guard against the live block-length-near-n bug caught earlier this
# session. block_length=20, matching the lag already established as the
# diagnostic horizon for squared-residual autocorrelation (LAGS=[1,5,10,
# 20] throughout) -- gives ~145-160 blocks per contract, far above the
# MIN_BOOTSTRAP_BLOCKS=4 floor.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv_contributions  # noqa: E402
from csr.study.recovery import (  # noqa: E402
    MIN_BOOTSTRAP_BLOCKS,
    _block_length_ok,
    _circular_block_indices,
)

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
BLOCK_LENGTH = 20
N_BOOT = 2000
ALPHA = 0.01
POWER_TARGET = 0.80
Z_ALPHA2 = float(norm.ppf(1.0 - ALPHA / 2.0))
Z_POWER = float(norm.ppf(POWER_TARGET))
MDE_FACTOR = Z_ALPHA2 + Z_POWER
SEED = 0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                          c=before["curve"]["c"], quadratic=True)

sys3 = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)
ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)), np.ascontiguousarray(x0), np.ascontiguousarray(P0))

Z_arr, d_arr = sys3["Z"], sys3["d"]
pred = np.einsum('tia,ta->ti', Z_arr, xp) + d_arr
e_honest = log_f - pred
e_honest[0, :] = np.nan

rw_error = np.full_like(log_f, np.nan)
rw_error[1:] = np.diff(log_f, axis=0)

mean_H = np.nanmean(sys3["Hdiag"], axis=0)


def _block_bootstrap_se(series, block_length, n_boot, seed):
    """Bootstrap SE of the sample mean under serial dependence -- resample
    blocks WITH replacement from the (uncentered) series, take the std of
    the resampled means across replicates. Standard moving/circular block
    bootstrap SE estimation (Politis-Romano); reuses the exact block
    machinery already validated against the live near-n bug."""
    series = np.asarray(series, float)
    nn = len(series)
    if not _block_length_ok(nn, block_length):
        return None
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_indices(rng, nn, block_length)
        boot_means[b] = series[idx].mean()
    return float(np.std(boot_means, ddof=1))


# %% [markdown]
# ## Step 1: absolute-error loss, block-bootstrapped DM

# %%
print(f"{'=' * 118}\nSTEP 1: ABSOLUTE-ERROR LOSS, BLOCK-BOOTSTRAPPED (block_length={BLOCK_LENGTH}, n_boot={N_BOOT})\n{'=' * 118}")
abs_results = []
hdr = (f"{'':4s}{'n':>6s}{'mean|e_f|':>11s}{'mean|e_rw|':>11s}{'gap_abs':>10s}"
      f"{'SE_boot':>10s}{'z':>8s}{'p':>10s}{'flag':>14s}")
print(hdr)
for j in range(18):
    ef = np.abs(e_honest[:, j])
    erw = np.abs(rw_error[:, j])
    valid = np.isfinite(ef) & np.isfinite(erw)
    d_abs = (ef - erw)[valid]   # negative mean = filter has smaller absolute error
    se_boot = _block_bootstrap_se(d_abs, BLOCK_LENGTH, N_BOOT, SEED + j)
    mean_d = float(d_abs.mean())
    z = mean_d / se_boot if se_boot else float("nan")
    pval = float(2.0 * (1.0 - norm.cdf(abs(z)))) if se_boot else float("nan")
    gap_abs = -mean_d   # mean|e_rw| - mean|e_f|, positive = filter wins
    flag = "FILTER WINS" if (gap_abs > 0 and pval < 0.01) else (
           "RW WINS" if (gap_abs < 0 and pval < 0.01) else "no sig. difference")
    abs_results.append({"contract": j + 1, "n": int(valid.sum()), "mean_abs_filter": float(ef[valid].mean()),
                        "mean_abs_rw": float(erw[valid].mean()), "gap_abs": gap_abs, "se_boot": se_boot,
                        "z": z, "p_value": pval, "flag": flag})
    print(f"C{j + 1:2d}{int(valid.sum()):6d}{float(ef[valid].mean()) * 1e3:11.4f}"
         f"{float(erw[valid].mean()) * 1e3:11.4f}{gap_abs * 1e3:10.4f}{se_boot * 1e3:10.4f}"
         f"{z:8.3f}{pval:10.4g}  {flag}")
print("(mean|e|/gap/SE in units of 1e-3 log-price, i.e. roughly 10bp)")

# %% [markdown]
# ## Step 2: sign test on daily win rate, block-bootstrapped

# %%
print(f"\n{'=' * 100}\nSTEP 2: SIGN TEST ON DAILY WIN RATE (H0: rate=0.5)\n{'=' * 100}")
sign_results = []
print(f"{'':4s}{'n':>6s}{'win_rate':>10s}{'SE_boot':>10s}{'z':>8s}{'p':>10s}{'flag':>16s}")
for j in range(18):
    ef = np.abs(e_honest[:, j])
    erw = np.abs(rw_error[:, j])
    valid = np.isfinite(ef) & np.isfinite(erw)
    win = (ef[valid] < erw[valid]).astype(float)
    se_boot = _block_bootstrap_se(win, BLOCK_LENGTH, N_BOOT, SEED + 100 + j)
    rate = float(win.mean())
    z = (rate - 0.5) / se_boot if se_boot else float("nan")
    pval = float(2.0 * (1.0 - norm.cdf(abs(z)))) if se_boot else float("nan")
    flag = "FILTER WINS MAJORITY" if (rate > 0.5 and pval < 0.01) else (
           "RW WINS MAJORITY" if (rate < 0.5 and pval < 0.01) else "no sig. difference")
    sign_results.append({"contract": j + 1, "n": int(valid.sum()), "win_rate": rate, "se_boot": se_boot,
                         "z": z, "p_value": pval, "flag": flag})
    print(f"C{j + 1:2d}{int(valid.sum()):6d}{rate:10.4f}{se_boot:10.4f}{z:8.3f}{pval:10.4g}  {flag}")

# %% [markdown]
# ## Step 3: MDE and power under both losses, same format as explore/55

# %%
print(f"\n{'=' * 118}\nSTEP 3: MDE/POWER UNDER BOTH LOSSES (alpha={ALPHA}, power_target={POWER_TARGET}, "
     f"MDE_factor={MDE_FACTOR:.3f})\n{'=' * 118}")
print("Absolute loss: predicted effect derived from the squared-loss theory (gap~H) via a "
     "same-SHAPE scaling assumption -- E|e| = c*SD(e) for a distribution-shape constant c, "
     "calibrated from the random walk's OWN empirical MAD/RMS ratio (no Gaussian assumption). "
     "Approximate by construction; reported as such.")
print(f"{'':4s}{'MSE(rw)':>11s}{'H':>9s}{'predicted':>11s}{'MDE_abs':>10s}{'power@pred':>11s}"
     f"{'detectable':>11s}   {'win-rate MDE (pp)':>18s}")
combined = []
for j in range(18):
    mse_rw = float(np.nanmean(rw_error[:, j] ** 2))
    h_j = float(mean_H[j])
    c_j = float(np.nanmean(np.abs(rw_error[:, j]))) / np.sqrt(mse_rw)   # empirical MAD/RMS ratio
    sd_rw = np.sqrt(mse_rw)
    sd_filter_theory = np.sqrt(max(mse_rw - h_j, 0.0))
    predicted_abs = c_j * (sd_rw - sd_filter_theory)
    se_abs = abs_results[j]["se_boot"]
    mde_abs = MDE_FACTOR * se_abs if se_abs else float("nan")
    power_at_pred = float(norm.cdf(predicted_abs / se_abs - Z_ALPHA2)) if se_abs else float("nan")
    detectable = predicted_abs > mde_abs if se_abs else False

    se_sign = sign_results[j]["se_boot"]
    mde_sign_pp = MDE_FACTOR * se_sign * 100 if se_sign else float("nan")

    combined.append({"contract": j + 1, "mse_rw": mse_rw, "fitted_H": h_j, "c_shape": c_j,
                     "predicted_abs_effect": predicted_abs, "SE_abs": se_abs, "MDE_abs": mde_abs,
                     "power_at_predicted_abs": power_at_pred, "detectable_abs": bool(detectable),
                     "MDE_win_rate_pp": mde_sign_pp})
    print(f"C{j + 1:2d}{mse_rw * 1e6:11.4f}{h_j * 1e6:9.4f}{predicted_abs * 1e3:11.4f}"
         f"{mde_abs * 1e3:10.4f}{power_at_pred:11.3f}{'YES' if detectable else 'no':>11s}"
         f"   {mde_sign_pp:18.3f}")
print("(MSE(rw)/H in 1e-6 log-price^2; predicted/MDE_abs in 1e-3 log-price; MDE win-rate in percentage points)")

# %% [markdown]
# ## Decision rule applied, at C1

# %%
c1_sign = sign_results[0]
c1_abs = abs_results[0]
c1_combined = combined[0]
print(f"\n{'=' * 100}\nDECISION RULE, APPLIED AT C1\n{'=' * 100}")
print(f"C1 sign test: win_rate={c1_sign['win_rate']:.4f}  p={c1_sign['p_value']:.4g}  -> {c1_sign['flag']}")
print(f"C1 absolute-loss DM: gap_abs={c1_abs['gap_abs'] * 1e3:.4f}(1e-3)  p={c1_abs['p_value']:.4g}  -> {c1_abs['flag']}")
print(f"C1 abs-loss power@predicted={c1_combined['power_at_predicted_abs']:.3f}  "
     f"detectable={c1_combined['detectable_abs']}")

if c1_sign["p_value"] < 0.01 and c1_sign["win_rate"] > 0.5:
    verdict = "REAL THIRD LEG -- mechanism confirmed by the test designed for it."
elif c1_sign["p_value"] < 0.01 and c1_sign["win_rate"] < 0.5:
    verdict = "CLEANER NEGATIVE than anything so far -- filter loses the majority-of-days test at C1."
else:
    verdict = ("NULL under both losses -- if MDE remains above the predicted effect, the existing "
              "wording stands verbatim: consistent with, does not strongly confirm.")
print(f"\nVERDICT: {verdict}")

# %%
out = {
    "block_length": BLOCK_LENGTH, "n_boot": N_BOOT, "alpha": ALPHA, "power_target": POWER_TARGET,
    "mde_factor": MDE_FACTOR,
    "absolute_loss_dm": abs_results, "sign_test": sign_results, "power_both_losses": combined,
    "verdict_at_c1": verdict,
}
out_path = ROOT / "results" / "c1c18_denoising_robust_tests.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
