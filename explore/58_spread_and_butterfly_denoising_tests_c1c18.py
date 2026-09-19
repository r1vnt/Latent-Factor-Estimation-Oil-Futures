# %% [markdown]
# # The spread test: same machinery, the quantity where xi's unit root cancels
#
# The level test conflated "does the filter estimate state well" with "is
# the level forecastable at all" -- front-month WTI is close to a
# martingale, so no model beats a random walk on levels regardless of
# whether the filter is well- or mis-specified. s_t = ln F_j - ln F_{j+1}
# cancels xi identically (loading exactly 1 on both legs) and leaves
# chi1/chi2, which mean-revert (kappa1=1.887, ~134-day half-life,
# estimated and stable). Butterflies (1,-2,1) cancel xi too and are
# further still from a unit root.
#
# Nothing refitted. The spread/butterfly forecast errors are EXACT linear
# combinations of the same per-contract e_honest (filter, shared x_pred,
# no same-day leakage) and rw_error (random walk) arrays used in
# explore/54-57:
#   spread error   = e[j]   - e[j+1]
#   butterfly error = e[j] - 2*e[j+1] + e[j+2]
# for both the filter (e_honest) and the random-walk benchmark (rw_error)
# -- so this is genuinely "one run on output already in hand," not a new
# filter pass.
#
# Predicted effect (for power, pre-registered before reading results):
# H_combo = the noise-removal-only prediction, Var(a-b)=Var(a)+Var(b) for
# independent per-contract measurement noise (spread: H_j+H_{j+1};
# butterfly: H_j+4*H_{j+1}+H_{j+2}). This is explicitly a CONSERVATIVE
# floor -- it omits whatever mean-reversion itself contributes, which the
# model also claims and which has no clean closed form to pre-register
# without assuming away the exact thing being tested. Finding an effect
# at or above this floor confirms noise-removal; finding nothing even
# against this lower bound is evidence against the model on the spread,
# not merely "insufficiently powerful."
#
# Same three tests as explore/56, block-bootstrapped throughout
# (block_length=20, reusing csr.study.recovery's validated machinery):
# sign test on daily win rate, absolute-loss DM, squared-loss DM
# (reported, not relied on).

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
from csr.study.recovery import _block_length_ok, _circular_block_indices  # noqa: E402

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


def _run_battery(label, filt_err, rw_err, h_combo, seed):
    valid = np.isfinite(filt_err) & np.isfinite(rw_err)
    filt_err, rw_err = filt_err[valid], rw_err[valid]
    nobs = len(filt_err)

    mse_rw = float(np.mean(rw_err ** 2))
    mse_f = float(np.mean(filt_err ** 2))

    # squared loss (reported, not relied on)
    d_sq = filt_err ** 2 - rw_err ** 2
    se_sq = _block_bootstrap_se(d_sq, BLOCK_LENGTH, N_BOOT, seed)
    z_sq = -d_sq.mean() / se_sq if se_sq else float("nan")   # gap = mse_rw - mse_f = -mean(d_sq)
    p_sq = float(2.0 * (1.0 - norm.cdf(abs(z_sq)))) if se_sq else float("nan")

    # absolute loss
    ef, erw = np.abs(filt_err), np.abs(rw_err)
    d_abs = ef - erw
    se_abs = _block_bootstrap_se(d_abs, BLOCK_LENGTH, N_BOOT, seed + 1)
    gap_abs = -float(d_abs.mean())
    z_abs = gap_abs / se_abs if se_abs else float("nan")
    p_abs = float(2.0 * (1.0 - norm.cdf(abs(z_abs)))) if se_abs else float("nan")

    # sign test
    win = (ef < erw).astype(float)
    se_win = _block_bootstrap_se(win, BLOCK_LENGTH, N_BOOT, seed + 2)
    rate = float(win.mean())
    z_win = (rate - 0.5) / se_win if se_win else float("nan")
    p_win = float(2.0 * (1.0 - norm.cdf(abs(z_win)))) if se_win else float("nan")

    # power/MDE, against the noise-removal-only floor
    c_shape = float(np.mean(erw)) / np.sqrt(mse_rw)
    sd_rw = np.sqrt(mse_rw)
    sd_filter_theory = np.sqrt(max(mse_rw - h_combo, 0.0))
    predicted_abs = c_shape * (sd_rw - sd_filter_theory)
    mde_sq = MDE_FACTOR * se_sq if se_sq else float("nan")
    mde_abs = MDE_FACTOR * se_abs if se_abs else float("nan")
    power_sq = float(norm.cdf(h_combo / se_sq - Z_ALPHA2)) if se_sq else float("nan")
    power_abs = float(norm.cdf(predicted_abs / se_abs - Z_ALPHA2)) if se_abs else float("nan")

    def _flag(gap, pval, win_label):
        if pval >= 0.01:
            return "no sig. difference"
        return f"FILTER WINS ({win_label})" if gap > 0 else f"RW WINS ({win_label})"

    row = {
        "label": label, "n": nobs, "mse_rw": mse_rw, "mse_filter": mse_f, "H_combo": h_combo,
        "H_combo_frac_mse_rw": h_combo / mse_rw,
        "squared": {"gap": mse_rw - mse_f, "se": se_sq, "z": z_sq, "p": p_sq,
                   "mde": mde_sq, "power_at_H": power_sq, "flag": _flag(mse_rw - mse_f, p_sq, "sq")},
        "absolute": {"gap": gap_abs, "se": se_abs, "z": z_abs, "p": p_abs, "predicted_effect": predicted_abs,
                    "mde": mde_abs, "power_at_predicted": power_abs, "flag": _flag(gap_abs, p_abs, "abs")},
        "sign": {"win_rate": rate, "se": se_win, "z": z_win, "p": p_win,
                "flag": _flag(rate - 0.5, p_win, "sign")},
    }
    return row


# %% [markdown]
# ## Spreads: s = ln F_j - ln F_{j+1}, j = 0..16 (17 adjacent pairs)

# %%
print(f"{'=' * 130}\nFRONT SPREADS: filter vs. random walk, one-step-ahead\n{'=' * 130}")
hdr = (f"{'':10s}{'n':>6s}{'mse_rw':>10s}{'H_combo':>9s}{'H/mse':>8s}   "
      f"{'sq: z':>8s}{'p':>9s}   {'abs: z':>8s}{'p':>9s}   {'win_rate':>9s}{'z':>8s}{'p':>9s}  {'flag'}")
print(hdr)
spread_results = []
for j in range(17):
    filt_err = e_honest[:, j] - e_honest[:, j + 1]
    rw_err = rw_error[:, j] - rw_error[:, j + 1]
    h_combo = float(mean_H[j] + mean_H[j + 1])
    row = _run_battery(f"C{j + 1}-C{j + 2}", filt_err, rw_err, h_combo, seed=SEED + 1000 + j)
    spread_results.append(row)
    print(f"{row['label']:10s}{row['n']:6d}{row['mse_rw'] * 1e6:10.4f}{h_combo * 1e6:9.4f}"
         f"{row['H_combo_frac_mse_rw'] * 100:7.3f}%   "
         f"{row['squared']['z']:8.3f}{row['squared']['p']:9.4g}   "
         f"{row['absolute']['z']:8.3f}{row['absolute']['p']:9.4g}   "
         f"{row['sign']['win_rate']:9.4f}{row['sign']['z']:8.3f}{row['sign']['p']:9.4g}  {row['sign']['flag']}")

n_filter_wins_sign = sum(1 for r in spread_results if "FILTER WINS" in r["sign"]["flag"])
n_rw_wins_sign = sum(1 for r in spread_results if "RW WINS" in r["sign"]["flag"])
print(f"\nspreads: sign test filter wins {n_filter_wins_sign}/17, rw wins {n_rw_wins_sign}/17, "
     f"neither {17 - n_filter_wins_sign - n_rw_wins_sign}/17")

# %% [markdown]
# ## Butterflies: b = ln F_j - 2 ln F_{j+1} + ln F_{j+2}, j = 0..15 (16 triples)

# %%
print(f"\n{'=' * 130}\nBUTTERFLIES: filter vs. random walk, one-step-ahead\n{'=' * 130}")
print(hdr)
fly_results = []
for j in range(16):
    filt_err = e_honest[:, j] - 2 * e_honest[:, j + 1] + e_honest[:, j + 2]
    rw_err = rw_error[:, j] - 2 * rw_error[:, j + 1] + rw_error[:, j + 2]
    h_combo = float(mean_H[j] + 4 * mean_H[j + 1] + mean_H[j + 2])
    row = _run_battery(f"C{j + 1}-2C{j + 2}+C{j + 3}", filt_err, rw_err, h_combo, seed=SEED + 2000 + j)
    fly_results.append(row)
    print(f"{row['label']:10s}{row['n']:6d}{row['mse_rw'] * 1e6:10.4f}{h_combo * 1e6:9.4f}"
         f"{row['H_combo_frac_mse_rw'] * 100:7.3f}%   "
         f"{row['squared']['z']:8.3f}{row['squared']['p']:9.4g}   "
         f"{row['absolute']['z']:8.3f}{row['absolute']['p']:9.4g}   "
         f"{row['sign']['win_rate']:9.4f}{row['sign']['z']:8.3f}{row['sign']['p']:9.4g}  {row['sign']['flag']}")

n_filter_wins_fly = sum(1 for r in fly_results if "FILTER WINS" in r["sign"]["flag"])
n_rw_wins_fly = sum(1 for r in fly_results if "RW WINS" in r["sign"]["flag"])
print(f"\nbutterflies: sign test filter wins {n_filter_wins_fly}/16, rw wins {n_rw_wins_fly}/16, "
     f"neither {16 - n_filter_wins_fly - n_rw_wins_fly}/16")

# %% [markdown]
# ## Decision rule

# %%
print(f"\n{'=' * 90}\nDECISION RULE\n{'=' * 90}")
if n_filter_wins_sign >= 1 and n_rw_wins_sign == 0:
    verdict = ("FILTER WINS on the front spreads -- the denoising claim is confirmed on the "
              "quantity the framework is actually about. The level result becomes what it "
              "should have been: WTI levels are a martingale, which nobody disputes.")
elif n_rw_wins_sign >= 1:
    verdict = ("FILTER LOSES on (some) spreads too -- decisive, since the martingale confound "
              "is gone and there was real predictable structure to find. The existing write-up "
              "stands, and stands more firmly.")
else:
    verdict = "No significant result either way on the spreads -- genuinely uninformative here, unlike the level test."
print(verdict)

# %%
out = {"spreads": spread_results, "butterflies": fly_results, "verdict": verdict,
      "block_length": BLOCK_LENGTH, "n_boot": N_BOOT, "alpha": ALPHA, "power_target": POWER_TARGET}
out_path = ROOT / "results" / "c1c18_spread_butterfly_denoising_tests.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
