# %% [markdown]
# # Cell B vs. the random walk: the same spread/butterfly test, on the free-T fit
#
# Identical machinery to explore/58 (S3), applied to cell B's fitted
# (T, c, Q, curve, nu) instead -- same frozen Z/d, same honest shared-
# x_pred one-step forecast (no same-day leakage), same random-walk
# benchmark, same three block-bootstrapped tests (sign test, absolute-
# loss DM, squared-loss DM reported-not-relied-on), same power/MDE
# calculation against the noise-removal-only floor.
#
# The random walk is itself a corner of this model class -- T=I, Z=I,
# H=0. If cell B (free T, still the affine cross-section) can't beat it
# either, that's the finding stated exhaustively rather than inferred:
# the most flexible transition this class can offer, still anchored to
# a genuine no-arbitrage cross-section, still loses to its own most
# restricted member.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams, hdiag_from_tau  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.pricing.varT import VarTState, frozen_Z_d, weighted_filter_varT  # noqa: E402
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
p_frozen = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
Z, d = frozen_Z_d(p_frozen, tau, DT)
x0, P0 = initial_state3(log_f, tau, p_frozen)

cellB = json.loads((ROOT / "results" / "c1c18_cellB_varT.json").read_text())["student_t"]
state = VarTState(T=np.array(cellB["T"]), c=np.array(cellB["c"]), Q=np.array(cellB["Q"]))
smooth = SmoothMeasParams(a=cellB["curve"]["a"], b=cellB["curve"]["b"], c=cellB["curve"]["c"], quadratic=True)
print(f"cell B: ll={cellB['ll']:.2f}  nu_hat={cellB['nu_hat']:.3f}")
print(f"eigenvalue moduli of T: {cellB['eigenvalues_moduli']}")

ll, xf, resid, n_obs = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
print(f"cross-check: ll recomputed={ll:.2f}  n_obs={n_obs}")

# %% [markdown]
# ## Honest one-step forecast, shared x_pred (need P_pred/x_pred, so call filter_uv_contributions directly)

# %%
Hdiag = hdiag_from_tau(smooth, tau)
Bu = np.zeros((n, 3))
c_tiled = np.tile(state.c, (n, 1))
Q_tiled = np.tile(state.Q[None, :, :], (n, 1, 1))
ll_c, xf_c, Pf_c, xp, Pp_c, resid_c, ll_cell, n_obs_c = filter_uv_contributions(
    np.ascontiguousarray(log_f), np.ascontiguousarray(Z), np.ascontiguousarray(d),
    np.ascontiguousarray(state.T), np.ascontiguousarray(c_tiled), np.ascontiguousarray(Q_tiled),
    np.ascontiguousarray(Hdiag), np.ascontiguousarray(Bu), np.ascontiguousarray(x0), np.ascontiguousarray(P0))
print(f"cross-check (contributions): ll={ll_c:.2f}")

pred = np.einsum('tia,ta->ti', Z, xp) + d
e_honest = log_f - pred
e_honest[0, :] = np.nan

rw_error = np.full_like(log_f, np.nan)
rw_error[1:] = np.diff(log_f, axis=0)

mean_H = np.nanmean(Hdiag, axis=0)


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

    d_sq = filt_err ** 2 - rw_err ** 2
    se_sq = _block_bootstrap_se(d_sq, BLOCK_LENGTH, N_BOOT, seed)
    z_sq = -d_sq.mean() / se_sq if se_sq else float("nan")
    p_sq = float(2.0 * (1.0 - norm.cdf(abs(z_sq)))) if se_sq else float("nan")

    ef, erw = np.abs(filt_err), np.abs(rw_err)
    d_abs = ef - erw
    se_abs = _block_bootstrap_se(d_abs, BLOCK_LENGTH, N_BOOT, seed + 1)
    gap_abs = -float(d_abs.mean())
    z_abs = gap_abs / se_abs if se_abs else float("nan")
    p_abs = float(2.0 * (1.0 - norm.cdf(abs(z_abs)))) if se_abs else float("nan")

    win = (ef < erw).astype(float)
    se_win = _block_bootstrap_se(win, BLOCK_LENGTH, N_BOOT, seed + 2)
    rate = float(win.mean())
    z_win = (rate - 0.5) / se_win if se_win else float("nan")
    p_win = float(2.0 * (1.0 - norm.cdf(abs(z_win)))) if se_win else float("nan")

    c_shape = float(np.mean(erw)) / np.sqrt(mse_rw)
    sd_rw = np.sqrt(mse_rw)
    sd_filter_theory = np.sqrt(max(mse_rw - h_combo, 0.0))
    predicted_abs = c_shape * (sd_rw - sd_filter_theory)
    mde_abs = MDE_FACTOR * se_abs if se_abs else float("nan")
    power_abs = float(norm.cdf(predicted_abs / se_abs - Z_ALPHA2)) if se_abs else float("nan")

    def _flag(gap, pval, tag):
        if pval >= 0.01:
            return "no sig. difference"
        return f"FILTER WINS ({tag})" if gap > 0 else f"RW WINS ({tag})"

    return {
        "label": label, "n": nobs, "mse_rw": mse_rw, "mse_filter": mse_f, "H_combo": h_combo,
        "H_combo_frac_mse_rw": h_combo / mse_rw,
        "squared": {"gap": mse_rw - mse_f, "se": se_sq, "z": z_sq, "p": p_sq, "flag": _flag(mse_rw - mse_f, p_sq, "sq")},
        "absolute": {"gap": gap_abs, "se": se_abs, "z": z_abs, "p": p_abs, "predicted_effect": predicted_abs,
                    "mde": mde_abs, "power_at_predicted": power_abs, "flag": _flag(gap_abs, p_abs, "abs")},
        "sign": {"win_rate": rate, "se": se_win, "z": z_win, "p": p_win, "flag": _flag(rate - 0.5, p_win, "sign")},
    }


# %% [markdown]
# ## Spreads and butterflies, same battery as explore/58

# %%
print(f"\n{'=' * 130}\nCELL B: FRONT SPREADS vs. random walk\n{'=' * 130}")
hdr = (f"{'':10s}{'n':>6s}{'mse_rw':>10s}{'H_combo':>9s}{'H/mse':>8s}   "
      f"{'sq: z':>8s}{'p':>9s}   {'abs: z':>8s}{'p':>9s}   {'win_rate':>9s}{'z':>8s}{'p':>9s}  {'flag'}")
print(hdr)
spread_results = []
for j in range(17):
    filt_err = e_honest[:, j] - e_honest[:, j + 1]
    rw_err = rw_error[:, j] - rw_error[:, j + 1]
    h_combo = float(mean_H[j] + mean_H[j + 1])
    row = _run_battery(f"C{j + 1}-C{j + 2}", filt_err, rw_err, h_combo, seed=SEED + 3000 + j)
    spread_results.append(row)
    print(f"{row['label']:10s}{row['n']:6d}{row['mse_rw'] * 1e6:10.4f}{h_combo * 1e6:9.4f}"
         f"{row['H_combo_frac_mse_rw'] * 100:7.3f}%   "
         f"{row['squared']['z']:8.3f}{row['squared']['p']:9.4g}   "
         f"{row['absolute']['z']:8.3f}{row['absolute']['p']:9.4g}   "
         f"{row['sign']['win_rate']:9.4f}{row['sign']['z']:8.3f}{row['sign']['p']:9.4g}  {row['sign']['flag']}")

n_fw = sum(1 for r in spread_results if "FILTER WINS" in r["sign"]["flag"])
n_rw = sum(1 for r in spread_results if "RW WINS" in r["sign"]["flag"])
print(f"\nspreads: sign test filter wins {n_fw}/17, rw wins {n_rw}/17, neither {17 - n_fw - n_rw}/17")

print(f"\n{'=' * 130}\nCELL B: BUTTERFLIES vs. random walk\n{'=' * 130}")
print(hdr)
fly_results = []
for j in range(16):
    filt_err = e_honest[:, j] - 2 * e_honest[:, j + 1] + e_honest[:, j + 2]
    rw_err = rw_error[:, j] - 2 * rw_error[:, j + 1] + rw_error[:, j + 2]
    h_combo = float(mean_H[j] + 4 * mean_H[j + 1] + mean_H[j + 2])
    row = _run_battery(f"C{j + 1}-2C{j + 2}+C{j + 3}", filt_err, rw_err, h_combo, seed=SEED + 4000 + j)
    fly_results.append(row)
    print(f"{row['label']:10s}{row['n']:6d}{row['mse_rw'] * 1e6:10.4f}{h_combo * 1e6:9.4f}"
         f"{row['H_combo_frac_mse_rw'] * 100:7.3f}%   "
         f"{row['squared']['z']:8.3f}{row['squared']['p']:9.4g}   "
         f"{row['absolute']['z']:8.3f}{row['absolute']['p']:9.4g}   "
         f"{row['sign']['win_rate']:9.4f}{row['sign']['z']:8.3f}{row['sign']['p']:9.4g}  {row['sign']['flag']}")

n_fw_fly = sum(1 for r in fly_results if "FILTER WINS" in r["sign"]["flag"])
n_rw_fly = sum(1 for r in fly_results if "RW WINS" in r["sign"]["flag"])
print(f"\nbutterflies: sign test filter wins {n_fw_fly}/16, rw wins {n_rw_fly}/16, neither {16 - n_fw_fly - n_rw_fly}/16")

# %% [markdown]
# ## Decision rule

# %%
print(f"\n{'=' * 90}\nDECISION RULE\n{'=' * 90}")
total_fw, total_rw, total_n = n_fw + n_fw_fly, n_rw + n_rw_fly, 33
if total_fw > 0 and total_rw == 0:
    verdict = "CELL B BEATS THE RANDOM WALK -- the transition restriction was the binding one."
elif total_rw > 0 and total_fw == 0:
    verdict = ("CELL B ALSO LOSES -- the free transition (still anchored to a genuine "
              "no-arbitrage cross-section) loses to its own most restricted corner (the "
              "random walk: T=I, Z=I, H=0). Nothing in this class helps on this data, "
              "established exhaustively.")
else:
    verdict = f"MIXED: filter wins {total_fw}/{total_n}, rw wins {total_rw}/{total_n} -- no clean sweep either way."
print(verdict)

# %%
out = {"cellB_fit": {"ll": cellB["ll"], "nu_hat": cellB["nu_hat"],
                    "eigenvalue_moduli": cellB["eigenvalues_moduli"]},
      "spreads": spread_results, "butterflies": fly_results, "verdict": verdict}
out_path = ROOT / "results" / "c1c18_cellB_spread_butterfly_test.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
