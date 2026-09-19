# %% [markdown]
# # Rank correlation (H/MSE(rw) vs. sign-test z), and the H-sensitivity sweep at C1
#
# Two closes on the same mechanism:
#
# 1. Formalise "the filter loses exactly where H is large" as a number:
#    Spearman rank correlation between H/MSE(rw) (explore/55) and the
#    sign-test z-statistic (explore/56) across all 18 contracts. Expect
#    rho < 0 -- more claimed noise, more negative z (worse for the filter).
#
# 2. Causal check: shrink H at C1 alone (structural params, curve, and
#    every other contract's H held fixed -- no refit, just a different
#    Hdiag fed into the same filter pass) and re-run the sign test at C1
#    across a sweep. If win rate climbs back toward/past 0.5 as H falls,
#    the disadvantage is a monotone function of the noise the filter is
#    told to remove, not an unrelated C1 idiosyncrasy.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

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
Z_ALPHA2 = float(norm.ppf(1.0 - ALPHA / 2.0))
SEED = 0

# %% [markdown]
# ## 1. Rank correlation: H/MSE(rw) vs. sign-test z, across all 18 contracts

# %%
power = json.loads((ROOT / "results" / "c1c18_denoising_power_analysis.json").read_text())["results"]
robust = json.loads((ROOT / "results" / "c1c18_denoising_robust_tests.json").read_text())
sign = robust["sign_test"]

h_frac = [r["H_frac_of_MSE_rw"] for r in power]
z_sign = [r["z"] for r in sign]
assert [r["contract"] for r in power] == [r["contract"] for r in sign]

rho, p_rho = spearmanr(h_frac, z_sign)
print(f"{'=' * 90}\nRANK CORRELATION: H/MSE(rw) vs. sign-test z, n=18 contracts\n{'=' * 90}")
for j in range(18):
    print(f"  C{j + 1:2d}: H/MSE(rw)={h_frac[j] * 100:7.3f}%   sign-test z={z_sign[j]:+7.3f}")
print(f"\nSpearman rho = {rho:+.4f}   p = {p_rho:.4g}")
print("(rho < 0 with a small p: more claimed noise -> more negative z, i.e. the filter's "
     "disadvantage worsens monotonically with H/MSE(rw), not just at the two endpoints.)")

# %% [markdown]
# ## 2. H-sensitivity sweep at C1 -- structural params, curve, all other H unchanged

# %%
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
Hdiag_base = sys3["Hdiag"]
H1_baseline = float(np.nanmean(Hdiag_base[:, 0]))

rw_error_c1 = np.full(n, np.nan)
rw_error_c1[1:] = np.diff(log_f[:, 0])
mse_rw_c1 = float(np.nanmean(rw_error_c1 ** 2))


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


SHRINK_FACTORS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
print(f"\n{'=' * 110}\nH-SENSITIVITY SWEEP AT C1 (baseline H1={H1_baseline * 1e6:.4f}e-6, "
     f"MSE(rw) C1={mse_rw_c1 * 1e6:.4f}e-6)\n{'=' * 110}")
print(f"{'shrink':>8s}{'H1':>10s}{'H1/MSE(rw)':>12s}{'win_rate':>10s}{'SE_boot':>10s}"
     f"{'z':>8s}{'p':>10s}{'flag':>20s}")
sweep_results = []
for shrink in SHRINK_FACTORS:
    Hdiag_mod = Hdiag_base.copy()
    Hdiag_mod[:, 0] = Hdiag_base[:, 0] * shrink
    ll, xf, Pf, xp_mod, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
        np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
        np.ascontiguousarray(Hdiag_mod), np.zeros((sys3["n"], 3)),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0))

    pred_c1 = np.einsum('ta,ta->t', sys3["Z"][:, 0, :], xp_mod) + sys3["d"][:, 0]
    e_honest_c1 = log_f[:, 0] - pred_c1
    e_honest_c1[0] = np.nan

    ef = np.abs(e_honest_c1)
    erw = np.abs(rw_error_c1)
    valid = np.isfinite(ef) & np.isfinite(erw)
    win = (ef[valid] < erw[valid]).astype(float)
    se_boot = _block_bootstrap_se(win, BLOCK_LENGTH, N_BOOT, SEED)
    rate = float(win.mean())
    z = (rate - 0.5) / se_boot if se_boot else float("nan")
    pval = float(2.0 * (1.0 - norm.cdf(abs(z)))) if se_boot else float("nan")
    flag = "FILTER WINS MAJORITY" if (rate > 0.5 and pval < 0.01) else (
           "RW WINS MAJORITY" if (rate < 0.5 and pval < 0.01) else "no sig. difference")
    h1_shrunk = H1_baseline * shrink
    sweep_results.append({"shrink": shrink, "H1": h1_shrunk, "H1_frac_mse_rw": h1_shrunk / mse_rw_c1,
                          "win_rate": rate, "se_boot": se_boot, "z": z, "p_value": pval, "flag": flag,
                          "n": int(valid.sum())})
    print(f"{shrink:8.3f}{h1_shrunk * 1e6:10.4f}{h1_shrunk / mse_rw_c1 * 100:11.3f}%{rate:10.4f}"
         f"{se_boot:10.4f}{z:8.3f}{pval:10.4g}  {flag}")

monotone_toward_half = all(
    sweep_results[i]["win_rate"] <= sweep_results[i + 1]["win_rate"] + 1e-9
    for i in range(len(sweep_results) - 1)
)
print(f"\nwin rate rises monotonically (or ties) as H1 shrinks: {monotone_toward_half}")
crosses_half = any(r["win_rate"] > 0.5 for r in sweep_results)
print(f"win rate ever crosses 0.5: {crosses_half}")
first_not_significant = next((r for r in sweep_results if r["flag"] == "no sig. difference"), None)
if first_not_significant:
    print(f"first shrink factor where the RW-wins result stops being significant: "
         f"{first_not_significant['shrink']} (H1/MSE(rw) = {first_not_significant['H1_frac_mse_rw'] * 100:.3f}%)")

# %%
out = {
    "rank_correlation": {"spearman_rho": float(rho), "p_value": float(p_rho),
                        "h_frac_by_contract": h_frac, "z_by_contract": z_sign},
    "h_sensitivity_sweep_c1": sweep_results,
    "monotone_toward_half": bool(monotone_toward_half),
    "crosses_half": bool(crosses_half),
}
out_path = ROOT / "results" / "c1c18_h_sensitivity_and_rank_correlation.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
