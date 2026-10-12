"""Fourteen results figures for article5.tex, single-column IEEEtran width
(COLWIDTH ~3.45in, 300dpi), built from the actual committed results/*.json
files -- no numbers re-typed from the paper text. Two panels need a fresh
(fast, numba-enabled) computation rather than a stored array:
  * Filtered factors: results/*.json never stores the full filtered state
    time series, so it's recomputed from the committed S3 parameters.
  * H2' scatter: results/c1c18_h2_reconstruction.json stores only summary
    correlations; the underlying (surprise, RV) pairs are recomputed by
    rerunning explore/64's own deseasonalisation pipeline verbatim.

Run with the project's venv Python (not bare `python`).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\bradl\Documents\GitHub\Latent-Factor-Estimation-Oil-Futures")
sys.path.insert(0, str(ROOT / "src"))
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"

COLWIDTH = 3.45

plt.rcParams.update({
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


C = [f"C{i}" for i in range(1, 19)]

# =======================================================================
# 1. Measurement-error profile, free versus smooth (sec:firstcontact / sec:smoothcurve)
# =======================================================================
free = load("c1c8_floor_released_student_t.json")["released_gaussian_bp"]
smooth = load("c1c8_smooth_vs_free.json")["smooth"]["s_bp_day0"]
x = np.arange(1, 9)
fig, ax = plt.subplots(figsize=(COLWIDTH, 2.1))
w = 0.38
ax.bar(x - w / 2, free, width=w, label="Free $s_i$ (degenerate)", color="#b23b3b")
ax.bar(x + w / 2, smooth, width=w, label="Smooth $s(\\tau)$", color="#2c6fa8")
ax.set_yscale("log")
ax.set_xticks(x, [f"C{i}" for i in x])
ax.set_ylabel("bp (log scale)", fontsize=7)
ax.tick_params(labelsize=6.5)
ax.set_title("Measurement-error profile: free vs. smooth (C1\u2013C8)", fontsize=7.5, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper right")
ax.grid(alpha=0.25, which="both", linewidth=0.4)
save(fig, "fig_measurement_profile.png")

# =======================================================================
# 2. Kappa stability battery (Table battery / sec:kappaslide)
# =======================================================================
battery = load("c1c18_perturbation_battery.json")
seen = {}
for r in battery["results"]:
    seen.setdefault(r["label"], r["kappa"])  # keep first occurrence
baseline = battery["baseline_kappa"]

loo = [(f"C{i}", seen[f"drop C{i}"]) for i in range(1, 19)]
block = [("C1\u2013C4", seen["drop C1-C4"]), ("C5\u2013C12", seen["drop C5-C12"]),
        ("C13\u2013C18", seen["drop C13-C18"])]
sub = [("2010\u201316", seen["C1-C18, 2010-2016"]), ("2017\u201322", seen["C1-C18, 2017-2022"])]

fig, ax = plt.subplots(figsize=(COLWIDTH, 2.3))
xs = list(range(18)) + [20, 21, 22] + [24, 25]
labels = [l for l, _ in loo] + [l for l, _ in block] + [l for l, _ in sub]
vals = [v for _, v in loo] + [v for _, v in block] + [v for _, v in sub]
colors = ["#6a8fb5"] * 18 + ["#b23b3b"] * 3 + ["#5a9e6f"] * 2
ax.bar(xs, vals, color=colors, width=0.8)
ax.axhline(baseline, color="#333333", linewidth=0.8, linestyle="--", label=f"full panel: {baseline:.3f}")
ax.set_xticks(xs, labels, rotation=90, fontsize=5.3)
ax.set_ylabel("$\\hat\\kappa$ (two-factor)", fontsize=7)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_title("$\\kappa$ across 26 panel perturbations", fontsize=7.5, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.grid(alpha=0.25, axis="y", linewidth=0.4)
save(fig, "fig_kappa_battery.png")

# =======================================================================
# 3. Filtered factors with backwardation shading -- fresh filter run
# =======================================================================
from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv  # noqa: E402

DT = 1.0 / 252.0
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau, dates = panel.log_f, panel.tau, panel.dates

s3 = load("c1c18_threefactor.json")
p3 = ThreeFactorParams(
    kappa1=s3["kappa1"], kappa2=s3["kappa2"], sigma1=s3["sigma1"], sigma2=s3["sigma2"],
    sigma_xi=s3["sigma_xi"], rho12=0.0, rho1x=s3["rho1x"], rho2x=s3["rho2x"],
    mu_xi=s3["mu_xi"], mu_xi_star=s3["mu_xi_star"], lambda1=s3["lambda1"], lambda2=s3["lambda2"],
    gamma_v=0.0,
)
smooth3 = SmoothMeasParams(a=s3["curve"]["a"], b=s3["curve"]["b"], c=s3["curve"]["c"], quadratic=True)
sys3 = _raw_system3(p3, smooth3, tau, DT)
x0, P0 = initial_state3(log_f, tau, p3)
ll, x_filt, P_filt, x_pred, P_pred, resid, n_obs = filter_uv(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]), np.ascontiguousarray(sys3["d"]),
    np.ascontiguousarray(sys3["T"]), np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((panel.n, 3)),
    np.ascontiguousarray(x0), np.ascontiguousarray(P0))
chi1_filt = x_filt[:, 0]
backwardation = np.isfinite(log_f[:, 0]) & np.isfinite(log_f[:, 1]) & (log_f[:, 0] > log_f[:, 1])

fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
in_back = False
start = None
for i, b in enumerate(backwardation):
    if b and not in_back:
        start, in_back = dates[i], True
    elif not b and in_back:
        ax.axvspan(start, dates[i], color="#f4c7c3", alpha=0.6, linewidth=0)
        in_back = False
if in_back:
    ax.axvspan(start, dates[-1], color="#f4c7c3", alpha=0.6, linewidth=0)
ax.plot(dates, chi1_filt, color="#1a1a1a", linewidth=0.5)
ax.axhline(0, color="#888888", linewidth=0.5)
ax.set_ylabel("$\\hat\\chi_{1,t}$ (filtered)", fontsize=7)
ax.tick_params(labelsize=6.5)
ax.set_title("Filtered fast factor; shaded = backwardation (C1>C2)", fontsize=7.3, loc="left")
save(fig, "fig_filtered_factors_backwardation.png")

# =======================================================================
# 4. Held-out RMSE, C13-C18 (sec:threefit)
# =======================================================================
ho = load("c1c18_heldout_contract_prediction.json")["rmse_bp"]
x = np.arange(13, 19)
fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
w = 0.38
ax.bar(x - w / 2, ho["per_contract_2F"], width=w, label="Two-factor", color="#b23b3b")
ax.bar(x + w / 2, ho["per_contract_3F"], width=w, label="Three-factor", color="#2c6fa8")
ax.set_xticks(x, [f"C{i}" for i in x])
ax.set_ylabel("RMSE (bp)", fontsize=7)
ax.tick_params(labelsize=6.5)
overall_gain = 100 * (1 - ho["three_factor"] / ho["two_factor"])
ax.set_title(f"Held-out prediction, C13\u2013C18 ({overall_gain:.1f}% overall gain)",
            fontsize=7.3, loc="left")
ax.legend(fontsize=6.5, frameon=False)
ax.grid(alpha=0.25, axis="y", linewidth=0.4)
save(fig, "fig_heldout_rmse.png")

# =======================================================================
# 5. The order-reversal artefact (sec:denoisetest retraction)
# =======================================================================
orr = load("c1c18_order_reversal_and_honest_forecast.json")["order_reversal"]
fwd = np.array(orr["raw_v2_fwd"])
rev = np.array(orr["raw_v2_rev_same_contract"])
x = np.arange(1, 19)
fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
ax.plot(x, fwd, "o-", color="#b23b3b", markersize=3, linewidth=1, label="Processed first (forward order)")
ax.plot(x, rev, "s-", color="#2c6fa8", markersize=3, linewidth=1, label="Processed last (reversed order)")
ax.set_yscale("log")
ax.set_xticks(x, C, fontsize=5.5, rotation=90)
ax.set_ylabel("Implied squared-error\nratio vs. RW (log)", fontsize=6.5)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_title("Same statistic, two processing orders", fontsize=7.5, loc="left")
ax.legend(fontsize=5.8, frameon=False, loc="upper right")
ax.grid(alpha=0.25, which="both", linewidth=0.4)
save(fig, "fig_order_reversal.png")

# =======================================================================
# 6. Win rate by contract and the variance-shrink sweep (Table denoisetest, sec:denoisetest)
# =======================================================================
sign_test = load("c1c18_denoising_robust_tests.json")["sign_test"]
sweep = load("c1c18_h_sensitivity_and_rank_correlation.json")["h_sensitivity_sweep_c1"]

fig, axes = plt.subplots(2, 1, figsize=(COLWIDTH, 3.5))
wr = [r["win_rate"] for r in sign_test]
axes[0].bar(range(1, 19), wr, color=["#b23b3b" if w < 0.5 else "#2c6fa8" for w in wr])
axes[0].axhline(0.5, color="#333333", linewidth=0.6, linestyle="--")
axes[0].set_xticks(range(1, 19), C, fontsize=5.3, rotation=90)
axes[0].set_ylabel("Filter win rate", fontsize=6.8)
axes[0].tick_params(axis="y", labelsize=6.5)
axes[0].set_title("Win rate by contract (RW wins below 0.5)", fontsize=7.3, loc="left")
axes[0].grid(alpha=0.25, axis="y", linewidth=0.4)

shrink = [r["shrink"] for r in sweep]
wr_sweep = [r["win_rate"] for r in sweep]
axes[1].plot(shrink, wr_sweep, "o-", color="#1a1a1a", markersize=3, linewidth=1)
axes[1].axhline(0.5, color="#888888", linewidth=0.6, linestyle="--")
axes[1].set_xscale("log")
axes[1].invert_xaxis()
axes[1].set_xlabel("Assumed C1 variance, fraction of fitted $H$ (log)", fontsize=6.8)
axes[1].set_ylabel("C1 win rate", fontsize=6.8)
axes[1].tick_params(labelsize=6.5)
axes[1].set_title("Causal sweep: shrinking C1's measurement variance", fontsize=7.3, loc="left")
axes[1].grid(alpha=0.25, linewidth=0.4)
save(fig, "fig_winrate_and_sweep.png")

# =======================================================================
# 7. Spread and butterfly win rates
# =======================================================================
sb = load("c1c18_spread_butterfly_denoising_tests.json")
spreads = sb["spreads"]
butterflies = sb["butterflies"]
fig, axes = plt.subplots(2, 1, figsize=(COLWIDTH, 3.3))
wr_s = [r["sign"]["win_rate"] for r in spreads]
labels_s = [r["label"] for r in spreads]
axes[0].bar(range(len(wr_s)), wr_s, color="#b23b3b")
axes[0].axhline(0.5, color="#333333", linewidth=0.6, linestyle="--")
axes[0].set_xticks(range(len(wr_s)), labels_s, fontsize=4.6, rotation=90)
axes[0].set_ylabel("Win rate", fontsize=6.8)
axes[0].tick_params(axis="y", labelsize=6.5)
axes[0].set_title("Adjacent spreads (17)", fontsize=7.3, loc="left")
axes[0].grid(alpha=0.25, axis="y", linewidth=0.4)

wr_b = [r["sign"]["win_rate"] for r in butterflies]
labels_b = [r["label"] for r in butterflies]
axes[1].bar(range(len(wr_b)), wr_b, color="#2c6fa8")
axes[1].axhline(0.5, color="#333333", linewidth=0.6, linestyle="--")
axes[1].set_xticks(range(len(wr_b)), labels_b, fontsize=4.2, rotation=90)
axes[1].set_ylabel("Win rate", fontsize=6.8)
axes[1].tick_params(axis="y", labelsize=6.5)
axes[1].set_title("Butterflies (16)", fontsize=7.3, loc="left")
axes[1].grid(alpha=0.25, axis="y", linewidth=0.4)
save(fig, "fig_spread_butterfly_winrates.png")

# =======================================================================
# 8. H3a gain profile (Table h3a)
# =======================================================================
h3a = load("c1c18_h3a_leaveoneout.json")["per_contract"]
gains = [r["gain_pct"] for r in h3a]
fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
ax.bar(range(1, 19), gains, color=["#b23b3b" if g < 0 else "#2c6fa8" for g in gains])
ax.axhline(0, color="#333333", linewidth=0.7)
ax.set_xticks(range(1, 19), C, fontsize=5.3, rotation=90)
ax.set_ylabel("Filter gain over NS (%)", fontsize=6.8)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_title("H3a: leave-one-out gain vs. Nelson--Siegel", fontsize=7.3, loc="left")
ax.grid(alpha=0.25, axis="y", linewidth=0.4)
save(fig, "fig_h3a_gain_profile.png")

# =======================================================================
# 9. Rung-two gain profile versus the placebo (sec:rungs, gates 1-2)
# =======================================================================
diag = load("c1c18_fourfactor_front_diagnosis.json")["per_contract"]
gate2 = load("c1c18_fourfactor_front_gate2_placebo.json")
ll_gain = [r["ll_gain"] for r in diag]

fig, axes = plt.subplots(1, 2, figsize=(COLWIDTH, 1.9), gridspec_kw={"width_ratios": [3, 1]})
axes[0].bar(range(1, 19), ll_gain, color="#5a9e6f")
axes[0].set_xticks(range(1, 19), C, fontsize=5.0, rotation=90)
axes[0].set_ylabel("ll gain (nats)", fontsize=6.8)
axes[0].tick_params(axis="y", labelsize=6.5)
axes[0].set_title("Front-loaded: per-contract gain", fontsize=6.8, loc="left")
axes[0].grid(alpha=0.25, axis="y", linewidth=0.4)

axes[1].bar(["front-\nloaded", "back-\nloaded\n(placebo)"],
           [gate2["front_gain"], gate2["placebo_gain"]], color=["#5a9e6f", "#999999"])
axes[1].set_ylabel("Total gain (nats)", fontsize=6.5)
axes[1].tick_params(labelsize=6)
axes[1].set_title("Totals", fontsize=6.8)
save(fig, "fig_rungtwo_gain_vs_placebo.png")

# =======================================================================
# 10. Squared-innovation ACF and kurtosis (sec:varchannels)
# =======================================================================
garch = load("c1c18_garch_check.json")
acf_sq = garch["mean_acf_squared"]
kurt = load("c1c18_frontspread_channel.json")["acceptance_3"]

fig, axes = plt.subplots(1, 2, figsize=(COLWIDTH, 1.9), gridspec_kw={"width_ratios": [2, 1]})
axes[0].bar(range(len(acf_sq)), acf_sq, color="#333333")
axes[0].set_xlabel("Lag", fontsize=6.8)
axes[0].set_ylabel("Mean ACF, $\\hat z_t^2$", fontsize=6.8)
axes[0].tick_params(labelsize=6.3)
axes[0].set_title("Squared-innovation ACF\n(pooled, all contracts)", fontsize=6.8, loc="left")
axes[0].grid(alpha=0.25, axis="y", linewidth=0.4)

axes[1].bar(["before", "after"], [kurt["excess_kurt_before"], kurt["excess_kurt_after"]],
           color=["#b23b3b", "#2c6fa8"])
axes[1].set_ylabel("Excess kurtosis", fontsize=6.8)
axes[1].tick_params(labelsize=6.3)
axes[1].set_title("Front-spread\nchannel", fontsize=6.8)
save(fig, "fig_squared_innov_acf_kurtosis.png")

# =======================================================================
# 11. H2' scatter -- rerun explore/64's deseasonalisation pipeline
# =======================================================================
import statsmodels.api as sm  # noqa: E402
from csr.curve.expiry import nearby_expiries  # noqa: E402

RV_WINDOW, SURPRISE_WINDOW = 20, 22
news = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet").set_index("date")
news_aligned = news.reindex(dates)
I_t = news_aligned["I_t"].to_numpy(dtype=float)
log_I = np.log1p(I_t)
valid_log = np.isfinite(log_I)
dow = dates.dayofweek.to_numpy()
dow_dummies = pd.get_dummies(dow, prefix="dow", drop_first=True).to_numpy(dtype=float)
X_season = sm.add_constant(dow_dummies, has_constant="add")
res_season = sm.OLS(log_I[valid_log], X_season[valid_log]).fit()
deseason = np.full(panel.n, np.nan)
deseason[valid_log] = res_season.resid

surprise = np.full(panel.n, np.nan)
for t in range(SURPRISE_WINDOW, panel.n):
    window = deseason[t - SURPRISE_WINDOW:t]
    if np.all(np.isfinite(window)):
        surprise[t] = deseason[t] - np.median(window)

ret = np.full(panel.n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_today = np.full(panel.n, np.nan)
for t in range(RV_WINDOW, panel.n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_today[t] = np.std(window, ddof=1)

is_wed = dow == 2
expiries = nearby_expiries(dates, 1, "CL")[:, 0]
is_expiry = np.array([pd.Timestamp(dates[t]).normalize() == pd.Timestamp(expiries[t]).normalize()
                      for t in range(panel.n)])
scheduled = is_wed | is_expiry
valid = np.isfinite(surprise) & np.isfinite(rv_today)

fig, ax = plt.subplots(figsize=(COLWIDTH, 2.3))
ax.scatter(surprise[valid & ~scheduled], rv_today[valid & ~scheduled], s=3, alpha=0.35,
          color="#999999", label="non-scheduled")
ax.scatter(surprise[valid & scheduled], rv_today[valid & scheduled], s=5, alpha=0.7,
          color="#b23b3b", label="scheduled release")
ax.set_xlabel("Deseasonalised surprise intensity", fontsize=7)
ax.set_ylabel("Contemporaneous RV", fontsize=7)
ax.tick_params(labelsize=6.3)
ax.set_title("H2$'$: surprise intensity vs. RV", fontsize=7.5, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper right")
ax.grid(alpha=0.25, linewidth=0.4)
save(fig, "fig_h2prime_scatter.png")

# =======================================================================
# 12. Study A disagreement runs (sec:studyA)
# =======================================================================
gate = load("c1c18_study_a3_full.json")["identifying_gate"]
horizons = sorted(gate.keys(), key=int)
n_runs = [gate[h]["n_independent_runs"] for h in horizons]
required = gate[horizons[0]]["required_n_reference"]

fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
ax.bar(horizons, n_runs, color="#b23b3b")
ax.axhline(required, color="#333333", linewidth=0.8, linestyle="--",
          label=f"required: {required:.1f}")
ax.set_xlabel("Horizon (trading days)", fontsize=7)
ax.set_ylabel("Independent\ndisagreement runs", fontsize=6.8)
ax.tick_params(labelsize=6.5)
ax.set_title("Study A: identifying sample vs. requirement", fontsize=7.3, loc="left")
ax.legend(fontsize=6.5, frameon=False)
ax.grid(alpha=0.25, axis="y", linewidth=0.4)
save(fig, "fig_studyA_disagreement_runs.png")

# =======================================================================
# 13. Placebo LR histogram (sec:newsresult)
# =======================================================================
placebo = load("c1c18_news_placebo.json")
lrs = placebo["placebo_LRs"]
real_lr = placebo["real_LR_joint_em"]

fig, ax = plt.subplots(figsize=(COLWIDTH, 2.0))
ax.hist(lrs, bins=30, color="#999999", edgecolor="white", linewidth=0.3)
ax.axvline(real_lr, color="#b23b3b", linewidth=1.3, label=f"observed LR = {real_lr:.1f}")
ax.set_xlabel("Block-shuffled placebo LR", fontsize=7)
ax.set_ylabel("Count (of 1,000 draws)", fontsize=6.8)
ax.tick_params(labelsize=6.3)
ax.set_title(f"Placebo distribution, $p={placebo['empirical_p']:.3f}$", fontsize=7.3, loc="left")
ax.legend(fontsize=6.3, frameon=False)
save(fig, "fig_placebo_lr_histogram.png")

# =======================================================================
# 14. News-loading heatmap across block-outs (sec:newsresult)
# =======================================================================
sm_full = load("c1c18_news_SM.json")
feat = sm_full["feature_cols"]
blocks = {
    "Full panel": sm_full["G"],
    "Drop C1\u2013C4": load("c1c18_news_SM_drop_C1_C4.json")["G"],
    "Drop C5\u2013C12": load("c1c18_news_SM_drop_C5_C12.json")["G"],
    "Drop C13\u2013C18": load("c1c18_news_SM_drop_C13_C18.json")["G"],
}
rows = []
row_labels = []
for feat_i, fname in enumerate(feat):
    for cls, cname in [(0, "fast"), (1, "slow")]:
        row_labels.append(f"{fname} ({cname})")
        rows.append([blocks[b][feat_i][cls] for b in blocks])
mat = np.array(rows)

fig, ax = plt.subplots(figsize=(COLWIDTH, 3.3))
vmax = np.abs(mat).max()
im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(blocks)), list(blocks.keys()), fontsize=5.8, rotation=25, ha="right")
ax.set_yticks(range(len(row_labels)), row_labels, fontsize=5.8)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=4.8,
               color="white" if abs(mat[i, j]) > vmax * 0.5 else "black")
ax.set_title("$\\mathbf{G}$ loadings across maturity block-outs", fontsize=7.3, loc="left")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=5.5)
save(fig, "fig_news_loading_heatmap.png")

print("\nAll 14 figures written to", FIGS)
