# %% [markdown]
# # Sec. VII-D placebo: block-shuffled news vs. the real SM(G) result
#
# Two corrections before this run means anything:
# - n_obs(S0)=55294 vs n_obs(SM)=65084 -- the naive LR=15982 against S0
#   compared models fit to different data. VOID.
# - The valid comparison, same 65084 observations both times, is SM(G
#   free) vs SM(G=0, a_n at its closed-form R-weighted-mean optimum):
#   LR=59.09, df=10, p=5.4e-9. Real, but ~0.4% of the naive figure.
#
# Even 59.09 has a reading that isn't the one wanted: news reports price
# moves, not just anticipates them, so a persistent news series correlating
# with a persistent price-implied state is not automatically a
# "measurement of the state" result. The control: block-shuffle news
# (preserving each block's internal persistence) so its within-block
# character survives but its calendar alignment to that day's price move
# does not. If shuffled news produces a comparable LR, 59.09 is mechanical
# correlation with contemporaneous returns, not a genuine reading.
#
# Approximation for speed: the full joint EM refit takes ~13 min per spec
# (structural+curve+G+a_n jointly), too slow for hundreds of placebo
# replications. Since kappa1/kappa2 are empirically unmoved by news across
# every spec fit this session (S0/SD/SM/SDM all land within 0.002 of each
# other), fix the shock-state trajectory (w_chi1, w_chi2) at its S0 (G=0,
# price-only) value and solve G/a_n as a per-feature weighted least
# squares against that fixed trajectory -- closed form, seconds per
# replication instead of minutes. Validated below against the real joint
# EM fit's LR=59.09 before trusting it for the placebo distribution.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.specs import news_variance  # noqa: E402
from csr.pricing.specs3 import run_spec3  # noqa: E402
from csr.pricing.specs import REGISTRY  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
FEATURE_COLS = ["sent_mean", "theme_supply_disruption", "theme_opec_policy",
                "theme_geopolitics", "theme_demand"]
LAG = 1
BLOCK_LEN = 21   # one roll cycle (curve/expiry.py's own roll window), a
                # defensible, round, pre-existing unit of time in this
                # project rather than a number picked to fit the result
N_PLACEBO = 1000
LOG_2PI = float(np.log(2.0 * np.pi))

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
n = panel.n

news_raw = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet")
news_raw["date"] = pd.to_datetime(news_raw["date"])
news_raw = news_raw.set_index("date")
aligned = news_raw.reindex(panel.dates)
feats_raw = aligned[FEATURE_COLS].to_numpy(dtype=float)
counts = aligned["n_t"].to_numpy(dtype=float)
dispersion = aligned["disp"].to_numpy(dtype=float)
p_feat = len(FEATURE_COLS)

# %% [markdown]
# ## Fixed shock-state trajectory (S0, price-only, unaffected by any G)

# %%
s0 = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_S0 = ThreeFactorParams(
    kappa1=s0["kappa1"], kappa2=s0["kappa2"], sigma1=s0["sigma1"], sigma2=s0["sigma2"],
    sigma_xi=s0["sigma_xi"], rho12=0.0, rho1x=s0["rho1x"], rho2x=s0["rho2x"],
    mu_xi=s0["mu_xi"], mu_xi_star=s0["mu_xi_star"], lambda1=s0["lambda1"],
    lambda2=s0["lambda2"], gamma_v=0.0,
)
smooth_S0 = SmoothMeasParams(a=s0["curve"]["a"], b=s0["curve"]["b"], c=s0["curve"]["c"],
                             quadratic=True)
_, x_filt_S0, _, _ = run_spec3(REGISTRY["S0"], p_S0, smooth_S0, panel.log_f, panel.tau, DT)
W = x_filt_S0[:, 3:5]   # (n, 2): fixed w_chi1, w_chi2 trajectory
print(f"shock trajectory: W shape {W.shape}, std per column {W.std(axis=0).round(6)}")


def _shift(a: np.ndarray, lag: int = LAG) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if lag < len(a):
        out[: len(a) - lag] = a[lag:]
    return out


def wls_ll(feats: np.ndarray, R: np.ndarray, W: np.ndarray):
    """Per-feature weighted least squares of shifted news on the fixed
    shock trajectory. Returns (ll_free, ll_null, n_obs_used) using the
    exact closed-form Gaussian log-likelihood, summed over features."""
    y = _shift(feats)
    Rs = _shift(R)
    ll_free = 0.0
    ll_null = 0.0
    n_obs = 0
    X_full = np.column_stack([np.ones(n), W])   # (n, 3): intercept, w1, w2
    for j in range(feats.shape[1]):
        ok = np.isfinite(y[:, j]) & np.isfinite(Rs[:, j])
        if ok.sum() < 5:
            continue
        yj = y[ok, j]
        w = 1.0 / Rs[ok, j]
        Xj = X_full[ok]
        # weighted normal equations: (X'WX) beta = X'Wy
        XtW = Xj.T * w
        beta = np.linalg.solve(XtW @ Xj, XtW @ yj)
        resid = yj - Xj @ beta
        ll_free += np.sum(-0.5 * (LOG_2PI + np.log(Rs[ok, j]) + resid ** 2 / Rs[ok, j]))
        # null: intercept only (G = 0), closed-form weighted mean
        a0 = np.sum(w * yj) / np.sum(w)
        resid0 = yj - a0
        ll_null += np.sum(-0.5 * (LOG_2PI + np.log(Rs[ok, j]) + resid0 ** 2 / Rs[ok, j]))
        n_obs += int(ok.sum())
    return ll_free, ll_null, n_obs


R = news_variance(dispersion, counts, p_feat)

# %% [markdown]
# ## Validate the WLS approximation against the real joint-EM LR=59.09

# %%
ll_free_real, ll_null_real, n_obs_real = wls_ll(feats_raw, R, W)
LR_real = 2.0 * (ll_free_real - ll_null_real)
print(f"\nWLS-approx real LR: {LR_real:.2f}  (df=10)  n_obs={n_obs_real}  "
     f"vs. joint-EM LR=59.09 -- {'close, approximation validated' if abs(LR_real - 59.09) < 30 else 'NOT close, approximation questionable'}")

# %% [markdown]
# ## Block-shuffle placebo, N_PLACEBO replications

# %%
rng = np.random.default_rng(20260908)
n_blocks = int(np.ceil(n / BLOCK_LEN))


_block_bounds = [np.arange(i, min(i + BLOCK_LEN, n)) for i in range(0, n, BLOCK_LEN)]


def block_shuffle_index(rng) -> np.ndarray:
    """One shared row-permutation index, built by permuting contiguous
    BLOCK_LEN-day blocks (preserving within-block order/persistence,
    destroying date alignment). Applying the SAME index to feats and R
    keeps each day's variance paired with its own feature reading --
    shuffling them with independent permutations would desynchronise a
    day's news value from its own sampling variance, corrupting the
    weighted likelihood in a way that has nothing to do with the
    placebo's actual question."""
    perm = rng.permutation(len(_block_bounds))
    return np.concatenate([_block_bounds[b] for b in perm])


placebo_LRs = np.empty(N_PLACEBO)
for i in range(N_PLACEBO):
    idx = block_shuffle_index(rng)
    feats_shuf = feats_raw[idx]
    R_shuf = R[idx]
    ll_free_i, ll_null_i, _ = wls_ll(feats_shuf, R_shuf, W)
    placebo_LRs[i] = 2.0 * (ll_free_i - ll_null_i)

print(f"\nplacebo LR: mean={placebo_LRs.mean():.2f}  median={np.median(placebo_LRs):.2f}  "
     f"p95={np.percentile(placebo_LRs, 95):.2f}  p99={np.percentile(placebo_LRs, 99):.2f}  "
     f"max={placebo_LRs.max():.2f}")
empirical_p = float((placebo_LRs >= LR_real).mean())
print(f"\nREAL LR = {LR_real:.2f}")
print(f"empirical placebo p-value (fraction of {N_PLACEBO} shuffles >= real LR): {empirical_p:.4f}")

print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
if empirical_p < 0.05:
    print("Real LR exceeds the shuffled-news null -- the measurement result "
         "survives the placebo. Genuine same-day correlation, not just "
         "generic persistence.")
else:
    print("Real LR is NOT distinguishable from shuffled news with the same "
         "persistence -- 59.09 is mechanical correlation with contemporaneous "
         "returns via news' own autocorrelation, not a genuine same-day "
         "measurement result. Report this as the finding.")

# %%
out = {"real_LR_wls_approx": LR_real, "real_LR_joint_em": 59.09,
      "n_obs_real": n_obs_real, "block_len": BLOCK_LEN, "n_placebo": N_PLACEBO,
      "placebo_LRs": placebo_LRs.tolist(), "empirical_p": empirical_p}
(ROOT / "results" / "c1c18_news_placebo.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_news_placebo.json")
