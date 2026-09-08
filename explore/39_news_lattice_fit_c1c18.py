# %% [markdown]
# # Study B, Phase 2: fit one news spec (SD/SM/SDM) on C1-C18
#
# Usage: python explore/39_news_lattice_fit_c1c18.py SPEC
# where SPEC in {SD, SM, SDM}. Warm-started from explore/38's validated
# S0 fit (kappa1=1.8871, kappa2=0.3889, matching the price-only
# three-factor fit to 4 decimals); B/G/a_n warm-started at (near-)zero.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF, pack3_smooth  # noqa: E402
from csr.pricing.fit3_news import (  # noqa: E402
    fit3_spec_gaussian,
    fit3_spec_student_t,
    pack3_news,
    unpack3_news,
)
from csr.pricing.specs import NewsData, REGISTRY, lr_degrees_of_freedom  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

SPEC_NAME = sys.argv[1] if len(sys.argv) > 1 else "SD"
ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
QUADRATIC = True
FEATURE_COLS = ["sent_mean", "theme_supply_disruption", "theme_opec_policy",
                "theme_geopolitics", "theme_demand"]

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)

news_raw = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet")
news_raw["date"] = pd.to_datetime(news_raw["date"])
news_raw = news_raw.set_index("date")
aligned = news_raw.reindex(panel.dates)
feats_raw = aligned[FEATURE_COLS].to_numpy(dtype=float)
# NOT standardised: news_variance's R = dispersion^2/counts is calibrated to
# sent_mean's own raw scale (the sampling variance of a day's mean document
# sentiment). Standardising features to unit variance while R stays on the
# raw scale is a units mismatch -- confirmed directly: at G=0 exactly (which
# the SM/S0 nesting test guarantees must reproduce S0's price-side ll plus
# only the news rows' own flat contribution), standardised features gave
# ll=-239664 against S0's +261977, an -500k-nat gap entirely inside the news
# rows once traced through _neg_loglik3_news by hand. B/G loadings below are
# therefore in raw feature units; feat_std (still computed, unused in the
# fit) lets anyone convert to a per-SD loading for reporting.
p_feat = len(FEATURE_COLS)
feat_mean_report = np.nanmean(feats_raw, axis=0)
feat_std_report = np.nanstd(feats_raw, axis=0)
news = NewsData(features=feats_raw, counts=aligned["n_t"].to_numpy(dtype=float),
               dispersion=aligned["disp"].to_numpy(dtype=float), intensity=None)

three_factor = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_init = ThreeFactorParams(
    kappa1=three_factor["kappa1"], kappa2=three_factor["kappa2"],
    sigma1=three_factor["sigma1"], sigma2=three_factor["sigma2"],
    sigma_xi=three_factor["sigma_xi"], rho12=0.0,
    rho1x=three_factor["rho1x"], rho2x=three_factor["rho2x"],
    mu_xi=three_factor["mu_xi"], mu_xi_star=three_factor["mu_xi_star"],
    lambda1=three_factor["lambda1"], lambda2=three_factor["lambda2"], gamma_v=0.0,
)
c3 = three_factor["curve"]
theta_base0 = pack3_smooth(p_init, c3["a"], c3["b"], c3["c"], quadratic=QUADRATIC)
n_base = len(theta_base0)

spec = REGISTRY[SPEC_NAME]
kw = {}
if spec.driver:
    kw["B"] = np.zeros((2, p_feat))
if spec.measurement:
    kw["G"] = 0.01 * np.ones((p_feat, 2))   # small, not exactly 0 (a flat G=0
                                            # start is a stationary point of
                                            # the news rows' own gradient)
    kw["a_n"] = np.zeros(p_feat)
theta0 = pack3_news(theta_base0, spec, **kw)

print(f"[{time.strftime('%H:%M:%S')}] fitting {SPEC_NAME}  n_base={n_base}  "
     f"p_feat={p_feat}  theta_len={len(theta0)}")
t0 = time.time()
p_g, smooth_g, B_g, G_g, a_n_g, _, _, _, ll_g = fit3_spec_gaussian(
    spec, panel.log_f, panel.tau, DT, theta0, news=news, quadratic=QUADRATIC,
    n_base=n_base, p_features=p_feat)
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian pre-fit done ({t1 - t0:.1f}s)  "
     f"ll={ll_g:.2f}  kappa1={p_g.kappa1:.4f}  kappa2={p_g.kappa2:.4f}")

theta0_em_kw = {}
if spec.driver:
    theta0_em_kw["B"] = B_g
if spec.measurement:
    theta0_em_kw["G"] = G_g
    theta0_em_kw["a_n"] = a_n_g
theta0_em = pack3_news(pack3_smooth(p_g, smooth_g.a, smooth_g.b, smooth_g.c,
                                    quadratic=QUADRATIC), spec, **theta0_em_kw)

p_hat, smooth_hat, B_hat, G_hat, a_n_hat, x_filt, resid, weight, n_iters, ll = \
    fit3_spec_student_t(spec, panel.log_f, panel.tau, DT, STUDENT_T_DF, theta0_em,
                        news=news, quadratic=QUADRATIC, n_base=n_base,
                        p_features=p_feat)
dt = time.time() - t0
converged = n_iters < 200

print(f"[{time.strftime('%H:%M:%S')}] {SPEC_NAME} done ({dt:.1f}s, "
     f"{n_iters}/200 EM iters, {'CONVERGED' if converged else 'HIT CAP'})")
print(f"kappa1={p_hat.kappa1:.4f}  kappa2={p_hat.kappa2:.4f}  ll={ll:.2f}")
if B_hat is not None:
    print(f"B (rows=chi1,chi2; cols={FEATURE_COLS}):\n{B_hat.round(5)}")
if G_hat is not None:
    print(f"G (rows={FEATURE_COLS}; cols=chi1,chi2):\n{G_hat.round(5)}")
    print(f"a_n: {a_n_hat.round(5)}")

s0 = json.loads((ROOT / "results" / "c1c18_news_S0_sanity.json").read_text())
ll_S0 = s0["S0_ll"]
lr = 2.0 * (ll - ll_S0)
df = lr_degrees_of_freedom(SPEC_NAME, "S0", p_feat)
pval = float(chi2.sf(lr, df)) if lr >= 0 else float("nan")
print(f"\nLR vs S0: ll_S0={ll_S0:.2f}  ll_{SPEC_NAME}={ll:.2f}  LR={lr:.2f}  "
     f"df={df}  p={pval:.4g}" + ("  (LR<0!)" if lr < 0 else ""))

out = {
    "spec": SPEC_NAME, "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2, "ll": ll,
    "n_em_iters": n_iters, "converged": converged, "seconds": dt,
    "B": B_hat.tolist() if B_hat is not None else None,
    "G": G_hat.tolist() if G_hat is not None else None,
    "a_n": a_n_hat.tolist() if a_n_hat is not None else None,
    "lr_vs_S0": lr, "df_vs_S0": df, "p_vs_S0": pval,
    "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
    "feature_cols": FEATURE_COLS,
}
out_path = ROOT / "results" / f"c1c18_news_{SPEC_NAME}.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
