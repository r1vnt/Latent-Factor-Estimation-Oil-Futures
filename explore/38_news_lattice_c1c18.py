# %% [markdown]
# # Study B, Phase 2: news lattice (S0/SD/SM/SDM) on the three-factor C1-C18 model
#
# Features: sent_mean + 4 theme shares (supply_disruption, opec_policy,
# geopolitics, demand) -- p=5, matching the pre-registered spec (sentiment
# mean plus the four theme shares, 2 persistence classes -> B is 2x5=10
# parameters). All 5 standardised (zero mean, unit variance over the
# panel-matched, non-missing rows) so B/G loadings are comparable across
# features and not conflated with each feature's own raw scale.
#
# S0 through this path is fit FIRST as a sanity check against the
# existing price-only three-factor fit (results/c1c18_threefactor.json) --
# if it doesn't land close, the news-lattice plumbing has a bug and
# nothing past this point should be trusted.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import STUDENT_T_DF, pack3_smooth  # noqa: E402
from csr.pricing.fit3_news import (  # noqa: E402
    fit3_spec_gaussian,
    fit3_spec_student_t,
    pack3_news,
    unpack3_news,
)
from csr.pricing.smoothmeas import fit_smooth_to_vector, sigma_from_tau  # noqa: E402
from csr.pricing.specs import NewsData, REGISTRY  # noqa: E402
from csr.pricing.specs3 import run_spec3  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
QUADRATIC = True
FEATURE_COLS = ["sent_mean", "theme_supply_disruption", "theme_opec_policy",
                "theme_geopolitics", "theme_demand"]

# %% [markdown]
# ## Panel + news, aligned

# %%
raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
print(f"panel: {panel.n} dates x {panel.m} contracts, "
     f"{panel.dates.min().date()} .. {panel.dates.max().date()}")

news_raw = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet")
news_raw["date"] = pd.to_datetime(news_raw["date"])
news_raw = news_raw.set_index("date")
aligned = news_raw.reindex(panel.dates)
print(f"news coverage on panel dates: {aligned[FEATURE_COLS[0]].notna().sum()}/{panel.n} days")

feats_raw = aligned[FEATURE_COLS].to_numpy(dtype=float)
feat_mean = np.nanmean(feats_raw, axis=0)
feat_std = np.nanstd(feats_raw, axis=0)
feats_std = (feats_raw - feat_mean) / feat_std
p_feat = len(FEATURE_COLS)
print(f"feature means (raw): {feat_mean.round(4).tolist()}")
print(f"feature stds  (raw): {feat_std.round(4).tolist()}")

news = NewsData(
    features=feats_std,
    counts=aligned["n_t"].to_numpy(dtype=float),
    dispersion=aligned["disp"].to_numpy(dtype=float),
    intensity=None,
)

# %% [markdown]
# ## S0 sanity check against the existing price-only three-factor fit

# %%
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
print(f"n_base (structural + curve) = {n_base}")

p_S0, smooth_S0, _, _, _, x_filt_S0, resid_S0, weight_S0, n_iters_S0, ll_S0 = \
    fit3_spec_student_t(REGISTRY["S0"], panel.log_f, panel.tau, DT, STUDENT_T_DF,
                        pack3_news(theta_base0, REGISTRY["S0"]), news=None,
                        quadratic=QUADRATIC, n_base=n_base, p_features=p_feat)
print(f"S0 through specs3: kappa1={p_S0.kappa1:.4f}  kappa2={p_S0.kappa2:.4f}  "
     f"ll={ll_S0:.2f}")
print(f"price-only three-factor (results/c1c18_threefactor.json): "
     f"kappa1={three_factor['kappa1']:.4f}  kappa2={three_factor['kappa2']:.4f}  "
     f"ll={three_factor['ll']:.2f}")
print(f"delta ll: {ll_S0 - three_factor['ll']:.4f}  "
     f"(should be ~0 -- this is the pipeline sanity check)")

# %%
out = {"n_base": n_base, "p_feat": p_feat,
      "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
      "S0_ll": ll_S0, "S0_kappa1": p_S0.kappa1, "S0_kappa2": p_S0.kappa2,
      "news_coverage_days": int(aligned[FEATURE_COLS[0]].notna().sum())}
(ROOT / "results" / "c1c18_news_S0_sanity.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved to results/c1c18_news_S0_sanity.json")
