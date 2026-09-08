# %% [markdown]
# # SM block-out robustness: does G survive the same maturity block-outs
# # that made kappa1 slide?
#
# Usage: python explore/41_news_block_out_battery.py BLOCK
# where BLOCK in {C1-C4, C5-C12, C13-C18} (the contracts DROPPED).
#
# Warm start: that block-out's own (kappa1, kappa2, curve) from the
# price-only three-factor battery (results/c1c18_threefactor_battery.json)
# -- the block-out-specific values that already accommodate that panel's
# own maturity range -- combined with the full-panel SM fit's other
# structural params (sigma1/sigma2/sigma_xi/rho1x/rho2x/mu_xi/mu_xi_star/
# lambda1/lambda2), since those weren't saved per block-out. The Gaussian
# pre-fit stage refines from there regardless, same as every other fit
# this session.

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
)
from csr.pricing.specs import NewsData, REGISTRY  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

BLOCK = sys.argv[1] if len(sys.argv) > 1 else "C1-C4"
ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
QUADRATIC = True
FEATURE_COLS = ["sent_mean", "theme_supply_disruption", "theme_opec_policy",
                "theme_geopolitics", "theme_demand"]
p_feat = len(FEATURE_COLS)

BLOCK_COLS = {
    "C1-C4": [j for j in range(18) if j not in range(0, 4)],
    "C5-C12": [j for j in range(18) if j not in range(4, 12)],
    "C13-C18": [j for j in range(18) if j not in range(12, 18)],
}
cols = BLOCK_COLS[BLOCK]

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f_sub = panel.log_f[:, cols]
tau_sub = panel.tau[:, cols]

news_raw = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet")
news_raw["date"] = pd.to_datetime(news_raw["date"])
news_raw = news_raw.set_index("date")
aligned = news_raw.reindex(panel.dates)
feats_raw = aligned[FEATURE_COLS].to_numpy(dtype=float)
news = NewsData(features=feats_raw, counts=aligned["n_t"].to_numpy(dtype=float),
               dispersion=aligned["disp"].to_numpy(dtype=float), intensity=None)

# %% [markdown]
# ## Warm start: block-out kappa1/kappa2/curve + full-panel's other structural params

# %%
battery = json.loads((ROOT / "results" / "c1c18_threefactor_battery.json").read_text())
block_fit = next(r for r in battery["results"]
                 if r["label"] == f"drop {BLOCK}" and not r["cold"])
full_sm = json.loads((ROOT / "results" / "c1c18_news_SM.json").read_text())
full_3f = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())

p_init = ThreeFactorParams(
    kappa1=block_fit["kappa1"], kappa2=block_fit["kappa2"],
    sigma1=full_3f["sigma1"], sigma2=full_3f["sigma2"], sigma_xi=full_3f["sigma_xi"],
    rho12=0.0, rho1x=full_3f["rho1x"], rho2x=full_3f["rho2x"],
    mu_xi=full_3f["mu_xi"], mu_xi_star=full_3f["mu_xi_star"],
    lambda1=full_3f["lambda1"], lambda2=full_3f["lambda2"], gamma_v=0.0,
)
bc = block_fit["curve"]
theta_base0 = pack3_smooth(p_init, bc["a"], bc["b"], bc["c"], quadratic=QUADRATIC)
n_base = len(theta_base0)

spec = REGISTRY["SM"]
theta0 = pack3_news(theta_base0, spec, G=0.01 * np.ones((p_feat, 2)), a_n=np.zeros(p_feat))

print(f"[{time.strftime('%H:%M:%S')}] fitting SM, drop {BLOCK}  m={len(cols)}  "
     f"warm k1={block_fit['kappa1']:.4f}  k2={block_fit['kappa2']:.4f}")
t0 = time.time()
p_g, smooth_g, B_g, G_g, a_n_g, _, _, _, ll_g = fit3_spec_gaussian(
    spec, log_f_sub, tau_sub, DT, theta0, news=news, quadratic=QUADRATIC,
    n_base=n_base, p_features=p_feat)
t1 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Gaussian pre-fit done ({t1 - t0:.1f}s)  "
     f"ll={ll_g:.2f}  k1={p_g.kappa1:.4f}  k2={p_g.kappa2:.4f}")

theta0_em = pack3_news(pack3_smooth(p_g, smooth_g.a, smooth_g.b, smooth_g.c,
                                    quadratic=QUADRATIC), spec, G=G_g, a_n=a_n_g)
p_hat, smooth_hat, B_hat, G_hat, a_n_hat, x_filt, resid, weight, n_iters, ll = \
    fit3_spec_student_t(spec, log_f_sub, tau_sub, DT, STUDENT_T_DF, theta0_em,
                        news=news, quadratic=QUADRATIC, n_base=n_base, p_features=p_feat)
dt = time.time() - t0
converged = n_iters < 200

print(f"[{time.strftime('%H:%M:%S')}] SM drop {BLOCK} done ({dt:.1f}s, "
     f"{n_iters}/200 EM iters, {'CONVERGED' if converged else 'HIT CAP'})")
print(f"kappa1={p_hat.kappa1:.4f}  kappa2={p_hat.kappa2:.4f}  ll={ll:.2f}")
print(f"G:\n{G_hat.round(5)}")
print(f"a_n: {a_n_hat.round(5)}")

print(f"\nfull-panel SM G (for comparison):\n{np.array(full_sm['G']).round(5)}")
g_diff = G_hat - np.array(full_sm["G"])
print(f"G difference (this block-out minus full panel):\n{g_diff.round(5)}")
print(f"max |G difference|: {np.max(np.abs(g_diff)):.5f}  "
     f"vs. full G's own max |value|: {np.max(np.abs(np.array(full_sm['G']))):.5f}")

out = {
    "block_dropped": BLOCK, "m": len(cols),
    "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2, "ll": ll,
    "n_em_iters": n_iters, "converged": converged, "seconds": dt,
    "G": G_hat.tolist(), "a_n": a_n_hat.tolist(),
    "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
}
out_path = ROOT / "results" / f"c1c18_news_SM_drop_{BLOCK.replace('-', '_')}.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
