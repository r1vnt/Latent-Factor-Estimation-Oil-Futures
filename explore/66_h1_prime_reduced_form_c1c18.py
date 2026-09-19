# %% [markdown]
# # H1': news content discriminates transient from permanent shocks (reduced form)
#
# Pre-registered in preregistrations/2026-09-19_h1_h2_h3_revisions.md.
# Original H1 asked whether yesterday's public news predicts today's
# move -- a strong-form efficiency test the model was never built to
# win (p=0.99, and a positive result at that horizon would itself be
# suspicious). H1' asks the question a two-factor decomposition model
# actually speaks to: conditional on the SIZE of today's move, does news
# content carry information about how much of it survives at horizon h?
#
# No refitting, no state-space machinery: for each h in {5,22,66} and
# each news feature, regress the h-ahead cumulative front-contract return
# on today's own return, interacted with that feature (contemporaneous --
# this is about characterising today's move, not forecasting from
# yesterday's news). A positive, significant interaction means
# news-heavy/news-flavoured moves persist MORE than news-quiet moves of
# the same size; negative means they revert faster (a transient-content
# signature, e.g. supply-disruption headlines on an inventory blip).
#
# Falsifier, stated in advance: no interaction term significant at
# p<0.01 across all h x features -> the reduced form shows nothing and
# the structural version (splitting chi/xi attribution by news content)
# is not worth building.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (5, 22, 66)
FEATURE_COLS = ["sent_mean", "theme_supply_disruption", "theme_opec_policy",
                "theme_geopolitics", "theme_demand", "intensity"]

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, dates = panel.log_f, panel.dates
n = panel.n

news = pd.read_parquet(ROOT / "data" / "stage4_daily.parquet").set_index("date")
news_aligned = news.reindex(dates)

ret_today = np.full(n, np.nan)
ret_today[1:] = np.diff(log_f[:, 0])

# %% [markdown]
# ## The regression, h x feature grid

# %%
print(f"{'=' * 100}\nH1' REDUCED FORM: cumret_h ~ const + ret_t + feature_t + ret_t*feature_t\n{'=' * 100}")
print(f"{'h':>4s}{'feature':>26s}{'n':>6s}{'coef(interact)':>15s}{'p':>10s}{'R2':>8s}   flag")
rows = []
any_significant = False
for h in HORIZONS:
    cumret_h = np.full(n, np.nan)
    for t in range(n - h):
        cumret_h[t] = log_f[t + h, 0] - log_f[t, 0]

    for feat in FEATURE_COLS:
        feat_vals = news_aligned[feat].to_numpy(dtype=float)
        valid = np.isfinite(cumret_h) & np.isfinite(ret_today) & np.isfinite(feat_vals)
        if valid.sum() < 100:
            continue

        interact = ret_today[valid] * feat_vals[valid]
        X = sm.add_constant(np.column_stack([ret_today[valid], feat_vals[valid], interact]),
                            has_constant="add")
        res = sm.OLS(cumret_h[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
        coef_interact, p_interact = float(res.params[3]), float(res.pvalues[3])
        significant = p_interact < 0.01
        any_significant |= significant
        rows.append({"h": h, "feature": feat, "n": int(res.nobs), "coef_interact": coef_interact,
                    "p_interact": p_interact, "r_squared": float(res.rsquared)})
        flag = "PERSISTS MORE" if (significant and coef_interact > 0) else (
               "REVERTS FASTER" if (significant and coef_interact < 0) else "")
        print(f"{h:4d}{feat:>26s}{int(res.nobs):6d}{coef_interact:+15.5f}{p_interact:10.4g}"
             f"{float(res.rsquared):8.4f}   {flag}")

# %% [markdown]
# ## Verdict

# %%
print(f"\n{'=' * 90}\nVERDICT (pre-registered: any interaction at p<0.01 keeps H1' alive)\n{'=' * 90}")
if any_significant:
    sig_rows = [r for r in rows if r["p_interact"] < 0.01]
    print(f"{len(sig_rows)}/{len(rows)} (h, feature) combinations significant at p<0.01 -- "
         f"the reduced form shows something. Worth building the structural version "
         f"(splitting chi1/chi2/xi attribution by news content).")
    for r in sig_rows:
        print(f"  h={r['h']:3d}  {r['feature']:>26s}  coef={r['coef_interact']:+.5f}  p={r['p_interact']:.4g}")
else:
    print("No interaction significant at p<0.01 across all h x features -- the reduced form shows "
         "nothing. Per the pre-registered rule, the structural version is not worth building.")

# %%
import json  # noqa: E402
out_path = ROOT / "results" / "c1c18_h1_prime_reduced_form.json"
out_path.write_text(json.dumps({"rows": rows, "any_significant": bool(any_significant)}, indent=2, default=float))
print(f"\nsaved to {out_path}")
