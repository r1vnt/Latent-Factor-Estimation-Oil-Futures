# %% [markdown]
# # H2 reconstruction, step 1: the one-line reverse-causality diagnostic
#
# Pre-registered in preregistrations/2026-09-19_h1_h2_h3_revisions.md
# before this ran. corr(I_{t-1}, RV_{t-1}) = 0.108 was already reported
# (explore/62). The new check: corr(I_t, RV_{t-1}) -- does TODAY's
# (unlagged) news intensity correlate with YESTERDAY's trailing realised
# vol more strongly than intensity correlates with ITS OWN same-day RV
# reading? If so, I_t is substantially just recent volatility relabeled,
# which would explain explore/62's wrong-signed, null coefficient as a
# controlling-for-the-mediator flip rather than genuine absence of a
# news-volatility relationship.
#
# Falsifier stated in advance: >0.3 confirms the specification was
# broken; staying near 0.108 means the mediation story doesn't explain
# the null and something else (level-vs-surprise construction, or the
# lag discarding the contemporaneous effect) is the more likely culprit.

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RV_WINDOW = 20

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
intensity_today = news_aligned["intensity"].to_numpy(dtype=float)          # I_t, unlagged
intensity_lag = np.concatenate([[np.nan], intensity_today[:-1]])           # I_{t-1}, as before

ret = np.full(n, np.nan)
ret[1:] = np.diff(log_f[:, 0])
rv_trailing = np.full(n, np.nan)
for t in range(RV_WINDOW, n):
    window = ret[t - RV_WINDOW + 1: t + 1]
    if np.all(np.isfinite(window)):
        rv_trailing[t] = np.std(window, ddof=1)
rv_today = rv_trailing                                                     # RV_t (ends today)
rv_lag = np.concatenate([[np.nan], rv_trailing[:-1]])                       # RV_{t-1}, as before

# %% [markdown]
# ## The three correlations

# %%
def _corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid], b[valid])[0, 1]), int(valid.sum())

c_lag_lag, n1 = _corr(intensity_lag, rv_lag)          # already reported: 0.108
c_today_lag, n2 = _corr(intensity_today, rv_lag)      # new: I_t vs RV_{t-1}
c_today_today, n3 = _corr(intensity_today, rv_today)  # new: I_t vs RV_t (contemporaneous)

print(f"{'=' * 90}\nH2 REVERSE-CAUSALITY DIAGNOSTIC\n{'=' * 90}")
print(f"corr(I_t-1, RV_t-1)  = {c_lag_lag:+.4f}   n={n1}   (already reported, explore/62)")
print(f"corr(I_t,   RV_t-1)  = {c_today_lag:+.4f}   n={n2}   (new: does today's news track YESTERDAY's vol?)")
print(f"corr(I_t,   RV_t)    = {c_today_today:+.4f}   n={n3}   (new: today's news vs today's OWN vol, contemporaneous)")

# %% [markdown]
# ## Verdict, against the pre-registered falsifier

# %%
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
if c_today_lag > 0.3:
    print(f"CONFIRMED BROKEN: corr(I_t, RV_t-1)={c_today_lag:.3f} >> 0.108 -- today's intensity is "
         f"substantially a lagged read on recent volatility, exactly the mediation pattern that "
         f"would flip a coefficient once RV_t-1 is controlled for. Proceed to full reconstruction.")
elif c_today_today > c_today_lag + 0.1:
    print(f"Contemporaneous relationship ({c_today_today:.3f}) clearly exceeds the lagged one "
         f"({c_today_lag:.3f}) -- consistent with news tracking CURRENT, not past, volatility. "
         f"The reverse-causality-through-the-control story is not strongly supported; the lag "
         f"structure discarding the contemporaneous effect (fix 3) is the more likely culprit.")
else:
    print(f"Neither correlation is materially above 0.108 -- the mediation story doesn't explain "
         f"the null on its own. Level-vs-surprise construction (fix 2) is the next most likely cause.")

# %%
import json  # noqa: E402
out = {
    "corr_Ilag_RVlag": c_lag_lag, "corr_Itoday_RVlag": c_today_lag, "corr_Itoday_RVtoday": c_today_today,
    "n_Ilag_RVlag": n1, "n_Itoday_RVlag": n2, "n_Itoday_RVtoday": n3,
}
out_path = ROOT / "results" / "c1c18_h2_reverse_causality_diagnostic.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nsaved to {out_path}")
