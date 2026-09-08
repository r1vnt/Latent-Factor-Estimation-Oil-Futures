# %% [markdown]
# # Study A, three-factor: full panel + all three maturity block-outs
#
# Usage: python explore/42_study_a3_c1c18.py [BLOCK]
# BLOCK in {full, C1-C4, C5-C12, C13-C18}. Runs the whole recovery3
# battery -- encompassing regressions (realised_vol primary, cross-
# maturity secondary) at h in {22,66,132,252}, the identifying-count/
# McNemar gate BEFORE any disagreement-conditioned test, the implied-
# spread confidence split, and the xi/chi-split-vs-log-price checks --
# against that panel's own price-only three-factor state.
#
# Block-outs reuse the saved kappa1/kappa2/curve from
# results/c1c18_threefactor_battery.json (the only per-block-out fields
# saved) combined with the full panel's other structural params (not
# saved per block-out) -- same approximation used for the news block-out
# battery (explore/41), refined by nothing further here since Study A
# only needs the filtered state, not a fresh optimisation.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import _raw_system3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.state.fast import filter_uv  # noqa: E402
from csr.study.recovery3 import (  # noqa: E402
    HORIZONS,
    NONOVERLAP_HORIZONS,
    chi_split_encompasses_log_price,
    confidence_split_regression,
    disagreement_event_study,
    filtered_label,
    identifying_count_gate,
    naive_label,
    run_encompassing_battery,
    xi_encompasses_log_price,
)

BLOCK = sys.argv[1] if len(sys.argv) > 1 else "full"
ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

BLOCK_COLS = {
    "full": list(range(18)),
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
log_f = panel.log_f[:, cols]
tau = panel.tau[:, cols]

# %% [markdown]
# ## Reconstruct that panel's own price-only three-factor state

# %%
full_3f = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
if BLOCK == "full":
    kappa1, kappa2 = full_3f["kappa1"], full_3f["kappa2"]
    curve = full_3f["curve"]
else:
    battery = json.loads((ROOT / "results" / "c1c18_threefactor_battery.json").read_text())
    block_fit = next(r for r in battery["results"]
                     if r["label"] == f"drop {BLOCK}" and not r["cold"])
    kappa1, kappa2 = block_fit["kappa1"], block_fit["kappa2"]
    curve = block_fit["curve"]

p = ThreeFactorParams(
    kappa1=kappa1, kappa2=kappa2, sigma1=full_3f["sigma1"], sigma2=full_3f["sigma2"],
    sigma_xi=full_3f["sigma_xi"], rho12=0.0, rho1x=full_3f["rho1x"], rho2x=full_3f["rho2x"],
    mu_xi=full_3f["mu_xi"], mu_xi_star=full_3f["mu_xi_star"],
    lambda1=full_3f["lambda1"], lambda2=full_3f["lambda2"], gamma_v=0.0,
)
smooth = SmoothMeasParams(a=curve["a"], b=curve["b"], c=curve["c"], quadratic=True)
print(f"BLOCK={BLOCK}  m={len(cols)}  kappa1={p.kappa1:.4f}  kappa2={p.kappa2:.4f}")

sys3 = _raw_system3(p, smooth, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)
n = sys3["n"]
ll, x_filt, P_filt, x_pred, P_pred, resid, n_obs = filter_uv(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
    np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
    np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((n, 3)),
    np.ascontiguousarray(x0), np.ascontiguousarray(P0),
)
print(f"filtered: ll={ll:.2f}  n_obs={n_obs}")

l_obs = naive_label(log_f)
l_filt = filtered_label(p, x_filt, tau)

# %% [markdown]
# ## Label agreement

# %%
from csr.study.recovery3 import label_agreement  # noqa: E402
agree = label_agreement(l_obs, l_filt)
print(f"\nlabel agreement: {agree}")

# %% [markdown]
# ## Encompassing battery: realised_vol (primary), cross-maturity (secondary)

# %%
print(f"\n{'=' * 90}\nENCOMPASSING BATTERY\n{'=' * 90}")
battery_results = run_encompassing_battery(p, x_filt, log_f, tau, horizons=HORIZONS)
for target, by_h in battery_results.items():
    print(f"\n{target}:")
    for h, r in by_h.items():
        print(f"  {r}")

# %% [markdown]
# ## Identifying-count / McNemar gate, THEN the disagreement-conditioned test

# %%
print(f"\n{'=' * 90}\nIDENTIFYING-COUNT GATE (before any disagreement test)\n{'=' * 90}")
for h in HORIZONS:
    gate = identifying_count_gate(l_obs, l_filt, log_f, h)
    print(f"h={h:4d}  n_disagree={gate['n_disagree']:4d}  n_runs={gate['n_runs']:3d}  "
         f"n_independent_runs={gate['n_independent_runs']:3d}  "
         f"p_effect={gate['p_effect']:.4f}  "
         f"required_n(observed, restates effect)={gate['required_n_observed']:.1f}  "
         f"required_n(@ref={gate['reference_effect_size']})={gate['required_n_reference']:.1f}  "
         f"sufficient(disagree)={gate['sufficient_disagree_days']}  "
         f"sufficient(indep_runs)={gate['sufficient_independent_runs']}")
    if gate["sufficient_independent_runs"]:
        event = disagreement_event_study(l_obs, l_filt, log_f, h)
        print(f"    -> RUNNING disagreement_event_study: {event}")
    else:
        print(f"    -> NOT RUN (insufficient identifying observations at this horizon)")

# %% [markdown]
# ## Confidence split (implied-spread variance) and xi/chi-split vs. log price

# %%
print(f"\n{'=' * 90}\nCONFIDENCE SPLIT (filter-as-estimator)\n{'=' * 90}")
for h in HORIZONS:
    r_low, r_high = confidence_split_regression(p, x_filt, P_filt, log_f, tau, h)
    print(f"h={h:4d}  high_uncertainty: {r_low}")
    print(f"          high_confidence:  {r_high}")

print(f"\n{'=' * 90}\nXI ENCOMPASSES LOG PRICE\n{'=' * 90}")
for h in HORIZONS:
    r = xi_encompasses_log_price(x_filt, log_f, h)
    print(f"  {r}")

print(f"\n{'=' * 90}\nCHI1/CHI2 SPLIT ENCOMPASSES LOG PRICE\n{'=' * 90}")
for h in HORIZONS:
    r = chi_split_encompasses_log_price(x_filt, log_f, h)
    print(f"  {r}")

# %%
out = {
    "block": BLOCK, "m": len(cols), "kappa1": p.kappa1, "kappa2": p.kappa2,
    "ll": ll, "label_agreement": agree,
    "battery": {target: {h: {"b1": r.b1, "b2": r.b2, "p_b1": r.p_b1, "p_b2": r.p_b2,
                            "n_obs": r.n_obs, "pattern": r.pattern()}
                        for h, r in by_h.items()}
               for target, by_h in battery_results.items()},
    "identifying_gate": {h: identifying_count_gate(l_obs, l_filt, log_f, h) for h in HORIZONS},
}
out_path = ROOT / "results" / f"c1c18_study_a3_{BLOCK.replace('-', '_')}.json"
out_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nsaved to {out_path}")
