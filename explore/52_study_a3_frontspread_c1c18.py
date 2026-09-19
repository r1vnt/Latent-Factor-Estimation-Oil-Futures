# %% [markdown]
# # The payoff test: does Study A's answer change on the front-spread state?
#
# explore/42 ran the full Study A3 battery against the plain price-only
# three-factor state (no variance channels) and found Gate 4 fails
# robustly. Since then: the front-spread H channel (explore/48-51) is the
# first extension in this whole investigation that genuinely worked --
# kurtosis 34.3 -> 1.86, nu_hat=13.7 confirming real tail-thinning, 73% of
# a 10,329-nat gain surviving the log(F)/resid^2 decomposition. That is a
# materially better state estimate. Step 5 of the original 5-step plan
# was always "re-run Study A on the new state" -- it has never happened.
# This is that run, on the full panel only (not the block-out battery --
# one decisive run, not a re-fit under every block-out).
#
# Everything below is byte-for-byte the same recovery3 battery explore/42
# calls -- only the filtered state and the params feeding it differ (now
# built via _raw_system3(..., rv_lag=rv_lag, h_driver=fs_driver) instead
# of the plain call). filtered_label/confidence_split_regression/etc.
# consume p's structural fields (sigma1, sigma2, sigma_xi, rho1x, rho2x,
# mu_xi_star, lambda1, lambda2) for the continuous-time bias correction,
# which do not depend on gamma_v/eta_v -- only x_filt itself (via the
# Kalman gain) changes.

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
    label_agreement,
    naive_label,
    run_encompassing_battery,
    xi_encompasses_log_price,
)

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0
MIN_HISTORY = 60

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

# %% [markdown]
# ## Rebuild the two drivers exactly as explore/48-51 did (deterministic, no randomness)

# %%
rv_lag = np.load(ROOT / "data" / "c1c18_log_rv_causal.npy")

spread = log_f[:, 0] - log_f[:, 1]
abs_spread_lag = np.concatenate([[np.nan], np.abs(spread[:-1])])
fs_driver = np.zeros(n)
running_sum, running_sumsq, count = 0.0, 0.0, 0
last_valid = 0.0
for t in range(n):
    if count >= MIN_HISTORY:
        mean = running_sum / count
        var = running_sumsq / count - mean ** 2
        std = np.sqrt(max(var, 1e-12))
        if np.isfinite(abs_spread_lag[t]):
            last_valid = (abs_spread_lag[t] - mean) / std
        fs_driver[t] = last_valid
    if np.isfinite(abs_spread_lag[t]):
        running_sum += abs_spread_lag[t]
        running_sumsq += abs_spread_lag[t] ** 2
        count += 1

# %% [markdown]
# ## Load the joint-nu front-spread state (explore/51's final fit)

# %%
fit = json.loads((ROOT / "results" / "c1c18_joint_nu_estimation.json").read_text())
fp = fit["params"]
p = ThreeFactorParams(
    kappa1=fit["kappa1"], kappa2=fit["kappa2"], sigma1=fp["sigma1"], sigma2=fp["sigma2"],
    sigma_xi=fp["sigma_xi"], rho12=0.0, rho1x=fp["rho1x"], rho2x=fp["rho2x"],
    mu_xi=fp["mu_xi"], mu_xi_star=fp["mu_xi_star"], lambda1=fp["lambda1"], lambda2=fp["lambda2"],
    gamma_v=fit["gamma_v"],
)
smooth = SmoothMeasParams(a=fp["curve"]["a"], b=fp["curve"]["b"], c=fp["curve"]["c"], quadratic=True,
                          eta_v=fit["eta_v"], lambda_v=fp["curve"]["lambda_v"])
print(f"front-spread state: nu_hat={fit['nu_hat']:.3f}  kappa1={p.kappa1:.4f}  kappa2={p.kappa2:.4f}  "
     f"eta_v={fit['eta_v']:.4f}  gamma_v={p.gamma_v:.4f}")

sys3 = _raw_system3(p, smooth, tau, DT, rv_lag=rv_lag, h_driver=fs_driver)
x0, P0 = initial_state3(log_f, tau, p)
ll, x_filt, P_filt, x_pred, P_pred, resid, n_obs = filter_uv(
    np.ascontiguousarray(log_f), np.ascontiguousarray(sys3["Z"]),
    np.ascontiguousarray(sys3["d"]), np.ascontiguousarray(sys3["T"]),
    np.ascontiguousarray(sys3["c"]), np.ascontiguousarray(sys3["Q"]),
    np.ascontiguousarray(sys3["Hdiag"]), np.zeros((sys3["n"], 3)),
    np.ascontiguousarray(x0), np.ascontiguousarray(P0),
)
print(f"filtered: ll={ll:.2f}  n_obs={n_obs}")

l_obs = naive_label(log_f)
l_filt = filtered_label(p, x_filt, tau)

# %% [markdown]
# ## Label agreement, vs. explore/42's plain-state baseline

# %%
agree = label_agreement(l_obs, l_filt)
baseline = json.loads((ROOT / "results" / "c1c18_study_a3_full.json").read_text())
print(f"\nlabel agreement (front-spread state): {agree}")
print(f"label agreement (baseline, plain state): {baseline['label_agreement']}")

# %% [markdown]
# ## Encompassing battery: realised_vol (primary), cross-maturity (secondary)

# %%
print(f"\n{'=' * 90}\nENCOMPASSING BATTERY\n{'=' * 90}")
battery_results = run_encompassing_battery(p, x_filt, log_f, tau, horizons=HORIZONS)
for target, by_h in battery_results.items():
    print(f"\n{target}:")
    for h, r in by_h.items():
        base_pattern = baseline["battery"].get(target, {}).get(str(h), {}).get("pattern", "?")
        print(f"  {r}   (baseline pattern: {base_pattern})")

# %% [markdown]
# ## Identifying-count / McNemar gate, THEN the disagreement-conditioned test

# %%
print(f"\n{'=' * 90}\nIDENTIFYING-COUNT GATE (before any disagreement test)\n{'=' * 90}")
gate_results = {}
for h in HORIZONS:
    gate = identifying_count_gate(l_obs, l_filt, log_f, h)
    gate_results[h] = gate
    base_gate = baseline["identifying_gate"].get(str(h), {})
    print(f"h={h:4d}  n_disagree={gate['n_disagree']:4d}  n_runs={gate['n_runs']:3d}  "
         f"n_independent_runs={gate['n_independent_runs']:3d}  "
         f"(baseline n_disagree={base_gate.get('n_disagree', '?')}  "
         f"n_independent_runs={base_gate.get('n_independent_runs', '?')})")
    print(f"         p_effect={gate['p_effect']:.4f}  "
         f"required_n(observed)={gate['required_n_observed']:.1f}  "
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
    "state": "front-spread H channel + RV Q channel, joint nu (explore/51)",
    "nu_hat": fit["nu_hat"], "kappa1": p.kappa1, "kappa2": p.kappa2,
    "ll": ll, "label_agreement": agree,
    "battery": {target: {h: {"b1": r.b1, "b2": r.b2, "p_b1": r.p_b1, "p_b2": r.p_b2,
                            "n_obs": r.n_obs, "pattern": r.pattern()}
                        for h, r in by_h.items()}
               for target, by_h in battery_results.items()},
    "identifying_gate": {h: gate_results[h] for h in HORIZONS},
}
out_path = ROOT / "results" / "c1c18_study_a3_frontspread_full.json"
out_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nsaved to {out_path}")
