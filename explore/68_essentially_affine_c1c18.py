# %% [markdown]
# # Essentially-affine kappa1: multi-start, before trusting anything
#
# kappa1_phys is notoriously weakly identified in affine models (the
# user's own warning, matching Duffee's literature). Multi-start FIRST,
# on kappa1_phys specifically -- kappa2_phys starts at kappa2 (nested,
# beta2=0) throughout, since the diagnosed defect (front decays fast
# along the curve, barely reverts in time) is specifically about the
# FAST factor. A kappa1_phys that lands in different places from
# different starts is a failure to report, not a result to interpret.
#
# Gaussian only (nu held, matching Fix 5's discipline -- a conditional-
# variance-like reinterpretation of kappa competes with Student-t for
# explaining the same residual structure if fit jointly with nu free).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import fit3_gaussian_smooth, pack3_smooth  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p_base = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
c0 = before["curve"]
print(f"S3 baseline: ll={before['ll']:.2f}  kappa1(=kappa1^Q)={p_base.kappa1:.4f}  "
     f"kappa2={p_base.kappa2:.4f}")

# %% [markdown]
# ## Multi-start on kappa1_phys, kappa2_phys nested at kappa2 (beta2=0)

# %%
STARTS = [1.0, 0.5, 0.1, 0.02, 0.002]   # fraction of kappa1
results = []
for frac in STARTS:
    k1_phys_start = p_base.kappa1 * frac
    p_start = p_base.replace(kappa1_phys=k1_phys_start, kappa2_phys=p_base.kappa2)
    theta0 = pack3_smooth(p_start, c0["a"], c0["b"], c0["c"], quadratic=True, ea_extension=True)
    p_hat, smooth_hat, x_filt, resid, n_obs, ll = fit3_gaussian_smooth(
        log_f, tau, DT, theta0, quadratic=True, ea_extension=True)
    results.append({
        "start_frac": frac, "kappa1_phys_start": k1_phys_start, "ll": ll,
        "kappa1_phys_hat": p_hat.kappa1_phys, "kappa2_phys_hat": p_hat.kappa2_phys,
        "kappa1": p_hat.kappa1, "kappa2": p_hat.kappa2,
        "beta1": p_hat.beta1, "beta2": p_hat.beta2,
        "half_life_fast_phys_days": p_hat.half_life_fast_phys_days,
    })
    print(f"start frac={frac:6.3f}  k1_phys_start={k1_phys_start:.4f}  ->  "
         f"ll={ll:.2f}  k1_phys_hat={p_hat.kappa1_phys:.4f}  k2_phys_hat={p_hat.kappa2_phys:.4f}  "
         f"beta1={p_hat.beta1:.3f}  half_life_phys={p_hat.half_life_fast_phys_days:.1f}d")

# %% [markdown]
# ## Convergence-consistency verdict

# %%
k1_phys_hats = [r["kappa1_phys_hat"] for r in results]
lls = [r["ll"] for r in results]
spread = max(k1_phys_hats) - min(k1_phys_hats)
best_idx = int(np.argmax(lls))
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"kappa1_phys_hat across {len(STARTS)} starts: {[round(x, 4) for x in k1_phys_hats]}")
print(f"spread = {spread:.4f}  (baseline kappa1^Q = {p_base.kappa1:.4f}, spread/kappa1^Q = {spread / p_base.kappa1:.1%})")
print(f"ll across starts: {[round(x, 2) for x in lls]}  (max spread: {max(lls) - min(lls):.4f})")

if spread / p_base.kappa1 > 0.3 and (max(lls) - min(lls)) < 5.0:
    print("\nUNSTABLE: kappa1_phys lands in materially different places across starts at "
         "essentially the same likelihood -- per the pre-stated standard, this is a failure "
         "to report, not a result. Flat-ridge identification failure, matching the weak-lambda1 "
         "warning. Do not interpret the point estimate; profile likelihood next to confirm.")
else:
    print(f"\nSTABLE: kappa1_phys converges to a consistent value (spread {spread / p_base.kappa1:.1%} "
         f"of kappa1^Q) across starts. Best fit: start_frac={results[best_idx]['start_frac']}, "
         f"kappa1_phys={results[best_idx]['kappa1_phys_hat']:.4f}, "
         f"beta1={results[best_idx]['beta1']:.3f}, "
         f"half_life^P={results[best_idx]['half_life_fast_phys_days']:.1f} days "
         f"(vs half_life^Q={p_base.half_life_fast_days:.1f} days).")

# %%
out = {"starts": results, "spread_k1_phys": spread, "spread_frac_of_kappa1Q": spread / p_base.kappa1,
      "ll_spread_across_starts": max(lls) - min(lls), "best_start_idx": best_idx}
out_path = ROOT / "results" / "c1c18_essentially_affine_multistart.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
