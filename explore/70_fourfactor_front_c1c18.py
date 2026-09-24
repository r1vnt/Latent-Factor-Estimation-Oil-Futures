# %% [markdown]
# # Rung 2: a genuinely front-concentrated fourth factor, essentially affine
#
# kappa_f^Q fixed at 12.3 (loadings [0.59, 0.21, 0.08, 0.03, 0.01, ...] at
# the actual C1-C18 tau grid -- negligible by C4-C5, unlike chi1's own
# slower decay). kappa2/kappa3 (the old chi1/chi2, relabelled) stay
# completely affine -- rung 1 already tested freeing those. Multi-start
# on kappa_f_P first, same standard as rung 1: identical convergence
# across starts or the result isn't trustworthy.
#
# Warm start: chi_f initialised near-zero (sigma_f small), kappa2/kappa3
# at the S3 baseline's own kappa1/kappa2, curve at the S3 baseline's own
# curve -- letting the optimiser discover how much of a front factor it
# wants, rather than assuming.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit4_front import fit4_front_gaussian, pack4_front  # noqa: E402
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402

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
c0 = before["curve"]
print(f"S3 baseline: ll={before['ll']:.2f}  kappa1={before['kappa1']:.4f}  kappa2={before['kappa2']:.4f}")
print(f"kappa_f^Q fixed at {KAPPA_F_Q_ANCHOR}")

# %% [markdown]
# ## Multi-start on kappa_f_P

# %%
STARTS = [12.3, 3.0, 1.0, 0.3, 0.03]   # from "completely affine" down toward near-unit-root
SIGMA_F_START = 0.05

results = []
for k_f_p_start in STARTS:
    p_start = FourFactorFrontParams(
        kappa2=before["kappa1"], kappa3=before["kappa2"],
        sigma_f=SIGMA_F_START, sigma2=before["sigma1"], sigma3=before["sigma2"], sigma_xi=before["sigma_xi"],
        kappa_f_Q=KAPPA_F_Q_ANCHOR, kappa_f_P=k_f_p_start,
        rho_fx=0.0, rho2x=before["rho1x"], rho3x=before["rho2x"],
        mu_xi=before["mu_xi"], mu_xi_star=before["mu_xi_star"],
        lambda_f=0.0, lambda2=before["lambda1"], lambda3=before["lambda2"],
    )
    theta0 = pack4_front(p_start, c0["a"], c0["b"], c0["c"], quadratic=True)
    p_hat, smooth_hat, x_filt, resid, n_obs, ll = fit4_front_gaussian(
        log_f, tau, DT, theta0, quadratic=True, kappa_f_q=KAPPA_F_Q_ANCHOR)
    results.append({
        "start": k_f_p_start, "ll": ll, "kappa_f_P_hat": p_hat.kappa_f_P,
        "sigma_f_hat": p_hat.sigma_f, "beta_f": p_hat.beta_f,
        "half_life_front_phys_days": p_hat.half_life_front_phys_days,
        "kappa2_hat": p_hat.kappa2, "kappa3_hat": p_hat.kappa3,
    })
    print(f"start={k_f_p_start:6.3f}  ->  ll={ll:.2f}  kappa_f_P_hat={p_hat.kappa_f_P:.4f}  "
         f"sigma_f_hat={p_hat.sigma_f:.4f}  beta_f={p_hat.beta_f:.3f}  "
         f"half_life^P={p_hat.half_life_front_phys_days:.1f}d")

# %% [markdown]
# ## Convergence-consistency verdict

# %%
k_f_p_hats = [r["kappa_f_P_hat"] for r in results]
lls = [r["ll"] for r in results]
spread = max(k_f_p_hats) - min(k_f_p_hats)
best_idx = int(np.argmax(lls))
print(f"\n{'=' * 90}\nVERDICT\n{'=' * 90}")
print(f"kappa_f_P_hat across {len(STARTS)} starts: {[round(x, 4) for x in k_f_p_hats]}")
print(f"sigma_f_hat across starts: {[round(r['sigma_f_hat'], 4) for r in results]}")
print(f"ll across starts: {[round(x, 2) for x in lls]}")

stable = spread < 0.5 and (max(lls) - min(lls)) < 5.0
if not stable:
    print("\nUNSTABLE across starts -- per the pre-stated standard, this is a failure to report, "
         "not a result to interpret.")
else:
    best = results[best_idx]
    print(f"\nSTABLE. Best fit: kappa_f_P={best['kappa_f_P_hat']:.4f} vs fixed kappa_f_Q={KAPPA_F_Q_ANCHOR}  "
         f"(beta_f={best['beta_f']:.3f}, half_life^P={best['half_life_front_phys_days']:.1f} days "
         f"vs half_life^Q={np.log(2)/KAPPA_F_Q_ANCHOR*365.25:.1f} days)")
    if best["kappa_f_P_hat"] < KAPPA_F_Q_ANCHOR * 0.3:
        print("kappa_f_P << kappa_f_Q: the front factor decays fast along the curve but reverts "
             "much more slowly in time -- the hypothesised direction, this time on a factor "
             "that's actually front-exclusive.")
    elif best["kappa_f_P_hat"] > KAPPA_F_Q_ANCHOR:
        print("kappa_f_P > kappa_f_Q, same wrong direction as rung 1 -- now on a factor that "
             "IS front-exclusive, which rules out the 'chi1 is too diffuse' explanation for "
             "why rung 1 pointed the wrong way.")

# %%
out = {"kappa_f_Q": KAPPA_F_Q_ANCHOR, "starts": results, "spread": spread,
      "stable": bool(stable), "best_start_idx": best_idx}
out_path = ROOT / "results" / "c1c18_fourfactor_front_multistart.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
