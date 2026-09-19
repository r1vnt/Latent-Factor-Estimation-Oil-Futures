# %% [markdown]
# # Cell B: restricted Z (frozen at S3), free VAR(1) T -- fit on C1-C18
#
# PCA (explore/59) ruled out the loading shape: affine residual ~ PCA-3
# ceiling (median ratio 0.83x). Three separate pieces of evidence already
# pointed at the transition instead -- kappa1 unstable at 0.7574 across
# block-outs, the H-sweep residual that never closes even as H1 -> 0, and
# untouched volatility clustering. Cell B tests that directly: same Z/d
# (frozen at the S3 baseline, still a genuine no-arbitrage cross-section),
# free T (9 params, spectral radius <= 1 permitted-not-forced unit root).
#
# Sequence, matching every prior fitting stage's discipline:
#   1. Nesting check on the REAL panel (not just the synthetic test panel
#      in tests/test_varT.py): T/c/Q set to reproduce S3's own OU
#      structure exactly, confirm the likelihood matches to high
#      precision before touching the optimiser.
#   2. Gaussian fit, T/c/Q/curve free.
#   3. Student-t EM with JOINT nu estimation (explore/50's lesson: fixed
#      nu and a variance channel double-count fat tails -- same logic
#      applies to freeing T, not just freeing H).

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.curve.build import Panel  # noqa: E402
from csr.pricing.fit3 import weighted_filter3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.pricing.varT import (  # noqa: E402
    fit_varT_gaussian,
    fit_varT_student_t,
    frozen_Z_d,
    pack_varT,
    unpack_varT,
    weighted_filter_varT,
)

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 252.0

raw = pd.read_parquet(ROOT / "panel_delivery" / "panel_c1_c24.parquet")
raw = raw[raw.index >= "2010-01-01"]
raw.columns = [int(c[1:]) for c in raw.columns]
raw18 = raw.iloc[:, :18]
panel = Panel.from_prices(raw18, m=18, prefix="CL", basis="business",
                          source="panel_delivery_c1_c24_truncated_c18", verbose=False)
log_f, tau = panel.log_f, panel.tau
n = panel.n

before = json.loads((ROOT / "results" / "c1c18_threefactor.json").read_text())
p = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth_s3 = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                             c=before["curve"]["c"], quadratic=True)
ll_s3, x_filt_s3, resid_s3, n_obs_s3 = weighted_filter3(p, smooth_s3, log_f, tau, DT, np.ones_like(log_f))
print(f"S3 baseline (reference): ll={ll_s3:.6f}  n_obs={n_obs_s3}")

# %% [markdown]
# ## Step 1: nesting check on the real panel

# %%
Z, d = frozen_Z_d(p, tau, DT)
x0, P0 = initial_state3(log_f, tau, p)

k1, k2 = p.kappa1, p.kappa2
T_ou = np.diag([np.exp(-k1 * DT), np.exp(-k2 * DT), 1.0])
c_ou = np.array([0.0, 0.0, p.mu_xi * DT])
q11 = -np.expm1(-2.0 * k1 * DT) * p.sigma1 ** 2 / (2.0 * k1)
q22 = -np.expm1(-2.0 * k2 * DT) * p.sigma2 ** 2 / (2.0 * k2)
q33 = p.sigma_xi ** 2 * DT
q13 = -np.expm1(-k1 * DT) * p.rho1x * p.sigma1 * p.sigma_xi / k1
q23 = -np.expm1(-k2 * DT) * p.rho2x * p.sigma2 * p.sigma_xi / k2
Q_ou = np.array([[q11, 0.0, q13], [0.0, q22, q23], [q13, q23, q33]])

theta0 = pack_varT(T_ou, c_ou, Q_ou, a=smooth_s3.a, b=smooth_s3.b, c_curve=smooth_s3.c)
state0, smooth0 = unpack_varT(theta0)
ll_nest, x_filt_nest, resid_nest, n_obs_nest = weighted_filter_varT(
    state0, smooth0, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
print(f"nesting check: ll={ll_nest:.6f}  (S3: {ll_s3:.6f})  "
     f"|diff|={abs(ll_nest - ll_s3):.2e}  n_obs match: {n_obs_nest == n_obs_s3}")
assert abs(ll_nest - ll_s3) < 1e-3, "nesting failed on the real panel -- stop, do not fit from here"

# %% [markdown]
# ## Step 2: Gaussian fit, T/c/Q/curve free

# %%
state_g, smooth_g, x_filt_g, resid_g, n_obs_g, ll_g, theta_g = fit_varT_gaussian(
    log_f, Z, d, tau, x0, P0, theta0)
eigvals_g = np.linalg.eigvals(state_g.T)
print(f"\n{'=' * 90}\nGAUSSIAN FIT, T/c/Q/curve free\n{'=' * 90}")
print(f"ll={ll_g:.2f}  (S3 Gaussian-equivalent: {ll_s3:.2f}, LR={2 * (ll_g - ll_s3):.2f})")
print(f"T=\n{state_g.T}")
print(f"eigenvalues of T: {eigvals_g}  (moduli: {np.abs(eigvals_g)})")
print(f"c={state_g.c}")
print(f"Q=\n{state_g.Q}")
print(f"curve: a={smooth_g.a:.4f}  b={smooth_g.b:.4f}  c={smooth_g.c:.4f}")

# %% [markdown]
# ## Step 3: Student-t EM, joint nu estimation

# %%
state_hat, smooth_hat, x_filt_hat, resid_hat, weight_hat, n_iters, ll_hat, nu_hat, theta_hat = \
    fit_varT_student_t(log_f, Z, d, tau, x0, P0, theta_g, df0=5.0, estimate_df=True)
eigvals_hat = np.linalg.eigvals(state_hat.T)
print(f"\n{'=' * 90}\nSTUDENT-T EM, JOINT NU\n{'=' * 90}")
print(f"ll={ll_hat:.2f}  n_em_iters={n_iters}  nu_hat={nu_hat:.3f}")
print(f"T=\n{state_hat.T}")
print(f"eigenvalues of T: {eigvals_hat}  (moduli: {np.abs(eigvals_hat)})")
print(f"c={state_hat.c}")
print(f"Q=\n{state_hat.Q}")
print(f"curve: a={smooth_hat.a:.4f}  b={smooth_hat.b:.4f}  c={smooth_hat.c:.4f}")

# %%
out = {
    "ll_s3_reference": ll_s3, "nesting_ll_diff": abs(ll_nest - ll_s3),
    "gaussian": {"ll": ll_g, "T": state_g.T.tolist(), "c": state_g.c.tolist(), "Q": state_g.Q.tolist(),
                "eigenvalues_moduli": np.abs(eigvals_g).tolist(),
                "curve": {"a": smooth_g.a, "b": smooth_g.b, "c": smooth_g.c}, "theta": theta_g.tolist()},
    "student_t": {"ll": ll_hat, "nu_hat": nu_hat, "n_em_iters": n_iters,
                 "T": state_hat.T.tolist(), "c": state_hat.c.tolist(), "Q": state_hat.Q.tolist(),
                 "eigenvalues_moduli": np.abs(eigvals_hat).tolist(),
                 "curve": {"a": smooth_hat.a, "b": smooth_hat.b, "c": smooth_hat.c},
                 "theta": theta_hat.tolist()},
}
out_path = ROOT / "results" / "c1c18_cellB_varT.json"
out_path.write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved to {out_path}")
