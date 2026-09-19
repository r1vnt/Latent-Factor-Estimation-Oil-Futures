# %% [markdown]
# # Is cell B's eigenvalue=1.000001 a genuine near-unit-root estimate, or a
# # wall the optimiser is straining against?
#
# explore/60's Gaussian fit landed with the largest eigenvalue modulus at
# 1.000001 -- suspiciously close to SPECTRAL_RADIUS_MAX (1.0 + 1e-6). That
# is exactly the signature explore/44 hit (identical convergence at a
# clipped boundary, not a genuine interior optimum) unless the gradient
# at that point is genuinely near zero rather than pointing further out.
# Rerun the SAME Gaussian fit from the SAME warm start with the stability
# wall loosened by several orders of magnitude (radius <= 5 instead of
# 1+1e-6). If it lands back near 1.0 anyway, this is a genuine (if noisy,
# as near-unit-root MLE always is) estimate, not an artifact of the wall.
# If it runs away to something much larger, the wall was doing real work
# and explore/60's fit cannot be trusted for the forecast test as-is.

# %%
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csr.pricing.varT as varT_module  # noqa: E402
from csr.curve.build import Panel  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402

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
p = ThreeFactorParams(
    kappa1=before["kappa1"], kappa2=before["kappa2"], sigma1=before["sigma1"],
    sigma2=before["sigma2"], sigma_xi=before["sigma_xi"], rho12=0.0,
    rho1x=before["rho1x"], rho2x=before["rho2x"], mu_xi=before["mu_xi"],
    mu_xi_star=before["mu_xi_star"], lambda1=before["lambda1"], lambda2=before["lambda2"],
    gamma_v=0.0,
)
smooth_s3 = SmoothMeasParams(a=before["curve"]["a"], b=before["curve"]["b"],
                             c=before["curve"]["c"], quadratic=True)
Z, d = varT_module.frozen_Z_d(p, tau, DT)
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
theta0 = varT_module.pack_varT(T_ou, c_ou, Q_ou, a=smooth_s3.a, b=smooth_s3.b, c_curve=smooth_s3.c)

for radius_max in (1.0 + 1e-6, 1.01, 1.1, 2.0, 5.0):
    varT_module.SPECTRAL_RADIUS_MAX = radius_max
    state, smooth, x_filt, resid, n_obs, ll, theta = varT_module.fit_varT_gaussian(
        log_f, Z, d, tau, x0, P0, theta0)
    eigvals = np.linalg.eigvals(state.T)
    print(f"radius_max={radius_max:<10.6f} ll={ll:.2f}  max|eig|={np.max(np.abs(eigvals)):.6f}  "
         f"eigs_moduli={np.round(np.abs(eigvals), 6)}")

# reset for any subsequent import in the same process
varT_module.SPECTRAL_RADIUS_MAX = 1.0 + 1e-6
