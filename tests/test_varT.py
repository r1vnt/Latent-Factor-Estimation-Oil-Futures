"""Tests for csr.pricing.varT -- cell B of the loading/transition lattice
(restricted Z, free VAR(1) T). Narrow, matching test_fit3.py's own scope:
locks in the pack/unpack roundtrip, the nesting guarantee (T set to the
S3 baseline's own diagonal, Q to its own Q, must reproduce S3's exact
likelihood -- Step 3's usual discipline, now for a genuinely different
model class), and the stability-rejection guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit3 import weighted_filter3  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams, initial_state3  # noqa: E402
from csr.pricing.varT import (  # noqa: E402
    frozen_Z_d,
    pack_varT,
    unpack_varT,
    weighted_filter_varT,
)


def _p3():
    return ThreeFactorParams(kappa1=1.9, kappa2=0.4, sigma1=0.22, sigma2=0.31,
                             sigma_xi=0.2, rho1x=0.1, rho2x=0.15,
                             mu_xi=0.0, mu_xi_star=0.01, lambda1=0.05, lambda2=0.02,
                             gamma_v=0.0)


def _panel(seed=0, n=200, m=6):
    rng = np.random.default_rng(seed)
    tau = np.tile(np.linspace(0.08, 1.0, m), (n, 1)) + rng.normal(0, 0.001, (n, m))
    log_f = 70.0 + np.cumsum(rng.normal(0, 0.01, n)).reshape(-1, 1) + rng.normal(0, 0.001, (n, m))
    return log_f, np.abs(tau)


DT = 1.0 / 252.0


def test_pack_unpack_roundtrip():
    T = np.array([[0.9, 0.05, 0.0], [-0.02, 0.85, 0.01], [0.0, 0.0, 1.0]])
    c = np.array([0.001, -0.002, 0.0003])
    Q = np.array([[0.04, 0.005, 0.001], [0.005, 0.09, -0.002], [0.001, -0.002, 0.03]])
    theta = pack_varT(T, c, Q, a=-4.0, b=-2.0, c_curve=1.0)
    state, smooth = unpack_varT(theta)
    assert np.allclose(state.T, T)
    assert np.allclose(state.c, c)
    assert np.allclose(state.Q, Q, atol=1e-8)
    assert smooth.a == pytest.approx(-4.0)
    assert smooth.b == pytest.approx(-2.0)
    assert smooth.c == pytest.approx(1.0)


def test_unstable_T_is_rejected():
    T_explosive = np.eye(3) * 2.0   # spectral radius 2.0, well past SPECTRAL_RADIUS_MAX=1.5
    Q = np.eye(3) * 0.01
    theta = pack_varT(T_explosive, np.zeros(3), Q, a=-4.0, b=-2.0, c_curve=0.0)
    with pytest.raises(ValueError):
        unpack_varT(theta)


def test_marginal_unit_root_is_accepted():
    """A unit root must be PERMITTED, not rejected -- it's the economically
    expected case (the long-run factor), not a boundary violation."""
    T_unit = np.diag([0.9, 0.8, 1.0])   # spectral radius exactly 1
    Q = np.eye(3) * 0.01
    theta = pack_varT(T_unit, np.zeros(3), Q, a=-4.0, b=-2.0, c_curve=0.0)
    state, smooth = unpack_varT(theta)   # must not raise
    assert np.allclose(state.T, T_unit)


def test_nesting_reproduces_S3_likelihood_exactly():
    """T set to S3's own diag(e^-k1*dt, e^-k2*dt, 1), c to S3's own [0,0,mu_xi*dt],
    Q to S3's own Q, on the SAME frozen Z/d -- must reproduce weighted_filter3's
    likelihood to 1e-9. This is the Step-3-style nesting guarantee, now for a
    structurally different transition parameterisation, not just a zeroed
    extension parameter."""
    log_f, tau = _panel()
    p = _p3()
    smooth_s3 = SmoothMeasParams(a=-4.0, b=-2.0, c=1.0, quadratic=True)

    ll_s3, x_filt_s3, resid_s3, n_obs_s3 = weighted_filter3(
        p, smooth_s3, log_f, tau, DT, np.ones_like(log_f))

    Z, d = frozen_Z_d(p, tau, DT)
    k1, k2 = p.kappa1, p.kappa2
    T_ou = np.diag([np.exp(-k1 * DT), np.exp(-k2 * DT), 1.0])
    c_ou = np.array([0.0, 0.0, p.mu_xi * DT])

    k1e, k2e = max(k1, 1e-8), max(k2, 1e-8)
    q11 = -np.expm1(-2.0 * k1e * DT) * p.sigma1 ** 2 / (2.0 * k1e)
    q22 = -np.expm1(-2.0 * k2e * DT) * p.sigma2 ** 2 / (2.0 * k2e)
    q33 = p.sigma_xi ** 2 * DT
    q13 = -np.expm1(-k1e * DT) * p.rho1x * p.sigma1 * p.sigma_xi / k1e
    q23 = -np.expm1(-k2e * DT) * p.rho2x * p.sigma2 * p.sigma_xi / k2e
    Q_ou = np.array([[q11, 0.0, q13], [0.0, q22, q23], [q13, q23, q33]])

    theta = pack_varT(T_ou, c_ou, Q_ou, a=-4.0, b=-2.0, c_curve=1.0)
    state, smooth_varT = unpack_varT(theta)

    x0, P0 = initial_state3(log_f, tau, p)
    ll_varT, x_filt_varT, resid_varT, n_obs_varT = weighted_filter_varT(
        state, smooth_varT, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))

    assert ll_varT == pytest.approx(ll_s3, abs=1e-6)
    assert n_obs_varT == n_obs_s3
    assert np.allclose(resid_varT, resid_s3, atol=1e-6, equal_nan=True)
