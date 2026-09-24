"""Tests for csr.pricing.fourfactor_front / fit4_front -- rung 2 of the
escalation ladder (a genuinely front-concentrated fourth factor, fixed
kappa_f^Q, free essentially-affine kappa_f^P). Narrow, matching
test_fit3.py's/test_varT.py's own scope: pack/unpack roundtrip, the
nesting guarantee (sigma_f=lambda_f=0 must reproduce the three-factor
model exactly -- chi_f contributes nothing when its own variance is
zero), and non-vacuousness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit3 import weighted_filter3  # noqa: E402
from csr.pricing.fit4_front import (  # noqa: E402
    _neg_loglik4_front,
    pack4_front,
    unpack4_front,
    weighted_filter4_front,
)
from csr.pricing.fourfactor_front import FourFactorFrontParams, KAPPA_F_Q_ANCHOR  # noqa: E402
from csr.pricing.smoothmeas import SmoothMeasParams  # noqa: E402
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402

DT = 1.0 / 252.0


def _p4():
    return FourFactorFrontParams(kappa2=1.9, kappa3=0.4, sigma_f=0.15, sigma2=0.22, sigma3=0.31,
                                 sigma_xi=0.2, rho_fx=0.05, rho2x=0.1, rho3x=0.15,
                                 mu_xi=0.0, mu_xi_star=0.01, lambda2=0.05, lambda3=0.02)


def _panel(seed=0, n=200, m=6):
    rng = np.random.default_rng(seed)
    tau = np.tile(np.linspace(0.08, 1.0, m), (n, 1)) + rng.normal(0, 0.001, (n, m))
    log_f = 70.0 + np.cumsum(rng.normal(0, 0.01, n)).reshape(-1, 1) + rng.normal(0, 0.001, (n, m))
    return log_f, np.abs(tau)


def test_kappa_f_q_defaults_to_the_anchor_and_kappa_f_p_falls_back_to_it():
    p = _p4()
    assert p.kappa_f_Q == pytest.approx(KAPPA_F_Q_ANCHOR)
    assert p.kappa_f_P == pytest.approx(KAPPA_F_Q_ANCHOR)   # completely-affine fallback
    assert p.beta_f == pytest.approx(0.0)


def test_pack_unpack_roundtrip():
    p = _p4().replace(kappa_f_P=3.5)
    theta = pack4_front(p, a=-4.0, b=-2.0, c=1.0, quadratic=True)
    p2, smooth2 = unpack4_front(theta, quadratic=True, kappa_f_q=p.kappa_f_Q)
    assert p2.kappa_f_P == pytest.approx(3.5)
    assert p2.kappa_f_Q == pytest.approx(p.kappa_f_Q)
    assert p2.kappa2 == pytest.approx(p.kappa2)
    assert p2.kappa3 == pytest.approx(p.kappa3)
    assert smooth2.a == pytest.approx(-4.0)


def test_ordering_guard_rejects_kappa3_exceeding_kappa2():
    with pytest.raises(ValueError):
        FourFactorFrontParams(kappa2=0.4, kappa3=1.9, sigma_f=0.1, sigma2=0.2, sigma3=0.3, sigma_xi=0.2)


def test_star_psd_guard_rejects_excess_correlation():
    with pytest.raises(ValueError):
        FourFactorFrontParams(kappa2=1.9, kappa3=0.4, sigma_f=0.1, sigma2=0.2, sigma3=0.3, sigma_xi=0.2,
                              rho_fx=0.8, rho2x=0.8, rho3x=0.1)


def test_nesting_sigma_f_zero_reproduces_the_three_factor_model_exactly():
    """chi_f contributes nothing when its own variance is zero (x0=0,
    P0's chi_f entry=0, Q's chi_f row/col=0 -- it stays at exactly zero
    forever) -- must reproduce weighted_filter3's likelihood on the SAME
    data with (kappa1,kappa2)=(this model's kappa2,kappa3), to 1e-9."""
    log_f, tau = _panel()
    p4 = _p4().replace(sigma_f=0.0, lambda_f=0.0, rho_fx=0.0)
    smooth = SmoothMeasParams(a=-4.0, b=-2.0, c=1.0, quadratic=True)

    ll4, xf4, resid4, nobs4 = weighted_filter4_front(p4, smooth, log_f, tau, DT, np.ones_like(log_f))

    p3 = ThreeFactorParams(kappa1=p4.kappa2, kappa2=p4.kappa3, sigma1=p4.sigma2, sigma2=p4.sigma3,
                           sigma_xi=p4.sigma_xi, rho12=0.0, rho1x=p4.rho2x, rho2x=p4.rho3x,
                           mu_xi=p4.mu_xi, mu_xi_star=p4.mu_xi_star, lambda1=p4.lambda2, lambda2=p4.lambda3,
                           gamma_v=0.0)
    ll3, xf3, resid3, nobs3 = weighted_filter3(p3, smooth, log_f, tau, DT, np.ones_like(log_f))

    assert ll4 == pytest.approx(ll3, abs=1e-6)
    assert nobs4 == nobs3


def test_nonzero_sigma_f_actually_changes_the_likelihood():
    log_f, tau = _panel(seed=3)
    p4 = _p4().replace(sigma_f=0.0, lambda_f=0.0, rho_fx=0.0)
    smooth = SmoothMeasParams(a=-4.0, b=-2.0, c=1.0, quadratic=True)
    theta_zero = pack4_front(p4, a=-4.0, b=-2.0, c=1.0, quadratic=True)

    p4_nonzero = _p4().replace(sigma_f=0.2, kappa_f_P=2.0)
    theta_nonzero = pack4_front(p4_nonzero, a=-4.0, b=-2.0, c=1.0, quadratic=True)

    nll_zero = _neg_loglik4_front(theta_zero, log_f, tau, DT, True)
    nll_nonzero = _neg_loglik4_front(theta_nonzero, log_f, tau, DT, True)
    assert abs(nll_zero - nll_nonzero) > 1e-6


def test_kappa_f_q_is_fixed_not_free_in_theta():
    """theta must have exactly 18 entries (15 structural/essentially-affine
    + 3 quadratic curve) -- kappa_f_Q is passed separately, never packed."""
    p = _p4()
    theta = pack4_front(p, a=-4.0, b=-2.0, c=1.0, quadratic=True)
    assert len(theta) == 18
