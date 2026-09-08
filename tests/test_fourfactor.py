"""Nesting, identification guards, and the joint-PSD bug class fit3.py hit.

test_star_correlation_psd_bound_is_the_sum_of_squares is the one that
matters most here: it's the exact failure mode that crashed the three-
factor battery (rho1x, rho2x each in (-1, 1) individually, jointly non-PSD),
generalised to three rates and pinned before fit4.py can hit it blind.
"""

from __future__ import annotations

import numpy as np
import pytest

from csr.pricing.fourfactor import (A_tau4, FourFactorParams, base_system4,
                                    from_three_factor, initial_state4)
from csr.pricing.threefactor import A_tau3, ThreeFactorParams, base_system3

DT = 1.0 / 252.0
TAU18 = np.arange(1, 19) / 12.0


def _p4(**kw):
    base = dict(kappa1=12.0, kappa2=2.0, kappa3=0.4, sigma1=0.4, sigma2=0.3,
               sigma3=0.2, sigma_xi=0.18, mu_xi=0.02, mu_xi_star=0.01,
               lambda1=0.05)
    base.update(kw)
    return FourFactorParams(**base)


# ----------------------------------------------------------------- nesting --

def test_sigma3_zero_reproduces_three_factor_A_tau():
    p3 = ThreeFactorParams(kappa1=2.0, kappa2=0.4, sigma1=0.3, sigma2=0.15,
                           sigma_xi=0.18, rho1x=0.2, rho2x=-0.1, mu_xi=0.02,
                           mu_xi_star=0.01, lambda1=0.05, lambda2=0.02)
    p4 = FourFactorParams(kappa1=p3.kappa1, kappa2=p3.kappa2, kappa3=p3.kappa2 * 0.1,
                          sigma1=p3.sigma1, sigma2=p3.sigma2, sigma3=0.0,
                          sigma_xi=p3.sigma_xi, rho1x=p3.rho1x, rho2x=p3.rho2x,
                          rho3x=0.0, mu_xi=p3.mu_xi, mu_xi_star=p3.mu_xi_star,
                          lambda1=p3.lambda1, lambda2=p3.lambda2, lambda3=0.0)
    tau = np.linspace(0.02, 1.5, 60)
    np.testing.assert_allclose(A_tau4(p4, tau), A_tau3(p3, tau), atol=1e-12)


def test_sigma3_zero_reproduces_three_factor_system():
    p3 = ThreeFactorParams(kappa1=2.0, kappa2=0.4, sigma1=0.3, sigma2=0.15,
                           sigma_xi=0.18, rho1x=0.2, rho2x=-0.1, mu_xi=0.02,
                           mu_xi_star=0.01, lambda1=0.05, lambda2=0.02)
    p4 = FourFactorParams(kappa1=p3.kappa1, kappa2=p3.kappa2, kappa3=p3.kappa2 * 0.1,
                          sigma1=p3.sigma1, sigma2=p3.sigma2, sigma3=0.0,
                          sigma_xi=p3.sigma_xi, rho1x=p3.rho1x, rho2x=p3.rho2x,
                          rho3x=0.0, mu_xi=p3.mu_xi, mu_xi_star=p3.mu_xi_star,
                          lambda1=p3.lambda1, lambda2=p3.lambda2, lambda3=0.0)
    tau = np.tile(TAU18, (40, 1))
    s3, s4 = base_system3(p3, tau, DT), base_system4(p4, tau, DT)

    np.testing.assert_allclose(s4["d"], s3["d"], atol=1e-12)
    np.testing.assert_allclose(s4["Z"][:, :, 0], s3["Z"][:, :, 0], atol=1e-14)
    np.testing.assert_allclose(s4["Z"][:, :, 1], s3["Z"][:, :, 1], atol=1e-14)
    np.testing.assert_allclose(s4["Z"][:, :, 3], s3["Z"][:, :, 2], atol=1e-14)
    # the dormant third factor contributes no process noise in any direction
    np.testing.assert_allclose(s4["Q"][:, 2, :], 0.0, atol=1e-16)
    np.testing.assert_allclose(s4["Q"][:, :, 2], 0.0, atol=1e-16)
    np.testing.assert_allclose(s4["Q"][0, 0, 0], s3["Q"][0, 0, 0], atol=1e-16)
    np.testing.assert_allclose(s4["Q"][0, 1, 1], s3["Q"][0, 1, 1], atol=1e-16)
    np.testing.assert_allclose(s4["Q"][0, 3, 3], s3["Q"][0, 2, 2], atol=1e-16)
    np.testing.assert_allclose(s4["Q"][0, 0, 3], s3["Q"][0, 0, 2], atol=1e-16)
    np.testing.assert_allclose(s4["Q"][0, 1, 3], s3["Q"][0, 1, 2], atol=1e-16)


def test_A_tau4_is_zero_at_zero_maturity():
    assert A_tau4(_p4(), np.array([0.0])) == pytest.approx(0.0, abs=1e-14)


# ------------------------------------------------------ identification guards --

def test_label_switching_is_blocked():
    with pytest.raises(ValueError, match="label switching"):
        _p4(kappa1=0.4, kappa2=2.0, kappa3=12.0)
    with pytest.raises(ValueError, match="label switching"):
        _p4(kappa1=2.0, kappa2=2.0, kappa3=0.4)
    with pytest.raises(ValueError, match="label switching"):
        _p4(kappa1=12.0, kappa2=0.4, kappa3=0.4)


def test_star_correlation_psd_bound_is_the_sum_of_squares():
    """The exact bug that crashed the three-factor battery, generalised.

    Each rho_ix individually well inside (-1, 1) -- 0.7 each -- but jointly
    infeasible: 0.7^2 * 3 = 1.47 > 1. Must be rejected on the sum, not
    accepted because no single coordinate looks extreme.
    """
    with pytest.raises(ValueError, match="not PSD"):
        _p4(rho1x=0.7, rho2x=0.7, rho3x=0.7)
    _p4(rho1x=0.5, rho2x=0.5, rho3x=0.5)   # 0.75 <= 1, must be accepted
    _p4(rho1x=0.9, rho2x=0.0, rho3x=0.0)   # single large rho is fine alone


def test_loadings_separate_all_three_classes():
    Z = base_system4(_p4(), np.tile(TAU18, (5, 1)), DT)["Z"][0]
    ev = np.linalg.eigvalsh(Z.T @ Z)
    assert ev[0] > 1e-4, "four-factor loadings are collinear on this grid"


def test_priors_are_stationary_for_all_three_reverting_factors():
    p = _p4()
    tau = np.tile(TAU18, (3, 1))
    x0, P0 = initial_state4(np.full((3, 18), np.log(70.0)), tau, p)
    assert x0[0] == 0.0 and x0[1] == 0.0 and x0[2] == 0.0
    assert P0[0, 0] == pytest.approx(p.sigma1 ** 2 / (2 * p.kappa1))
    assert P0[1, 1] == pytest.approx(p.sigma2 ** 2 / (2 * p.kappa2))
    assert P0[2, 2] == pytest.approx(p.sigma3 ** 2 / (2 * p.kappa3))


def test_warm_start_from_three_factor_orders_and_uses_the_acf_rate():
    p3 = ThreeFactorParams(kappa1=1.887, kappa2=0.389, sigma1=0.2154,
                           sigma2=0.3297, sigma_xi=0.2042)
    p4 = from_three_factor(p3, kappa_fast=12.3214)
    assert p4.kappa1 == pytest.approx(12.3214)
    assert p4.kappa2 == pytest.approx(1.887)
    assert p4.kappa3 == pytest.approx(0.389)
    assert p4.kappa1 > p4.kappa2 > p4.kappa3


def test_warm_start_rejects_a_fast_rate_below_the_old_kappa1():
    p3 = ThreeFactorParams(kappa1=1.887, kappa2=0.389, sigma1=0.2154,
                           sigma2=0.3297, sigma_xi=0.2042)
    with pytest.raises(ValueError):
        from_three_factor(p3, kappa_fast=1.0)
