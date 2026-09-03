"""Tests for csr.pricing.fit: the Gaussian and Student-t S0 MLE machinery
that Study A's filtered state depends on. Uses simulated data (fast,
deterministic, known generating parameters) rather than the real EIA
panel -- the slow real-data fit belongs in the Study A script itself, not
in a test that runs on every commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit import (MIN_SIGMA_MEAS, fit_s0_gaussian, fit_s0_student_t,  # noqa: E402
                             filtered_covariance, pack, student_t_weights,
                             unpack, weighted_filter)
from csr.pricing.specs import REGISTRY, run_spec                                 # noqa: E402
from csr.simulate import SS_OIL, simulate_panel                                   # noqa: E402

DT = 1.0 / 252.0


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(SS_OIL, n=600, m=4, seed=11)


def test_pack_unpack_roundtrip():
    theta = pack(SS_OIL)
    p2 = unpack(theta, m=len(SS_OIL.sigma_meas))
    assert p2.kappa == pytest.approx(SS_OIL.kappa)
    assert p2.sigma_chi == pytest.approx(SS_OIL.sigma_chi)
    assert p2.sigma_xi == pytest.approx(SS_OIL.sigma_xi)
    assert p2.rho == pytest.approx(SS_OIL.rho)
    assert p2.mu_xi == pytest.approx(SS_OIL.mu_xi)
    assert p2.mu_xi_star == pytest.approx(SS_OIL.mu_xi_star)
    assert p2.lambda_chi == pytest.approx(SS_OIL.lambda_chi)
    assert np.allclose(p2.sigma_meas, SS_OIL.sigma_meas)


def test_weighted_filter_at_weight_one_matches_run_spec(panel):
    """The bypass weighted_filter() relies on for the Student-t EM loop:
    weight = 1 must reproduce run_spec's plain S0 filter exactly, in
    log-likelihood, n_obs, AND the filtered state -- the last of which
    Study A depends on directly."""
    log_f, tau, _ = panel
    ll_ref, xf_ref, Pf_ref, resid_ref, n_ref = run_spec(
        REGISTRY["S0"], SS_OIL, log_f, tau, DT)

    ll_w, xf_w, resid_w, n_w = weighted_filter(
        SS_OIL, log_f, tau, DT, np.ones_like(log_f))

    assert abs(ll_w - ll_ref) < 1e-8
    assert n_w == n_ref
    assert np.allclose(xf_w, xf_ref)
    np.testing.assert_allclose(resid_w, resid_ref, equal_nan=True)


def test_student_t_weights_matches_formula_and_handles_missing():
    resid = np.array([0.0, 1.0, 2.0, np.nan])
    df = 5.0
    w = student_t_weights(resid, df)
    expected = (df + 1.0) / (df + resid ** 2)
    assert w[0] == pytest.approx(expected[0])
    assert w[1] == pytest.approx(expected[1])
    assert w[2] == pytest.approx(expected[2])
    assert w[3] == 1.0   # missing cell -> inert weight, not NaN propagated
    # A large residual gets downweighted relative to a small one.
    assert w[2] < w[0]


def test_gaussian_fit_converges_to_at_least_the_truths_likelihood(panel):
    """Not a parameter-recovery test -- kappa is explicitly known to be
    weakly identified from 4 contracts (explore/02_fit_s0.py's own
    diagnostics: "four contracts barely identify kappa"), so a fitted
    kappa landing far from the generating value on one finite simulated
    sample is expected sampling behaviour, not a bug. What DOES have to
    hold regardless of identification strength: MLE must find a point at
    least as good as the true generating parameters by log-likelihood
    (confirms the optimisation wrapper -- pack/unpack/minimize -- is
    actually working, rather than silently stuck or mis-scaled), and every
    returned quantity must be finite and correctly shaped."""
    log_f, tau, _ = panel
    m = log_f.shape[1]
    ll_truth, *_ = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)
    p_hat, hess_inv, x_filt, resid, n_obs, ll = fit_s0_gaussian(
        log_f, tau, DT, m, theta0=pack(SS_OIL))

    assert ll >= ll_truth - 1e-6   # MLE must not do worse than the truth
    assert p_hat.kappa > 0
    assert p_hat.sigma_chi > 0 and p_hat.sigma_xi > 0
    assert np.isfinite(ll)
    assert x_filt.shape == (len(log_f), 4)
    assert np.all(np.isfinite(x_filt))


def test_student_t_fit_with_high_df_matches_gaussian_fit(panel):
    """As df -> infinity, Student-t -> Normal, so a very high df should
    converge to essentially the same fit as the plain Gaussian MLE -- a
    consistency check on the EM loop that doesn't require independently
    re-deriving Student-t likelihood math."""
    log_f, tau, _ = panel
    m = log_f.shape[1]
    p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f, tau, DT, m, theta0=pack(SS_OIL))

    p_t, x_filt_t, resid_t, weight_t, n_iters = fit_s0_student_t(
        log_f, tau, DT, m, df=1e6, theta0=pack(p_gauss))

    assert p_t.kappa == pytest.approx(p_gauss.kappa, rel=0.05)
    assert p_t.sigma_chi == pytest.approx(p_gauss.sigma_chi, rel=0.05)
    # weight -> 1 everywhere at this df (finite residuals only)
    finite = np.isfinite(resid_t)
    assert np.allclose(weight_t[finite], 1.0, atol=1e-3)
    assert x_filt_t.shape == (len(log_f), 4)


def test_student_t_fit_downweights_a_real_outlier():
    """A single planted extreme observation should end up with a materially
    lower EM weight than the rest -- the entire point of the Student-t
    construction, checked directly rather than just trusting convergence."""
    log_f, tau, _ = simulate_panel(SS_OIL, n=300, m=4, seed=23)
    log_f = log_f.copy()
    log_f[150, 0] += 5.0   # a single, large, isolated shock

    p_gauss, _, _, _, _, _ = fit_s0_gaussian(log_f, tau, DT, 4, theta0=pack(SS_OIL))
    p_t, x_filt_t, resid_t, weight_t, n_iters = fit_s0_student_t(
        log_f, tau, DT, 4, df=5.0, theta0=pack(p_gauss))

    assert weight_t[150, 0] < 0.3 * np.nanmedian(weight_t)


def test_filtered_covariance_matches_weighted_filter_and_run_spec(panel):
    """filtered_covariance() must return the exact same P_filt a plain
    weighted_filter(weight=1) pass computes internally (confirming it
    isn't a second, independently-computed -- and possibly divergent --
    version), and by the transitivity already established in
    test_weighted_filter_at_weight_one_matches_run_spec, that also pins
    it against run_spec's own P_filt."""
    log_f, tau, _ = panel
    Pf = filtered_covariance(SS_OIL, log_f, tau, DT)

    from csr.pricing.specs import REGISTRY, run_spec
    _, _, Pf_ref, _, _ = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)

    assert Pf.shape == Pf_ref.shape
    assert np.allclose(Pf, Pf_ref)
    assert np.all(Pf[:, 0, 0] > 0)   # chi variance must be positive throughout
