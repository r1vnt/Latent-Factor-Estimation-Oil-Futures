"""Tests for csr.pricing.fit3 -- the three-factor smooth-curve fitting loop.

fit3.py has been relied on all session via ad hoc smoke-test scripts, not
committed tests. This file is deliberately narrow: it locks in the
pack3_smooth/unpack3_smooth round trip and the RV-extension nesting
guarantee (Step 3 of the volatility-clustering follow-up: all new
parameters at zero must reproduce the current fit to 1e-9), not a full
re-test of the Gaussian/Student-t fitting loops themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.fit3 import (  # noqa: E402
    _neg_loglik3,
    _neg_t_loglik_df,
    fit3_student_t_smooth,
    pack3_smooth,
    unpack3_smooth,
    weighted_filter3,
)
from csr.pricing.threefactor import ThreeFactorParams  # noqa: E402


def _p3():
    return ThreeFactorParams(kappa1=1.9, kappa2=0.4, sigma1=0.22, sigma2=0.31,
                             sigma_xi=0.2, rho1x=0.1, rho2x=0.15,
                             mu_xi=0.0, mu_xi_star=0.01, lambda1=0.05, lambda2=0.02,
                             gamma_v=0.3)


def _panel(seed=0, n=200, m=6):
    rng = np.random.default_rng(seed)
    tau = np.tile(np.linspace(0.08, 1.0, m), (n, 1)) + rng.normal(0, 0.001, (n, m))
    log_f = 70.0 + np.cumsum(rng.normal(0, 0.01, n)).reshape(-1, 1) + rng.normal(0, 0.001, (n, m))
    return log_f, np.abs(tau)


# ------------------------------------------------------- pack/unpack ------
def test_pack_unpack_roundtrip_without_rv_extension():
    p = _p3()
    theta = pack3_smooth(p, a=-4.0, b=-2.0, c=1.5, quadratic=True)
    p2, smooth2 = unpack3_smooth(theta, quadratic=True, rv_extension=False)
    assert p2.kappa1 == pytest.approx(p.kappa1)
    assert p2.kappa2 == pytest.approx(p.kappa2)
    assert p2.gamma_v == pytest.approx(p.gamma_v)
    assert smooth2.a == pytest.approx(-4.0)
    assert smooth2.b == pytest.approx(-2.0)
    assert smooth2.c == pytest.approx(1.5)
    assert smooth2.eta_v == 0.0 and smooth2.lambda_v == 0.0


def test_pack_unpack_roundtrip_with_rv_extension():
    p = _p3()
    theta = pack3_smooth(p, a=-4.0, b=-2.0, c=1.5, quadratic=True,
                         eta_v=2.5, lambda_v=1.3, rv_extension=True)
    p2, smooth2 = unpack3_smooth(theta, quadratic=True, rv_extension=True)
    assert smooth2.eta_v == pytest.approx(2.5)
    assert smooth2.lambda_v == pytest.approx(1.3)
    assert p2.kappa1 == pytest.approx(p.kappa1)


def test_theta_length_grows_by_exactly_two_for_the_rv_extension():
    p = _p3()
    theta_off = pack3_smooth(p, a=-4.0, b=-2.0, c=1.5, quadratic=True, rv_extension=False)
    theta_on = pack3_smooth(p, a=-4.0, b=-2.0, c=1.5, quadratic=True,
                            eta_v=1.0, lambda_v=1.0, rv_extension=True)
    assert len(theta_on) == len(theta_off) + 2


# ------------------------------------------- Step 3: nesting at zero ------
def test_rv_extension_at_zero_reproduces_the_baseline_likelihood_exactly():
    """The core Step-3 requirement: eta_v = lambda_v = 0 (with rv_extension
    True, an rv_lag series actually supplied) must give the IDENTICAL
    negative log-likelihood as rv_extension=False entirely -- not just
    approximately, since exp(0 * anything) = 1 is an exact identity, and
    a numerical mismatch here would mean the wiring, not the maths, is
    wrong. gamma_v=0 here specifically, so that supplying rv_lag to one
    branch and not the other (True baseline: no RV signal anywhere) can't
    also activate the Q-side channel asymmetrically and contaminate this
    H-side-only check -- that interaction is covered separately below."""
    log_f, tau = _panel()
    p = _p3().replace(gamma_v=0.0)
    rng = np.random.default_rng(1)
    rv_lag = rng.normal(0, 1, len(log_f))

    theta_off = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True, rv_extension=False)
    theta_on = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True,
                            eta_v=0.0, lambda_v=0.0, rv_extension=True)

    nll_off = _neg_loglik3(theta_off, log_f, tau, 1 / 252, True, rv_lag=None, rv_extension=False)
    nll_on = _neg_loglik3(theta_on, log_f, tau, 1 / 252, True, rv_lag=rv_lag, rv_extension=True)
    assert nll_off == pytest.approx(nll_on, abs=1e-9)


def test_rv_extension_at_zero_reproduces_the_baseline_even_with_gamma_v_active():
    """Same guarantee, with the Q-side gamma_v channel ALSO active and fed
    the same rv_lag -- the two channels (Q via gamma_v, H via eta_v) must
    nest independently; eta_v/lambda_v=0 must not depend on whatever
    gamma_v is doing."""
    log_f, tau = _panel(seed=2)
    p = _p3()   # gamma_v = 0.3, nonzero
    rng = np.random.default_rng(3)
    rv_lag = rng.normal(0, 1, len(log_f))

    theta_off = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False, rv_extension=False)
    theta_on = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                            eta_v=0.0, lambda_v=0.0, rv_extension=True)

    nll_off = _neg_loglik3(theta_off, log_f, tau, 1 / 252, False, rv_lag=rv_lag, rv_extension=False)
    nll_on = _neg_loglik3(theta_on, log_f, tau, 1 / 252, False, rv_lag=rv_lag, rv_extension=True)
    assert nll_off == pytest.approx(nll_on, abs=1e-9)


def test_nonzero_eta_v_actually_changes_the_likelihood():
    """Guard against the extension being vacuous: a nonzero eta_v with a
    real rv_lag series must change the likelihood relative to eta_v=0."""
    log_f, tau = _panel(seed=4)
    p = _p3()
    rng = np.random.default_rng(5)
    rv_lag = rng.normal(0, 1, len(log_f))

    theta_zero = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                              eta_v=0.0, lambda_v=1.0, rv_extension=True)
    theta_nonzero = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                                 eta_v=2.0, lambda_v=1.0, rv_extension=True)

    nll_zero = _neg_loglik3(theta_zero, log_f, tau, 1 / 252, False, rv_lag=rv_lag, rv_extension=True)
    nll_nonzero = _neg_loglik3(theta_nonzero, log_f, tau, 1 / 252, False, rv_lag=rv_lag, rv_extension=True)
    assert abs(nll_zero - nll_nonzero) > 1e-6


# ------------------------------------------- Fix 4: bounded, not clipped --
def test_gamma_v_and_eta_v_are_bounded_by_construction_not_by_clipping():
    """No finite theta value can push gamma_v or eta_v outside their
    economic bounds -- tanh saturates, it doesn't overflow, so there is
    no plateau for an optimiser to run into and park on. This is the
    actual fix for the runaway multi-start explore/44 hit: the ±20
    exponent clip was a numerical guard standing in for an economic
    constraint, and bounding the PARAMETER itself removes that plateau
    rather than relocating it."""
    from csr.pricing.fit3 import ETA_V_MAX, GAMMA_V_MAX

    p = _p3().replace(gamma_v=2.9)   # near the bound already
    for huge in (10.0, 100.0, 1e6):
        theta = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                             eta_v=huge, lambda_v=1.0, rv_extension=True)
        p_out, smooth_out = unpack3_smooth(theta, quadratic=False, rv_extension=True)
        assert abs(smooth_out.eta_v) < ETA_V_MAX
        assert abs(p_out.gamma_v) < GAMMA_V_MAX


def test_binding_constraint_is_detectable_from_the_fitted_value():
    """A caller must be able to tell 'estimate' from 'parked at the
    bound' just from the returned value -- ratio to MAX close to 1 is
    the tell the acceptance step checks for explicitly."""
    from csr.pricing.fit3 import ETA_V_MAX

    p = _p3()
    theta_at_bound = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                                  eta_v=ETA_V_MAX * 0.999, lambda_v=1.0, rv_extension=True)
    _, smooth_at_bound = unpack3_smooth(theta_at_bound, quadratic=False, rv_extension=True)
    assert abs(smooth_at_bound.eta_v) / ETA_V_MAX > 0.95

    theta_interior = pack3_smooth(p, a=-4.0, b=-2.0, quadratic=False,
                                  eta_v=0.5, lambda_v=1.0, rv_extension=True)
    _, smooth_interior = unpack3_smooth(theta_interior, quadratic=False, rv_extension=True)
    assert abs(smooth_interior.eta_v) / ETA_V_MAX < 0.5


# ------------------------------------------- essentially-affine extension --
def test_ea_extension_pack_unpack_roundtrip():
    p = _p3().replace(kappa1_phys=1.2, kappa2_phys=0.25)
    theta = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=True)
    p2, _ = unpack3_smooth(theta, quadratic=True, ea_extension=True)
    assert p2.kappa1_phys == pytest.approx(1.2)
    assert p2.kappa2_phys == pytest.approx(0.25)
    assert p2.kappa1 == pytest.approx(p.kappa1)   # kappa^Q unaffected
    assert p2.kappa2 == pytest.approx(p.kappa2)


def test_theta_length_grows_by_exactly_two_for_ea_extension():
    p = _p3()
    theta_off = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=False)
    theta_on = pack3_smooth(p.replace(kappa1_phys=1.0, kappa2_phys=0.3), a=-4.0, b=-2.0, c=1.0,
                            quadratic=True, ea_extension=True)
    assert len(theta_on) == len(theta_off) + 2


def test_ea_extension_at_kappa_phys_equals_kappa_reproduces_baseline_exactly():
    """The Step-3-style nesting guarantee, now for the market-price-of-risk
    restriction: kappa_phys = kappa (completely affine) must reproduce the
    exact likelihood of the model fit without this extension at all -- not
    approximately, since kappa_phys defaulting to kappa is an EXACT
    reparameterisation (T/Q's kappa1_phys/kappa2_phys are literally the
    same floats as kappa1/kappa2 in this case), not a limiting case."""
    log_f, tau = _panel()
    p = _p3().replace(gamma_v=0.0)   # isolate from the RV channel, same discipline as rv_extension's own nesting test

    theta_off = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=False)
    p_ea = p.replace(kappa1_phys=p.kappa1, kappa2_phys=p.kappa2)
    theta_on = pack3_smooth(p_ea, a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=True)

    nll_off = _neg_loglik3(theta_off, log_f, tau, 1 / 252, True, ea_extension=False)
    nll_on = _neg_loglik3(theta_on, log_f, tau, 1 / 252, True, ea_extension=True)
    assert nll_off == pytest.approx(nll_on, abs=1e-9)


def test_nonequal_kappa_phys_actually_changes_the_likelihood():
    """Guard against the extension being vacuous."""
    log_f, tau = _panel(seed=6)
    p = _p3().replace(gamma_v=0.0)

    theta_equal = pack3_smooth(p.replace(kappa1_phys=p.kappa1, kappa2_phys=p.kappa2),
                               a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=True)
    theta_diff = pack3_smooth(p.replace(kappa1_phys=p.kappa1 * 0.3, kappa2_phys=p.kappa2 * 0.3),
                              a=-4.0, b=-2.0, c=1.0, quadratic=True, ea_extension=True)

    nll_equal = _neg_loglik3(theta_equal, log_f, tau, 1 / 252, True, ea_extension=True)
    nll_diff = _neg_loglik3(theta_diff, log_f, tau, 1 / 252, True, ea_extension=True)
    assert abs(nll_equal - nll_diff) > 1e-6


# --------------------------------------------- joint nu estimation (CM-step 2) --
def test_neg_t_loglik_df_recovers_the_true_df_from_simulated_t_data():
    """The CM-step 2 objective is exactly sum(log t_pdf(z; df)) -- confirm
    it actually recovers a known df from simulated data, not just that it
    runs. n=8000 keeps sampling noise well below the 1.0 tolerance."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import t as student_t_dist

    rng = np.random.default_rng(42)
    z = student_t_dist.rvs(df=3.0, size=8000, random_state=rng)
    res = minimize_scalar(_neg_t_loglik_df, bounds=(2.1, 100.0), method="bounded", args=(z,))
    assert abs(res.x - 3.0) < 1.0


def test_neg_t_loglik_df_pushes_toward_the_bound_for_gaussian_data():
    """Gaussian data (zero excess kurtosis) has no finite best-fit df --
    the CM-step should push the estimate toward the upper bound, the
    opposite of the fat-tailed case above."""
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(7)
    z = rng.normal(size=8000)
    res = minimize_scalar(_neg_t_loglik_df, bounds=(2.1, 100.0), method="bounded", args=(z,))
    assert res.x > 50.0


def test_fit3_student_t_smooth_estimate_df_false_keeps_df_byte_for_byte_fixed():
    """Default behaviour (estimate_df=False) must reproduce every existing
    caller's assumption that df never moves -- the 8th return value must
    equal the input df exactly, not just approximately."""
    log_f, tau = _panel(seed=10)
    p = _p3()
    theta0 = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True)
    df_start = 3.0
    *_, df_out = fit3_student_t_smooth(log_f, tau, 1 / 252, df_start, theta0,
                                       quadratic=True, n_outer=5)
    assert df_out == df_start


def test_fit3_student_t_smooth_estimate_df_true_moves_df_on_gaussian_data():
    """Wiring check on the real EM loop (not just the standalone helper):
    starting from a deliberately low df on Gaussian-simulated data, the
    estimate must move -- and move UP, since Gaussian residuals have no
    finite best-fit df."""
    log_f, tau = _panel(seed=11)
    p = _p3()
    theta0 = pack3_smooth(p, a=-4.0, b=-2.0, c=1.0, quadratic=True)
    df_start = 3.0
    *_, df_out = fit3_student_t_smooth(log_f, tau, 1 / 252, df_start, theta0,
                                       quadratic=True, n_outer=20, estimate_df=True)
    assert df_out > df_start + 1.0
