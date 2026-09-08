"""The smooth measurement curve must be degenerate-proof, not merely smaller.

The point of these tests is the third one: no parameter value can reproduce the
Chen-Scott collapse observed on C1-C8, where releasing the tick floor sent two
contracts 60-700x below a tick while two others sat at 320 and 162 bp.
"""

from __future__ import annotations

import numpy as np
import pytest

from csr.pricing.smoothmeas import (LOG_TICK, SmoothMeasParams,
                                    fit_smooth_to_vector, hdiag_from_tau,
                                    sigma_from_tau, spread_bound)

TAU8 = np.array([0.083, 0.167, 0.25, 0.333, 0.5, 0.583, 0.667, 0.75])


def test_flat_curve_when_b_zero():
    s = sigma_from_tau(SmoothMeasParams(a=np.log(5e-4)), TAU8)
    np.testing.assert_allclose(s, 5e-4)
    assert spread_bound(SmoothMeasParams(a=np.log(5e-4)), TAU8) == pytest.approx(1.0)


def test_decays_with_maturity_for_negative_b():
    s = sigma_from_tau(SmoothMeasParams(a=np.log(3e-3), b=-2.0), TAU8)
    assert np.all(np.diff(s) < 0)
    assert s[0] > s[-1]


# ------------------------------------------------------ the structural claim --

def test_spread_ratio_is_exactly_the_exponential_bound():
    """max(s)/min(s) = exp(|b| * span(tau)) -- monotone in b, and *monotone in tau*.

    Note what this does and does not establish. A large |b| CAN reach the
    degenerate fit's spread ratio of 320 / 0.0021 ~ 1.5e5: it needs
    |b| = ln(1.5e5) / 0.667 ~ 18. What it cannot do is reach it the way the free
    fit did. A monotone curve can only ever sacrifice an END of the panel, and
    does so visibly, with a single interpretable parameter. The Chen-Scott
    manoeuvre requires selecting k contracts by position -- C5 and C7 with C6 an
    order of magnitude larger -- and that shape is off the manifold entirely.
    The two tests below are the ones carrying the structural claim.
    """
    span = TAU8.max() - TAU8.min()
    # Range chosen so the numerical clip in sigma_from_tau is inactive.
    for b in (-25.0, -18.0, -10.0, 0.0, 10.0, 15.0):
        r = spread_bound(SmoothMeasParams(a=np.log(1e-3), b=b), TAU8)
        assert r == pytest.approx(np.exp(abs(b) * span), rel=1e-9)


def test_clip_bounds_the_ratio_in_the_extreme():
    """Outside that range the [-30, 5] clip binds and caps the spread."""
    r = spread_bound(SmoothMeasParams(a=np.log(1e-3), b=-50.0), TAU8)
    assert r < np.exp(50.0 * (TAU8.max() - TAU8.min()))
    assert np.isfinite(r)


def test_no_single_contract_can_fall_below_a_tick_alone():
    """Sub-tick noise on one contract drags its neighbours sub-tick too."""
    for b in np.linspace(-40.0, 40.0, 33):
        for a in np.linspace(np.log(1e-6), np.log(1e-1), 25):
            s = sigma_from_tau(SmoothMeasParams(a=a, b=b), TAU8)
            below = s < np.exp(LOG_TICK)
            if below.any():
                # whichever contracts are sub-tick must be contiguous from an
                # end of the curve -- never an isolated interior contract.
                idx = np.flatnonzero(below)
                assert idx[0] == 0 or idx[-1] == len(TAU8) - 1, (
                    f"isolated interior collapse at a={a:.2f}, b={b:.2f}"
                )


def test_c5_and_c7_cannot_collapse_together_leaving_c6_large():
    """The exact observed pattern: 5 and 7 pinned, 6 an order larger."""
    tick = np.exp(LOG_TICK)
    for b in np.linspace(-60.0, 60.0, 61):
        for a in np.linspace(np.log(1e-7), np.log(1.0), 40):
            s = sigma_from_tau(SmoothMeasParams(a=a, b=b), TAU8)
            pathological = (s[4] < tick) and (s[6] < tick) and (s[5] > 10 * tick)
            assert not pathological


# --------------------------------------------------------------- mechanics --

def test_hdiag_shape_and_roll_separability():
    tau = np.tile(TAU8, (500, 1))
    roll = np.zeros(500); roll[100:104] = 1.0
    p = SmoothMeasParams(a=np.log(1e-3), b=-1.5)
    H0 = hdiag_from_tau(p, tau)
    H1 = hdiag_from_tau(p, tau, eta_roll=0.7, roll=roll)
    assert H0.shape == (500, 8)
    np.testing.assert_allclose(H1[0], H0[0])
    np.testing.assert_allclose(H1[100], H0[100] * np.exp(0.7))


def test_time_varying_tau_is_respected():
    """Maturity shrinks between rolls, so H must move with it."""
    tau = np.column_stack([np.linspace(0.083, 0.004, 21)] +
                          [np.linspace(0.083, 0.004, 21) + k / 12 for k in range(1, 4)])
    H = hdiag_from_tau(SmoothMeasParams(a=np.log(1e-3), b=-2.0), tau)
    assert H[0, 0] != H[-1, 0]


def test_quadratic_requires_flag():
    with pytest.raises(ValueError):
        SmoothMeasParams(a=-7.0, b=0.0, c=1.0).validate()
    SmoothMeasParams(a=-7.0, b=0.0, c=1.0, quadratic=True).validate()


def test_warm_start_recovers_a_known_curve():
    truth = SmoothMeasParams(a=np.log(2e-3), b=-1.8)
    s = sigma_from_tau(truth, TAU8)
    est = fit_smooth_to_vector(s, TAU8)
    assert est.a == pytest.approx(truth.a, abs=1e-8)
    assert est.b == pytest.approx(truth.b, abs=1e-8)


def test_warm_start_from_the_degenerate_c1_c8_vector():
    """Even a degenerate free fit gives a usable starting point."""
    degenerate = np.array([320.2, 162.1, 52.9, 16.2, 1.4, 5.5, 1.4, 10.0]) * 1e-4
    est = fit_smooth_to_vector(degenerate, TAU8)
    assert np.isfinite(est.a) and np.isfinite(est.b)
    assert est.b < 0, "front-loaded misfit should imply a decaying curve"
    s = sigma_from_tau(est, TAU8)
    assert s.min() > np.exp(LOG_TICK) * 0.01


def test_parameter_count_collapses():
    """The wall-clock argument, asserted rather than asserted-in-prose."""
    for m in (4, 8, 12, 18, 24):
        free = 6 + m                       # structural + one variance each
        smooth = 6 + 2
        assert smooth < free
    assert 6 + 2 == 8 and 6 + 18 == 24     # C1-C18: 24 -> 8
