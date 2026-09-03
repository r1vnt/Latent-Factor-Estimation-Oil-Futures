"""Tests for the AR(1) pricing-error state (Runbook Step 1, S0P).

test_sigma_p_zero_reproduces_s0_exactly is the gating assertion: if it
fails, every LR statistic built on S0P downstream is void, and nothing
else in the runbook is meaningful. It runs first in this file and every
other test in the module depends on this one having already passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.twofactor import base_system, initial_state, scale_variances  # noqa: E402
from csr.simulate import SS_OIL, simulate_panel                                 # noqa: E402
from csr.state.augment import add_pricing_error_prior, add_pricing_error_state  # noqa: E402
from csr.state.fast import filter_uv                                            # noqa: E402

DT = 1.0 / 252.0


def _s0_filter(p, log_f, tau):
    """Plain 2-dim S0 filter, no pricing-error state -- the reference."""
    sys = scale_variances(base_system(p, tau, DT), p)
    x0, P0 = initial_state(log_f, tau, p)
    n = sys["n"]
    Bu = np.zeros((n, 2))
    return filter_uv(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys["Z"]),
        np.ascontiguousarray(sys["d"]), np.ascontiguousarray(sys["T"]),
        np.ascontiguousarray(np.broadcast_to(sys["c"], (n, 2))),
        np.ascontiguousarray(sys["Q"] if sys["Q"].ndim == 3
                            else np.repeat(sys["Q"][None], n, 0)),
        np.ascontiguousarray(sys["Hdiag"]), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0), np.ascontiguousarray(P0),
    )


def _s0p_filter(p, log_f, tau, phi, sigma_p, contract_index=0):
    """3-dim S0P filter: base_system extended with the pricing-error state."""
    sys = scale_variances(base_system(p, tau, DT), p)
    sys_p = add_pricing_error_state(sys, phi, sigma_p, contract_index)
    x0, P0 = initial_state(log_f, tau, p)
    x0p, P0p = add_pricing_error_prior(x0, P0, phi, sigma_p)
    n = sys_p["n"]
    Bu = np.zeros((n, 3))
    return filter_uv(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys_p["Z"]),
        np.ascontiguousarray(sys_p["d"]), np.ascontiguousarray(sys_p["T"]),
        np.ascontiguousarray(sys_p["c"]),
        np.ascontiguousarray(sys_p["Q"]),
        np.ascontiguousarray(sys_p["Hdiag"]), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0p), np.ascontiguousarray(P0p),
    )


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(SS_OIL, n=900, m=4, seed=101)


# ---------------------------------------------------------- THE gate test
def test_sigma_p_zero_reproduces_s0_exactly(panel):
    """sigma_p = 0 must reproduce plain S0's likelihood, filtered states,
    and n_obs exactly (to 1e-9) -- p_t is deterministically 0 for all t
    with no process noise and a point-mass prior, so it must contribute
    nothing to any observation regardless of phi's value. If this fails,
    every LR statistic in the runbook is void."""
    log_f, tau, _ = panel
    ll_ref, xf_ref, Pf_ref, xp_ref, Pp_ref, resid_ref, n_ref = _s0_filter(SS_OIL, log_f, tau)

    for phi in (0.0, 0.5, 0.9, -0.3):   # phi must not matter when sigma_p=0
        ll_p, xf_p, Pf_p, xp_p, Pp_p, resid_p, n_p = _s0p_filter(
            SS_OIL, log_f, tau, phi=phi, sigma_p=0.0)

        assert abs(ll_p - ll_ref) < 1e-9, f"phi={phi}: loglik diverged"
        assert n_p == n_ref
        assert np.allclose(xf_p[:, :2], xf_ref, atol=1e-9), f"phi={phi}: chi/xi diverged"
        assert np.allclose(xf_p[:, 2], 0.0, atol=1e-12), f"phi={phi}: p_t not exactly 0"
        np.testing.assert_allclose(resid_p, resid_ref, atol=1e-9, equal_nan=True)


def test_sigma_p_zero_reproduces_s0_at_different_contract_indices(panel):
    """The nesting property must hold regardless of which contract the
    (inert, sigma_p=0) pricing-error state is attached to."""
    log_f, tau, _ = panel
    ll_ref, *_ = _s0_filter(SS_OIL, log_f, tau)
    for idx in range(4):
        ll_p, *_ = _s0p_filter(SS_OIL, log_f, tau, phi=0.8, sigma_p=0.0,
                               contract_index=idx)
        assert abs(ll_p - ll_ref) < 1e-9


# ---------------------------------------------------- basic sanity, sigma_p > 0
def test_positive_sigma_p_changes_the_likelihood(panel):
    """The flip side of the nesting test: a genuinely nonzero sigma_p must
    actually change the likelihood (confirms the extension isn't a no-op
    for any sigma_p, only exactly at sigma_p=0)."""
    log_f, tau, _ = panel
    ll_ref, *_ = _s0_filter(SS_OIL, log_f, tau)
    ll_p, *_ = _s0p_filter(SS_OIL, log_f, tau, phi=0.9, sigma_p=0.01)
    assert abs(ll_p - ll_ref) > 1e-3


def test_pricing_error_state_only_loads_the_named_contract(panel):
    """Z's new column must be a one-hot vector on contract_index, zero on
    every other contract."""
    _, tau, _ = panel
    sys = scale_variances(base_system(SS_OIL, tau, DT), SS_OIL)
    sys_p = add_pricing_error_state(sys, phi=0.8, sigma_p=0.01, contract_index=2)
    loading = sys_p["Z"][:, :, -1]   # (n, m), the new state's column
    assert np.all(loading[:, 2] == 1.0)
    other = np.delete(loading, 2, axis=1)
    assert np.all(other == 0.0)


def test_pricing_error_prior_variance_matches_stationary_formula():
    x0 = np.array([0.1, 4.0])
    P0 = np.diag([0.01, 1.0])
    phi, sigma_p = 0.9, 0.02
    x_new, P_new = add_pricing_error_prior(x0, P0, phi, sigma_p)
    assert x_new[-1] == 0.0
    expected_var = sigma_p ** 2 / (1 - phi ** 2)
    assert P_new[-1, -1] == pytest.approx(expected_var)
    assert np.allclose(P_new[:2, :2], P0)
    assert np.allclose(P_new[:2, 2], 0.0) and np.allclose(P_new[2, :2], 0.0)


def test_pricing_error_state_generic_over_input_dimension():
    """Must work identically whether appended to a raw 2-dim system or an
    already-augmented 4-dim one -- the composability Step 4 depends on."""
    _, tau, _ = simulate_panel(SS_OIL, n=50, m=4, seed=3)
    sys2 = scale_variances(base_system(SS_OIL, tau, DT), SS_OIL)
    sys3 = add_pricing_error_state(sys2, phi=0.7, sigma_p=0.01)
    assert sys3["T"].shape == (3, 3)
    assert sys3["Z"].shape[-1] == 3

    from csr.state.augment import augment_system
    sys4 = augment_system(sys2)
    sys5 = add_pricing_error_state(sys4, phi=0.7, sigma_p=0.01)
    assert sys5["T"].shape == (5, 5)
    assert sys5["Z"].shape[-1] == 5
    # the pricing-error block sits LAST, not interleaved with the news block
    assert sys5["T"][4, 4] == pytest.approx(0.7)
    assert np.all(sys5["T"][4, :4] == 0.0)
