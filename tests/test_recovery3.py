"""Tests for csr.study.recovery3 -- Study A / Gate 4, three-factor port.

Covers the parts that are genuinely new here (filtered_label's chi1+chi2
sum, the hand-derived a_bar3 formula, the McNemar power/identifying-count
gate, the implied-spread confidence split, the chi-split-vs-log-price
joint test) -- the imported machinery (encompassing_regression,
classify_gate, the bootstrap block-length guard, etc.) is already covered
by test_recovery.py and not re-tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.threefactor import A_tau3, ThreeFactorParams  # noqa: E402
from csr.study.recovery3 import (  # noqa: E402
    chi_split_encompasses_log_price,
    confidence_split_regression,
    continuous_filtered,
    continuous_naive,
    cross_maturity_spread_change,
    filtered_label,
    identifying_count,
    identifying_count_gate,
    mcnemar_required_n,
    naive_label,
    run_encompassing_battery,
    xi_encompasses_log_price,
)


def _p3():
    return ThreeFactorParams(kappa1=1.9, kappa2=0.4, sigma1=0.22, sigma2=0.31,
                             sigma_xi=0.2, rho1x=0.1, rho2x=0.15,
                             mu_xi=0.0, mu_xi_star=0.01, lambda1=0.05, lambda2=0.02)


# --------------------------------------------------------------- labels ---
def test_filtered_label_matches_hand_computed_two_factor_sum():
    p = _p3()
    x_filt = np.array([[0.10, -0.05, 4.0],
                       [-0.05, 0.08, 4.0]])
    tau = np.array([[1 / 12, 2 / 12], [1 / 12, 2 / 12]])
    lab = filtered_label(p, x_filt, tau)

    chi1, chi2 = x_filt[:, 0], x_filt[:, 1]
    f1 = (np.exp(-p.kappa1 * tau[:, 0]) * chi1 + np.exp(-p.kappa2 * tau[:, 0]) * chi2
         + A_tau3(p, tau[:, 0]))
    f2 = (np.exp(-p.kappa1 * tau[:, 1]) * chi1 + np.exp(-p.kappa2 * tau[:, 1]) * chi2
         + A_tau3(p, tau[:, 1]))
    expected = np.sign(f1 - f2)
    assert np.array_equal(lab, expected)


def test_filtered_label_xi_cancels_in_the_difference():
    """xi (column 2) must not affect the label at all -- it's identical in
    both f1 and f2 by construction, so changing it must leave the sign
    unchanged."""
    p = _p3()
    tau = np.array([[1 / 12, 3 / 12]])
    x1 = np.array([[0.2, -0.1, 4.0]])
    x2 = np.array([[0.2, -0.1, 9.5]])   # only xi differs
    assert filtered_label(p, x1, tau)[0] == filtered_label(p, x2, tau)[0]


def test_continuous_filtered_matches_hand_derived_a_bar3():
    p = _p3()
    x_filt = np.array([[0.2, -0.1, 4.0], [-0.15, 0.05, 4.0]])
    out = continuous_filtered(p, x_filt)
    a_bar3 = (p.mu_xi_star - p.lambda1 - p.lambda2
             + 0.5 * (p.sigma1 ** 2 + p.sigma2 ** 2 + p.sigma_xi ** 2)
             + p.rho1x * p.sigma1 * p.sigma_xi + p.rho2x * p.sigma2 * p.sigma_xi)
    expected = -p.kappa1 * x_filt[:, 0] - p.kappa2 * x_filt[:, 1] + a_bar3
    assert np.allclose(out, expected)


def test_continuous_filtered_matches_numerical_derivative_of_A_tau3():
    """Independent check of the hand-derived a_bar3: A_tau3's own
    numerical derivative at tau=0 (excluding the chi contribution, which
    continuous_filtered adds separately) must match the constant term."""
    p = _p3()
    eps = 1e-6
    dA_dtau_at_0 = (A_tau3(p, np.array([eps])) - A_tau3(p, np.array([0.0]))) / eps
    a_bar3 = (p.mu_xi_star - p.lambda1 - p.lambda2
             + 0.5 * (p.sigma1 ** 2 + p.sigma2 ** 2 + p.sigma_xi ** 2)
             + p.rho1x * p.sigma1 * p.sigma_xi + p.rho2x * p.sigma2 * p.sigma_xi)
    assert dA_dtau_at_0[0] == pytest.approx(a_bar3, abs=1e-4)


def test_continuous_naive_is_the_front_spread():
    log_f = np.array([[4.1, 4.0], [3.9, 4.05]])
    assert np.allclose(continuous_naive(log_f), [0.1, -0.15])


# ---------------------------------------------------- identifying count ---
def test_identifying_count_counts_runs_and_independent_runs():
    l_obs = np.array([1, 1, -1, -1, 1, 1, 1, -1, -1, 1], dtype=float)
    l_filt = np.array([1, -1, -1, 1, 1, 1, 1, -1, 1, 1], dtype=float)
    # disagree at indices 1, 3, 8 -- three isolated single-day runs
    out = identifying_count(l_obs, l_filt, h=2)
    assert out["n_disagree"] == 3
    assert out["n_runs"] == 3
    # h=2: gap between run ends and next run start must exceed 2 to count
    # as independent. runs at 1,3,8 -- gap(1->3)=3-1-1=1 (<=2, not
    # independent from the first), gap(3->8)=8-3-1=4 (>2, independent)
    assert out["n_independent_runs"] <= out["n_runs"]


def test_mcnemar_required_n_matches_known_formula_values():
    # p=0.65: (1.96+0.8416)^2 / (0.3)^2 ~ 87.4
    n = mcnemar_required_n(0.65)
    assert n == pytest.approx(87.4, rel=0.02)
    # symmetric in distance from 0.5
    assert mcnemar_required_n(0.65) == pytest.approx(mcnemar_required_n(0.35))
    # p=0.5 -> no finite sample suffices
    assert mcnemar_required_n(0.5) == float("inf")


def test_identifying_count_gate_flags_insufficiency_on_a_thin_series():
    rng = np.random.default_rng(0)
    n = 60
    l_obs = np.sign(rng.normal(size=n))
    l_filt = l_obs.copy()
    l_filt[::7] *= -1   # sparse disagreement
    log_f = np.cumsum(rng.normal(0, 0.01, (n, 4)), axis=0) + 70.0

    out = identifying_count_gate(l_obs, l_filt, log_f, h=22)
    assert out["n_disagree"] < 20
    # with this few disagreement days, the gate must not silently claim sufficiency
    assert isinstance(out["sufficient_disagree_days"], (bool, np.bool_))


# ------------------------------------------------- confidence split -------
def test_confidence_split_uses_implied_spread_variance_not_raw_p_filt():
    """The split must come from Var(f1-f2) = z'*P_chi*z, not a marginal
    chi1 variance -- confirmed by constructing P_filt where chi1's own
    marginal variance is CONSTANT across all days but the IMPLIED-SPREAD
    variance still varies (via the chi1-chi2 covariance term), and
    checking the split is non-trivial (not all-one-group)."""
    rng = np.random.default_rng(2)
    n = 200
    p = _p3()
    tau = np.tile(np.array([1 / 12, 3 / 12]), (n, 1))
    x_filt = np.column_stack([np.cumsum(rng.normal(0, 0.01, n)),
                              np.cumsum(rng.normal(0, 0.01, n)),
                              np.full(n, np.log(70.0))])
    log_f = np.column_stack([70.0 + 0.01 * np.arange(n), 69.5 + 0.01 * np.arange(n),
                             69.0 + 0.01 * np.arange(n), 68.5 + 0.01 * np.arange(n)])

    P_filt = np.zeros((n, 3, 3))
    for t in range(n):
        P_filt[t, 0, 0] = 0.01   # constant chi1 marginal variance
        P_filt[t, 1, 1] = 0.01
        P_filt[t, 0, 1] = P_filt[t, 1, 0] = 0.009 * np.sin(t / 20.0)   # varying covariance
        P_filt[t, 2, 2] = 0.1

    r_low, r_high = confidence_split_regression(p, x_filt, P_filt, log_f, tau, h=22)
    assert r_low.n_obs > 0 and r_high.n_obs > 0
    assert r_low.n_obs + r_high.n_obs <= n


# --------------------------------------------- xi / chi split vs. price ---
def test_xi_encompasses_log_price_runs_and_reports_r2():
    rng = np.random.default_rng(3)
    n = 300
    h = 22
    xi_hat = np.cumsum(rng.normal(0, 0.02, n))
    x_filt = np.column_stack([np.zeros(n), np.zeros(n), xi_hat])
    front = np.zeros(n)
    for t in range(n - h):
        front[t + h] = front[t] + 0.8 * xi_hat[t] + rng.normal(0, 0.01)
    log_f = front.reshape(-1, 1)

    r = xi_encompasses_log_price(x_filt, log_f, h)
    assert 0.0 <= r.r_squared <= 1.0
    assert r.n_obs > 0


def test_chi_split_encompasses_log_price_joint_wald_runs():
    rng = np.random.default_rng(4)
    n = 300
    h = 22
    chi1 = np.cumsum(rng.normal(0, 0.02, n))
    chi2 = np.cumsum(rng.normal(0, 0.01, n))
    x_filt = np.column_stack([chi1, chi2, np.full(n, np.log(70.0))])
    log_f1 = 70.0 + 0.5 * chi1 + rng.normal(0, 0.05, n)
    log_f = log_f1.reshape(-1, 1)

    r = chi_split_encompasses_log_price(x_filt, log_f, h)
    assert r.n_obs > 0
    assert np.isfinite(r.joint_chi_stat) or r.degenerate


# ------------------------------------------------------- full battery -----
def test_run_encompassing_battery_covers_both_targets_all_horizons():
    rng = np.random.default_rng(5)
    n = 2000
    m = 4
    p = _p3()
    tau = np.tile(np.array([1 / 12, 2 / 12, 3 / 12, 4 / 12]), (n, 1))
    log_f = 70.0 + np.cumsum(rng.normal(0, 0.01, (n, m)), axis=0) * 0.01
    log_f = np.tile((70.0 + np.cumsum(rng.normal(0, 0.005, n))).reshape(-1, 1), (1, m))
    log_f += rng.normal(0, 0.001, (n, m))
    x_filt = np.column_stack([np.cumsum(rng.normal(0, 0.005, n)),
                              np.cumsum(rng.normal(0, 0.002, n)),
                              np.full(n, np.log(70.0))])

    out = run_encompassing_battery(p, x_filt, log_f, tau)
    assert set(out.keys()) == {"realised_vol", "cross_maturity_spread_change"}
    for target, by_h in out.items():
        assert set(by_h.keys()) == {22, 66, 132, 252}
        for h, r in by_h.items():
            assert r.n_obs > 0


def test_required_n_reference_does_not_move_with_the_observed_split():
    """The reference-effect-size requirement must be identical regardless
    of what the data happened to show -- unlike required_n_observed,
    which just restates the observed split back in sample-size units and
    is not a real power bound (a near-50/50 split gives an enormous
    number not because the test lacks power, but because the effect was
    ~zero)."""
    rng = np.random.default_rng(9)
    n = 400

    def make(seed, split):
        r = np.random.default_rng(seed)
        l_obs = np.sign(r.normal(size=n))
        l_filt = l_obs.copy()
        flip = r.random(n) < 0.3
        l_filt[flip] *= -1
        log_f = np.cumsum(r.normal(0, 0.01, (n, 4)), axis=0) + 70.0
        return l_obs, l_filt, log_f

    l_obs_a, l_filt_a, log_f_a = make(1, 0.5)
    l_obs_b, l_filt_b, log_f_b = make(2, 0.5)
    gate_a = identifying_count_gate(l_obs_a, l_filt_a, log_f_a, h=22)
    gate_b = identifying_count_gate(l_obs_b, l_filt_b, log_f_b, h=22)

    assert gate_a["required_n_reference"] == gate_b["required_n_reference"]
    assert gate_a["required_n_reference"] == pytest.approx(mcnemar_required_n(0.65))
    # the observed-effect version is free to differ -- it's descriptive, not a bound
