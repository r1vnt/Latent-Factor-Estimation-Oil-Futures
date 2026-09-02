"""Tests for csr.study.recovery -- Study A / Gate 4.

Covers the label construction (including the asymmetry that a missing
front-month price kills the naive label but not the filtered one -- the
mechanism Gate 4 is actually testing), the target variables' NaN handling,
that the regression's HAC standard errors are doing real work (not
silently falling back to understated OLS SEs under overlap), and every
branch of the pass/partial/fail/inconclusive gate classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.specs import REGISTRY, run_spec                    # noqa: E402
from csr.pricing.twofactor import TwoFactorParams                    # noqa: E402
from csr.simulate import SS_OIL, simulate_panel                       # noqa: E402
from csr.study.recovery import (RegressionResult, classify_gate,       # noqa: E402
                                continuous_filtered, continuous_naive,
                                cross_maturity_spread_change,
                                encompassing_regression, filtered_label,
                                forward_spread_level, front_return,
                                lag_labels, naive_label, realised_vol,
                                reversion_fraction, run_corrected_battery,
                                run_horizon_robustness, run_study_a,
                                spread_change)

DT = 1.0 / 252.0


# --------------------------------------------------------------- labels ---
def test_naive_label_sign_and_nan_propagation():
    log_f = np.array([
        [4.10, 4.05],   # backwardation: front > second
        [4.00, 4.08],   # contango
        [np.nan, 4.02],   # front missing -- must be NaN, not a guess
    ])
    lab = naive_label(log_f)
    assert lab[0] == 1.0
    assert lab[1] == -1.0
    assert np.isnan(lab[2])


def test_filtered_label_matches_hand_computed_discrete_form():
    p = SS_OIL.replace(kappa=1.5)
    x_filt = np.array([[0.10, 4.0, 0.0, 0.0],
                       [-0.05, 4.0, 0.0, 0.0]])
    tau = np.array([[1 / 12, 2 / 12],
                    [1 / 12, 2 / 12]])
    lab = filtered_label(p, x_filt, tau)

    from csr.pricing.twofactor import A_tau
    chi = x_filt[:, 0]
    f1 = np.exp(-p.kappa * tau[:, 0]) * chi + A_tau(p, tau[:, 0])
    f2 = np.exp(-p.kappa * tau[:, 1]) * chi + A_tau(p, tau[:, 1])
    expected = np.sign(f1 - f2)
    assert np.array_equal(lab, expected)


def test_filtered_label_survives_a_missing_price_the_naive_label_cannot():
    """The concrete mechanism Gate 4 tests: a Kalman filter's predict step
    runs every period regardless of that period's observations, so a day
    with NO price reading at all still has a finite filtered chi_hat -- the
    filtered label can form a view where the naive one has nothing to read."""
    log_f, tau, _ = simulate_panel(SS_OIL, n=100, m=4, seed=5)
    log_f = log_f.copy()
    log_f[50, 0] = np.nan   # front-month missing, like 2020-04-20

    ll, x_filt, Pf, resid, n_obs = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)
    l_obs = naive_label(log_f)
    l_filt = filtered_label(SS_OIL, x_filt, tau)

    assert np.isnan(l_obs[50])
    assert np.isfinite(l_filt[50])
    assert l_filt[50] in (-1.0, 1.0)


def test_continuous_filtered_matches_a_bar_formula():
    p = SS_OIL
    x_filt = np.array([[0.2, 4.0, 0.0, 0.0], [-0.1, 4.0, 0.0, 0.0]])
    out = continuous_filtered(p, x_filt)
    a_bar = (p.mu_xi_star - p.lambda_chi
            + 0.5 * (p.sigma_chi ** 2 + p.sigma_xi ** 2
                    + 2 * p.rho * p.sigma_chi * p.sigma_xi))
    expected = -p.kappa * x_filt[:, 0] + a_bar
    assert np.allclose(out, expected)


def test_continuous_naive_is_the_front_spread():
    log_f = np.array([[4.1, 4.0], [3.9, 4.05]])
    assert np.allclose(continuous_naive(log_f), [0.1, -0.15])


# -------------------------------------------------------------- targets ---
def test_spread_change_hand_computed():
    log_f = np.array([[4.0, 3.9], [4.1, 3.8], [4.2, 3.7], [4.0, 3.9]])
    # s = [0.1, 0.3, 0.5, 0.1]
    y = spread_change(log_f, h=2)
    assert y[0] == pytest.approx(0.5 - 0.1)
    assert y[1] == pytest.approx(0.1 - 0.3)
    assert np.isnan(y[2]) and np.isnan(y[3])   # no t+2 available


def test_front_return_hand_computed():
    log_f = np.array([[4.0, 0.0], [4.2, 0.0], [4.1, 0.0]])
    y = front_return(log_f, h=1)
    assert y[0] == pytest.approx(0.2)
    assert y[1] == pytest.approx(-0.1)
    assert np.isnan(y[2])


def test_realised_vol_nan_on_any_missing_return_in_window():
    log_f = np.zeros((10, 1))
    rng = np.random.default_rng(0)
    log_f[:, 0] = np.cumsum(rng.normal(0, 0.01, 10))
    log_f[5, 0] = np.nan   # kills the return into AND out of day 5

    y = realised_vol(log_f, h=3)
    # windows (t, t+3] that include day 5's undefined return must be NaN
    assert np.isnan(y[3])   # window (3,6] includes the day-5 return (ret[4:7] incl ret[5])
    assert np.isnan(y[4])
    # a window entirely clear of day 5 should be finite
    assert np.isfinite(y[6])


def test_realised_vol_matches_manual_std():
    log_f = np.zeros((6, 1))
    log_f[:, 0] = [0.0, 0.01, 0.03, 0.02, 0.05, 0.04]
    y = realised_vol(log_f, h=2)
    ret = np.diff(log_f[:, 0])   # [0.01, 0.02, -0.01, 0.03, -0.01]
    expected0 = np.std([ret[0], ret[1]], ddof=1)
    assert y[0] == pytest.approx(expected0)


# ------------------------------------------------------------ regression ---
def test_encompassing_regression_recovers_known_coefficients():
    rng = np.random.default_rng(42)
    n = 2000
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    alpha_true, b1_true, b2_true = 0.01, 0.02, -0.15
    y = alpha_true + b1_true * l_obs + b2_true * l_filt + rng.normal(0, 0.05, n)

    r = encompassing_regression(y, l_obs, l_filt, h=1, target="t", subset="s")
    assert r.b1 == pytest.approx(b1_true, abs=0.02)
    assert r.b2 == pytest.approx(b2_true, abs=0.02)
    assert r.n_obs == n
    assert r.b2_significant
    assert r.pattern() in ("b2_only", "both")


def test_encompassing_regression_drops_nan_rows():
    rng = np.random.default_rng(1)
    n = 200
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    y = 0.1 * l_filt + rng.normal(0, 0.01, n)
    y[:20] = np.nan

    r = encompassing_regression(y, l_obs, l_filt, h=1)
    assert r.n_obs == n - 20


def test_hac_lag_adjustment_actually_changes_the_standard_error():
    """The entire reason HAC is mandatory here: an overlapping h-day target
    (spread_change on a random-walk log_f, exactly the real construction)
    has genuine MA(h-1) serial correlation in its regression residual. This
    isolates the LAG-WEIGHTING mechanism specifically -- same data, same
    HAC estimator, maxlags=0 (no autocorrelation adjustment, only
    heteroskedasticity-robust) vs maxlags=h-1 -- rather than comparing
    against a separately-configured non-robust OLS call, which would
    conflate the two adjustments and be a less precise test of what h
    actually controls."""
    rng = np.random.default_rng(7)
    n = 1500
    h = 22
    log_f = np.column_stack([
        np.cumsum(rng.normal(0, 0.02, n)),
        np.cumsum(rng.normal(0, 0.02, n)),
    ])
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    y = spread_change(log_f, h)

    r_no_lag_adjustment = encompassing_regression(y, l_obs, l_filt, h=1)   # maxlags=0
    r_full_lag_adjustment = encompassing_regression(y, l_obs, l_filt, h=h)  # maxlags=21

    assert r_full_lag_adjustment.fit.cov_type == "HAC"
    assert r_full_lag_adjustment.fit.cov_kwds["maxlags"] == h - 1
    assert r_full_lag_adjustment.se_b2 != pytest.approx(r_no_lag_adjustment.se_b2, rel=1e-6)


def test_maxlags_scales_with_horizon():
    """maxlags passed to HAC must be h - 1, not a fixed constant --
    confirmed by checking the fitted result's own cov_kwds rather than
    just trusting the call site."""
    rng = np.random.default_rng(3)
    n = 300
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    y = rng.normal(0, 1, n)

    r5 = encompassing_regression(y, l_obs, l_filt, h=5)
    r22 = encompassing_regression(y, l_obs, l_filt, h=22)
    assert r5.fit.cov_kwds["maxlags"] == 4
    assert r22.fit.cov_kwds["maxlags"] == 21


# ------------------------------------------------------------------ gate ---
def _result(h, pattern):
    p_b1, p_b2 = {"b2_only": (0.5, 0.01), "both": (0.01, 0.01),
                 "b1_only": (0.01, 0.5), "neither": (0.5, 0.5)}[pattern]
    return RegressionResult(target="spread_change", horizon=h, subset="full",
                            n_obs=100, alpha=0.0, b1=0.1, b2=-0.1,
                            se_b1=0.05, se_b2=0.05, p_b1=p_b1, p_b2=p_b2,
                            r_squared=0.1)


def test_gate_pass_when_b2_only_on_both_horizons():
    results = {5: _result(5, "b2_only"), 22: _result(22, "b2_only")}
    assert classify_gate(results) == "pass"


def test_gate_fail_when_b1_only_on_either_horizon():
    results = {5: _result(5, "b1_only"), 22: _result(22, "b2_only")}
    assert classify_gate(results) == "fail"
    results2 = {5: _result(5, "b2_only"), 22: _result(22, "b1_only")}
    assert classify_gate(results2) == "fail"


def test_gate_partial_when_both_significant_on_a_horizon():
    results = {5: _result(5, "b2_only"), 22: _result(22, "both")}
    assert classify_gate(results) == "partial"


def test_gate_inconclusive_when_neither_significant():
    results = {5: _result(5, "b2_only"), 22: _result(22, "neither")}
    assert classify_gate(results) == "inconclusive"


# --------------------------------------------------------- end-to-end ------
def test_run_study_a_end_to_end_smoke():
    """Full battery on simulated data with SS_OIL's own filtered state (no
    MLE fit needed for a structural smoke test) -- confirms run_study_a
    executes cleanly end to end and returns a coherently shaped result."""
    log_f, tau, _ = simulate_panel(SS_OIL, n=400, m=4, seed=99)
    ll, x_filt, Pf, resid, n_obs = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)
    dates = pd.bdate_range("2015-01-02", periods=len(log_f))

    result = run_study_a(SS_OIL, x_filt, log_f, tau, dates)

    assert result.gate in ("pass", "partial", "fail", "inconclusive")
    assert set(result.primary.keys()) == {5, 22}
    assert set(result.continuous.keys()) == {5, 22}
    assert len(result.all_results) > 10
    for r in result.all_results:
        assert np.isfinite(r.b1) and np.isfinite(r.b2)


# ------------------------------------------ shared-measurement-error fix ---
def test_cross_maturity_spread_change_hand_computed():
    log_f = np.array([
        [0.0, 0.0, 4.00, 3.90],
        [0.0, 0.0, 4.10, 3.80],
        [0.0, 0.0, 4.20, 3.70],
    ])
    # s_far = C3 - C4 = [0.10, 0.30, 0.50]
    y = cross_maturity_spread_change(log_f, h=1, far=(2, 3))
    assert y[0] == pytest.approx(0.30 - 0.10)
    assert y[1] == pytest.approx(0.50 - 0.30)
    assert np.isnan(y[2])


def test_cross_maturity_default_uses_contracts_3_and_4():
    log_f = np.array([[0.0, 0.0, 4.0, 3.9], [0.0, 0.0, 4.1, 3.8]])
    y_default = cross_maturity_spread_change(log_f, h=1)
    y_explicit = cross_maturity_spread_change(log_f, h=1, far=(2, 3))
    np.testing.assert_array_equal(y_default, y_explicit)


def test_forward_spread_level_does_not_depend_on_s_t():
    """The whole point: s_t must not enter this target at all. Proven
    directly -- change ONLY row t (s_t) while holding row t+h (s_{t+h})
    fixed, at h=2 so the two rows are distinct, and y_t (which depends only
    on s_{t+h}) must not move."""
    log_f_a = np.array([[4.0, 3.9], [4.3, 4.1], [4.5, 4.0]])   # s = [0.1, 0.2, 0.5]
    log_f_b = log_f_a.copy()
    log_f_b[0] = [9.0, 1.0]   # s_0 changed drastically; s_2 (row 2) untouched

    y_a = forward_spread_level(log_f_a, h=2)
    y_b = forward_spread_level(log_f_b, h=2)
    assert y_a[0] == pytest.approx(0.5)          # = s_2, unaffected by s_0
    assert y_b[0] == pytest.approx(y_a[0])        # changing s_0 didn't move it


def test_forward_spread_level_hand_computed():
    log_f = np.array([[4.0, 3.9], [4.5, 4.0], [4.2, 3.7]])
    # s = [0.1, 0.5, 0.5]
    y = forward_spread_level(log_f, h=1)
    assert y[0] == pytest.approx(0.5)   # s_1, not s_1 - s_0
    assert y[1] == pytest.approx(0.5)   # s_2, not s_2 - s_1
    assert np.isnan(y[2])


def test_lag_labels_shifts_by_one_session():
    l_obs = np.array([1.0, -1.0, 1.0, 1.0])
    l_filt = np.array([-1.0, 1.0, -1.0, 1.0])
    lo, lf = lag_labels(l_obs, l_filt)
    assert np.isnan(lo[0]) and np.isnan(lf[0])
    assert np.array_equal(lo[1:], l_obs[:-1])
    assert np.array_equal(lf[1:], l_filt[:-1])


def test_run_corrected_battery_end_to_end_smoke():
    # n=800, not 400: at n=400/seed=13 the filtered label never flips sign
    # in this particular short simulated window (SS_OIL's ~170-day
    # half-life), making it exactly collinear with the intercept -- a
    # short-window artefact of the smoke test's own sample, not something
    # the real ~3,600-day fitted run on WTI exhibits.
    log_f, tau, _ = simulate_panel(SS_OIL, n=800, m=4, seed=13)
    ll, x_filt, Pf, resid, n_obs = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)

    result = run_corrected_battery(SS_OIL, x_filt, log_f, tau)

    assert result.gate in ("pass", "partial", "fail", "inconclusive")
    assert set(result.cross_maturity.keys()) == {5, 22}
    assert set(result.forward_level.keys()) == {5, 22}
    assert set(result.lagged_labels.keys()) == {5, 22}
    for r in result.all_results():
        assert np.isfinite(r.b1) and np.isfinite(r.b2)


def test_shared_noise_artefact_vanishes_under_cross_maturity_with_a_perfect_label():
    """Direct proof the fix works, not just documentation. Simulated data
    has a KNOWN true state (states[:,0] = true chi_t), so a "perfect"
    filtered label built from it -- no estimation noise at all -- has, by
    construction, no genuine information the naive label doesn't already
    approximate. Any apparent naive-label edge under the ORIGINAL
    spread_change target is therefore pure shared-measurement-error
    artefact: s_t appears (negated) in both sign(s_t) and s_{t+h}-s_t. The
    same comparison under cross_maturity_spread_change, which shares no
    measurement-error component with either label, must not reproduce it."""
    log_f, tau, states = simulate_panel(SS_OIL, n=1500, m=4, seed=17)
    x_true = np.zeros((len(states), 4))
    x_true[:, 0] = states[:, 0]   # true chi_t -- a "perfect" filtered state

    l_obs = naive_label(log_f)
    l_filt_perfect = filtered_label(SS_OIL, x_true, tau)
    h = 22

    r_orig = encompassing_regression(spread_change(log_f, h), l_obs, l_filt_perfect, h)
    r_cross = encompassing_regression(cross_maturity_spread_change(log_f, h),
                                      l_obs, l_filt_perfect, h)

    # The artefact's signature, reproduced: strongly significant negative
    # b1 despite a PERFECT second label that has denoised away everything
    # genuinely informative.
    assert r_orig.b1 < 0
    assert r_orig.b1_significant
    # Under cross-maturity that mechanical bias must not survive.
    assert abs(r_cross.b1) < abs(r_orig.b1) * 0.5


# ------------------------------------------------- horizon robustness ---
def test_reversion_fraction_hand_computed():
    kappa = 1.3144
    assert reversion_fraction(kappa, 0) == pytest.approx(0.0)
    assert reversion_fraction(kappa, 252) == pytest.approx(1 - np.exp(-kappa))
    # A short horizon spans a small fraction; increases monotonically in h.
    assert reversion_fraction(kappa, 5) < reversion_fraction(kappa, 22)
    assert reversion_fraction(kappa, 22) < reversion_fraction(kappa, 252)


def test_encompassing_regression_maxlags_override_changes_lags_used():
    """The Sec. 7 override (floor(1.5h) instead of h-1) must actually take
    effect, and the default (no override) must be untouched -- every
    existing call site in Sec. 3-6 relies on that default."""
    rng = np.random.default_rng(4)
    n = 500
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    y = rng.normal(0, 1, n)

    r_default = encompassing_regression(y, l_obs, l_filt, h=22)
    assert r_default.fit.cov_kwds["maxlags"] == 21   # unchanged: h - 1

    r_override = encompassing_regression(y, l_obs, l_filt, h=22, maxlags=33)
    assert r_override.fit.cov_kwds["maxlags"] == 33   # floor(1.5*22)


def test_nonoverlapping_regression_uses_every_hth_row_and_plain_ols():
    rng = np.random.default_rng(9)
    n = 300
    l_obs = rng.choice([-1.0, 1.0], n)
    l_filt = rng.choice([-1.0, 1.0], n)
    y = 0.2 * l_filt + rng.normal(0, 0.1, n)
    h = 10

    from csr.study.recovery import _nonoverlapping_regression
    r = _nonoverlapping_regression(y, l_obs, l_filt, h)
    assert r.n_obs == len(np.arange(0, n, h))
    assert r.fit.cov_type == "nonrobust"   # plain OLS, no HAC
    assert r.subset == "nonoverlap"


def test_run_horizon_robustness_end_to_end_smoke():
    log_f, tau, _ = simulate_panel(SS_OIL, n=2000, m=4, seed=21)
    ll, x_filt, Pf, resid, n_obs = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)

    rows = run_horizon_robustness(SS_OIL, x_filt, log_f, tau,
                                  horizons=(5, 22, 66), nonoverlap_horizons=(66,))

    assert [r.h for r in rows] == [5, 22, 66]
    for r in rows:
        assert r.hac_maxlags == int(np.floor(1.5 * r.h))
        assert r.t_over_h == pytest.approx(r.n_obs / r.h, rel=1e-2)
        assert 0.0 <= r.reversion_frac <= 1.0
    assert rows[0].nonoverlap is None
    assert rows[1].nonoverlap is None
    assert rows[2].nonoverlap is not None
    # reversion_frac must strictly increase with h at a fixed kappa
    assert rows[0].reversion_frac < rows[1].reversion_frac < rows[2].reversion_frac
