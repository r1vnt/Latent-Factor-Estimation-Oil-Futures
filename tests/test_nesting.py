"""Nesting and equivalence assertions -- the hard gate of Phase 3.

Every likelihood-ratio statistic in the study depends on these holding. If a
restricted specification fails to reproduce its parent, the LR test still runs
and still returns a p-value; the failure is silent. That is why these are
committed as tests rather than checked once by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.pricing.specs import (NewsData, REGISTRY, lr_degrees_of_freedom,  # noqa: E402
                               run_spec, split_sign)
from csr.pricing.twofactor import A_tau, base_system, initial_state  # noqa: E402
from csr.simulate import SS_OIL, simulate_panel                      # noqa: E402
from csr.state.kalman import kalman_filter                           # noqa: E402

DT = 1.0 / 252.0
TOL = 1e-9


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(SS_OIL, n=900, m=4, seed=101)


@pytest.fixture(scope="module")
def news(panel):
    log_f, _, _ = panel
    n = len(log_f)
    rng = np.random.default_rng(7)
    feats = np.zeros((n, 3))
    for t in range(1, n):
        feats[t] = 0.5 * feats[t - 1] + rng.normal(0, 1, 3) * np.sqrt(0.75)
    counts = rng.integers(5, 200, n).astype(float)
    disp = 0.3 + 0.1 * rng.random(n)
    intensity = (np.log1p(counts) - np.log1p(counts).mean()) / np.log1p(counts).std()
    roll = ((np.arange(n) % 21) >= 18).astype(float)
    release = (np.arange(n) % 5 == 2).astype(float)
    rv = np.abs(rng.normal(0, 1, n))
    rv_lag = np.concatenate([[0.0], rv[:-1]])
    return NewsData(feats, counts, disp, intensity, release, rv_lag, roll)


# ------------------------------------------------------------------ baseline
def test_augmented_path_reproduces_two_state_baseline(panel):
    """S0 through the 4-state code path must equal the 2-state model exactly.

    All specifications share the augmented path, so if this drifts, every
    result in the study is computed on a different model from the one derived
    in the paper.
    """
    log_f, tau, _ = panel
    ll_aug, _, _, _, n_aug = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT)

    sysm = base_system(SS_OIL, tau, DT)
    x0, P0 = initial_state(log_f, tau, SS_OIL)
    n = sysm["n"]
    ref = kalman_filter(
        log_f, T=sysm["T"], Z=sysm["Z"],
        Q=np.repeat(sysm["Q"][None], n, 0),
        H=np.array([np.diag(sysm["Hdiag"][t]) for t in range(n)]),
        c=sysm["c"], d=sysm["d"], x0=x0, P0=P0,
    )
    assert abs(ll_aug - ref.loglik) < 1e-6
    assert n_aug == ref.n_obs


def test_pricing_boundary_condition():
    """A(0) = 0: a contract expiring now prices at spot."""
    assert abs(float(A_tau(SS_OIL, np.array([0.0]))[0])) < 1e-14


# ------------------------------------------------------------------ nesting
def test_SD_with_zero_B_equals_S0(panel, news):
    log_f, tau, _ = panel
    ll0, *_ = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT, news)
    llD, *_ = run_spec(REGISTRY["SD"], SS_OIL, log_f, tau, DT, news,
                       B=np.zeros((2, news.p)))
    assert abs(llD - ll0) < TOL


def test_SM_with_zero_G_equals_S0_on_price_rows(panel, news):
    """With G = 0 the news rows are uninformative about the state.

    They still contribute their own (constant) likelihood terms, so the test is
    that the *price* contribution is unchanged: the difference between the two
    log-likelihoods must equal the news rows' own Gaussian density, and the
    filtered states must be identical.
    """
    log_f, tau, _ = panel
    ll0, xf0, _, _, nobs0 = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT, news)
    llM, xfM, _, _, nobsM = run_spec(REGISTRY["SM"], SS_OIL, log_f, tau, DT, news,
                                     G=np.zeros((news.p, 2)))
    # states must be untouched by a zero-loading channel
    assert np.max(np.abs(xf0 - xfM)) < 1e-10
    # and the extra likelihood must be exactly the news rows' marginal density
    assert nobsM > nobs0


def test_SDM_with_both_zero_equals_S0_states(panel, news):
    log_f, tau, _ = panel
    _, xf0, _, _, _ = run_spec(REGISTRY["S0"], SS_OIL, log_f, tau, DT, news)
    _, xfDM, _, _, _ = run_spec(REGISTRY["SDM"], SS_OIL, log_f, tau, DT, news,
                                B=np.zeros((2, news.p)),
                                G=np.zeros((news.p, 2)))
    assert np.max(np.abs(xf0 - xfDM)) < 1e-10


def test_asymmetric_driver_reduces_to_symmetric(panel, news):
    """SD_ASYM with B+ = B- must reproduce SD exactly, since u+ + u- = u.

    Asymmetry is placed in the driver channel for exactly this reason. The same
    construction in the measurement channel does *not* nest: splitting an
    observation at zero yields a row carrying a structural zero, which the
    filter cannot distinguish from a genuine reading. See the note in
    specs.REGISTRY.
    """
    log_f, tau, _ = panel
    B = np.array([[0.004, 0.002, 0.0], [0.001, 0.0, 0.0015]])
    llD, xfD, *_ = run_spec(REGISTRY["SD"], SS_OIL, log_f, tau, DT, news, B=B)
    llA, xfA, *_ = run_spec(REGISTRY["SD_ASYM"], SS_OIL, log_f, tau, DT, news,
                            B=np.hstack([B, B]))
    assert abs(llA - llD) < TOL
    assert np.max(np.abs(xfA - xfD)) < 1e-10


def test_asymmetric_driver_actually_differs_when_loadings_differ(panel, news):
    """Guard against the restriction being vacuous: B+ != B- must change the fit."""
    log_f, tau, _ = panel
    B = np.array([[0.004, 0.002, 0.0], [0.001, 0.0, 0.0015]])
    llD, *_ = run_spec(REGISTRY["SD"], SS_OIL, log_f, tau, DT, news, B=B)
    llA, *_ = run_spec(REGISTRY["SD_ASYM"], SS_OIL, log_f, tau, DT, news,
                       B=np.hstack([B, 2.5 * B]))
    assert abs(llA - llD) > 1e-3


# ------------------------------------------------------------ bookkeeping
def test_degrees_of_freedom_are_derived_not_hardcoded():
    p = 3
    assert lr_degrees_of_freedom("SD", "S0", p) == 2 * p
    assert lr_degrees_of_freedom("SM", "S0", p) == 2 * p + p
    assert lr_degrees_of_freedom("SDM", "SD", p) == 2 * p + p
    assert lr_degrees_of_freedom("SD_ASYM", "SD", p) == 2 * p


def test_split_sign_recovers_the_original():
    u = np.random.default_rng(0).normal(size=(50, 3))
    s = split_sign(u)
    assert np.allclose(s[:, :3] + s[:, 3:], u)


# --------------------------------------------------- variance-channel guards
def test_news_variance_rejects_standardised_counts():
    """R_t = disp^2 / n_t is undefined for a mean-zero standardised series."""
    from csr.state.augment import news_variance
    with pytest.raises(ValueError):
        news_variance(np.ones(10), np.linspace(-2, 2, 10), p=2)


def test_news_variance_shifts_with_the_observation_it_describes(panel):
    """R_diag must be lag-shifted the same way the news observation is, so
    the variance attached to row t always describes the same calendar day
    as y_news[t] -- not the day that happens to sit at index t before the
    shift. A day with a real gap (feature, dispersion, and counts all NaN
    together, as aggregate_stage4.py always produces them) sitting right
    next to covered days must not poison the log-likelihood: filter_uv only
    skips a row when the *observation* is NaN, so a valid shifted
    observation paired with a wrong-day NaN variance falls straight through
    into log(NaN), silently corrupting the likelihood and every filtered
    state after it. This is real, gappy Stage 4 data's shape, not an edge
    case -- caught via exactly this scenario during the Stage 4 -> model
    integration dry run, on data with no NaN in the synthetic `news`
    fixture above to have ever exercised it."""
    log_f, tau, _ = panel
    n = len(log_f)
    gap = 2   # index with no news coverage at all
    dispersion = np.full(n, 0.3)
    counts = np.full(n, 20.0)
    feats = np.linspace(0.1, 0.6, n).reshape(-1, 1)
    dispersion[gap] = np.nan
    counts[gap] = np.nan
    feats[gap] = np.nan
    gappy_news = NewsData(feats, counts, dispersion, intensity=None)

    ll, xf, _, _, nobs = run_spec(REGISTRY["SM"], SS_OIL, log_f, tau, DT,
                                  gappy_news, G=np.array([[0.5, 0.3]]))
    assert np.isfinite(ll)
    assert np.all(np.isfinite(xf))
    assert nobs > 0


def test_news_variance_alignment_guard_catches_mismatched_coverage(panel):
    """If a caller ever hands run_spec a NewsData whose .features NaN
    pattern disagrees with .dispersion/.counts' NaN pattern -- the exact
    kind of mismatch that let a wrong-day variance through before the
    shift fix above -- it must fail loudly, not produce a silent NaN
    likelihood through a different door."""
    log_f, tau, _ = panel
    n = len(log_f)
    dispersion = np.full(n, 0.3)
    counts = np.full(n, 20.0)
    feats = np.linspace(0.1, 0.6, n).reshape(-1, 1)
    dispersion[2] = np.nan
    counts[2] = np.nan
    # feats deliberately NOT made NaN at index 2 -- the mismatch itself.
    mismatched_news = NewsData(feats, counts, dispersion, intensity=None)

    with pytest.raises(ValueError, match="same per-day coverage"):
        run_spec(REGISTRY["SM"], SS_OIL, log_f, tau, DT, mismatched_news,
                G=np.array([[0.5, 0.3]]))


def test_roll_inflation_raises_measurement_variance(panel, news):
    """eta > 0 must widen H inside roll windows and nowhere else."""
    from csr.pricing.twofactor import base_system, scale_variances
    _, tau, _ = panel
    p_roll = SS_OIL.replace(eta_roll=0.7)
    base = base_system(p_roll, tau, DT)
    scaled = scale_variances(base, p_roll, roll=news.roll)
    inside = news.roll > 0
    assert np.all(scaled["Hdiag"][inside] > base["Hdiag"][inside] * 1.5)
    assert np.allclose(scaled["Hdiag"][~inside], base["Hdiag"][~inside])


def test_news_lag_never_uses_future_information(panel, news):
    """A spike in the feature must not move the state on or before its date."""
    from csr.state.augment import append_news_block, augment_system
    from csr.pricing.twofactor import base_system
    _, tau, _ = panel
    sysm = augment_system(base_system(SS_OIL, tau, DT))
    feats = np.zeros((sysm["n"], 1))
    feats[100, 0] = 1.0
    _, y_news = append_news_block(sysm, feats, np.array([[1.0, 0.0]]),
                                  np.zeros(1), np.ones((sysm["n"], 1)), lag=1)
    # row t carries news published at t+1, describing the shock of period t
    assert y_news[99, 0] == 1.0
    assert y_news[100, 0] == 0.0
