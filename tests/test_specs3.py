"""Nesting guards for the three-factor news lattice (specs3.py).

Mirrors tests/test_nesting.py's own guards for the two-factor lattice
exactly (B=0 -> S0; G=0 -> price rows untouched) -- "the LR test still
runs and still returns a p-value" if these don't hold, silently, per
that file's own docstring. Same discipline, ported base.
"""

from __future__ import annotations

import numpy as np
import pytest

from csr.pricing.fit3 import _raw_system3
from csr.pricing.smoothmeas import SmoothMeasParams
from csr.pricing.specs import NewsData, REGISTRY
from csr.pricing.specs3 import run_spec3
from csr.pricing.threefactor import ThreeFactorParams, base_system3, initial_state3

DT = 1.0 / 252.0
TOL = 1e-9


def _p3():
    return ThreeFactorParams(kappa1=1.9, kappa2=0.4, sigma1=0.22, sigma2=0.31,
                             sigma_xi=0.2, rho1x=0.1, rho2x=0.15,
                             mu_xi=0.0, mu_xi_star=0.01, lambda1=0.05, lambda2=0.02)


def _smooth():
    return SmoothMeasParams(a=np.log(0.01), b=-3.0, c=1.5, quadratic=True)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(11)
    n, m = 300, 8
    tau = np.tile(np.linspace(0.08, 1.2, m), (n, 1)) + rng.normal(0, 0.001, (n, m))
    tau = np.abs(tau)
    p, smooth = _p3(), _smooth()
    sys = base_system3(p, tau, DT)
    x0, P0 = initial_state3(np.zeros((1, m)), tau, p)
    x = x0.copy()
    log_f = np.empty((n, m))
    from csr.pricing.smoothmeas import sigma_from_tau
    for t in range(n):
        w = rng.multivariate_normal(np.zeros(3), sys["Q"][t])
        x = sys["T"] @ x + sys["c"][t] + w
        s = sigma_from_tau(smooth, tau[t])
        log_f[t] = sys["d"][t] + sys["Z"][t] @ x + rng.normal(0, s)
    return log_f, tau, p, smooth


@pytest.fixture(scope="module")
def news(panel):
    log_f, *_ = panel
    n = len(log_f)
    rng = np.random.default_rng(13)
    feats = np.zeros((n, 5))
    for t in range(1, n):
        feats[t] = 0.5 * feats[t - 1] + rng.normal(0, 1, 5) * np.sqrt(0.75)
    counts = rng.integers(5, 200, n).astype(float)
    disp = 0.3 + 0.1 * rng.random(n)
    return NewsData(feats, counts, disp, intensity=None)


def test_S0_through_specs3_matches_weighted_filter3_exactly(panel):
    """The 5-dim augmentation, with no news spec active, must reproduce
    fit3.weighted_filter3's own 3-dim (unaugmented) likelihood exactly --
    not approximately. Pins the bug where augment_system3_news copied
    only the chi1/chi2 sub-block into the augmented state's own base
    block, silently zeroing xi's process variance and its correlation
    with chi1/chi2; S0 landed ~31000 nats away from the known-good fit
    before this was caught (explore/38)."""
    from csr.pricing.fit3 import weighted_filter3

    log_f, tau, p, smooth = panel
    ll_ref, xf_ref, resid_ref, n_ref = weighted_filter3(
        p, smooth, log_f, tau, DT, np.ones_like(log_f))
    ll_new, xf_new, resid_new, n_new = run_spec3(REGISTRY["S0"], p, smooth,
                                                 log_f, tau, DT)
    assert abs(ll_new - ll_ref) < 1e-6
    assert n_new == n_ref
    np.testing.assert_allclose(xf_new[:, :3], xf_ref, atol=1e-9)


def test_SD_with_zero_B_equals_S0(panel, news):
    log_f, tau, p, smooth = panel
    ll0, *_ = run_spec3(REGISTRY["S0"], p, smooth, log_f, tau, DT, news)
    llD, *_ = run_spec3(REGISTRY["SD"], p, smooth, log_f, tau, DT, news,
                        B=np.zeros((2, news.p)))
    assert abs(llD - ll0) < TOL


def test_SD_with_zero_B_equals_S0_even_with_news_gaps(panel):
    """Same NaN*0=NaN guard as test_nesting.py's two-factor version --
    _driver_term is shared code, but confirm the 5-dim Bu wiring doesn't
    reopen the failure mode via a different path."""
    log_f, tau, p, smooth = panel
    n = len(log_f)
    feats = np.zeros((n, 1))
    feats[3] = np.nan
    gappy_news = NewsData(feats, counts=np.full(n, 20.0),
                          dispersion=np.full(n, 0.3), intensity=None)

    ll0, xf0, *_ = run_spec3(REGISTRY["S0"], p, smooth, log_f, tau, DT)
    llD, xfD, *_ = run_spec3(REGISTRY["SD"], p, smooth, log_f, tau, DT,
                             gappy_news, B=np.zeros((2, 1)))
    assert np.isfinite(llD)
    assert abs(llD - ll0) < TOL
    assert np.max(np.abs(xfD - xf0)) < 1e-10


def test_SM_with_zero_G_equals_S0_on_price_rows(panel, news):
    log_f, tau, p, smooth = panel
    ll0, xf0, _, nobs0 = run_spec3(REGISTRY["S0"], p, smooth, log_f, tau, DT, news)
    llM, xfM, _, nobsM = run_spec3(REGISTRY["SM"], p, smooth, log_f, tau, DT, news,
                                   G=np.zeros((news.p, 2)))
    assert np.max(np.abs(xf0 - xfM)) < 1e-10
    assert nobsM > nobs0


def test_SDM_with_both_zero_equals_S0_states(panel, news):
    log_f, tau, p, smooth = panel
    _, xf0, *_ = run_spec3(REGISTRY["S0"], p, smooth, log_f, tau, DT, news)
    _, xfDM, *_ = run_spec3(REGISTRY["SDM"], p, smooth, log_f, tau, DT, news,
                            B=np.zeros((2, news.p)), G=np.zeros((news.p, 2)))
    assert np.max(np.abs(xf0 - xfDM)) < 1e-10


def test_xi_has_no_direct_news_channel(panel):
    """Bu[:, 2] (xi's column) must be identically zero regardless of B --
    the driver term only ever writes into indices 0, 1 (chi1, chi2).

    This is a structural fact about the driver term itself, checked
    directly on Bu -- NOT via filtered xi trajectories. An earlier version
    of this test asserted xi's filtered value stays identical to the B=0
    case and failed: even with rho1x = rho2x = 0 (no process-noise
    correlation at all), xi's filtered estimate still moves once chi1's
    prediction shifts, because every contract's price loads on BOTH chi1
    and xi (the same shared observation), so the Kalman correction step
    updates xi's posterior from the very same innovation that updates
    chi1's -- states sharing an observation become entangled through
    filtering regardless of any prior independence. That is correct
    Kalman-filter behaviour, not a leak, and there is no direct-channel
    claim that survives being tested on filtered output; the only true
    claim is about Bu's own construction, tested here on Bu itself."""
    log_f, tau, p, smooth = panel
    n = len(log_f)
    feats = np.zeros((n, 1))
    feats[10:20] = 1.0
    strong_news = NewsData(feats, counts=np.full(n, 20.0),
                           dispersion=np.full(n, 0.3), intensity=None)
    _, xf0, *_ = run_spec3(REGISTRY["S0"], p, smooth, log_f, tau, DT)
    _, xfD, *_ = run_spec3(REGISTRY["SD"], p, smooth, log_f, tau, DT, strong_news,
                           B=np.array([[0.5], [0.3]]))
    assert np.max(np.abs(xfD[:, 0] - xf0[:, 0])) > 1e-6   # chi1 moved

    from csr.pricing.specs import _driver_term, split_sign
    Bu = _driver_term(np.array([[0.5], [0.3]]), np.atleast_2d(strong_news.features), n, 5)
    np.testing.assert_array_equal(Bu[:, 2], 0.0)
    np.testing.assert_array_equal(Bu[:, 3], 0.0)
    np.testing.assert_array_equal(Bu[:, 4], 0.0)
