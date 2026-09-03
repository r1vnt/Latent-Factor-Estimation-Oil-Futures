"""State augmentation exposing current-period innovations (Section XI-C).

News is definitionally the *unexpected* component, so a sentiment flow on a
given day is evidence about that day's shock, not about the accumulated
equilibrium level. Pairing a flow variable with a stock variable is a category
error, so the measurement channel must observe the innovation.

That is achieved by carrying the current disturbances in the state:

    z_t = ( chi_t , xi_t , w^chi_t , w^xi_t )'

with

    T4 = [[a,0,0,0],
          [0,1,0,0],
          [0,0,0,0],      <- the shock block is i.i.d., simply exposed
          [0,0,0,0]]

The disturbance driving z_t is (w^chi, w^xi, w^chi, w^xi)': the *same* draw
appears in both blocks, so

    Q4 = [[Q2, Q2],
          [Q2, Q2]]

which has rank 2 and is therefore singular. That is harmless -- a Kalman filter
requires only that Q be positive *semi*-definite -- but it is why the augmented
system must never be handed to code that assumes Q is invertible.

Price observations load on the level block only; news observations load on the
shock block only:

    y_t = d_t + [ Z_t | 0 ] z_t + v_t
    n_t = a_n + [ 0 | G ] z_{t-lag} + e_t

The system remains linear and Gaussian, so the filter stays exact.
"""

from __future__ import annotations

import numpy as np


def augment_system(sys: dict) -> dict:
    """Lift a two-dimensional system to the four-dimensional augmented form.

    ``sys`` is the output of :func:`csr.pricing.twofactor.base_system`, after
    any variance scaling. Returns matrices in the same convention with k = 4.
    """
    n, m = sys["n"], sys["m"]
    T2, c2, Z2 = sys["T"], np.asarray(sys["c"], float), sys["Z"]
    Q2 = sys["Q"]
    if Q2.ndim == 2:
        Q2 = np.repeat(Q2[None, :, :], n, axis=0)

    T4 = np.zeros((4, 4))
    T4[:2, :2] = T2

    c4 = np.zeros((n, 4))
    c4[:, :2] = c2[None, :]

    Q4 = np.zeros((n, 4, 4))
    Q4[:, :2, :2] = Q2
    Q4[:, :2, 2:] = Q2
    Q4[:, 2:, :2] = Q2
    Q4[:, 2:, 2:] = Q2

    Z4 = np.zeros((n, m, 4))
    Z4[:, :, :2] = Z2

    return {"T": T4, "c": c4, "Q": Q4, "Z": Z4, "d": sys["d"],
            "Hdiag": sys["Hdiag"], "n": n, "m": m, "k": 4}


def augment_prior(x0: np.ndarray, P0: np.ndarray, Q2_0: np.ndarray):
    """Lift the prior. The shock block's prior is its own covariance.

    Because ``T4`` annihilates the shock block, the first prediction step
    overwrites it with ``Q2`` regardless; the value set here only matters for
    code that inspects the prior directly.
    """
    x4 = np.zeros(4)
    x4[:2] = x0
    P4 = np.zeros((4, 4))
    P4[:2, :2] = P0
    P4[2:, 2:] = Q2_0
    return x4, P4


def add_pricing_error_state(sys: dict, phi: float, sigma_p: float,
                            contract_index: int = 0) -> dict:
    """Extend a k-dimensional system to k+1 dimensions by appending an
    AR(1) pricing-error state p_t, loading 1 onto exactly one observation
    (``contract_index``) and 0 onto every other:

        p_{t+1} = phi * p_t + eps_t,   eps ~ N(0, sigma_p^2)
        y_{t,contract_index} += p_t

    A general third pricing factor is not identifiable in a shallow
    (four-contract) panel -- restricting it to load on a single, named
    contract is what makes it estimable at all: two parameters (phi,
    sigma_p) rather than a full loading vector, targeted at the
    front-contract misspecification the sigma_meas profile actually shows
    ([54.9, 6.5, 5.1, 16.6] bp -- the front contract absorbing eight times
    the error of the middle of the curve on the single most liquid
    instrument in the panel is a pricing-error signature, not noise).

    Works generically over the input dimension k -- called on the raw
    2-dim base_system output for S0P, or on the already news-augmented
    4-dim system for the full SxP lattice -- appending the new state as
    the LAST dimension either way, never interleaved with the existing
    ones. ``sigma_p = 0`` must reproduce the input system's likelihood
    exactly (p_t is then deterministically 0 for all t, contributing
    nothing to any observation) -- pinned in tests/test_pricing_error.py
    before this is trusted for any fit.
    """
    n, m = sys["n"], sys["m"]
    k = sys["T"].shape[0]
    T, c, Z = sys["T"], np.asarray(sys["c"], float), sys["Z"]
    Q = sys["Q"]
    if Q.ndim == 2:
        Q = np.repeat(Q[None, :, :], n, axis=0)

    T_new = np.zeros((k + 1, k + 1))
    T_new[:k, :k] = T
    T_new[k, k] = phi

    c_new = np.zeros((n, k + 1))
    c_new[:, :k] = c if c.ndim == 2 else c[None, :]

    Q_new = np.zeros((n, k + 1, k + 1))
    Q_new[:, :k, :k] = Q
    Q_new[:, k, k] = sigma_p ** 2

    Z_new = np.zeros((n, m, k + 1))
    Z_new[:, :, :k] = Z
    Z_new[:, contract_index, k] = 1.0

    out = dict(sys)
    out.update({"T": T_new, "c": c_new, "Q": Q_new, "Z": Z_new, "k": k + 1})
    return out


def add_pricing_error_prior(x0: np.ndarray, P0: np.ndarray,
                            phi: float, sigma_p: float):
    """Extend a k-dim prior to k+1 dims: p_0 ~ N(0, stationary variance),
    uncorrelated with the existing states at t=0. sigma_p=0 gives an exact
    point mass at 0, matching add_pricing_error_state's sigma_p=0 nesting
    property (p_t deterministically 0 for all t)."""
    k = len(x0)
    x_new = np.zeros(k + 1)
    x_new[:k] = x0
    P_new = np.zeros((k + 1, k + 1))
    P_new[:k, :k] = P0
    denom = max(1.0 - phi ** 2, 1e-12)   # phi = tanh(z) so |phi| < 1 strictly;
                                        # the floor only guards float error
                                        # right at the boundary, never reached
                                        # by the tanh parameterisation itself
    P_new[k, k] = sigma_p ** 2 / denom
    return x_new, P_new


def append_news_block(aug: dict, news: np.ndarray, G: np.ndarray,
                      a_n: np.ndarray, R_diag: np.ndarray,
                      lag: int = 1) -> dict:
    """Append the news observations as extra rows of the measurement system.

    Parameters
    ----------
    news   : (n, p)  observed news features; NaN where a day has no coverage
    G      : (p, 2)  loadings of each feature on (w^chi, w^xi)
    a_n    : (p,)    news intercepts
    R_diag : (n, p)  news measurement variances, ON THE SAME CALENDAR-DAY
                     INDEXING AS ``news`` -- i.e. R_diag[t] must be the
                     sampling-variance construction of eq. (23) for the SAME
                     day whose feature value is news[t], not pre-shifted by
                     the caller. This function applies the identical lag
                     shift to both, so the R value that ends up in row t of
                     Hdiag always describes the actual observation stored in
                     row t of y_news. Passing an already-shifted R_diag (or
                     a caller assuming this function only shifts ``news``)
                     silently pairs a valid shifted observation with a
                     wrong-day variance -- if that wrong-day variance is NaN
                     (a real possibility: the observation's own day had
                     coverage but the unrelated day supplying an unshifted
                     R_diag[t] didn't), filter_uv's per-observation missing
                     check (``if np.isnan(yi): continue``) does not catch
                     it, since yi itself is valid -- F silently becomes NaN,
                     which is not caught by `F <= 0.0` either (NaN
                     comparisons are always false), and NaN then poisons the
                     log-likelihood *and every filtered state from that
                     point forward*, for the whole remaining sample. This
                     was found and fixed via exactly that failure mode on
                     real (gappy) news data -- synthetic test fixtures with
                     no NaN dispersion/counts cannot exercise it.
    lag    : int     sessions by which news trails the innovation it measures.
                     Must be >= 1 for any specification used for inference;
                     the estimation grid may include 0 or negative values
                     purely as a timestamp diagnostic (Section XI-J).

    Notes
    -----
    Stacking the news rows into the same measurement equation is what makes the
    two channels commensurable: the filter weights each by its own noise and by
    current state uncertainty, rather than combining two separate estimates
    after the fact.
    """
    n, m = aug["n"], aug["m"]
    news = np.atleast_2d(np.asarray(news, float))
    p = news.shape[1]
    G = np.asarray(G, float).reshape(p, 2)
    R_diag = np.atleast_2d(np.asarray(R_diag, float))
    if R_diag.shape[0] == 1 and n != 1:
        R_diag = np.repeat(R_diag, n, axis=0)   # scalar-per-feature broadcast

    def _shift(a: np.ndarray) -> np.ndarray:
        """Row t -> the value attributable to the innovation of period t.
        lag >= 1 means today's state is measured by something published on
        a later session, which is not usable in real time; the shift below
        places each array at the period whose shock it describes. Applied
        identically to `news` and `R_diag` so a shifted observation and its
        own variance can never end up describing different calendar days."""
        out = np.full_like(a, np.nan)
        if lag >= 0:
            if lag < n:
                out[: n - lag] = a[lag:]
        else:
            s = -lag
            if s < n:
                out[s:] = a[: n - s]
        return out

    y_news = _shift(news)
    R_shifted = _shift(R_diag)

    Z_news = np.zeros((n, p, 4))
    Z_news[:, :, 2:] = G[None, :, :]

    Z = np.concatenate([aug["Z"], Z_news], axis=1)
    d = np.concatenate([aug["d"], np.repeat(np.asarray(a_n, float)[None, :], n, 0)],
                       axis=1)
    Hdiag = np.concatenate([aug["Hdiag"], R_shifted], axis=1)

    out = dict(aug)
    out.update({"Z": Z, "d": d, "Hdiag": Hdiag, "m": m + p,
                "m_price": m, "p_news": p})
    return out, y_news


def news_variance(dispersion: np.ndarray, counts: np.ndarray,
                  p: int, floor: float = 1e-8) -> np.ndarray:
    """News measurement variance from eq. (23): R_t = sigma^2_doc,t / n_t.

    This is the sampling variance of a mean over that day's documents, not a
    posited heteroskedasticity function -- both inputs are observed.

    ``counts`` must be the RAW article count after near-duplicate removal, not
    the standardised intensity used for the process-noise scaling: a
    standardised series is mean-zero and takes negative values, so dividing by
    it is undefined.
    """
    disp = np.asarray(dispersion, float)
    cnt = np.maximum(np.asarray(counts, float), 1.0)
    if np.any(np.asarray(counts, float) < 0):
        raise ValueError(
            "counts contains negative values -- pass raw article counts, "
            "not standardised intensity"
        )
    var = np.maximum(disp ** 2 / cnt, floor)
    return np.repeat(var[:, None], p, axis=1) if var.ndim == 1 else var
