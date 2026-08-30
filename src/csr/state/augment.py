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


def append_news_block(aug: dict, news: np.ndarray, G: np.ndarray,
                      a_n: np.ndarray, R_diag: np.ndarray,
                      lag: int = 1) -> dict:
    """Append the news observations as extra rows of the measurement system.

    Parameters
    ----------
    news   : (n, p)  observed news features; NaN where a day has no coverage
    G      : (p, 2)  loadings of each feature on (w^chi, w^xi)
    a_n    : (p,)    news intercepts
    R_diag : (n, p)  news measurement variances -- time-varying, carrying the
                     sampling-variance construction of eq. (23)
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

    # Shift news so that row t carries the observation attributable to the
    # innovation of period t. lag >= 1 means today's state is measured by
    # news published on a later session, which is not usable in real time;
    # the shift below places news at the period whose shock it describes.
    y_news = np.full((n, p), np.nan)
    if lag >= 0:
        if lag < n:
            y_news[: n - lag] = news[lag:]
    else:
        s = -lag
        if s < n:
            y_news[s:] = news[: n - s]

    Z_news = np.zeros((n, p, 4))
    Z_news[:, :, 2:] = G[None, :, :]

    Z = np.concatenate([aug["Z"], Z_news], axis=1)
    d = np.concatenate([aug["d"], np.repeat(np.asarray(a_n, float)[None, :], n, 0)],
                       axis=1)
    Hdiag = np.concatenate([aug["Hdiag"], np.asarray(R_diag, float)], axis=1)

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
