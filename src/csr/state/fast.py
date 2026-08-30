"""JIT-compiled Kalman filter, general state dimension, univariate updating.

The earlier prototype hard-coded the two-factor case. The augmented state of
Section XI-C is four-dimensional, so the kernel is generalised over ``k`` while
keeping the two properties that made the specialised version fast:

* **Univariate (sequential) updating.** With a diagonal measurement covariance
  the ``m`` observations at each date can be absorbed one at a time by rank-one
  updates. This is exact -- it yields the identical likelihood -- and removes
  every matrix inversion from the inner loop (Durbin & Koopman, ch. 6).
* **Explicit loops rather than array expressions.** At k = 2..4 the per-call
  overhead of NumPy dominates, so the arithmetic is written out.

Two capabilities the prototype lacked, both required by the design:

* ``Hdiag`` is time-varying ``(n, m)``, which is what lets the price-channel
  variance be inflated inside roll windows (eq. 25).
* ``Q`` is time-varying ``(n, k, k)``, which carries the process-noise
  scaling of eq. (24).

``tests/test_nesting.py`` pins this against the reference filter.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap(args[0]) if args and callable(args[0]) else wrap


_LOG_2PI = float(np.log(2.0 * np.pi))


@njit(cache=True)
def filter_uv(y, Z, d, T, c, Q, Hdiag, Bu, x0, P0):
    """Forward filter by univariate updating. Returns the exact log-likelihood.

    Parameters
    ----------
    y      : (n, m)      observations; NaN marks missing
    Z      : (n, m, k)   observation loadings
    d      : (n, m)      observation intercepts
    T      : (k, k)      transition matrix
    c      : (n, k)      transition intercept
    Q      : (n, k, k)   state noise covariance
    Hdiag  : (n, m)      measurement variances (diagonal H, time-varying)
    Bu     : (n, k)      exogenous contribution B @ u_t
    x0, P0 : (k,), (k,k) prior

    Returns
    -------
    (loglik, x_filt, P_filt, x_pred, P_pred, resid, n_obs)
    """
    n, m = y.shape
    k = x0.shape[0]

    x_filt = np.empty((n, k))
    P_filt = np.empty((n, k, k))
    x_pred = np.empty((n, k))
    P_pred = np.empty((n, k, k))
    resid = np.full((n, m), np.nan)

    x = x0.copy()
    P = P0.copy()
    xp = np.empty(k)
    Pp = np.empty((k, k))
    Pz = np.empty(k)
    tmp = np.empty((k, k))

    ll = 0.0
    n_obs = 0

    for t in range(n):
        # ---- predict: xp = c + T x + Bu -------------------------------
        for a in range(k):
            s = c[t, a] + Bu[t, a]
            for b in range(k):
                s += T[a, b] * x[b]
            xp[a] = s

        # Pp = T P T' + Q
        for a in range(k):
            for b in range(k):
                s = 0.0
                for e in range(k):
                    s += T[a, e] * P[e, b]
                tmp[a, b] = s
        for a in range(k):
            for b in range(k):
                s = 0.0
                for e in range(k):
                    s += tmp[a, e] * T[b, e]
                Pp[a, b] = s + Q[t, a, b]
        # enforce symmetry
        for a in range(k):
            for b in range(a + 1, k):
                v = 0.5 * (Pp[a, b] + Pp[b, a])
                Pp[a, b] = v
                Pp[b, a] = v

        for a in range(k):
            x_pred[t, a] = xp[a]
            for b in range(k):
                P_pred[t, a, b] = Pp[a, b]

        # ---- sequential update over contracts --------------------------
        for i in range(m):
            yi = y[t, i]
            if np.isnan(yi):
                continue

            # v = y - d - z'xp ;  Pz = Pp z ;  F = z'Pz + H
            v = yi - d[t, i]
            for a in range(k):
                v -= Z[t, i, a] * xp[a]

            F = Hdiag[t, i]
            for a in range(k):
                s = 0.0
                for b in range(k):
                    s += Pp[a, b] * Z[t, i, b]
                Pz[a] = s
                F += Z[t, i, a] * s

            if F <= 0.0:
                return -np.inf, x_filt, P_filt, x_pred, P_pred, resid, n_obs

            ll += -0.5 * (_LOG_2PI + np.log(F) + v * v / F)
            n_obs += 1
            resid[t, i] = v / np.sqrt(F)

            # rank-one update
            for a in range(k):
                ka = Pz[a] / F
                xp[a] += ka * v
                for b in range(k):
                    Pp[a, b] -= ka * Pz[b]

        for a in range(k):
            x[a] = xp[a]
            x_filt[t, a] = xp[a]
            for b in range(k):
                P[a, b] = Pp[a, b]
                P_filt[t, a, b] = Pp[a, b]

    return ll, x_filt, P_filt, x_pred, P_pred, resid, n_obs


@njit(cache=True)
def smooth_rts(x_filt, P_filt, x_pred, P_pred, T):
    """RTS backward pass for general state dimension.

    Filtered states are what was knowable in real time and are the only honest
    input to forecasting. Smoothed states use the whole sample and belong in
    descriptive output.
    """
    n, k = x_filt.shape
    xs = x_filt.copy()
    Ps = P_filt.copy()

    for t in range(n - 2, -1, -1):
        Pp = P_pred[t + 1]
        Pinv = np.linalg.pinv(Pp)

        # J = P_filt[t] T' Pinv
        M = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                s = 0.0
                for e in range(k):
                    s += P_filt[t, a, e] * T[b, e]
                M[a, b] = s
        J = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                s = 0.0
                for e in range(k):
                    s += M[a, e] * Pinv[e, b]
                J[a, b] = s

        for a in range(k):
            s = x_filt[t, a]
            for b in range(k):
                s += J[a, b] * (xs[t + 1, b] - x_pred[t + 1, b])
            xs[t, a] = s

        D = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                D[a, b] = Ps[t + 1, a, b] - Pp[a, b]
        JD = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                s = 0.0
                for e in range(k):
                    s += J[a, e] * D[e, b]
                JD[a, b] = s
        for a in range(k):
            for b in range(k):
                s = P_filt[t, a, b]
                for e in range(k):
                    s += JD[a, e] * J[b, e]
                Ps[t, a, b] = s

    return xs, Ps
