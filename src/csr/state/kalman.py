"""Linear-Gaussian state-space filtering.

More general than a textbook Kalman filter in three ways the news-augmented
specification needs:

1. **Exogenous drivers** -- a ``Bu_t`` term in the transition equation, so
   sentiment enters the *dynamics* of the latent factors (Channel A) rather
   than being bolted on as another observed price.
2. **Time-varying system matrices** -- ``Q_t`` varies with time, letting news
   intensity scale the diffusion of the short-run factor (Channel B).
3. **Missing observations** -- futures panels are ragged (contract rolls,
   settlement gaps, thin deferred maturities). NaN rows are dropped from the
   update, which is the exact-likelihood treatment, not interpolation.

Notation follows Durbin & Koopman (2012):

    x_t = c_t + T_t x_{t-1} + Bu_t + w_t,   w_t ~ N(0, Q_t)
    y_t = d_t + Z_t x_t        + v_t,       v_t ~ N(0, H_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class FilterResult:
    """Output of a forward pass."""

    loglik: float
    x_filt: np.ndarray       # (n, k)     E[x_t | y_{1:t}]
    P_filt: np.ndarray       # (n, k, k)
    x_pred: np.ndarray       # (n, k)     E[x_t | y_{1:t-1}]
    P_pred: np.ndarray       # (n, k, k)
    resid: np.ndarray        # (n, m)     standardised innovations, NaN where missing
    loglik_t: np.ndarray     # (n,)       per-period contribution
    n_obs: int = 0
    smoothed: bool = False
    x_smooth: np.ndarray | None = field(default=None)
    P_smooth: np.ndarray | None = field(default=None)


def _stack(mat, n: int) -> np.ndarray:
    """Broadcast a constant system *matrix* (2-D) to a time-indexed stack."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim <= 2:
        return np.broadcast_to(mat, (n,) + mat.shape)
    return mat


def _stack_vec(vec, n: int, k: int) -> np.ndarray:
    """Broadcast a system *vector* to (n, k).

    A 1-D input is constant across time; a 2-D input is already time-indexed.
    Intercepts ``c`` and ``d`` are vectors per period, so they need this rather
    than :func:`_stack` -- ``d`` in particular is normally (n, m) already,
    because A(tau) changes as maturities shrink.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.ndim == 1:
        return np.broadcast_to(vec, (n, k))
    return vec.reshape(n, k)


def kalman_filter(
    y: np.ndarray,
    *,
    T: np.ndarray,
    Z: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    c: np.ndarray | None = None,
    d: np.ndarray | None = None,
    Bu: np.ndarray | None = None,
    x0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult:
    """Forward filter returning the exact Gaussian log-likelihood.

    Parameters
    ----------
    y : (n, m)
        Observations; ``np.nan`` marks a missing element.
    T, Z, Q, H :
        System matrices, constant (2-D) or time-indexed (3-D). ``Z`` and ``d``
        are normally time-varying here because time-to-maturity shrinks between
        roll dates.
    Bu : (n, k), optional
        Pre-multiplied exogenous contribution for each period.
    x0, P0 :
        Prior mean and covariance for the state before the first transition.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n, m = y.shape
    k = int(np.asarray(x0).size)

    Ts, Zs, Qs, Hs = _stack(T, n), _stack(Z, n), _stack(Q, n), _stack(H, n)
    cs = np.zeros((n, k)) if c is None else _stack_vec(c, n, k)
    ds = np.zeros((n, m)) if d is None else _stack_vec(d, n, m)
    bu = np.zeros((n, k)) if Bu is None else np.asarray(Bu, float).reshape(n, k)

    x = np.asarray(x0, float).reshape(k).copy()
    P = np.asarray(P0, float).reshape(k, k).copy()

    x_filt = np.empty((n, k)); P_filt = np.empty((n, k, k))
    x_pred = np.empty((n, k)); P_pred = np.empty((n, k, k))
    resid = np.full((n, m), np.nan)
    loglik_t = np.zeros(n)
    total_obs = 0
    ll = 0.0
    eye = np.eye(k)

    for t in range(n):
        # ---- predict -----------------------------------------------------
        xp = cs[t] + Ts[t] @ x + bu[t]
        Pp = Ts[t] @ P @ Ts[t].T + Qs[t]
        Pp = 0.5 * (Pp + Pp.T)
        x_pred[t], P_pred[t] = xp, Pp

        obs = np.flatnonzero(np.isfinite(y[t]))
        if obs.size == 0:
            x, P = xp, Pp
            x_filt[t], P_filt[t] = xp, Pp
            continue

        Zt = Zs[t][obs, :]
        Ht = Hs[t][np.ix_(obs, obs)]
        v = y[t, obs] - ds[t, obs] - Zt @ xp
        S = Zt @ Pp @ Zt.T + Ht
        S = 0.5 * (S + S.T)

        try:
            chol = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            try:
                chol = np.linalg.cholesky(S + 1e-10 * np.eye(S.shape[0]))
            except np.linalg.LinAlgError:
                # Let the optimiser walk away rather than crashing the run.
                return FilterResult(-np.inf, x_filt, P_filt, x_pred, P_pred,
                                    resid, loglik_t, total_obs)

        alpha = np.linalg.solve(chol, v)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        contrib = -0.5 * (obs.size * _LOG_2PI + log_det + float(alpha @ alpha))
        loglik_t[t] = contrib
        ll += contrib
        total_obs += obs.size

        # Gain via the Cholesky factor rather than an explicit inverse.
        K = Pp @ Zt.T @ np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(obs.size)))
        x = xp + K @ v
        # Joseph form: keeps P positive semi-definite under MLE search.
        IKZ = eye - K @ Zt
        P = IKZ @ Pp @ IKZ.T + K @ Ht @ K.T
        P = 0.5 * (P + P.T)

        resid[t, obs] = alpha
        x_filt[t], P_filt[t] = x, P

    return FilterResult(float(ll), x_filt, P_filt, x_pred, P_pred,
                        resid, loglik_t, total_obs)


def rts_smoother(res: FilterResult, T: np.ndarray) -> FilterResult:
    """Rauch-Tung-Striebel backward pass; fills ``x_smooth`` / ``P_smooth``.

    Filtered states are what a trader could have known in real time, so they are
    the honest input to forecasting. Smoothed states use the whole sample and
    belong in descriptive output -- e.g. plotting the long-run factor against
    known geopolitical episodes.
    """
    n, k = res.x_filt.shape
    Ts = _stack(T, n)
    xs = res.x_filt.copy()
    Ps = res.P_filt.copy()

    for t in range(n - 2, -1, -1):
        Pp = res.P_pred[t + 1]
        try:
            J = res.P_filt[t] @ Ts[t + 1].T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            J = res.P_filt[t] @ Ts[t + 1].T @ np.linalg.pinv(Pp)
        xs[t] = res.x_filt[t] + J @ (xs[t + 1] - res.x_pred[t + 1])
        Ps[t] = res.P_filt[t] + J @ (Ps[t + 1] - Pp) @ J.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)

    res.x_smooth, res.P_smooth, res.smoothed = xs, Ps, True
    return res
