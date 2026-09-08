"""MLE fitting for the three-factor news lattice (S0/SD/SM/SDM on
specs3.run_spec3), extending fit3.py's pack3_smooth/unpack3_smooth with a
news-parameter slice sized by Spec.n_news_params.

Theta layout: fit3.pack3_smooth's own vector (structural incl. gamma_v,
then the smooth curve), followed by, in this order:
    driver (SD, SDM):       B raveled,  (2, p) row-major -> 2p entries
    measurement (SM, SDM):  G raveled,  (p, 2) row-major -> 2p entries,
                             then a_n -> p entries
S0 appends nothing. ``n_base`` (the length of the structural+curve prefix)
must be computed once by the caller via ``len(pack3_smooth(...))`` and
passed through everywhere -- kept explicit rather than hard-coded so this
file never silently drifts out of sync with fit3.py's own layout.

Student-t reweighting applies to the PRICE rows only, via the same
IRLS scheme as fit3.fit3_student_t_smooth (student_t_weights3 on the
price-row residuals). News rows stay Gaussian throughout -- see
specs3.run_spec3's ``weight`` docstring for why.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .fit3 import pack3_smooth, student_t_weights3, unpack3_smooth
from .specs import Spec
from .specs3 import run_spec3

__all__ = ["pack3_news", "unpack3_news", "fit3_spec_gaussian", "fit3_spec_student_t"]


def pack3_news(theta_base: np.ndarray, spec: Spec, B=None, G=None, a_n=None
              ) -> np.ndarray:
    extra = []
    if spec.driver:
        extra.append(np.asarray(B, float).ravel())
    if spec.measurement:
        extra.append(np.asarray(G, float).ravel())
        extra.append(np.asarray(a_n, float).ravel())
    return np.concatenate([theta_base] + extra) if extra else np.asarray(theta_base, float)


def unpack3_news(theta: np.ndarray, spec: Spec, quadratic: bool, p_features: int,
                 n_base: int):
    theta = np.asarray(theta, float)
    p, smooth = unpack3_smooth(theta[:n_base], quadratic)
    idx = n_base
    B = G = a_n = None
    if spec.driver:
        B = theta[idx: idx + 2 * p_features].reshape(2, p_features)
        idx += 2 * p_features
    if spec.measurement:
        G = theta[idx: idx + 2 * p_features].reshape(p_features, 2)
        idx += 2 * p_features
        a_n = theta[idx: idx + p_features]
        idx += p_features
    return p, smooth, B, G, a_n


def _neg_loglik3_news(theta, log_f, tau, dt, quadratic, n_base, spec, news,
                      p_features, weight, rv_lag):
    try:
        p, smooth, B, G, a_n = unpack3_news(theta, spec, quadratic, p_features, n_base)
    except ValueError:
        return 1e10   # same PSD/ordering guard as fit3.py's _neg_loglik3
    ll, *_ = run_spec3(spec, p, smooth, log_f, tau, dt, news=news, B=B, G=G,
                       a_n=a_n, rv_lag=rv_lag, weight=weight)
    return -ll if np.isfinite(ll) else 1e10


def fit3_spec_gaussian(spec: Spec, log_f, tau, dt, theta0: np.ndarray,
                       news=None, quadratic: bool = True, n_base: int | None = None,
                       p_features: int | None = None, rv_lag=None):
    if n_base is None or p_features is None:
        raise ValueError("n_base and p_features must be given explicitly")
    m = log_f.shape[1]
    weight = np.ones((log_f.shape[0], m))
    res = minimize(_neg_loglik3_news, theta0,
                   args=(log_f, tau, dt, quadratic, n_base, spec, news, p_features,
                        weight, rv_lag),
                   method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    if not res.success:
        res2 = minimize(_neg_loglik3_news, res.x,
                        args=(log_f, tau, dt, quadratic, n_base, spec, news,
                             p_features, weight, rv_lag),
                        method="Nelder-Mead",
                        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    p_hat, smooth_hat, B_hat, G_hat, a_n_hat = unpack3_news(
        res.x, spec, quadratic, p_features, n_base)
    ll, x_filt, resid, n_obs = run_spec3(spec, p_hat, smooth_hat, log_f, tau, dt,
                                        news=news, B=B_hat, G=G_hat, a_n=a_n_hat,
                                        rv_lag=rv_lag, weight=weight)
    return p_hat, smooth_hat, B_hat, G_hat, a_n_hat, x_filt, resid, n_obs, ll


def fit3_spec_student_t(spec: Spec, log_f, tau, dt, df: float, theta0: np.ndarray,
                        news=None, quadratic: bool = True, n_base: int | None = None,
                        p_features: int | None = None, rv_lag=None,
                        n_outer: int = 200, tol: float = 1e-6):
    if n_base is None or p_features is None:
        raise ValueError("n_base and p_features must be given explicitly")
    n, m = log_f.shape
    weight = np.ones((n, m))
    theta = np.asarray(theta0, float).copy()
    p, smooth, B, G, a_n = unpack3_news(theta, spec, quadratic, p_features, n_base)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik3_news, theta,
                       args=(log_f, tau, dt, quadratic, n_base, spec, news,
                            p_features, weight, rv_lag),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p, smooth, B, G, a_n = unpack3_news(theta, spec, quadratic, p_features, n_base)
        _, _, resid_raw, _ = run_spec3(spec, p, smooth, log_f, tau, dt, news=news,
                                       B=B, G=G, a_n=a_n, rv_lag=rv_lag,
                                       weight=np.ones((n, m)))
        price_resid = resid_raw[:, :m]
        new_weight = student_t_weights3(price_resid, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = run_spec3(spec, p, smooth, log_f, tau, dt,
                                             news=news, B=B, G=G, a_n=a_n,
                                             rv_lag=rv_lag, weight=np.ones((n, m)))
    return p, smooth, B, G, a_n, x_filt, resid_raw, weight, it + 1, ll
