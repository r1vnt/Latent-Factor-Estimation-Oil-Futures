"""Three-factor model: two mean-reverting factors plus a random walk.

    ln S_t = chi1_t + chi2_t + xi_t

    P:  d chi1 = -k1*chi1 dt + s1 dW1        (fast)
        d chi2 = -k2*chi2 dt + s2 dW2        (slow)
        d xi   =  mu_xi dt    + sx dWx       (unit root)
    Q:  drifts pick up -lam1, -lam2, mu_xi_star

    ln F(t,T) = e^{-k1 tau} chi1 + e^{-k2 tau} chi2 + xi + A(tau)

WHY THIS EXISTS. The C1-C18 perturbation battery separated two things that had
been tangled together all along. The free-diagonal Chen-Scott degeneracy is
gone -- zero pinned contracts in all 26 fits, cold starts reproducing warm to
four decimals. What remains is not an identification failure but a
specification one, and it has a signature:

    drop C13-C18  ->  kappa = 1.055      (front-heavy sample wants fast decay)
    full C1-C18   ->  kappa = 0.757
    drop C1-C4    ->  kappa = 0.694      (deep sample wants slow decay)
    drop C5-C12   ->  kappa = 0.636

Leave-one-contract-out barely moves it (spread 0.068 over 18 fits); dropping
whole segments moves it six times as much (spread 0.42). A single exponential
cannot match the curvature at the front and the flatness at the back at the
same time, so kappa lands wherever the panel's centre of mass puts it. That is
a model too rigid for the data, not data too thin for the model -- and the
remedy is a second decay rate rather than a deeper panel.

With k1 fast and k2 slow the front curvature and the back flattening are
carried by different factors, and neither has to compromise. The prediction the
fix must satisfy: **k1 and k2 stable across the same block-outs that moved a
single kappa by 0.42.** That is the acceptance test, and it is a direct test of
the defect rather than a p-value.

IDENTIFICATION NOTES, both of which will silently corrupt results if ignored.

* The two factors are exchangeable. Without an ordering constraint the
  optimiser may swap them between starts, and a multi-start check then reports
  spurious multimodality that is only label switching. ``ThreeFactorParams``
  enforces k1 > k2 at construction.
* Under sigma2 = 0 the parameters k2, lam2, rho12 and rho2x are unidentified.
  The LR statistic for adding the factor therefore does NOT have a chi2(5)
  null -- this is the Davies problem, a boundary null with unidentified
  nuisance parameters. Use a parametric bootstrap for the null distribution, or
  rely on the perturbation battery, which needs no null at all.

Setting sigma2 = 0 reproduces the two-factor model exactly, which is asserted
in tests/test_threefactor.py against the existing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EPS = 1e-8

__all__ = ["ThreeFactorParams", "A_tau3", "base_system3", "initial_state3",
           "from_two_factor"]


def _m1(k: float, tau: np.ndarray) -> np.ndarray:
    """(1 - e^{-k tau}) / k, stable as k -> 0."""
    k = max(k, EPS)
    return -np.expm1(-k * tau) / k


@dataclass(frozen=True)
class ThreeFactorParams:
    """Structural parameters. ``kappa1`` is the fast factor by construction."""

    kappa1: float
    kappa2: float
    sigma1: float
    sigma2: float
    sigma_xi: float
    rho12: float = 0.0
    rho1x: float = 0.0
    rho2x: float = 0.0
    mu_xi: float = 0.0
    mu_xi_star: float = 0.0
    lambda1: float = 0.0
    lambda2: float = 0.0
    gamma_v: float = 0.0      # lagged realised vol -> process noise (eq. 24)

    def __post_init__(self):
        if not (self.kappa1 > self.kappa2):
            raise ValueError(
                f"kappa1 must exceed kappa2 to break label switching; got "
                f"kappa1={self.kappa1}, kappa2={self.kappa2}. Parameterise as "
                f"kappa1 = kappa2 + exp(gap) so the optimiser cannot swap them."
            )
        if min(self.sigma1, self.sigma2, self.sigma_xi) < 0:
            raise ValueError("volatilities must be non-negative")
        ev = np.linalg.eigvalsh(self.corr())
        if ev.min() < -1e-10:
            raise ValueError(
                f"correlation matrix is not positive semi-definite "
                f"(min eigenvalue {ev.min():.2e}); rho12, rho1x, rho2x are not "
                f"free of one another"
            )

    def corr(self) -> np.ndarray:
        return np.array([[1.0, self.rho12, self.rho1x],
                         [self.rho12, 1.0, self.rho2x],
                         [self.rho1x, self.rho2x, 1.0]])

    @property
    def half_life_fast_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa1, EPS) * 365.25)

    @property
    def half_life_slow_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa2, EPS) * 365.25)

    def replace(self, **kw) -> "ThreeFactorParams":
        return replace(self, **kw)


def A_tau3(p: ThreeFactorParams, tau) -> np.ndarray:
    """Deterministic log-futures adjustment. A(0) = 0 exactly.

    Reduces to the two-factor ``A_tau`` term for term when sigma2 = lambda2 = 0.
    """
    tau = np.asarray(tau, float)
    k1, k2 = max(p.kappa1, EPS), max(p.kappa2, EPS)

    a1 = _m1(k1, tau)
    a2 = _m1(k2, tau)
    v11 = -np.expm1(-2.0 * k1 * tau) / (2.0 * k1)
    v22 = -np.expm1(-2.0 * k2 * tau) / (2.0 * k2)
    v12 = -np.expm1(-(k1 + k2) * tau) / (k1 + k2)

    drift = p.mu_xi_star * tau - a1 * p.lambda1 - a2 * p.lambda2
    var = (v11 * p.sigma1 ** 2
           + v22 * p.sigma2 ** 2
           + p.sigma_xi ** 2 * tau
           + 2.0 * v12 * p.rho12 * p.sigma1 * p.sigma2
           + 2.0 * a1 * p.rho1x * p.sigma1 * p.sigma_xi
           + 2.0 * a2 * p.rho2x * p.sigma2 * p.sigma_xi)
    return drift + 0.5 * var


def base_system3(p: ThreeFactorParams, tau, dt: float) -> dict:
    """Three-dimensional state-space matrices, state ordered (chi1, chi2, xi).

    ``Hdiag`` is left as ones: build it from
    :func:`csr.pricing.smoothmeas.hdiag_from_tau` and overwrite. Keeping the two
    apart is deliberate -- the measurement curve is what removed the Chen-Scott
    degeneracy and it should not be silently re-specified here.
    """
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = tau.shape
    k1, k2 = max(p.kappa1, EPS), max(p.kappa2, EPS)

    T = np.diag([np.exp(-k1 * dt), np.exp(-k2 * dt), 1.0])
    c = np.zeros((n, 3))
    c[:, 2] = p.mu_xi * dt

    q11 = -np.expm1(-2.0 * k1 * dt) * p.sigma1 ** 2 / (2.0 * k1)
    q22 = -np.expm1(-2.0 * k2 * dt) * p.sigma2 ** 2 / (2.0 * k2)
    q33 = p.sigma_xi ** 2 * dt
    q12 = -np.expm1(-(k1 + k2) * dt) * p.rho12 * p.sigma1 * p.sigma2 / (k1 + k2)
    q13 = -np.expm1(-k1 * dt) * p.rho1x * p.sigma1 * p.sigma_xi / k1
    q23 = -np.expm1(-k2 * dt) * p.rho2x * p.sigma2 * p.sigma_xi / k2
    Q = np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]])
    Q = np.repeat(Q[None, :, :], n, axis=0)

    Z = np.empty((n, m, 3))
    Z[:, :, 0] = np.exp(-k1 * tau)
    Z[:, :, 1] = np.exp(-k2 * tau)
    Z[:, :, 2] = 1.0

    return {"T": T, "c": c, "Q": Q, "Z": Z, "d": A_tau3(p, tau),
            "Hdiag": np.ones((n, m)), "n": n, "m": m, "k": 3}


def scale_variances3(sys: dict, p: ThreeFactorParams, *, rv_lag=None) -> dict:
    """Lagged-realised-vol process-noise scaling (eq. 24), ported from
    twofactor.scale_variances to the two mean-reverting factors here.

    Both chi1 and chi2 share the single gamma_v -- one common "how
    volatile is the market right now" regime scaling both factors'
    shock magnitude jointly, not two independent volatility channels.
    Only entries touching chi1/chi2 scale (by s or sqrt(s), matching
    twofactor.py's convention for how a scaled shock's own variance and
    its covariance with an unscaled variable transform); q33 (xi-xi) is
    untouched, and q12 scales by s but is 0 whenever rho12=0 regardless.

    ``rv_lag`` must already be lagged by the caller -- see
    twofactor.scale_variances's docstring for why (contemporaneous RV
    would make the state prior a function of the data being explained).
    """
    n = sys["n"]
    out = dict(sys)
    if rv_lag is None or not p.gamma_v:
        return out
    s = np.exp(np.clip(p.gamma_v * np.asarray(rv_lag, float), -20.0, 20.0))
    Q = sys["Q"].copy()
    Q[:, 0, 0] *= s
    Q[:, 1, 1] *= s
    Q[:, 0, 1] *= s
    Q[:, 1, 0] *= s
    sqrt_s = np.sqrt(s)
    Q[:, 0, 2] *= sqrt_s
    Q[:, 2, 0] *= sqrt_s
    Q[:, 1, 2] *= sqrt_s
    Q[:, 2, 1] *= sqrt_s
    out["Q"] = Q
    return out


def initial_state3(log_f, tau, p: ThreeFactorParams):
    """Stationary priors for both mean-reverting factors, diffuse for xi."""
    k1, k2 = max(p.kappa1, EPS), max(p.kappa2, EPS)
    row = np.atleast_2d(np.asarray(log_f, float))[0]
    t0 = np.atleast_2d(np.asarray(tau, float))[0]
    ok = np.isfinite(row)
    xi0 = float(np.mean(row[ok] - A_tau3(p, t0[ok]))) if ok.any() else 0.0
    x0 = np.array([0.0, 0.0, xi0])
    P0 = np.diag([p.sigma1 ** 2 / (2.0 * k1),
                  p.sigma2 ** 2 / (2.0 * k2),
                  1.0])
    return x0, P0


def from_two_factor(p2, *, kappa_ratio: float = 4.0, split: float = 0.5
                    ) -> ThreeFactorParams:
    """Warm start from a fitted two-factor model.

    Places the fitted ``kappa`` between the two new rates and splits the
    short-run variance between them, which is the sensible starting guess when
    a single exponential has been averaging two.
    """
    k = max(getattr(p2, "kappa", 1.0), EPS)
    r = max(kappa_ratio, 1.0 + 1e-6)
    k1, k2 = k * np.sqrt(r), k / np.sqrt(r)
    s = getattr(p2, "sigma_chi", 0.3)
    return ThreeFactorParams(
        kappa1=float(k1), kappa2=float(k2),
        sigma1=float(s * np.sqrt(split)),
        sigma2=float(s * np.sqrt(1.0 - split)),
        sigma_xi=float(getattr(p2, "sigma_xi", 0.2)),
        rho1x=float(getattr(p2, "rho", 0.0)),
        mu_xi=float(getattr(p2, "mu_xi", 0.0)),
        mu_xi_star=float(getattr(p2, "mu_xi_star", 0.0)),
        lambda1=float(getattr(p2, "lambda_chi", 0.0)),
    )
