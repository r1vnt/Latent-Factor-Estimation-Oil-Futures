"""Four-factor model: three mean-reverting factors plus a random walk.

    ln S_t = chi1_t + chi2_t + chi3_t + xi_t

    P:  d chi1 = -k1*chi1 dt + s1 dW1        (fast)
        d chi2 = -k2*chi2 dt + s2 dW2        (medium)
        d chi3 = -k3*chi3 dt + s3 dW3        (slow)
        d xi   =  mu_xi dt    + sx dWx       (unit root)

    ln F(t,T) = e^{-k1 tau} chi1 + e^{-k2 tau} chi2 + e^{-k3 tau} chi3 + xi + A(tau)

WHY THIS EXISTS. The three-factor model's own battery found the sliding
defect redistributed, not fixed: kappa2 (slow) stabilised across maturity
block-outs (spread 0.065, under the 0.15 target) but kappa1 (fast, 134d
half-life) got WORSE (spread 0.731, worse than the single-kappa two-factor
model's 0.42). Ljung-Box on the three-factor residuals rejected whiteness
in all 18 contracts at every lag through 20, and the fitted ACF decay
(explore/33) implies a component with a ~20-trading-day half-life -- about
6.5x faster than kappa1's 134 days. kappa1 was carrying two jobs: genuine
medium-run curve curvature, and (badly) standing in for a faster component
it has no rate to represent. This adds that third rate rather than asking
kappa1 to keep compromising.

Prediction the fix must satisfy, symmetric to threefactor.py's own
acceptance test: kappa1 (now the medium class) should stabilise across the
same maturity block-outs where the three-factor model's kappa1 slid.
kappa3 (slow) should keep passing where kappa2 already did. If kappa1 is
still sliding with a fast class in place, the missing dynamics are not
simply "one rate faster" and the diagnosis needs revisiting, not another
factor bolted on reflexively.

IDENTIFICATION NOTES, extending threefactor.py's:

* Label switching among three exchangeable mean-reverting factors is
  broken by construction via the SAME nested-gap reparameterisation
  threefactor.py uses for two: kappa3 free, kappa2 = kappa3 + exp(gap23),
  kappa1 = kappa2 + exp(gap12). FourFactorParams.__post_init__ still
  checks kappa1 > kappa2 > kappa3 directly for anyone constructing a
  ThreeFactorParams by hand.
* rho12 = rho13 = rho23 = 0, matching threefactor.py's choice to only let
  each mean-reverting factor correlate with the random walk, not with each
  other. The resulting correlation matrix is a "star": zero among
  chi1/chi2/chi3, rho_ix on each's edge to xi. Its PSD condition
  generalises threefactor.py's rho1x^2 + rho2x^2 <= 1 directly, by the same
  Schur-complement argument, to rho1x^2 + rho2x^2 + rho3x^2 <= 1 --
  FourFactorParams.__post_init__ checks this exactly, not the pairwise
  bounds alone (each rho_ix individually in (-1, 1) does not imply the sum
  of squares is; this is exactly the failure mode that crashed the
  three-factor battery once already -- see fit3.py's try/except).

Setting sigma3 = lambda3 = 0 must reproduce the three-factor model exactly
-- asserted in tests/test_fourfactor.py, the same nesting discipline
threefactor.py used against the two-factor model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EPS = 1e-8

__all__ = ["FourFactorParams", "A_tau4", "base_system4", "initial_state4",
          "from_three_factor"]


def _m1(k: float, tau: np.ndarray) -> np.ndarray:
    k = max(k, EPS)
    return -np.expm1(-k * tau) / k


@dataclass(frozen=True)
class FourFactorParams:
    """Structural parameters. kappa1 > kappa2 > kappa3 by construction."""

    kappa1: float
    kappa2: float
    kappa3: float
    sigma1: float
    sigma2: float
    sigma3: float
    sigma_xi: float
    rho1x: float = 0.0
    rho2x: float = 0.0
    rho3x: float = 0.0
    mu_xi: float = 0.0
    mu_xi_star: float = 0.0
    lambda1: float = 0.0
    lambda2: float = 0.0
    lambda3: float = 0.0

    def __post_init__(self):
        if not (self.kappa1 > self.kappa2 > self.kappa3):
            raise ValueError(
                f"kappa1 > kappa2 > kappa3 required to break label switching; "
                f"got kappa1={self.kappa1}, kappa2={self.kappa2}, "
                f"kappa3={self.kappa3}. Parameterise via nested gaps: "
                f"kappa3 free, kappa2 = kappa3 + exp(gap23), "
                f"kappa1 = kappa2 + exp(gap12)."
            )
        if min(self.sigma1, self.sigma2, self.sigma3, self.sigma_xi) < 0:
            raise ValueError("volatilities must be non-negative")
        s = self.rho1x ** 2 + self.rho2x ** 2 + self.rho3x ** 2
        if s > 1.0 + 1e-10:
            raise ValueError(
                f"rho1x^2 + rho2x^2 + rho3x^2 = {s:.6f} > 1 -- the star "
                f"correlation matrix (chi1, chi2, chi3 mutually uncorrelated, "
                f"each correlated with xi) is not PSD. Each rho_ix individually "
                f"in (-1, 1) does not imply this; check the sum of squares, not "
                f"the pairwise bounds alone."
            )

    @property
    def half_life_fast_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa1, EPS) * 365.25)

    @property
    def half_life_medium_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa2, EPS) * 365.25)

    @property
    def half_life_slow_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa3, EPS) * 365.25)

    def replace(self, **kw) -> "FourFactorParams":
        return replace(self, **kw)


def A_tau4(p: FourFactorParams, tau) -> np.ndarray:
    """Deterministic log-futures adjustment. A(0) = 0 exactly.

    Reduces to the three-factor A_tau3 when sigma3 = lambda3 = 0.
    """
    tau = np.asarray(tau, float)
    k1, k2, k3 = max(p.kappa1, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)

    a1, a2, a3 = _m1(k1, tau), _m1(k2, tau), _m1(k3, tau)
    v11 = -np.expm1(-2.0 * k1 * tau) / (2.0 * k1)
    v22 = -np.expm1(-2.0 * k2 * tau) / (2.0 * k2)
    v33 = -np.expm1(-2.0 * k3 * tau) / (2.0 * k3)

    drift = p.mu_xi_star * tau - a1 * p.lambda1 - a2 * p.lambda2 - a3 * p.lambda3
    var = (v11 * p.sigma1 ** 2 + v22 * p.sigma2 ** 2 + v33 * p.sigma3 ** 2
           + p.sigma_xi ** 2 * tau
           + 2.0 * a1 * p.rho1x * p.sigma1 * p.sigma_xi
           + 2.0 * a2 * p.rho2x * p.sigma2 * p.sigma_xi
           + 2.0 * a3 * p.rho3x * p.sigma3 * p.sigma_xi)
    return drift + 0.5 * var


def base_system4(p: FourFactorParams, tau, dt: float) -> dict:
    """Four-dimensional state-space matrices, state ordered (chi1, chi2, chi3, xi).

    Hdiag left as ones -- build it from smoothmeas.hdiag_from_tau, same
    separation threefactor.py insists on and for the same reason.
    """
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = tau.shape
    k1, k2, k3 = max(p.kappa1, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)

    T = np.diag([np.exp(-k1 * dt), np.exp(-k2 * dt), np.exp(-k3 * dt), 1.0])
    c = np.zeros((n, 4))
    c[:, 3] = p.mu_xi * dt

    q11 = -np.expm1(-2.0 * k1 * dt) * p.sigma1 ** 2 / (2.0 * k1)
    q22 = -np.expm1(-2.0 * k2 * dt) * p.sigma2 ** 2 / (2.0 * k2)
    q33 = -np.expm1(-2.0 * k3 * dt) * p.sigma3 ** 2 / (2.0 * k3)
    q44 = p.sigma_xi ** 2 * dt
    q14 = -np.expm1(-k1 * dt) * p.rho1x * p.sigma1 * p.sigma_xi / k1
    q24 = -np.expm1(-k2 * dt) * p.rho2x * p.sigma2 * p.sigma_xi / k2
    q34 = -np.expm1(-k3 * dt) * p.rho3x * p.sigma3 * p.sigma_xi / k3
    Q = np.array([[q11, 0.0, 0.0, q14],
                 [0.0, q22, 0.0, q24],
                 [0.0, 0.0, q33, q34],
                 [q14, q24, q34, q44]])
    Q = np.repeat(Q[None, :, :], n, axis=0)

    Z = np.empty((n, m, 4))
    Z[:, :, 0] = np.exp(-k1 * tau)
    Z[:, :, 1] = np.exp(-k2 * tau)
    Z[:, :, 2] = np.exp(-k3 * tau)
    Z[:, :, 3] = 1.0

    return {"T": T, "c": c, "Q": Q, "Z": Z, "d": A_tau4(p, tau),
            "Hdiag": np.ones((n, m)), "n": n, "m": m, "k": 4}


def initial_state4(log_f, tau, p: FourFactorParams):
    """Stationary priors for all three mean-reverting factors, diffuse for xi."""
    k1, k2, k3 = max(p.kappa1, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)
    row = np.atleast_2d(np.asarray(log_f, float))[0]
    t0 = np.atleast_2d(np.asarray(tau, float))[0]
    ok = np.isfinite(row)
    xi0 = float(np.mean(row[ok] - A_tau4(p, t0[ok]))) if ok.any() else 0.0
    x0 = np.array([0.0, 0.0, 0.0, xi0])
    P0 = np.diag([p.sigma1 ** 2 / (2.0 * k1),
                  p.sigma2 ** 2 / (2.0 * k2),
                  p.sigma3 ** 2 / (2.0 * k3),
                  1.0])
    return x0, P0


def from_three_factor(p3, *, kappa_fast: float = 12.3, split: float = 0.5
                      ) -> FourFactorParams:
    """Warm start from a fitted three-factor model.

    p3's (kappa1, kappa2) become the new (kappa2, kappa3) -- medium and
    slow -- and a genuinely fast rate is inserted above them. ``kappa_fast``
    defaults to explore/33's ACF-implied rate (~20.5 trading day half-life),
    not an arbitrary guess. Splits p3.sigma1 (the old fast factor's
    variance) between the new fast factor and a share left with the old
    one, on the view that the old kappa1 was partly standing in for the
    missing rate and its variance should partly transfer.
    """
    k2 = max(getattr(p3, "kappa1", 1.0), EPS)   # old fast -> new medium
    k3 = max(getattr(p3, "kappa2", 0.3), EPS)   # old slow -> new slow
    if not (kappa_fast > k2):
        raise ValueError(f"kappa_fast={kappa_fast} must exceed the old kappa1={k2}")
    s_old_fast = getattr(p3, "sigma1", 0.3)
    return FourFactorParams(
        kappa1=float(kappa_fast), kappa2=float(k2), kappa3=float(k3),
        sigma1=float(s_old_fast * np.sqrt(split)),
        sigma2=float(s_old_fast * np.sqrt(1.0 - split)),
        sigma3=float(getattr(p3, "sigma2", 0.2)),
        sigma_xi=float(getattr(p3, "sigma_xi", 0.2)),
        rho1x=0.0, rho2x=float(getattr(p3, "rho1x", 0.0)),
        rho3x=float(getattr(p3, "rho2x", 0.0)),
        mu_xi=float(getattr(p3, "mu_xi", 0.0)),
        mu_xi_star=float(getattr(p3, "mu_xi_star", 0.0)),
        lambda1=0.0, lambda2=float(getattr(p3, "lambda1", 0.0)),
        lambda3=float(getattr(p3, "lambda2", 0.0)),
    )
