"""Rung 2 of the escalation ladder: a genuinely front-concentrated fourth
factor, essentially-affine in ITS OWN right, rather than rung 1's
decoupling of the existing (front-weighted but curve-wide) chi1.

    ln S_t = chi_f + chi2 + chi3 + xi

    Q:  d chi_f = -k_f^Q*chi_f dt + s_f dW_f        (front, FIXED k_f^Q)
    P:  d chi_f = -k_f^P*chi_f dt + s_f dW_f        (front, FREE k_f^P)
        d chi2  = -k2*chi2 dt + s2 dW2              (medium, completely affine)
        d chi3  = -k3*chi3 dt + s3 dW3              (slow, completely affine)
        d xi    =  mu_xi dt   + sx dWx              (unit root)

    ln F(t,T) = e^{-k_f^Q tau} chi_f + e^{-k2 tau} chi2 + e^{-k3 tau} chi3 + xi + A(tau)

WHY THIS IS A DIFFERENT OBJECT FROM fourfactor.py's chi1 (explicitly ruled
out, explore/34: real fit, zero effect on Ljung-Box). That factor had a
FREELY estimated kappa1 governing BOTH its loading and its dynamics --
completely affine, same restriction as everything else in this project
until rung 1. Rung 1 (fit3.py's kappa1_phys/kappa2_phys, essentially-
affine on the EXISTING chi1/chi2) found a stable, well-identified kappa1^P
> kappa1^Q -- the WRONG direction for "front is closer to a random walk"
-- and 84% of its 3142-nat gain came from C5-C18, not the front. Diagnosis:
chi1 loads on every contract (front-weighted, not front-EXCLUSIVE), so
freeing its physical rate changes curve-wide dynamics, diluting any
C1-specific signal. This factor is built to not have that problem:

* kappa_f^Q is FIXED, not estimated, at 12.3 -- the exact anchor
  fourfactor.py's own from_three_factor() already uses (explore/33's
  ACF-implied rate). At the actual C1-C18 tau grid this gives loadings
  [0.59, 0.21, 0.08, 0.03, 0.01, ...] -- genuinely negligible by C4-C5,
  unlike chi1's own much slower e^{-1.9*tau} decay. Fixing it (rather
  than estimating it, which is what the refuted chi1 attempt did) keeps
  this factor pinned to the front by construction, so a freely-estimated
  kappa_f^P is a clean test of the SPECIFIC hypothesis (front dynamics
  decoupled from front loading), not entangled with "does the optimiser
  even want a front-concentrated shape."
* kappa_f^P is FREE (essentially-affine, same mechanism as fit3.py's
  kappa1_phys/kappa2_phys) -- the actual quantity of interest. Small/
  near-zero kappa_f^P would mean chi_f is close to a unit root under P
  while being sharply decaying under Q: exactly "decays fast along the
  curve, barely reverts in time," made representable because loading
  speed and time-series speed are no longer forced equal for this factor.
* kappa2/kappa3 (the OLD chi1/chi2, relabelled) stay completely affine
  (kappa^P = kappa^Q, unmodified) -- rung 1 already tested freeing THOSE
  and found it wasn't the right lever; re-testing it here would confound
  this rung's result. One thing at a time.
* rho12=rho13=rho23=0 (star correlation, matching fourfactor.py's own
  choice and threefactor.py's before it) -- chi_f correlates only with
  xi, not with chi2/chi3.

NESTING: sigma_f = lambda_f = 0 must reproduce the three-factor model
(chi2, chi3 as its own kappa1, kappa2) exactly -- same discipline
fourfactor.py's own docstring states for its chi3, asserted in
tests/test_fourfactor_front.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EPS = 1e-8
KAPPA_F_Q_ANCHOR = 12.3   # fixed, not estimated -- see module docstring

__all__ = ["FourFactorFrontParams", "A_tau4_front", "base_system4_front",
          "initial_state4_front", "KAPPA_F_Q_ANCHOR"]


def _m1(k: float, tau: np.ndarray) -> np.ndarray:
    k = max(k, EPS)
    return -np.expm1(-k * tau) / k


@dataclass(frozen=True)
class FourFactorFrontParams:
    """State ordered (chi_f, chi2, chi3, xi). kappa2 > kappa3 by
    construction (label-switching guard, same nested-gap discipline as
    threefactor.py/fourfactor.py); kappa_f_Q is fixed, not part of that
    ordering constraint (it's pinned far above kappa2's typical range by
    construction, not estimated)."""

    kappa2: float
    kappa3: float
    sigma_f: float
    sigma2: float
    sigma3: float
    sigma_xi: float
    kappa_f_Q: float = KAPPA_F_Q_ANCHOR
    kappa_f_P: float | None = None   # None -> kappa_f_Q (completely affine fallback)
    rho_fx: float = 0.0
    rho2x: float = 0.0
    rho3x: float = 0.0
    mu_xi: float = 0.0
    mu_xi_star: float = 0.0
    lambda_f: float = 0.0
    lambda2: float = 0.0
    lambda3: float = 0.0

    def __post_init__(self):
        if not (self.kappa2 > self.kappa3):
            raise ValueError(
                f"kappa2 must exceed kappa3 to break label switching; got "
                f"kappa2={self.kappa2}, kappa3={self.kappa3}."
            )
        if min(self.sigma_f, self.sigma2, self.sigma3, self.sigma_xi) < 0:
            raise ValueError("volatilities must be non-negative")
        s = self.rho_fx ** 2 + self.rho2x ** 2 + self.rho3x ** 2
        if s > 1.0 + 1e-10:
            raise ValueError(
                f"rho_fx^2 + rho2x^2 + rho3x^2 = {s:.6f} > 1 -- the star "
                f"correlation matrix is not PSD."
            )
        if self.kappa_f_P is None:
            object.__setattr__(self, "kappa_f_P", self.kappa_f_Q)
        if self.kappa_f_P <= 0:
            raise ValueError(f"kappa_f_P={self.kappa_f_P} must be positive")

    @property
    def half_life_front_phys_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa_f_P, EPS) * 365.25)

    @property
    def half_life_front_q_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa_f_Q, EPS) * 365.25)

    @property
    def beta_f(self) -> float:
        """Implied essentially-affine market-price-of-risk slope for the
        front factor: kappa_f_P = kappa_f_Q - sigma_f*beta_f."""
        return float((self.kappa_f_Q - self.kappa_f_P) / self.sigma_f) if self.sigma_f > 0 else 0.0

    @property
    def half_life_medium_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa2, EPS) * 365.25)

    @property
    def half_life_slow_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa3, EPS) * 365.25)

    def replace(self, **kw) -> "FourFactorFrontParams":
        return replace(self, **kw)


def A_tau4_front(p: FourFactorFrontParams, tau) -> np.ndarray:
    """Deterministic log-futures adjustment. Uses kappa_f_Q (risk-neutral)
    for chi_f's contribution -- pricing/drift side, untouched by the
    essentially-affine split, same convention as fit3.py's A_tau3."""
    tau = np.asarray(tau, float)
    kf, k2, k3 = max(p.kappa_f_Q, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)

    af, a2, a3 = _m1(kf, tau), _m1(k2, tau), _m1(k3, tau)
    vff = -np.expm1(-2.0 * kf * tau) / (2.0 * kf)
    v22 = -np.expm1(-2.0 * k2 * tau) / (2.0 * k2)
    v33 = -np.expm1(-2.0 * k3 * tau) / (2.0 * k3)

    drift = p.mu_xi_star * tau - af * p.lambda_f - a2 * p.lambda2 - a3 * p.lambda3
    var = (vff * p.sigma_f ** 2 + v22 * p.sigma2 ** 2 + v33 * p.sigma3 ** 2
           + p.sigma_xi ** 2 * tau
           + 2.0 * af * p.rho_fx * p.sigma_f * p.sigma_xi
           + 2.0 * a2 * p.rho2x * p.sigma2 * p.sigma_xi
           + 2.0 * a3 * p.rho3x * p.sigma3 * p.sigma_xi)
    return drift + 0.5 * var


def base_system4_front(p: FourFactorFrontParams, tau, dt: float) -> dict:
    """T/Q use kappa_f_P (physical) for chi_f; Z/A_tau4_front use kappa_f_Q
    (risk-neutral, fixed at the anchor) -- same essentially-affine split
    fit3.py's base_system3 makes for kappa1_phys/kappa2_phys, applied to
    a factor that's front-concentrated BY CONSTRUCTION rather than by
    estimation."""
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = tau.shape
    kf_q = max(p.kappa_f_Q, EPS)
    kf_p, k2, k3 = max(p.kappa_f_P, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)

    T = np.diag([np.exp(-kf_p * dt), np.exp(-k2 * dt), np.exp(-k3 * dt), 1.0])
    c = np.zeros((n, 4))
    c[:, 3] = p.mu_xi * dt

    qff = -np.expm1(-2.0 * kf_p * dt) * p.sigma_f ** 2 / (2.0 * kf_p)
    q22 = -np.expm1(-2.0 * k2 * dt) * p.sigma2 ** 2 / (2.0 * k2)
    q33 = -np.expm1(-2.0 * k3 * dt) * p.sigma3 ** 2 / (2.0 * k3)
    q44 = p.sigma_xi ** 2 * dt
    qf4 = -np.expm1(-kf_p * dt) * p.rho_fx * p.sigma_f * p.sigma_xi / kf_p
    q24 = -np.expm1(-k2 * dt) * p.rho2x * p.sigma2 * p.sigma_xi / k2
    q34 = -np.expm1(-k3 * dt) * p.rho3x * p.sigma3 * p.sigma_xi / k3
    Q = np.array([[qff, 0.0, 0.0, qf4],
                 [0.0, q22, 0.0, q24],
                 [0.0, 0.0, q33, q34],
                 [qf4, q24, q34, q44]])
    Q = np.repeat(Q[None, :, :], n, axis=0)

    Z = np.empty((n, m, 4))
    Z[:, :, 0] = np.exp(-kf_q * tau)   # kappa_f_Q, FIXED, risk-neutral loading
    Z[:, :, 1] = np.exp(-k2 * tau)
    Z[:, :, 2] = np.exp(-k3 * tau)
    Z[:, :, 3] = 1.0

    return {"T": T, "c": c, "Q": Q, "Z": Z, "d": A_tau4_front(p, tau),
            "Hdiag": np.ones((n, m)), "n": n, "m": m, "k": 4}


def initial_state4_front(log_f, tau, p: FourFactorFrontParams):
    """Stationary priors -- chi_f's uses kappa_f_P (physical), same
    reasoning as fit3.py's initial_state3."""
    kf_p, k2, k3 = max(p.kappa_f_P, EPS), max(p.kappa2, EPS), max(p.kappa3, EPS)
    row = np.atleast_2d(np.asarray(log_f, float))[0]
    t0 = np.atleast_2d(np.asarray(tau, float))[0]
    ok = np.isfinite(row)
    xi0 = float(np.mean(row[ok] - A_tau4_front(p, t0[ok]))) if ok.any() else 0.0
    x0 = np.array([0.0, 0.0, 0.0, xi0])
    P0 = np.diag([p.sigma_f ** 2 / (2.0 * kf_p),
                  p.sigma2 ** 2 / (2.0 * k2),
                  p.sigma3 ** 2 / (2.0 * k3),
                  1.0])
    return x0, P0
