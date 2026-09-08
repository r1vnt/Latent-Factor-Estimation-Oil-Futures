"""Smooth measurement-error curve: s(tau) = exp(a + b*tau).

Replaces the m free per-contract variances with two parameters, for one
structural reason.

WHY. With k latent factors, m > k contracts and a free diagonal H, the Gaussian
likelihood can always be raised by pricing exactly k contracts *perfectly* --
inverting the state from them -- and dumping all misfit onto the remaining
m - k. This is the Chen-Scott normalisation, and a free-s_i optimiser
rediscovers it unprompted, choosing which contracts to sacrifice by likelihood
rather than by the researcher. Observed directly on the C1-C8 panel: releasing
the tick floor sent s_5 to 0.0021 bp and s_7 to 0.0233 bp -- 60-700x below one
tick -- for 47 nats, while leaving C1 at 320 bp and C2 at 162 bp. Two factors,
two collapsed contracts. The front-end misfit is not a failure to fit; it is
the price the optimiser pays to price two contracts exactly.

The degeneracy cannot be reached under this parameterisation. Driving one
contract's noise to zero requires a + b*tau_i -> -inf, which drags its
neighbours down with it because tau is near-continuous across the panel. The
spread of s across the curve is bounded by exp(|b| * (tau_max - tau_min)), so
"price C5 exactly, abandon C1" is not a point in the parameter space.

Three further consequences, all in the right direction:

* Economically honest. Settlement error should vary smoothly with maturity and
  liquidity. The step profiles being read as evidence -- [54.9, 6.5, 5.1, 16.6]
  at C1-C4, [320, 162, 53, 16, 1.4, 5.5, 1.4, 10.0] at C1-C8 -- were never
  plausible as physical quantities, and both turned out to be artefacts.
* Cost. C1-C18 falls from ~24 free parameters to ~8. The Nelder-Mead cost that
  made a deep fit a 30-45 hour proposition is a function of dimension.
* Testable. Free-s_i *nests* this, so 2*(ll_free - ll_smooth) ~ chi2(m - 2)
  measures whether smoothness is rejected. A rejection is a finding about the
  curve; it is not a reason to revert, since the free fit is degenerate.

Use ``SmoothMeasParams(a, b)`` and ``sigma_from_tau``. ``b = 0`` gives a flat
curve (one parameter, all contracts equal); ``quadratic=True`` adds a c*tau^2
term for a hump, still degenerate-proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SmoothMeasParams", "sigma_from_tau", "hdiag_from_tau",
           "fit_smooth_to_vector", "spread_bound"]

LOG_TICK = np.log(0.01 / 70.0)      # one tick at $70, in log-price units


@dataclass(frozen=True)
class SmoothMeasParams:
    """log s(tau) = a + b*tau (+ c*tau^2 when ``quadratic``)."""

    a: float
    b: float = 0.0
    c: float = 0.0
    quadratic: bool = False

    def n_params(self) -> int:
        return 3 if self.quadratic else (1 if self.b == 0.0 else 2)

    def validate(self) -> None:
        for nm, v in (("a", self.a), ("b", self.b), ("c", self.c)):
            if not np.isfinite(v):
                raise ValueError(f"{nm} must be finite; got {v}")
        if not self.quadratic and self.c != 0.0:
            raise ValueError("c is nonzero but quadratic=False; set quadratic=True")


def sigma_from_tau(p: SmoothMeasParams, tau) -> np.ndarray:
    """Measurement-error standard deviations at maturities ``tau``.

    ``tau`` may be (m,) for a fixed grid or (n, m) for the genuinely
    time-varying grid the panel actually has -- maturity shrinks daily between
    rolls, so the (n, m) form is the correct one and costs nothing.
    """
    p.validate()
    t = np.asarray(tau, float)
    z = p.a + p.b * t
    if p.quadratic:
        z = z + p.c * t * t
    return np.exp(np.clip(z, -30.0, 5.0))


def hdiag_from_tau(p: SmoothMeasParams, tau, *, eta_roll: float = 0.0,
                   roll=None) -> np.ndarray:
    """Time-varying ``Hdiag`` (n, m) for the fast filter.

    The roll inflation of eq. (25) is applied on top and stays separable: it
    keys on a roll indicator, this on maturity, and the two are distinct
    regressors rather than competing explanations of the same variation.
    """
    t = np.atleast_2d(np.asarray(tau, float))
    H = sigma_from_tau(p, t) ** 2
    if roll is not None and eta_roll:
        H = H * np.exp(np.clip(eta_roll * np.asarray(roll, float), -20.0, 20.0))[:, None]
    return H


def spread_bound(p: SmoothMeasParams, tau) -> float:
    """Largest achievable ratio max(s)/min(s) across the panel.

    This is the quantity that makes the Chen-Scott degeneracy unreachable: it is
    finite for any finite ``b``, so no parameter value prices one contract
    exactly while leaving another at 320 bp.
    """
    s = sigma_from_tau(p, tau)
    return float(np.max(s) / max(np.min(s), 1e-300))


def fit_smooth_to_vector(sigma_meas, tau, *, quadratic: bool = False
                         ) -> SmoothMeasParams:
    """Least-squares starting values from an existing free-s_i fit.

    Regresses log s_i on mean maturity. Use it to warm-start the smooth fit from
    whatever the free fit returned -- even a degenerate free fit carries the
    right order of magnitude for the interior contracts.
    """
    s = np.asarray(sigma_meas, float)
    t = np.asarray(tau, float)
    t = t.mean(axis=0) if t.ndim == 2 else t
    ok = np.isfinite(s) & (s > 0) & np.isfinite(t)
    if ok.sum() < (3 if quadratic else 2):
        raise ValueError("not enough usable contracts to initialise")
    cols = [np.ones(ok.sum()), t[ok]] + ([t[ok] ** 2] if quadratic else [])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, np.log(s[ok]), rcond=None)
    return SmoothMeasParams(a=float(beta[0]), b=float(beta[1]),
                            c=float(beta[2]) if quadratic else 0.0,
                            quadratic=quadratic)
