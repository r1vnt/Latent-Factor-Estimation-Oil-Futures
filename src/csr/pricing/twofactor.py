"""Two-factor commodity pricing (Gabillon 1992 / Schwartz-Smith 2000).

    ln S_t = chi_t + xi_t

    P:  d chi = -kappa*chi dt + s_chi dW_chi
        d xi  =  mu_xi dt     + s_xi  dW_xi,        corr = rho
    Q:  d chi = (-kappa*chi - lam_chi) dt + s_chi dW^Q
        d xi  =  mu_xi_star dt           + s_xi  dW^Q

    ln F(t,T) = exp(-kappa*tau)*chi_t + xi_t + A(tau)

Linear in the latent state, hence a Gaussian state-space model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EPS = 1e-8


@dataclass
class TwoFactorParams:
    """Structural parameters of the two-factor model."""

    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float
    mu_xi: float
    mu_xi_star: float
    lambda_chi: float
    sigma_meas: np.ndarray

    # ---- optional variance-channel scalings (Sec. XI-F) -----------------
    gamma: float = 0.0        # news intensity      -> process noise
    gamma_r: float = 0.0      # scheduled releases  -> process noise
    gamma_v: float = 0.0      # lagged realised vol -> process noise
    eta_roll: float = 0.0     # roll window         -> measurement noise

    @property
    def half_life_days(self) -> float:
        return float(np.log(2.0) / max(self.kappa, EPS) * 365.25)

    def replace(self, **kw) -> "TwoFactorParams":
        return replace(self, **kw)


def A_tau(p: TwoFactorParams, tau: np.ndarray) -> np.ndarray:
    """Deterministic log-futures adjustment A(tau); A(0) = 0 exactly."""
    tau = np.asarray(tau, float)
    k = max(p.kappa, EPS)
    m1 = -np.expm1(-k * tau) / k                 # (1 - e^{-k tau}) / k
    m2 = -np.expm1(-2.0 * k * tau) / (2.0 * k)   # (1 - e^{-2k tau}) / (2k)
    return (
        p.mu_xi_star * tau
        - m1 * p.lambda_chi
        + 0.5 * (
            m2 * p.sigma_chi ** 2
            + p.sigma_xi ** 2 * tau
            + 2.0 * m1 * p.rho * p.sigma_chi * p.sigma_xi
        )
    )


def base_system(p: TwoFactorParams, tau: np.ndarray, dt: float) -> dict:
    """Two-dimensional state-space matrices.

    ``tau`` is (n, m): time to maturity shrinks each trading day between rolls,
    so Z and d are genuinely time-varying. Treating them as constant is the
    most common error in Schwartz-Smith implementations.
    """
    tau = np.atleast_2d(np.asarray(tau, float))
    n, m = tau.shape
    k = max(p.kappa, EPS)

    T = np.array([[np.exp(-k * dt), 0.0], [0.0, 1.0]])
    c = np.array([0.0, p.mu_xi * dt])

    q11 = -np.expm1(-2.0 * k * dt) * p.sigma_chi ** 2 / (2.0 * k)
    q22 = p.sigma_xi ** 2 * dt
    q12 = -np.expm1(-k * dt) * p.rho * p.sigma_chi * p.sigma_xi / k
    Q = np.array([[q11, q12], [q12, q22]])

    Z = np.empty((n, m, 2))
    Z[:, :, 0] = np.exp(-k * tau)
    Z[:, :, 1] = 1.0
    d = A_tau(p, tau)
    # sigma_meas is (m,) for the free per-contract parameterisation, or
    # already (n, m) when it came from a maturity-indexed curve (e.g.
    # csr.pricing.smoothmeas.sigma_from_tau) -- the latter is genuinely
    # time-varying (tau shrinks daily between rolls) so it is used as-is,
    # not re-broadcast from a single row.
    sigma_meas = np.asarray(p.sigma_meas, float)
    if sigma_meas.ndim == 1:
        Hdiag = np.repeat((sigma_meas ** 2)[None, :], n, axis=0)
    else:
        Hdiag = sigma_meas ** 2

    return {"T": T, "c": c, "Q": Q, "Z": Z, "d": d, "Hdiag": Hdiag,
            "n": n, "m": m}


def scale_variances(sys: dict, p: TwoFactorParams, *, intensity=None,
                    release=None, rv_lag=None, roll=None) -> dict:
    """Apply the time-varying variance scalings of Section XI-F.

    Process noise (eq. 24):
        sigma_chi,t^2 = sigma_chi^2 * exp(gamma*I_t + gamma_r*rel_t + gamma_v*RV_{t-1})

    Measurement noise (eq. 25):
        H_t = diag(s_i^2 * exp(eta * roll_t))

    ``rv_lag`` must already be lagged by the caller. A contemporaneous realised
    volatility would make the state prior a function of the data being
    explained, and the prediction-error decomposition would no longer be the
    likelihood of the sample.
    """
    n = sys["n"]
    out = dict(sys)

    expo = np.zeros(n)
    if intensity is not None and p.gamma:
        expo = expo + p.gamma * np.asarray(intensity, float)
    if release is not None and p.gamma_r:
        expo = expo + p.gamma_r * np.asarray(release, float)
    if rv_lag is not None and p.gamma_v:
        expo = expo + p.gamma_v * np.asarray(rv_lag, float)

    Q = np.repeat(sys["Q"][None, :, :], n, axis=0).copy()
    if np.any(expo):
        s = np.exp(np.clip(expo, -20.0, 20.0))
        Q[:, 0, 0] *= s
        Q[:, 0, 1] *= np.sqrt(s)
        Q[:, 1, 0] *= np.sqrt(s)
    out["Q"] = Q

    if roll is not None and p.eta_roll:
        out["Hdiag"] = sys["Hdiag"] * np.exp(
            np.clip(p.eta_roll * np.asarray(roll, float), -20.0, 20.0)
        )[:, None]
    return out


def initial_state(log_f, tau, p: TwoFactorParams):
    """Prior for (chi_0, xi_0): stationary for chi, diffuse for xi."""
    k = max(p.kappa, EPS)
    row = np.atleast_2d(np.asarray(log_f, float))[0]
    t0 = np.atleast_2d(np.asarray(tau, float))[0]
    ok = np.isfinite(row)
    xi0 = float(np.mean(row[ok] - A_tau(p, t0[ok]))) if ok.any() else 0.0
    x0 = np.array([0.0, xi0])
    P0 = np.diag([p.sigma_chi ** 2 / (2.0 * k), 1.0])
    return x0, P0
