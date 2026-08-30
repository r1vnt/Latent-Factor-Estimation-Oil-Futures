"""Ground-truth simulation of two-factor futures panels.

Used to validate the estimator before any inference on market data. If the
filter cannot recover parameters it generated itself, nothing downstream means
anything -- so this is the first thing reported in Part II.

The states evolve under the physical measure; prices are risk-neutral
expectations via the closed form in :mod:`nao.model`. Getting that split right
matters: simulating prices under P would make the risk-premium parameters
unidentifiable by construction and the recovery test meaningless.
"""

from __future__ import annotations

import numpy as np

from .pricing.twofactor import A_tau, TwoFactorParams

TRADING_DAYS = 252
DAYS_PER_ROLL = 21


def maturity_grid(n: int, m: int, days_per_roll: int = DAYS_PER_ROLL) -> np.ndarray:
    """(n, m) time-to-maturity in years for m nearby contracts under monthly roll.

    Time-to-maturity shrinks each day and jumps back at the roll, which is what
    makes the observation matrix genuinely time-varying.
    """
    day_in_cycle = np.arange(n) % days_per_roll
    days_to_first = (days_per_roll - day_in_cycle).astype(float)
    offsets = np.arange(m, dtype=float) * days_per_roll
    return (days_to_first[:, None] + offsets[None, :]) / TRADING_DAYS


def simulate_panel(
    params: TwoFactorParams,
    n: int = 2520,
    m: int = 4,
    dt: float = 1.0 / TRADING_DAYS,
    chi0: float = 0.0,
    xi0: float = np.log(60.0),
    seed: int = 0,
    days_per_roll: int = DAYS_PER_ROLL,
):
    """Simulate latent factors and the observed log-futures panel.

    Returns
    -------
    log_f : (n, m) observed log futures prices, including measurement error
    tau   : (n, m) time to maturity in years
    states: (n, 2) true (chi_t, xi_t)
    """
    rng = np.random.default_rng(seed)
    k = params.kappa

    # Exact discretisation of the bivariate OU / ABM system under P.
    a = np.exp(-k * dt)
    v_chi = (1.0 - np.exp(-2.0 * k * dt)) * params.sigma_chi ** 2 / (2.0 * k)
    v_xi = params.sigma_xi ** 2 * dt
    c_xy = (1.0 - np.exp(-k * dt)) * params.rho * params.sigma_chi * params.sigma_xi / k

    cov = np.array([[v_chi, c_xy], [c_xy, v_xi]])
    # Nudge onto the PSD cone if rho pushes the discretised covariance off it.
    w, V = np.linalg.eigh(cov)
    if np.any(w <= 0):
        cov = V @ np.diag(np.maximum(w, 1e-14)) @ V.T
    L = np.linalg.cholesky(cov)

    states = np.empty((n, 2))
    chi, xi = chi0, xi0
    for t in range(n):
        z = L @ rng.standard_normal(2)
        chi = a * chi + z[0]
        xi = xi + params.mu_xi * dt + z[1]
        states[t] = (chi, xi)

    tau = maturity_grid(n, m, days_per_roll)
    log_f = (np.exp(-k * tau) * states[:, 0:1] + states[:, 1:2] + A_tau(params, tau))
    log_f = log_f + rng.normal(0.0, 1.0, size=log_f.shape) * np.asarray(params.sigma_meas)
    return log_f, tau, states


def simulate_news(
    n: int,
    p: int = 3,
    seed: int = 0,
    persistence: float = 0.4,
):
    """Simulate standardised daily news features and a news-intensity series.

    Features are persistent (news arrives in clusters) and standardised, which
    is how the real pipeline in :mod:`nao.news` delivers them.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros((n, p))
    for t in range(1, n):
        u[t] = persistence * u[t - 1] + rng.normal(0, 1, p) * np.sqrt(1 - persistence ** 2)
    intensity = np.zeros(n)
    for t in range(1, n):
        intensity[t] = 0.7 * intensity[t - 1] + rng.normal(0, 1) * np.sqrt(1 - 0.49)
    return u, intensity


def simulate_with_news(
    params: TwoFactorParams,
    B: np.ndarray,
    n: int = 2520,
    m: int = 4,
    dt: float = 1.0 / TRADING_DAYS,
    xi0: float = np.log(60.0),
    seed: int = 0,
    gamma: float = 0.0,
):
    """Simulate a panel where news genuinely drives the factors (Channel A/B).

    ``B`` is (2, p): row 0 loads news onto the short-run factor, row 1 onto the
    long-run factor. ``gamma`` activates Channel B, scaling the short-run
    diffusion by exp(gamma * intensity).

    This is the data-generating process under which H1 and H2 are *true*, so a
    correct test should reject the baseline restriction on it -- and should not
    on a panel simulated with B = 0.
    """
    rng = np.random.default_rng(seed)
    p_feat = np.asarray(B).shape[1]
    u, intensity = simulate_news(n, p_feat, seed=seed + 1)

    k = params.kappa
    a = np.exp(-k * dt)
    v_chi0 = (1.0 - np.exp(-2.0 * k * dt)) * params.sigma_chi ** 2 / (2.0 * k)
    v_xi = params.sigma_xi ** 2 * dt

    states = np.empty((n, 2))
    chi, xi = 0.0, xi0
    for t in range(n):
        scale = np.exp(gamma * intensity[t])
        drive = B @ u[t - 1] if t > 0 else np.zeros(2)
        chi = a * chi + drive[0] + rng.normal(0, np.sqrt(v_chi0 * scale))
        xi = xi + params.mu_xi * dt + drive[1] + rng.normal(0, np.sqrt(v_xi))
        states[t] = (chi, xi)

    tau = maturity_grid(n, m)
    log_f = (np.exp(-k * tau) * states[:, 0:1] + states[:, 1:2] + A_tau(params, tau))
    log_f = log_f + rng.normal(0.0, 1.0, size=log_f.shape) * np.asarray(params.sigma_meas)
    return log_f, tau, states, u, intensity


# Parameters near the Schwartz-Smith (2000) crude oil estimates, used as the
# default ground truth so the recovery exercise is economically realistic.
SS_OIL = TwoFactorParams(
    kappa=1.49,
    sigma_chi=0.286,
    sigma_xi=0.145,
    rho=0.30,
    mu_xi=0.0,
    mu_xi_star=0.0115,
    lambda_chi=0.157,
    sigma_meas=np.array([0.008, 0.003, 0.003, 0.007]),
)
