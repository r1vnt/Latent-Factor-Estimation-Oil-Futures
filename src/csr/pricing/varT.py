"""Cell B of the loading/transition lattice: restricted (affine) Z, free
VAR(1) transition T.

    |                         | restricted Z (e^-k*tau) | free Z |
    | restricted T (OU+unit)  | S3 -- current           | A      |
    | free T (VAR(1))         | B -- this module        | C      |

Three independent pieces of evidence already point at the TRANSITION,
not the loading, as the source of the spread/butterfly forecast failure:
kappa1 unstable at 0.7574 across block-outs (explore/26), the H-sweep
residual that never closes even as H1 -> 0 (explore/57), and volatility
clustering that no exogenous variance channel touches (explore/46-51).
explore/59's PCA diagnostic separately confirmed the loading is NOT the
problem (affine residual ~ PCA-3 ceiling, median ratio 0.83x) -- so if
anything in this class is going to beat the random walk on spreads, the
lattice says look here first.

Z and d = A_tau3(p_frozen, tau) are FROZEN at the already-fitted S3
baseline (csr.pricing.threefactor.base_system3 / A_tau3) -- not
re-estimated. This is what keeps cell B "still a no-arbitrage pricing
model in the cross-section": given a state vector, the curve is priced
exactly as S3 already prices it, so cell B can still price an unseen
maturity, unlike cells A/C (free Z, no A(tau), no risk premia -- pure
diagnostics, per the module's own design note, not replacements).

Only T, c, and Q -- the discrete-time law of motion the filter uses to
predict one state forward -- are freed from the OU-plus-unit-root
restriction (T = diag(e^-k1*dt, e^-k2*dt, 1) in the S3 baseline).

T is a fully general 3x3 real matrix (9 free parameters). NOT forced to
have exactly one eigenvalue at unity by construction: Z is fixed here
(not free), so there is no rotational indeterminacy between Z and T
requiring an anchoring normalisation -- Z's fixed columns already pin
down what each state coordinate MEANS, unlike cells A/C where a free Z
would need "fix the upper 3x3 of Z to the identity" to break that
ambiguity. The only constraint on T is stability: spectral radius <=
SPECTRAL_RADIUS_MAX, PERMITTING (not forcing) a unit root, enforced by
rejecting any candidate that violates it with the same large-penalty
convention fit3.py already uses for invalid PSD draws (found the hard
way there: explore/28's battery crashed on an uncaught rejection). An
explosive root would fit beautifully in-sample and forecast nonsense --
exactly the failure mode this guard exists to prevent.

SPECTRAL_RADIUS_MAX is 1.5, not 1 + 1e-6 -- explore/60b found the tight
tolerance WAS the ETA_V_MAX/GAMMA_V_MAX plateau lesson again: at 1+1e-6
the optimiser parked exactly on the wall (max modulus 1.000001, a worse
likelihood) rather than reaching its true optimum. Loosened by even 1%
(radius<=1.01), it converges to a genuinely interior, STABLE point (max
modulus 0.9975) with a strictly better likelihood, identical across
every looser bound tested up to 5.0 -- confirming this is a real,
well-identified optimum the tight wall was hiding, not an explosive
root the wall was correctly holding back. 1.5 is a generous safety net
against genuine pathology, not a binding constraint on the answer.

theta layout (21 total):
    0-8:   T.flatten(), raw/unconstrained (3x3)
    9-11:  c, raw/unconstrained drift 3-vector
    12-14: log(L_00), log(L_11), log(L_22) -- Cholesky diagonal of Q,
           log-parameterised for positivity
    15-17: L_10, L_20, L_21 -- Cholesky off-diagonal of Q
    18-20: a, b, c -- smoothmeas.SmoothMeasParams curve (quadratic only;
           this diagnostic does not carry over the RV/front-spread
           channels -- minimal scope, matching "no refit needed" beyond
           what the lattice actually asks)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .fit3 import _neg_t_loglik_df, student_t_weights3
from .smoothmeas import SmoothMeasParams, hdiag_from_tau
from .threefactor import A_tau3, ThreeFactorParams, base_system3
from ..state.fast import filter_uv, filter_uv_contributions

SPECTRAL_RADIUS_MAX = 1.5
N_THETA = 21


@dataclass
class VarTState:
    T: np.ndarray      # (3, 3)
    c: np.ndarray       # (3,)
    Q: np.ndarray       # (3, 3), PSD


def pack_varT(T: np.ndarray, c: np.ndarray, Q: np.ndarray,
             a: float, b: float, c_curve: float) -> np.ndarray:
    L = np.linalg.cholesky(Q)
    theta = np.empty(N_THETA)
    theta[0:9] = np.asarray(T, float).ravel()
    theta[9:12] = np.asarray(c, float).ravel()
    theta[12] = np.log(max(L[0, 0], 1e-12))
    theta[13] = np.log(max(L[1, 1], 1e-12))
    theta[14] = np.log(max(L[2, 2], 1e-12))
    theta[15] = L[1, 0]
    theta[16] = L[2, 0]
    theta[17] = L[2, 1]
    theta[18] = a
    theta[19] = b
    theta[20] = c_curve
    return theta


def unpack_varT(theta: np.ndarray) -> tuple[VarTState, SmoothMeasParams]:
    theta = np.asarray(theta, float)
    T = theta[0:9].reshape(3, 3)

    eigvals = np.linalg.eigvals(T)
    if np.max(np.abs(eigvals)) > SPECTRAL_RADIUS_MAX:
        raise ValueError("T violates the spectral-radius-<=1 stability constraint")

    c = theta[9:12].copy()
    L = np.array([
        [np.exp(np.clip(theta[12], -30.0, 30.0)), 0.0, 0.0],
        [theta[15], np.exp(np.clip(theta[13], -30.0, 30.0)), 0.0],
        [theta[16], theta[17], np.exp(np.clip(theta[14], -30.0, 30.0))],
    ])
    Q = L @ L.T
    smooth = SmoothMeasParams(a=float(theta[18]), b=float(theta[19]), c=float(theta[20]), quadratic=True)
    return VarTState(T=T, c=c, Q=Q), smooth


def frozen_Z_d(p_frozen: ThreeFactorParams, tau: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Z and d, frozen at the already-fitted S3 baseline -- never re-estimated
    in cell B. dt is accepted for signature symmetry with base_system3 but
    unused: Z/d depend only on tau and the frozen structural params."""
    sys3 = base_system3(p_frozen, tau, dt)
    return sys3["Z"], sys3["d"]


def _raw_system_varT(state: VarTState, smooth: SmoothMeasParams, Z: np.ndarray, d: np.ndarray,
                     tau: np.ndarray, n: int) -> dict:
    c_tiled = np.tile(state.c, (n, 1))
    Q_tiled = np.tile(state.Q[None, :, :], (n, 1, 1))
    Hdiag = hdiag_from_tau(smooth, tau)
    return {"Z": Z, "d": d, "T": state.T, "c": c_tiled, "Q": Q_tiled, "Hdiag": Hdiag, "n": n}


def weighted_filter_varT(state: VarTState, smooth: SmoothMeasParams, log_f: np.ndarray,
                         Z: np.ndarray, d: np.ndarray, tau: np.ndarray, x0: np.ndarray,
                         P0: np.ndarray, weight: np.ndarray):
    n = log_f.shape[0]
    sys = _raw_system_varT(state, smooth, Z, d, tau, n)
    Hdiag = sys["Hdiag"] / weight
    Bu = np.zeros((n, 3))
    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f, dtype=np.float64), np.ascontiguousarray(sys["Z"], dtype=np.float64),
        np.ascontiguousarray(sys["d"], dtype=np.float64), np.ascontiguousarray(sys["T"], dtype=np.float64),
        np.ascontiguousarray(sys["c"], dtype=np.float64), np.ascontiguousarray(sys["Q"], dtype=np.float64),
        np.ascontiguousarray(Hdiag, dtype=np.float64), np.ascontiguousarray(Bu, dtype=np.float64),
        np.ascontiguousarray(x0, dtype=np.float64), np.ascontiguousarray(P0, dtype=np.float64))
    return float(ll), xf, resid, int(n_obs)


def _neg_loglik_varT(theta, log_f, Z, d, tau, x0, P0):
    try:
        state, smooth = unpack_varT(theta)
    except ValueError:
        return 1e10   # unstable T (or non-PD Cholesky, though the log-parameterisation prevents that)
    ll, *_ = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
    return -ll if np.isfinite(ll) else 1e10


def fit_varT_gaussian(log_f: np.ndarray, Z: np.ndarray, d: np.ndarray, tau: np.ndarray,
                      x0: np.ndarray, P0: np.ndarray, theta0: np.ndarray):
    res = minimize(_neg_loglik_varT, theta0, args=(log_f, Z, d, tau, x0, P0),
                   method="BFGS", options={"maxiter": 3000, "gtol": 1e-8})
    if not res.success:
        res2 = minimize(_neg_loglik_varT, res.x, args=(log_f, Z, d, tau, x0, P0),
                        method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-8})
        if res2.fun < res.fun:
            res.x = res2.x
    state, smooth = unpack_varT(res.x)
    ll, x_filt, resid, n_obs = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
    return state, smooth, x_filt, resid, n_obs, ll, res.x


def _neg_loglik_varT_weighted(theta, log_f, Z, d, tau, x0, P0, weight):
    try:
        state, smooth = unpack_varT(theta)
    except ValueError:
        return 1e10
    ll, *_ = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, weight)
    return -ll if np.isfinite(ll) else 1e10


def fit_varT_student_t(log_f: np.ndarray, Z: np.ndarray, d: np.ndarray, tau: np.ndarray,
                       x0: np.ndarray, P0: np.ndarray, theta0: np.ndarray, df0: float,
                       n_outer: int = 200, tol: float = 1e-6, estimate_df: bool = True,
                       df_bounds: tuple = (2.1, 100.0)):
    """Same ECM structure as fit3.fit3_student_t_smooth, including joint nu
    estimation by default -- explore/50's lesson (fixed nu and a variance
    channel double-count fat tails) applies here too: re-estimate nu in
    every cell rather than leaving it at the S3 baseline's fixed value."""
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    df = df0
    state, smooth = unpack_varT(theta)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik_varT_weighted, theta, args=(log_f, Z, d, tau, x0, P0, weight),
                       method="BFGS", options={"maxiter": 500})
        theta = res.x
        state, smooth = unpack_varT(theta)
        _, _, resid_raw, _ = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
        if estimate_df:
            z_finite = resid_raw[np.isfinite(resid_raw)]
            res_df = minimize_scalar(_neg_t_loglik_df, bounds=df_bounds, method="bounded", args=(z_finite,))
            df = float(res_df.x)
        new_weight = student_t_weights3(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, resid_raw, n_obs = weighted_filter_varT(state, smooth, log_f, Z, d, tau, x0, P0, np.ones_like(log_f))
    return state, smooth, x_filt, resid_raw, weight, it + 1, ll, df, theta
