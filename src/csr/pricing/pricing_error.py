"""S0P: S0 extended with an AR(1) front-contract pricing-error state
(Filter Repair Runbook, Step 1) -- REFUTED, kept for the record and for
re-use on a deep panel. Do not present S0P as the fitted model; see
"Why this is refuted" below before using anything in this module for a
result rather than a diagnostic.

The original hypothesis: the measurement-error profile under plain S0 --
s_hat = [54.9, 6.5, 5.1, 16.6] bp -- looked like a front-contract
misspecification signature (C1 absorbing eight times C3's error on the
single most liquid futures contract in the world), fixable with a state
restricted to load on one named contract:

    p_{t+1} = phi * p_t + eps_t,    eps ~ N(0, sigma_p^2)
    ln F(t, T_1) = e^{-kappa*tau_1} chi_t + xi_t + A(tau_1) + p_t + v_1t

Why this is refuted (Runbook Step 1, checks A/B/1.9 -- explore/12, 13, 14)
---------------------------------------------------------------------
Two of Gate 1's four criteria were numerically met (s_1: 54.9 -> 9.88bp;
LR vs S0: 15,912 on 2 df) and BOTH were void under their own
decomposition, not passed:

1. The LR gain is a boundary artefact, not the mechanism. Decomposed by
   contract (per_contract_loglik / per_contract_loglik_s0p), C4 supplies
   80.2% of the total likelihood gain and C3+C4 together supply 93.3%.
   C1 -- the contract this state was built for -- contributes 3.8%
   (+604 of +15,912). C4 landed on MIN_SIGMA_MEAS (1.43bp): the same
   boundary degeneracy documented for the Gaussian fits, reappearing
   under Student-t on a different contract.

2. The s_1 collapse is not caused by the pricing-error state. Plain S0
   (NO new state of any kind) refit on C1-C3 with C4 simply dropped
   already gets s_1 to 3.34bp -- lower than S0P achieves with or without
   C4. Adding the pricing-error state back on top of the C1-C3 panel then
   makes s_1 WORSE (3.34 -> 7.09bp) and pushes the exact same boundary
   pathology onto C3 instead (1.43bp, the floor again).

3. The perturbation battery (explore/14_perturbation_battery.py, Runbook
   1.9-1.10): refit plain S0 (Student-t, no pricing-error state) on every
   leave-one-contract-out subset of C1-C4 and on two non-overlapping time
   periods. The inflated entry of the s vector relocates depending on
   which contracts are in the panel -- C1 and C4 in the four-contract fit,
   C2/C3 once C4 is dropped, C3 to the floor once the state is added back.
   See that script's output for the full table and the kappa-spread
   comparison it was built to produce.

The actual finding: the measurement-error vector is not identified from a
four-contract panel. With lambda_min(Z'Z) = 0.024, the orientation of the
residual space the two factors fail to span is barely determined, so
WHERE the unexplained variance lands is an artefact of which contracts
happen to be in the panel, not a stable property of any one contract. The
U-shape in [54.9, 6.5, 5.1, 16.6] was over-read -- a mechanism was built
on the pattern of a vector whose pattern is not stable under panel
perturbation. That inference, not the AR(1) state itself, was the error.

Why this module is kept, not deleted
-------------------------------------
The code is correct for the hypothesis it was built to test, it nests
exactly (sigma_p=0 reproduces S0 to 1e-9, pinned in
tests/test_pricing_error.py), and a single-contract pricing-error state
becomes a meaningful, testable question again on a deep panel (Runbook
Step 2) where the measurement-error vector -- and therefore any single
contract's claim to be the misspecified one -- is actually identified.
Deleting this would also delete the record of how the front-contract
hypothesis was tested and settled.

This reuses csr.pricing.fit's pack/unpack/student_t_weights directly
(both are generic over the parameter vector or residual shape, no S0P-
specific logic needed there) and csr.state.augment's
add_pricing_error_state/prior for the state construction itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..state.augment import add_pricing_error_prior, add_pricing_error_state
from ..state.fast import filter_uv, filter_uv_contributions
from .fit import MIN_SIGMA_MEAS, pack, student_t_weights, unpack
from .twofactor import TwoFactorParams, base_system, initial_state, scale_variances


def pack_s0p(p: TwoFactorParams, phi: float, sigma_p: float,
            min_sigma_meas: float = MIN_SIGMA_MEAS) -> np.ndarray:
    """pack() extended with z=arctanh(phi), w=log(sigma_p) appended at the
    end -- same reparameterisation convention (log for positive scales,
    atanh for a (-1,1)-bounded parameter) as everything else in fit.py."""
    base = pack(p, min_sigma_meas)
    return np.concatenate([base, [
        np.arctanh(np.clip(phi, -0.999, 0.999)),
        np.log(max(sigma_p, 1e-12)),
    ]])


def unpack_s0p(theta: np.ndarray, m: int, min_sigma_meas: float = MIN_SIGMA_MEAS
              ) -> tuple[TwoFactorParams, float, float]:
    p = unpack(theta[:-2], m, min_sigma_meas)
    phi = float(np.tanh(theta[-2]))
    sigma_p = float(np.exp(np.clip(theta[-1], -50.0, 50.0)))
    return p, phi, sigma_p


def _s0p_system(p: TwoFactorParams, phi: float, sigma_p: float,
                log_f: np.ndarray, tau: np.ndarray, dt: float, contract_index: int):
    sys = scale_variances(base_system(p, tau, dt), p)
    sys_p = add_pricing_error_state(sys, phi, sigma_p, contract_index)
    x0, P0 = initial_state(log_f, tau, p)
    x0p, P0p = add_pricing_error_prior(x0, P0, phi, sigma_p)
    return sys_p, x0p, P0p


def weighted_filter_s0p(p: TwoFactorParams, phi: float, sigma_p: float,
                        log_f: np.ndarray, tau: np.ndarray, dt: float,
                        weight: np.ndarray, contract_index: int = 0):
    """Gaussian filter for S0P with Hdiag / weight -- the S0P analogue of
    csr.pricing.fit.weighted_filter, same EM/scale-mixture role, over the
    3-dim (chi, xi, p) state instead of the 2-dim one. weight == 1 must
    reproduce a plain (unweighted) S0P filter exactly, and (via the
    sigma_p=0 nesting gate already pinned in test_pricing_error.py)
    sigma_p == 0 must further reproduce plain S0 -- pinned again here at
    the weighted-filter level in tests/test_pricing_error.py.
    """
    sys_p, x0p, P0p = _s0p_system(p, phi, sigma_p, log_f, tau, dt, contract_index)
    n = sys_p["n"]
    Hdiag = sys_p["Hdiag"] / weight
    k = sys_p["T"].shape[0]
    Bu = np.zeros((n, k))
    ll, xf, Pf, xp, Pp, resid, n_obs = filter_uv(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys_p["Z"]),
        np.ascontiguousarray(sys_p["d"]), np.ascontiguousarray(sys_p["T"]),
        np.ascontiguousarray(sys_p["c"]), np.ascontiguousarray(sys_p["Q"]),
        np.ascontiguousarray(Hdiag), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0p), np.ascontiguousarray(P0p),
    )
    return float(ll), xf, Pf, resid, int(n_obs)


def _neg_loglik_s0p_weighted(theta, log_f, tau, dt, m, weight, min_sigma_meas,
                             contract_index):
    p, phi, sigma_p = unpack_s0p(theta, m, min_sigma_meas)
    ll, _, _, _, _ = weighted_filter_s0p(p, phi, sigma_p, log_f, tau, dt,
                                         weight, contract_index)
    return -ll if np.isfinite(ll) else 1e10


@dataclass
class S0PFit:
    p: TwoFactorParams
    phi: float
    sigma_p: float
    x_filt: np.ndarray
    P_filt: np.ndarray
    resid: np.ndarray
    weight: np.ndarray
    n_iters: int
    ll: float
    contract_index: int


def fit_s0p_student_t(log_f: np.ndarray, tau: np.ndarray, dt: float, m: int,
                      df: float, theta0: np.ndarray, contract_index: int = 0,
                      min_sigma_meas: float = MIN_SIGMA_MEAS,
                      n_outer: int = 30, tol: float = 1e-6) -> S0PFit:
    """EM/scale-mixture Student-t(df) fit of S0P -- identical procedure to
    csr.pricing.fit.fit_s0_student_t, extended to the (two-factor params,
    sigma_meas, phi, sigma_p) vector via pack_s0p/unpack_s0p. ``theta0``
    should come from pack_s0p(p_gauss_or_student_t, phi0, sigma_p0) --
    same "start from a converged simpler fit" discipline as fit_s0_student_t.
    """
    weight = np.ones_like(log_f)
    theta = np.asarray(theta0, float).copy()
    p, phi, sigma_p = unpack_s0p(theta, m, min_sigma_meas)
    it = 0
    for it in range(n_outer):
        res = minimize(_neg_loglik_s0p_weighted, theta,
                       args=(log_f, tau, dt, m, weight, min_sigma_meas, contract_index),
                       method="BFGS", options={"maxiter": 300})
        theta = res.x
        p, phi, sigma_p = unpack_s0p(theta, m, min_sigma_meas)
        _, _, _, resid_raw, _ = weighted_filter_s0p(
            p, phi, sigma_p, log_f, tau, dt, np.ones_like(log_f), contract_index)
        new_weight = student_t_weights(resid_raw, df)
        shift = float(np.nanmax(np.abs(new_weight - weight)))
        weight = new_weight
        if shift < tol:
            break

    ll, x_filt, P_filt, resid_raw, n_obs = weighted_filter_s0p(
        p, phi, sigma_p, log_f, tau, dt, np.ones_like(log_f), contract_index)
    return S0PFit(p=p, phi=phi, sigma_p=sigma_p, x_filt=x_filt, P_filt=P_filt,
                 resid=resid_raw, weight=weight, n_iters=it + 1, ll=ll,
                 contract_index=contract_index)


def multi_start_fit_s0p(log_f: np.ndarray, tau: np.ndarray, dt: float, m: int,
                        df: float, p_init: TwoFactorParams,
                        phi_grid: tuple[float, ...], sigma_p_grid: tuple[float, ...],
                        contract_index: int = 0,
                        min_sigma_meas: float = MIN_SIGMA_MEAS) -> dict:
    """Fit S0P from every (phi0, sigma_p0) start in the grid (Runbook 1.4).
    Reports the spread of final log-likelihoods across starts -- a spread
    above ~1 unit means the surface is multimodal and the reported optimum
    is a starting-value artefact, not a genuine global optimum, and the
    result should not be trusted without further investigation."""
    theta0_base = pack(p_init, min_sigma_meas)
    starts = []
    for phi0 in phi_grid:
        for sigma_p0 in sigma_p_grid:
            theta0 = np.concatenate([theta0_base,
                                     [np.arctanh(phi0), np.log(sigma_p0)]])
            fit = fit_s0p_student_t(log_f, tau, dt, m, df, theta0,
                                    contract_index, min_sigma_meas)
            starts.append({"phi0": phi0, "sigma_p0": sigma_p0, "fit": fit})

    lls = [s["fit"].ll for s in starts]
    spread = float(max(lls) - min(lls))
    best = max(starts, key=lambda s: s["fit"].ll)
    return {"best": best["fit"], "best_start": (best["phi0"], best["sigma_p0"]),
           "starts": starts, "ll_spread": spread}


def profile_likelihood_phi(log_f: np.ndarray, tau: np.ndarray, dt: float, m: int,
                           p_init: TwoFactorParams, phi_grid: np.ndarray,
                           sigma_p0: float = 1e-3, contract_index: int = 0,
                           min_sigma_meas: float = MIN_SIGMA_MEAS) -> dict:
    """Profile the (Gaussian, weight=1) likelihood in phi over a grid,
    refitting the two-factor params and sigma_p at each fixed phi
    (Runbook 1.5). A flat profile means phi isn't identified from the
    data on its own; the runbook's resolution is to fix phi at a
    reasonable value (e.g. 0.9) and free only sigma_p, so the eventual LR
    test against S0 carries 1 degree of freedom, not 2 -- getting this
    wrong inflates the test. This profiles the Gaussian (weight=1)
    likelihood rather than re-running the full Student-t EM loop at every
    grid point, which would be more faithful but is a materially heavier
    computation for a diagnostic whose purpose is to check identification,
    not to produce a final estimate.
    """
    theta0_base = pack(p_init, min_sigma_meas)

    def _neg_ll_fixed_phi(theta_reduced, phi_fixed):
        theta_base, w_sigma_p = theta_reduced[:-1], theta_reduced[-1]
        p = unpack(theta_base, m, min_sigma_meas)
        sigma_p = float(np.exp(np.clip(w_sigma_p, -50.0, 50.0)))
        ll, _, _, _, _ = weighted_filter_s0p(
            p, phi_fixed, sigma_p, log_f, tau, dt, np.ones_like(log_f), contract_index)
        return -ll if np.isfinite(ll) else 1e10

    lls = []
    for phi_fixed in phi_grid:
        theta0 = np.concatenate([theta0_base, [np.log(sigma_p0)]])
        res = minimize(_neg_ll_fixed_phi, theta0, args=(phi_fixed,),
                       method="BFGS", options={"maxiter": 500})
        lls.append(-res.fun)

    lls = np.array(lls)
    return {
        "phi_grid": np.asarray(phi_grid), "loglik": lls,
        "range": float(lls.max() - lls.min()),
        "argmax_phi": float(phi_grid[int(np.argmax(lls))]),
    }


def per_contract_loglik_s0p(p: TwoFactorParams, phi: float, sigma_p: float,
                            log_f: np.ndarray, tau: np.ndarray, dt: float,
                            contract_index: int = 0) -> np.ndarray:
    """Exact per-contract log-likelihood contribution under S0P -- the
    S0P analogue of csr.pricing.fit.per_contract_loglik, via
    filter_uv_contributions on the 3-dim (chi, xi, p) system. Compared
    against per_contract_loglik(p_s0, ...), 2*(this - that) decomposed by
    contract shows exactly where an S0-to-S0P likelihood gain is coming
    from -- the check that matters before trusting Gate 1's LR statistic
    at face value, since an extra state with free parameters can produce
    a large aggregate LR gain that turns out to be concentrated in a
    single contract hitting its measurement-error floor rather than
    reflecting the pricing-error mechanism working as intended.
    """
    sys_p, x0p, P0p = _s0p_system(p, phi, sigma_p, log_f, tau, dt, contract_index)
    n = sys_p["n"]
    k = sys_p["T"].shape[0]
    Bu = np.zeros((n, k))
    ll, xf, Pf, xp, Pp, resid, ll_cell, n_obs = filter_uv_contributions(
        np.ascontiguousarray(log_f), np.ascontiguousarray(sys_p["Z"]),
        np.ascontiguousarray(sys_p["d"]), np.ascontiguousarray(sys_p["T"]),
        np.ascontiguousarray(sys_p["c"]), np.ascontiguousarray(sys_p["Q"]),
        np.ascontiguousarray(sys_p["Hdiag"]), np.ascontiguousarray(Bu),
        np.ascontiguousarray(x0p), np.ascontiguousarray(P0p),
    )
    return np.nansum(ll_cell, axis=0)


def confounding_correlation(x_filt: np.ndarray) -> float:
    """corr(p_hat, xi_hat) from a fitted S0P's filtered state -- the two
    are separated only by persistence (p is AR(1), xi is a unit root), so
    |corr| above ~0.7 means that separation isn't working on real data,
    and the new state is stealing the long-run factor rather than
    absorbing pricing error (Runbook 1.6)."""
    p_hat = x_filt[:, 2]
    xi_hat = x_filt[:, 1]
    return float(np.corrcoef(p_hat, xi_hat)[0, 1])
