"""Study A / Gate 4, ported to the three-factor smooth-curve model.

Price-only; does not touch the news pipeline (same discipline as
recovery.py). Re-running against the state estimate this project actually
defends now, not patching the original C1-C4/two-factor/free-diagonal-H
run -- that panel, model, and noise profile are each separately
invalidated (deep panel investigation, this session). This is the first
run against a defensible state, not a fix to a failed one.

Four departures from recovery.py, corresponding to the four live issues
found re-examining the original design:

1. spread_change (the original PRIMARY target) is WITHDRAWN, not ported.
   It regresses a spread CHANGE on a label built from a LEVEL of the same
   two prices, so the label's own measurement error appears on both sides
   of the regression with opposite sign -- confirmed to manufacture a
   result from a pure random walk (recovery.py Sec. 6's own history).
   Not computed here at all, on methodological grounds.
2. PRIMARY_TARGET is realised_vol (promoted from secondary, post hoc --
   the one target in the original battery that shares no measurement-
   error component with either label, and the one clean win). SECONDARY
   is cross_maturity_spread_change (recovery.py Sec. 6's own fix, still
   valid here since it never touched C1/C2 at all).
3. HORIZONS drop h=5 (2.6% of kappa1's reversion, no power) and add
   h=252, matching recovery.py Sec. 7's own EXTENDED_HORIZONS logic but
   as the ONLY horizon set now, not an extension bolted onto a smaller
   default.
4. The identifying-day count (identifying_count, new) is computed and
   gated BEFORE any disagreement-conditioned test runs, not after --
   short of the McNemar requirement means the test is not run, and that
   shortfall is itself the reported result.

Model-independent machinery (encompassing_regression, RegressionResult,
classify_gate, reversion_fraction, label_agreement, disagreement_by_year,
disagreement_clustering, the circular-block-bootstrap primitives and
their live block-length guard, the Wald-test and VIF helpers) is
IMPORTED from recovery.py unchanged, not duplicated -- none of it reads a
TwoFactorParams or ThreeFactorParams field directly.

Everything here is meant to be run on EVERY maturity block-out, not just
the full panel -- kappa1's own 0.731 block-out spread (three-factor
battery, this session) means anything downstream of chi1 must show its
results survive that instability directly, not assume it because kappa2
and xi don't inherit it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

from ..pricing.threefactor import A_tau3, ThreeFactorParams
from .recovery import (  # noqa: F401 -- re-exported for callers of this module
    MIN_BOOTSTRAP_BLOCKS,
    GateVerdict,
    RegressionResult,
    _block_length_ok,
    _circular_block_indices,
    _label_corr_and_vif,
    _norm_cdf,
    _wald_tests,
    classify_gate,
    disagreement_by_year,
    disagreement_clustering,
    encompassing_regression,
    forward_spread_level,
    front_return,
    front_spread,
    label_agreement,
    lag_labels,
    naive_label,
    realised_vol,
    reversion_fraction,
    spread_change,  # imported for completeness/withdrawal-documentation only, see module docstring point 1
)

# ============================================================ 1. Labels ===


def filtered_label(p: ThreeFactorParams, x_filt: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """L_filt: sign of the model-implied spread at the same two maturities
    the naive label uses, from the FILTERED (never smoothed) state.

    f_i = exp(-k1*tau_i)*chi1_hat + exp(-k2*tau_i)*chi2_hat + A3(tau_i)
    L_filt = sign(f1 - f2)

    xi still cancels in the difference (same reasoning as recovery.py's
    two-factor version) -- only chi1_hat, chi2_hat, and A3(tau) matter.
    x_filt columns are (chi1, chi2, xi), matching threefactor.py's state
    ordering and fit3.py's own unpack3_smooth.
    """
    chi1_hat = x_filt[:, 0]
    chi2_hat = x_filt[:, 1]
    f1 = (np.exp(-p.kappa1 * tau[:, 0]) * chi1_hat
         + np.exp(-p.kappa2 * tau[:, 0]) * chi2_hat + A_tau3(p, tau[:, 0]))
    f2 = (np.exp(-p.kappa1 * tau[:, 1]) * chi1_hat
         + np.exp(-p.kappa2 * tau[:, 1]) * chi2_hat + A_tau3(p, tau[:, 1]))
    return np.sign(f1 - f2)


def continuous_naive(log_f: np.ndarray) -> np.ndarray:
    return log_f[:, 0] - log_f[:, 1]


def continuous_filtered(p: ThreeFactorParams, x_filt: np.ndarray) -> np.ndarray:
    """-k1*chi1_hat - k2*chi2_hat + a_bar3: the instantaneous risk-neutral
    drift of the front of the curve, three-factor analogue of recovery.py's
    continuous_filtered. a_bar3 = dA3/dtau at tau=0, derived by hand from
    A_tau3's closed form (m1'(0)=1, v_ii'(0)=1 for both factors):

        a_bar3 = mu_xi_star - lambda1 - lambda2
                + 0.5*(sigma1^2 + sigma2^2 + sigma_xi^2)
                + rho1x*sigma1*sigma_xi + rho2x*sigma2*sigma_xi

    (rho12's cross term is absent: fit3.py fixes rho12=0.)
    """
    a_bar3 = (p.mu_xi_star - p.lambda1 - p.lambda2
             + 0.5 * (p.sigma1 ** 2 + p.sigma2 ** 2 + p.sigma_xi ** 2)
             + p.rho1x * p.sigma1 * p.sigma_xi + p.rho2x * p.sigma2 * p.sigma_xi)
    return -p.kappa1 * x_filt[:, 0] - p.kappa2 * x_filt[:, 1] + a_bar3


# ============================================== 2. Targets (reused as-is) ===
# front_spread, spread_change (withdrawn, not used), front_return,
# realised_vol, forward_spread_level, lag_labels all imported from
# recovery.py unchanged -- none of them touch the model.


def cross_maturity_spread_change(log_f: np.ndarray, h: int,
                                 far: tuple[int, int] = (2, 3)) -> np.ndarray:
    """SECONDARY target (recovery.py Sec. 6's decisive spec, unmodified
    logic): C3-C4 by default, sharing no measurement-error component with
    either label (which always read C1/C2)."""
    s_far = log_f[:, far[0]] - log_f[:, far[1]]
    y = np.full_like(s_far, np.nan)
    if h < len(s_far):
        y[: len(s_far) - h] = s_far[h:] - s_far[:-h]
    return y


PRIMARY_TARGET = "realised_vol"
SECONDARY_TARGET = "cross_maturity_spread_change"
HORIZONS = (22, 66, 132, 252)
NONOVERLAP_HORIZONS = (66, 132)


def _target_fn(name: str):
    return {"realised_vol": realised_vol,
           "cross_maturity_spread_change": cross_maturity_spread_change,
           "front_return": front_return}[name]


# ===================================== 3. Identifying-count / McNemar gate ===


def identifying_count(l_obs: np.ndarray, l_filt: np.ndarray, h: int) -> dict:
    """Three-stage count, computed BEFORE any disagreement-conditioned
    test is run: disagreement days -> contiguous runs -> runs separated
    by more than h days (independent, non-overlapping evidence at this
    specific horizon's window length). Compared against the McNemar
    sample-size requirement (z_{a/2}+z_b)^2 / (2p-1)^2 at alpha=0.05,
    power=0.80 (z_{a/2}=1.96, z_b=0.8416), evaluated at the observed
    effect size p = filt_win_rate on disagreement days (a two-sided,
    standard McNemar power calculation for the discordant-pair proportion
    test -- see e.g. Fleiss et al., not this project's own derivation).

    Runs are treated as separated by more than h days when the gap
    between one run's END and the next run's START exceeds h -- inside
    that gap, the h-day forward windows from the two runs can still
    overlap, so they are not independent identifying observations of the
    kind McNemar's test assumes.
    """
    valid = np.isfinite(l_obs) & np.isfinite(l_filt)
    disagree = (l_obs != l_filt) & valid
    n_disagree = int(disagree.sum())

    d = disagree.astype(int)
    change = np.diff(np.concatenate([[0], d, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]   # exclusive end index of each run
    n_runs = len(starts)

    n_independent_runs = 0
    last_end = -10 ** 9
    for s, e in zip(starts, ends):
        if s - last_end > h:
            n_independent_runs += 1
            last_end = e
        else:
            last_end = max(last_end, e)

    return {
        "h": h, "n_disagree": n_disagree, "n_runs": n_runs,
        "n_independent_runs": n_independent_runs,
    }


def mcnemar_required_n(p_effect: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """(z_{a/2} + z_b)^2 / (2p - 1)^2 -- required discordant-pair count
    for McNemar's test to have `power` at significance `alpha`, given the
    observed effect size p (the winning label's win rate on discordant
    pairs). Returns inf if p == 0.5 (no effect, no finite sample suffices).
    """
    if abs(p_effect - 0.5) < 1e-12:
        return float("inf")
    z_alpha2 = 1.9599639845400545   # scipy.stats.norm.ppf(0.975)
    z_beta = 0.8416212335729143     # scipy.stats.norm.ppf(0.80)
    return (z_alpha2 + z_beta) ** 2 / (2.0 * p_effect - 1.0) ** 2


def identifying_count_gate(l_obs: np.ndarray, l_filt: np.ndarray, log_f: np.ndarray,
                           h: int, far: tuple[int, int] = (2, 3)) -> dict:
    """identifying_count plus the McNemar power comparison at the
    OBSERVED effect size (filt_win_rate on disagreement days, against the
    decisive cross-maturity target) -- the single gate that decides
    whether disagreement_event_study should run at this horizon at all.
    """
    y = cross_maturity_spread_change(log_f, h, far=far)
    valid = np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(y)
    disagree = (l_obs != l_filt) & valid
    counts = identifying_count(l_obs, l_filt, h)

    if disagree.sum() == 0:
        p_effect = float("nan")
        required_n = float("nan")
    else:
        filt_correct = np.sign(l_filt[disagree]) == np.sign(y[disagree])
        p_effect = float(filt_correct.mean())
        required_n = mcnemar_required_n(p_effect)

    counts["p_effect"] = p_effect
    counts["required_n"] = required_n
    counts["sufficient_disagree_days"] = (np.isfinite(required_n)
                                          and counts["n_disagree"] >= required_n)
    counts["sufficient_independent_runs"] = (np.isfinite(required_n)
                                             and counts["n_independent_runs"] >= required_n)
    return counts


# ===================================== 4. Horizon battery, both targets ===


def run_encompassing_battery(p: ThreeFactorParams, x_filt: np.ndarray, log_f: np.ndarray,
                             tau: np.ndarray, horizons: tuple[int, ...] = HORIZONS
                             ) -> dict[str, dict[int, RegressionResult]]:
    """PRIMARY (realised_vol) and SECONDARY (cross_maturity_spread_change)
    targets, full sample, at every horizon -- no April-2020 exclusion or
    vol-tercile split ported (those were built around spread_change,
    which this module withdraws; re-add if/when a specific reason
    surfaces for the new targets)."""
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)

    results: dict[str, dict[int, RegressionResult]] = {PRIMARY_TARGET: {}, SECONDARY_TARGET: {}}
    for h in horizons:
        y_primary = realised_vol(log_f, h)
        results[PRIMARY_TARGET][h] = encompassing_regression(
            y_primary, l_obs, l_filt, h, target=PRIMARY_TARGET, subset="full")
        y_secondary = cross_maturity_spread_change(log_f, h)
        results[SECONDARY_TARGET][h] = encompassing_regression(
            y_secondary, l_obs, l_filt, h, target=SECONDARY_TARGET, subset="full")
    return results


def print_encompassing_battery(results: dict[str, dict[int, RegressionResult]]) -> None:
    for target, by_h in results.items():
        print(f"\n{target}:")
        for h, r in by_h.items():
            print(f"  {r}")


# ===================================== 5. Confidence split (filter-as- =====
#                                          estimator, not classifier) =======


def confidence_split_regression(p: ThreeFactorParams, x_filt: np.ndarray,
                                P_filt: np.ndarray, log_f: np.ndarray,
                                tau: np.ndarray, h: int
                                ) -> tuple[RegressionResult, RegressionResult]:
    """Splits on the filter's own calibrated uncertainty about the
    IMPLIED SPREAD f1 - f2 (not an arbitrary marginal chi1 variance):
    Var(f1-f2) = z'*P_chi*z where z = (e^{-k1*tau1}-e^{-k1*tau2},
    e^{-k2*tau1}-e^{-k2*tau2}) and P_chi is the (chi1,chi2) 2x2 sub-block
    of P_filt -- the exact quantity filtered_label's sign decision is
    made from, so this is the filter's own confidence in the object being
    tested, not a proxy for it. The naive label has no equivalent at all,
    which is the point: this is a comparison the observed spread cannot
    enter, run on every day rather than only disagreement days.

    Returns (high_uncertainty, high_confidence), split at the median.
    """
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)
    y = cross_maturity_spread_change(log_f, h)

    k1, k2 = p.kappa1, p.kappa2
    z = np.column_stack([
        np.exp(-k1 * tau[:, 0]) - np.exp(-k1 * tau[:, 1]),
        np.exp(-k2 * tau[:, 0]) - np.exp(-k2 * tau[:, 1]),
    ])   # (n, 2)
    P_chi = P_filt[:, :2, :2]   # (n, 2, 2)
    var_spread = np.einsum("ti,tij,tj->t", z, P_chi, z)

    valid = np.isfinite(y) & np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(var_spread)
    median_v = np.median(var_spread[valid])
    high_uncertainty = valid & (var_spread >= median_v)
    high_confidence = valid & (var_spread < median_v)

    maxlags = max(h - 1, 0)
    r_low_conf = encompassing_regression(y, l_obs, l_filt, h,
                                         target=SECONDARY_TARGET, subset="high_uncertainty",
                                         extra_mask=high_uncertainty, maxlags=maxlags)
    r_high_conf = encompassing_regression(y, l_obs, l_filt, h,
                                          target=SECONDARY_TARGET, subset="high_confidence",
                                          extra_mask=high_confidence, maxlags=maxlags)
    return r_low_conf, r_high_conf


# ===================================== 6. xi / chi1-chi2 vs. log price =====


def xi_encompasses_log_price(x_filt: np.ndarray, log_f: np.ndarray, h: int) -> RegressionResult:
    """xi_hat (x_filt[:, 2] -- state order chi1, chi2, xi) vs. log_f1,
    encompassing form, same reasoning as recovery.py's two-factor version:
    if xi_hat subsumes log_f1, the permanent/transitory decomposition is
    doing real work; if log_f1 dominates, it's repackaged mean reversion
    in log(price), a well-known result the filter adds nothing to.
    """
    y = front_return(log_f, h)
    log_f1 = log_f[:, 0]
    xi_hat = x_filt[:, 2]
    return encompassing_regression(y, log_f1, xi_hat, h,
                                   target="xi_encompasses_log_price", subset="full")


@dataclass
class ChiSplitResult:
    """y ~ 1 + log_f1 + chi1_hat + chi2_hat, HAC. The chi1/chi2 SPLIT
    encompassed against log F1 -- a genuinely different question from
    xi_encompasses_log_price (that one is log_f1 vs. xi; this is whether
    decomposing chi into two persistence classes adds anything beyond
    log_f1, jointly). encompassing_regression is fixed at two regressors
    by construction, so this is its own small three-regressor
    implementation rather than a forced reuse."""

    h: int
    n_obs: int
    c_log_f1: float
    c_chi1: float
    c_chi2: float
    p_log_f1: float
    p_chi1: float
    p_chi2: float
    r_squared: float
    joint_chi_stat: float   # Wald: c_chi1 = c_chi2 = 0
    joint_chi_p: float
    degenerate: bool

    def __str__(self) -> str:
        note = "  (DEGENERATE -- regressors collinear over these rows)" if self.degenerate else ""
        return (f"[chi_split_vs_log_f1 h={self.h}] n={self.n_obs}  "
               f"c(log_f1)={self.c_log_f1:+.4f}(p={self.p_log_f1:.4f})  "
               f"c(chi1)={self.c_chi1:+.4f}(p={self.p_chi1:.4f})  "
               f"c(chi2)={self.c_chi2:+.4f}(p={self.p_chi2:.4f})  "
               f"R2={self.r_squared:.4f}  "
               f"joint(chi1=chi2=0): stat={self.joint_chi_stat:.3f} p={self.joint_chi_p:.4f}{note}")


def chi_split_encompasses_log_price(x_filt: np.ndarray, log_f: np.ndarray, h: int
                                    ) -> ChiSplitResult:
    y = front_return(log_f, h)
    log_f1 = log_f[:, 0]
    chi1_hat = x_filt[:, 0]
    chi2_hat = x_filt[:, 1]

    X = sm.add_constant(np.column_stack([log_f1, chi1_hat, chi2_hat]), has_constant="add")
    valid = np.isfinite(y) & np.isfinite(log_f1) & np.isfinite(chi1_hat) & np.isfinite(chi2_hat)
    res = sm.OLS(np.where(valid, y, np.nan), X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
    degenerate = res.model.rank < X.shape[1]
    alpha, c_lf1, c1, c2 = res.params
    if degenerate:
        joint_stat, joint_p = float("nan"), float("nan")
    else:
        joint = res.wald_test(np.array([[0, 0, 1, 0], [0, 0, 0, 1]]), scalar=True)
        joint_stat, joint_p = float(joint.statistic), float(joint.pvalue)
    return ChiSplitResult(
        h=h, n_obs=int(res.nobs), c_log_f1=float(c_lf1), c_chi1=float(c1), c_chi2=float(c2),
        p_log_f1=float(res.pvalues[1]), p_chi1=float(res.pvalues[2]), p_chi2=float(res.pvalues[3]),
        r_squared=float(res.rsquared), joint_chi_stat=joint_stat, joint_chi_p=joint_p,
        degenerate=degenerate,
    )


# ===================================== 7. Disagreement-conditioned test ===
# (only meaningful once identifying_count_gate has cleared -- callers must
# check that before running this, per module docstring point 4)


def disagreement_event_study(l_obs: np.ndarray, l_filt: np.ndarray,
                             log_f: np.ndarray, h: int,
                             far: tuple[int, int] = (2, 3)) -> dict:
    y = cross_maturity_spread_change(log_f, h, far=far)
    valid = np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(y)
    disagree = (l_obs != l_filt) & valid
    y_d, lo_d, lf_d = y[disagree], l_obs[disagree], l_filt[disagree]

    obs_correct = np.sign(lo_d) == np.sign(y_d)
    filt_correct = np.sign(lf_d) == np.sign(y_d)

    both_correct = int(np.sum(obs_correct & filt_correct))
    only_obs = int(np.sum(obs_correct & ~filt_correct))
    only_filt = int(np.sum(~obs_correct & filt_correct))
    both_wrong = int(np.sum(~obs_correct & ~filt_correct))
    table = np.array([[both_wrong, only_filt], [only_obs, both_correct]])

    result = mcnemar(table, exact=True)
    return {
        "h": h, "n_disagree": int(disagree.sum()),
        "obs_win_rate": float(obs_correct.mean()) if len(obs_correct) else float("nan"),
        "filt_win_rate": float(filt_correct.mean()) if len(filt_correct) else float("nan"),
        "n_only_obs_correct": only_obs, "n_only_filt_correct": only_filt,
        "n_both_correct": both_correct, "n_both_wrong": both_wrong,
        "mcnemar_stat": float(result.statistic), "mcnemar_p": float(result.pvalue),
    }
