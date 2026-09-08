"""Study A: state recovery and predictive validity -- Gate 4 of the design.

Price-only; does not touch the news pipeline.

The claim. On simulated data with a known state, the filtered regime label
recovered the truth 97.4% of the time against 84.3% for reading F1 > F2
directly, because measurement error is ~62% of the front spread's own
standard deviation. Study A asks whether that transfers to WTI.

Why it can't be tested directly. On market data there is no true state --
it is latent, which is the premise of the whole framework. So the test is
not accuracy against truth but PREDICTIVE VALIDITY: which label carries
more information about something independently observable in the future.
That needs no ground truth and is a real claim about oil markets.

Section map
-----------
1. The two labels (naive_label, filtered_label, continuous forms).
2. Target variables (spread_change, front_return, realised_vol).
3. The test: encompassing_regression, one regression with both labels on
   the right-hand side together, not two separate horse races -- they
   agree ~85% of the time, so testing them separately would mostly measure
   shared content. HAC (Newey-West) standard errors, >= h-1 lags: overlapping
   h-day windows autocorrelate the residuals, and OLS SEs under overlap are
   the standard way a result like this comes out significant and wrong.
4. Passing the gate: classify_gate.
5. run_study_a: the full reporting battery -- both horizons x three
   targets, with/without April 2020, split by realised-vol tercile, plus
   the continuous-form robustness check.

Three things that would silently invalidate this, all from the design's
own Sec. 1:
    - Filtered states, never smoothed. The smoother conditions on future
      data; using it here manufactures exactly the look-ahead the study is
      built to avoid. Every x_filt this module consumes must come from
      run_spec/fit_s0_student_t directly, never through state.fast.smooth_rts.
    - Parameters from the Student-t S0 fit (csr.pricing.fit.fit_s0_student_t),
      not the Gaussian one -- the Gaussian fits sit at a boundary and their
      kappa is unstable across sample windows (see csr.pricing.fit and
      explore/02_fit_s0.py's diagnostics).
    - A(tau) recomputed per date, because tau shrinks between rolls. Every
      function here that needs A(tau) takes the actual per-date tau array,
      never a single scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..pricing.twofactor import A_tau, TwoFactorParams

# ============================================================ 1. Labels ===


def naive_label(log_f: np.ndarray) -> np.ndarray:
    """L_obs: sign of the front log spread s_t = log_f[:,0] - log_f[:,1].
    +1 = backwardation. NaN wherever either of the two front contracts is
    missing that date (e.g. 2020-04-20's negative front-month settlement)
    -- there is no reading to take a sign of, so this must not silently
    become a stale or fabricated value."""
    s = log_f[:, 0] - log_f[:, 1]
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(s), np.nan, np.sign(s))


def filtered_label(p: TwoFactorParams, x_filt: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """L_filt: sign of the model-implied spread at the SAME two maturities
    the naive label uses, built from the FILTERED (never smoothed) state.

    f_i = exp(-kappa*tau_i)*chi_hat + A(tau_i);  L_filt = sign(f1 - f2)

    xi cancels in the difference, so only chi_hat and A(tau) matter. This
    is the discrete form, not the instantaneous slope (-kappa*chi_hat + a_bar,
    see continuous_filtered): they agree ~99.8% of the time, but the
    discrete form is measured at the same maturities as the naive label, so
    the comparison isolates the input (filtered state vs raw prices) rather
    than confounding it with a definitional difference. Unlike the naive
    label, this is finite even on a day the front-month PRICE is missing --
    the predict step of a Kalman filter runs unconditionally every period,
    so a day with a missing price observation still has a valid filtered
    chi_hat. That asymmetry is real and expected, not a bug: it is the
    concrete mechanism Gate 4 is testing.
    """
    chi_hat = x_filt[:, 0]
    f1 = np.exp(-p.kappa * tau[:, 0]) * chi_hat + A_tau(p, tau[:, 0])
    f2 = np.exp(-p.kappa * tau[:, 1]) * chi_hat + A_tau(p, tau[:, 1])
    return np.sign(f1 - f2)


def continuous_naive(log_f: np.ndarray) -> np.ndarray:
    """s_t, the continuous-form observed quantity for the Sec. 3 robustness
    check (replacing L_obs's sign with the underlying level)."""
    return log_f[:, 0] - log_f[:, 1]


def continuous_filtered(p: TwoFactorParams, x_filt: np.ndarray) -> np.ndarray:
    """-kappa*chi_hat + a_bar: the instantaneous risk-neutral drift of chi,
    for the Sec. 3 continuous-form robustness check ONLY. The binary label
    above uses the discrete, matching-maturity form instead -- this is the
    clean theoretical statement, not the like-for-like empirical one."""
    a_bar = (p.mu_xi_star - p.lambda_chi
            + 0.5 * (p.sigma_chi ** 2 + p.sigma_xi ** 2
                    + 2.0 * p.rho * p.sigma_chi * p.sigma_xi))
    return -p.kappa * x_filt[:, 0] + a_bar


# ==================================================== 2. Target variables ===


def front_spread(log_f: np.ndarray) -> np.ndarray:
    return log_f[:, 0] - log_f[:, 1]


def spread_change(log_f: np.ndarray, h: int) -> np.ndarray:
    """PRIMARY target. y_t^(h) = s_{t+h} - s_t. The direct test: chi mean-
    reverts, so a label that correctly identifies a positive chi should
    predict the spread narrowing. The target is a CHANGE while the naive
    label is a LEVEL, so there is no circularity between the two."""
    s = front_spread(log_f)
    y = np.full_like(s, np.nan)
    if h < len(s):
        y[: len(s) - h] = s[h:] - s[:-h]
    return y


def front_return(log_f: np.ndarray, h: int) -> np.ndarray:
    """SECONDARY target. r_t^(h) = log_f[t+h,0] - log_f[t,0]. A different
    object from the primary: level rather than slope -- tests whether the
    regime label says anything about direction as well as carry."""
    f = log_f[:, 0]
    y = np.full_like(f, np.nan)
    if h < len(f):
        y[: len(f) - h] = f[h:] - f[:-h]
    return y


def realised_vol(log_f: np.ndarray, h: int) -> np.ndarray:
    """SECONDARY target. v_t^(h) = std of daily front-contract log returns
    over (t, t+h]. Physical tightness is associated with higher subsequent
    volatility, so a label that genuinely identifies tightness should carry
    information about it. A window containing any missing daily return
    (e.g. spanning 2020-04-20, whose front-month log price is undefined) is
    NaN rather than computed over fewer, silently non-comparable days."""
    f = log_f[:, 0]
    n = len(f)
    ret = np.full(n, np.nan)
    ret[1:] = np.diff(f)
    y = np.full(n, np.nan)
    for t in range(n - h):
        window = ret[t + 1: t + h + 1]
        if np.all(np.isfinite(window)):
            y[t] = float(np.std(window, ddof=1))
    return y


# ===================================================== 3. The test ==========


@dataclass
class RegressionResult:
    """One encompassing regression: y_t^(h) = alpha + b1*L_obs + b2*L_filt + e."""

    target: str
    horizon: int
    subset: str
    n_obs: int
    alpha: float
    b1: float
    b2: float
    se_b1: float
    se_b2: float
    p_b1: float
    p_b2: float
    r_squared: float
    fit: object = field(repr=False, default=None)   # the raw statsmodels result
    degenerate: bool = False   # design matrix rank-deficient -- see __post_init__

    @property
    def b1_significant(self) -> bool:
        return self.p_b1 < 0.05

    @property
    def b2_significant(self) -> bool:
        return self.p_b2 < 0.05

    @property
    def n_regressors(self) -> int:
        """1 for a single-regressor fit (b1/p_b1 are NaN by construction,
        e.g. xi_monopoly_test -- no naive comparator fit at all), else 2."""
        return 1 if np.isnan(self.b1) else 2

    @property
    def r_squared_adj(self) -> float:
        """Adjusted R^2 -- penalises R^2 for the ratio of regressors to
        observations, which matters here specifically because several
        horizons run at T/h in the low tens (Sec. 7); raw R^2 alone
        overstates fit quality exactly where the effective sample is
        smallest."""
        n, k = self.n_obs, self.n_regressors
        if n - k - 1 <= 0:
            return float("nan")
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / (n - k - 1)

    def pattern(self) -> str:
        """Which of the four sig/insig combinations this regression landed
        in -- the building block classify_gate aggregates across horizons.
        "degenerate" is a fifth, deliberately-not-sig/insig outcome: the
        design matrix was rank-deficient (l_obs and l_filt collinear, most
        often because they were IDENTICAL over the fitted rows -- a real
        possibility on a small or sparsely-sampled subset, since the two
        labels agree the large majority of the time on daily data), so b1
        and b2 are not separately identified and neither their point
        estimates nor p-values mean what they'd normally mean."""
        if self.degenerate:
            return "degenerate"
        if self.b2_significant and not self.b1_significant:
            return "b2_only"
        if self.b2_significant and self.b1_significant:
            return "both"
        if self.b1_significant and not self.b2_significant:
            return "b1_only"
        return "neither"

    def __str__(self) -> str:
        note = "" if not self.degenerate else \
              "  (labels collinear over these rows -- b1/b2 not separately identified)"
        return (f"[{self.target} h={self.horizon} {self.subset}] n={self.n_obs}  "
               f"b1(naive)={self.b1:+.4f} (p={self.p_b1:.4f})  "
               f"b2(filt)={self.b2:+.4f} (p={self.p_b2:.4f})  "
               f"R2={self.r_squared:.4f}  [{self.pattern()}]{note}")


def encompassing_regression(y: np.ndarray, l_obs: np.ndarray, l_filt: np.ndarray,
                            h: int, target: str = "", subset: str = "full",
                            extra_mask: np.ndarray | None = None,
                            maxlags: int | None = None) -> RegressionResult:
    """OLS of y on [1, L_obs, L_filt] with HAC (Newey-West) standard errors.

    ``maxlags`` defaults to h - 1 (the minimum implied by h-day overlap) --
    every existing call site relies on this default and is unchanged by
    its addition. An explicit ``maxlags`` (e.g. floor(1.5*h), Sec. 7's
    horizon-robustness table) overrides it for callers that want a more
    conservative lag count than the bare minimum.

    Rows with any NaN among y/l_obs/l_filt, or excluded by `extra_mask`
    (e.g. April 2020, or outside a volatility tercile), are dropped
    listwise -- statsmodels' missing='drop' handles this uniformly rather
    than needing separate masking logic per caller.
    """
    y = np.asarray(y, float).copy()
    if extra_mask is not None:
        y = np.where(extra_mask, y, np.nan)
    X = sm.add_constant(np.column_stack([l_obs, l_filt]), has_constant="add")
    model = sm.OLS(y, X, missing="drop")
    if maxlags is None:
        maxlags = max(h - 1, 0)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha, b1, b2 = res.params
    se_b1, se_b2 = res.bse[1], res.bse[2]
    p_b1, p_b2 = res.pvalues[1], res.pvalues[2]
    # rank < 3 columns (const, l_obs, l_filt): l_obs and l_filt were
    # collinear over the FITTED rows -- most often because they were
    # exactly identical, a real possibility on a small or sparsely-sampled
    # subset since the two labels agree on the large majority of days.
    # b1/b2 are then not separately identified; statsmodels still returns
    # a (minimum-norm) split, which is not a meaningful result.
    degenerate = res.model.rank < X.shape[1]
    return RegressionResult(
        target=target, horizon=h, subset=subset, n_obs=int(res.nobs),
        alpha=float(alpha), b1=float(b1), b2=float(b2),
        se_b1=float(se_b1), se_b2=float(se_b2),
        p_b1=float(p_b1), p_b2=float(p_b2),
        r_squared=float(res.rsquared), fit=res, degenerate=degenerate,
    )


# =================================================== 4. Passing the gate ===

GateVerdict = str   # "pass" | "partial" | "fail" | "inconclusive" | "degenerate"


def classify_gate(results_by_horizon: dict[int, RegressionResult]) -> GateVerdict:
    """Sec. 4's rule, applied to the PRIMARY target's regressions across
    both required horizons (5 and 22 trading days).

    Pass: b2 significant at 5% with HAC errors, b1 insignificant (or
        materially smaller when both are significant), on BOTH horizons.
    Partial: b2 significant on one horizon only, or both labels significant
        on a horizon where the gate would otherwise pass. Report as such --
        the claim weakens rather than dies, and Study B continues.
    Fail: b1 dominates -- significant while b2 is not -- on at least one
        horizon. The central claim is wounded.
    Inconclusive: neither label is significant on a horizon; the design
        doc's three-way taxonomy doesn't cover this case explicitly, so it
        is surfaced rather than forced into pass/partial/fail.
    Degenerate: l_obs and l_filt were collinear (most often identical) over
        the fitted rows on at least one horizon -- b1/b2 aren't separately
        identified there, so NONE of the above verdicts can honestly be
        read off that horizon's numbers. Checked first, before any
        significance-based branch, since a degenerate fit's p-values are
        not meaningful inputs to a pass/fail/partial/inconclusive call.
    """
    patterns = {h: r.pattern() for h, r in results_by_horizon.items()}

    if any(p == "degenerate" for p in patterns.values()):
        return "degenerate"
    if all(p == "b2_only" for p in patterns.values()):
        return "pass"
    if any(p == "b1_only" for p in patterns.values()):
        return "fail"
    if any(p == "neither" for p in patterns.values()):
        return "inconclusive"
    # Remaining case: every horizon is "b2_only" or "both", with at least
    # one "both" -- b2 always clears the bar, b1 sometimes also does.
    return "partial"


# ===================================================== 5. Full battery =====

PRIMARY_TARGET = "spread_change"
HORIZONS = (5, 22)
VOL_TERCILE_LABELS = ("low_vol", "mid_vol", "high_vol")


def _target_fn(name: str):
    return {"spread_change": spread_change, "front_return": front_return,
           "realised_vol": realised_vol}[name]


def _vol_tercile_masks(vol: np.ndarray) -> dict[str, np.ndarray]:
    """Terciles of the (finite) realised-vol distribution -- the denoising
    effect should be strongest where measurement error matters most
    relative to signal, i.e. the high-vol tercile, a secondary prediction
    worth checking (Sec. 5)."""
    finite = vol[np.isfinite(vol)]
    if len(finite) < 3:
        return {}
    q1, q2 = np.nanpercentile(finite, [100 / 3, 200 / 3])
    return {
        "low_vol": vol <= q1,
        "mid_vol": (vol > q1) & (vol <= q2),
        "high_vol": vol > q2,
    }


@dataclass
class StudyAResult:
    gate: GateVerdict
    primary: dict[int, RegressionResult]                 # h -> full-sample result
    all_results: list[RegressionResult]
    continuous: dict[int, RegressionResult]


def run_study_a(p: TwoFactorParams, x_filt: np.ndarray, log_f: np.ndarray,
                tau: np.ndarray, dates: pd.DatetimeIndex) -> StudyAResult:
    """The full Sec. 5 reporting battery: both horizons x three targets,
    with/without April 2020, the primary target split by realised-vol
    tercile, and the continuous-form robustness check. Returns every
    regression run, not just the ones the gate decision needs -- Sec. 5 is
    explicit that every specification gets reported, not the best one.
    """
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)

    apr2020 = (dates >= pd.Timestamp("2020-04-01")) & (dates <= pd.Timestamp("2020-04-30"))
    not_apr2020 = ~apr2020

    all_results: list[RegressionResult] = []
    primary: dict[int, RegressionResult] = {}
    continuous: dict[int, RegressionResult] = {}

    for target_name in ("spread_change", "front_return", "realised_vol"):
        fn = _target_fn(target_name)
        for h in HORIZONS:
            y = fn(log_f, h)

            r_full = encompassing_regression(y, l_obs, l_filt, h,
                                             target=target_name, subset="full")
            all_results.append(r_full)
            if target_name == PRIMARY_TARGET:
                primary[h] = r_full

            r_ex = encompassing_regression(y, l_obs, l_filt, h,
                                           target=target_name,
                                           subset="excl_apr2020",
                                           extra_mask=not_apr2020)
            all_results.append(r_ex)

            if target_name == PRIMARY_TARGET:
                vol_h = realised_vol(log_f, h)
                for tercile_name, mask in _vol_tercile_masks(vol_h).items():
                    r_ter = encompassing_regression(
                        y, l_obs, l_filt, h, target=target_name,
                        subset=tercile_name, extra_mask=mask)
                    all_results.append(r_ter)

    c_obs = continuous_naive(log_f)
    c_filt = continuous_filtered(p, x_filt)
    for h in HORIZONS:
        y = spread_change(log_f, h)
        r_c = encompassing_regression(y, c_obs, c_filt, h,
                                      target="spread_change_continuous", subset="full")
        continuous[h] = r_c
        all_results.append(r_c)

    gate = classify_gate(primary)
    return StudyAResult(gate=gate, primary=primary, all_results=all_results,
                        continuous=continuous)


def print_report(result: StudyAResult) -> None:
    print(f"\n{'=' * 70}\nGATE 4 VERDICT: {result.gate.upper()}\n{'=' * 70}")
    print(f"\nPrimary target ({PRIMARY_TARGET}), full sample, both required horizons:")
    for h in HORIZONS:
        print(f"  {result.primary[h]}")
    print(f"\nContinuous-form robustness ({PRIMARY_TARGET}):")
    for h in HORIZONS:
        print(f"  {result.continuous[h]}")
    print(f"\nFull battery ({len(result.all_results)} regressions):")
    for r in result.all_results:
        print(f"  {r}")


# ======================================= 6. Shared-measurement-error fix ===
#
# The original PRIMARY target, spread_change, is s_{t+h} - s_t. The naive
# label is sign(s_t). The same s_t therefore appears in both the label and
# the target, negated -- so noise in s_t mechanically produces a negative
# naive-label coefficient with no true predictive content behind it, and
# the filtered label, having denoised exactly that s_t, structurally cannot
# exploit the same mechanical relationship. That artefact predicts every
# anomaly the original run_study_a battery showed: b1 negative, b2 positive
# on data with zero differential signal; naive dominating hardest in the
# high-vol tercile (where the shared noise is largest, not where the
# theory's denoising advantage should be largest); realised_vol (which
# shares no component with either label) being the one target where the
# filtered label won cleanly. Reproduced independently on a pure random
# walk with a perfect second label before this section was written.
#
# The three specifications below each break the shared-noise mechanism a
# different way, without touching naive_label, filtered_label, or any of
# the ORIGINAL targets/regressions above -- those 20 regressions stand as
# the naive-specification panel that motivated this section, not something
# this corrects retroactively.


def cross_maturity_spread_change(log_f: np.ndarray, h: int,
                                 far: tuple[int, int] = (2, 3)) -> np.ndarray:
    """DECISIVE spec. y_t^(h) built from a DIFFERENT contract pair (C3-C4
    by default) than the labels, which always use C1/C2 (naive_label,
    filtered_label). C3 and C4's measurement errors are independent of
    C1/C2's under the model's diagonal Hdiag, so this target shares no
    noise component with either label -- if the filtered label still loses
    here, the original Gate 4 fail is real, not an artefact."""
    s_far = log_f[:, far[0]] - log_f[:, far[1]]
    y = np.full_like(s_far, np.nan)
    if h < len(s_far):
        y[: len(s_far) - h] = s_far[h:] - s_far[:-h]
    return y


def forward_spread_level(log_f: np.ndarray, h: int) -> np.ndarray:
    """y_t^(h) = s_{t+h}, a LEVEL rather than a change. s_t itself never
    enters the target, so it cannot appear (negated) in both the naive
    label sign(s_t) and y the way it did in the original spread_change."""
    s = front_spread(log_f)
    y = np.full_like(s, np.nan)
    if h < len(s):
        y[: len(s) - h] = s[h:]
    return y


def lag_labels(l_obs: np.ndarray, l_filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift both labels back one session: L_{t-1} predicts y_t, so the
    label's own measurement noise (from day t-1) and the target's (from
    day t, via s_t in the original spread_change construction) come from
    different days -- independent under the model's i.i.d. measurement-
    error assumption, rather than the same shared s_t."""
    l_obs_lag = np.full_like(l_obs, np.nan)
    l_filt_lag = np.full_like(l_filt, np.nan)
    l_obs_lag[1:] = l_obs[:-1]
    l_filt_lag[1:] = l_filt[:-1]
    return l_obs_lag, l_filt_lag


@dataclass
class CorrectedResult:
    cross_maturity: dict[int, RegressionResult]
    forward_level: dict[int, RegressionResult]
    lagged_labels: dict[int, RegressionResult]
    gate: GateVerdict   # decided on cross_maturity, the decisive spec

    def all_results(self) -> list[RegressionResult]:
        return (list(self.cross_maturity.values())
               + list(self.forward_level.values())
               + list(self.lagged_labels.values()))


def run_corrected_battery(p: TwoFactorParams, x_filt: np.ndarray, log_f: np.ndarray,
                          tau: np.ndarray) -> CorrectedResult:
    """The three shared-measurement-error-free specifications, each run at
    both required horizons. Reuses encompassing_regression and
    classify_gate unmodified -- only the label/target CONSTRUCTION differs
    from the original battery, not the test machinery itself."""
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)
    l_obs_lag, l_filt_lag = lag_labels(l_obs, l_filt)

    cross_maturity: dict[int, RegressionResult] = {}
    forward_level: dict[int, RegressionResult] = {}
    lagged_labels: dict[int, RegressionResult] = {}

    for h in HORIZONS:
        cross_maturity[h] = encompassing_regression(
            cross_maturity_spread_change(log_f, h), l_obs, l_filt, h,
            target="cross_maturity_spread_change", subset="full")
        forward_level[h] = encompassing_regression(
            forward_spread_level(log_f, h), l_obs, l_filt, h,
            target="forward_spread_level", subset="full")
        lagged_labels[h] = encompassing_regression(
            spread_change(log_f, h), l_obs_lag, l_filt_lag, h,
            target="spread_change", subset="lagged_labels")

    gate = classify_gate(cross_maturity)
    return CorrectedResult(cross_maturity=cross_maturity, forward_level=forward_level,
                           lagged_labels=lagged_labels, gate=gate)


def print_corrected_report(result: CorrectedResult) -> None:
    print(f"\n{'=' * 70}\nCORRECTED GATE 4 VERDICT (cross-maturity, decisive): "
         f"{result.gate.upper()}\n{'=' * 70}")
    print("\nCross-maturity (labels from C1-C2, target from C3-C4):")
    for h in HORIZONS:
        print(f"  {result.cross_maturity[h]}")
    print("\nForward-only (y = s_{t+h} as a level, s_t not in the target):")
    for h in HORIZONS:
        print(f"  {result.forward_level[h]}")
    print("\nLagged labels (L_{t-1} predicting y_t):")
    for h in HORIZONS:
        print(f"  {result.lagged_labels[h]}")


# =========================== 7. Horizon robustness (the third spec flaw) ===
#
# h in {5, 22} were chosen by convention, never checked against the
# fitted model's own half-life. Against kappa ~1.31 (half-life ~193
# trading days), those horizons span only 1.8%/7.6% of chi's reversion --
# the shared-measurement-error bias fixed in Sec. 6 is O(1) in h (a single
# noise draw, the same size at every horizon), so the original test asked
# the filtered label to beat a full-strength artefact using at most 7.6%
# of the signal it exists to detect. Cross-maturity (Sec. 6) inherits the
# same horizons and the same power problem -- which is why it came back
# "inconclusive" rather than resolving cleanly, and why that can't be the
# last word.
#
# This section reruns the decisive cross-maturity spec at horizons chosen
# against the ACTUAL fitted half-life: 5, 22, 66, 132, 252 trading days.
# The theory's sharp, falsifiable, monotone prediction: filtered
# performance relative to naive should IMPROVE with horizon, since signal
# accumulates and the shared-noise bias doesn't.
#
# Three things change with h, not just its value:
#   - HAC lags scale as floor(1.5h), a more conservative choice than the
#     bare h-1 minimum used for the original battery.
#   - A non-overlapping check at h=66 and h=132 (every h-th row -> plain
#     OLS, no HAC needed since there is no overlap left to autocorrelate).
#     If HAC and non-overlapping diverge in sign or significance, the
#     non-overlapping result is the one to trust.
#   - Effective (independent-block) sample size T/h is reported alongside
#     every row -- at h=252 that's ~14 blocks, and the standard errors are
#     not to be believed at that count regardless of what they print.

EXTENDED_HORIZONS = (5, 22, 66, 132, 252)
NONOVERLAP_HORIZONS = (66, 132)


def reversion_fraction(kappa: float, h: int) -> float:
    """1 - exp(-kappa*h/252): fraction of chi's mean-reversion a horizon of
    h trading days spans, at the fitted kappa. What should grow with h if
    the filtered label's advantage is real signal accumulation rather than
    an artefact -- the diagnostic that makes the horizon-choice power
    problem legible to a reader instead of something to take on faith."""
    return 1.0 - float(np.exp(-kappa * h / 252.0))


def _nonoverlapping_regression(y: np.ndarray, l_obs: np.ndarray, l_filt: np.ndarray,
                               h: int, target: str = "") -> RegressionResult:
    """Every h-th row starting at 0: genuinely independent (t, t+h] windows,
    so plain (non-HAC) OLS is valid -- there is no overlap left for HAC to
    correct for."""
    idx = np.arange(0, len(y), h)
    y_sub = y[idx]
    X = sm.add_constant(np.column_stack([l_obs[idx], l_filt[idx]]), has_constant="add")
    res = sm.OLS(y_sub, X, missing="drop").fit()
    alpha, b1, b2 = res.params
    degenerate = res.model.rank < X.shape[1]   # see encompassing_regression's note
    return RegressionResult(
        target=target, horizon=h, subset="nonoverlap", n_obs=int(res.nobs),
        alpha=float(alpha), b1=float(b1), b2=float(b2),
        se_b1=float(res.bse[1]), se_b2=float(res.bse[2]),
        p_b1=float(res.pvalues[1]), p_b2=float(res.pvalues[2]),
        r_squared=float(res.rsquared), fit=res, degenerate=degenerate,
    )


def _wald_tests(fit) -> tuple[float, float, float, float]:
    """Joint (b1=b2=0, 'does the pair carry any information') and
    difference (b1=b2, 'can the two labels be told apart', the study's
    actual discrimination question) Wald tests, using whatever covariance
    estimator (HAC or plain) is already attached to `fit`. Matrix-form
    restrictions rather than string formulas naming 'x1'/'x2' -- this must
    not depend on statsmodels' auto-generated column names, since the
    design matrix is always [const, l_obs, l_filt] in that order by
    construction, not by name.

    Under near-collinearity the individual p-values can both be
    insignificant while the joint test rejects -- the classic collinearity
    signature -- and the difference test directly answers whether the
    encompassing design can tell the two labels apart at all, which
    neither individual p-value does on its own.
    """
    joint = fit.wald_test(np.array([[0, 1, 0], [0, 0, 1]]), scalar=True)
    diff = fit.wald_test(np.array([[0, 1, -1]]), scalar=True)
    return (float(joint.statistic), float(joint.pvalue),
           float(diff.statistic), float(diff.pvalue))


def _label_corr_and_vif(l_obs: np.ndarray, l_filt: np.ndarray,
                        valid: np.ndarray) -> tuple[float, float, float]:
    """Pearson correlation and variance inflation factors for the two
    labels over the valid (non-NaN) rows actually used in the fit --
    VIF > ~10 is the conventional "problematic multicollinearity" line;
    near-collinear labels inflate the individual coefficient variances
    even when the pair jointly carries real information."""
    lo, lf = l_obs[valid], l_filt[valid]
    corr = float(np.corrcoef(lo, lf)[0, 1])
    X = sm.add_constant(np.column_stack([lo, lf]))
    return corr, float(variance_inflation_factor(X, 1)), float(variance_inflation_factor(X, 2))


@dataclass
class HorizonRow:
    h: int
    n_obs: int
    t_over_h: float
    reversion_frac: float
    hac_maxlags: int
    hac: RegressionResult
    nonoverlap: RegressionResult | None = None
    joint_wald_stat: float = float("nan")
    joint_wald_p: float = float("nan")
    diff_wald_stat: float = float("nan")
    diff_wald_p: float = float("nan")
    label_corr: float = float("nan")
    vif_obs: float = float("nan")
    vif_filt: float = float("nan")

    @property
    def hac_degenerate(self) -> bool:
        return self.hac.degenerate

    def __str__(self) -> str:
        line = (f"h={self.h:4d}  n={self.n_obs:5d}  T/h={self.t_over_h:6.1f}  "
               f"1-e^(-kh/252)={self.reversion_frac:.3f}  "
               f"HAC[lags={self.hac_maxlags}]: "
               f"b1={self.hac.b1:+.5f}(p={self.hac.p_b1:.4f})  "
               f"b2={self.hac.b2:+.5f}(p={self.hac.p_b2:.4f})"
               + ("  [HAC DEGENERATE -- labels collinear, not usable]"
                 if self.hac.degenerate else ""))
        if not self.hac.degenerate:
            line += (f"\n          joint(b1=b2=0): stat={self.joint_wald_stat:.3f} "
                    f"p={self.joint_wald_p:.4f}   "
                    f"diff(b1=b2): stat={self.diff_wald_stat:.3f} "
                    f"p={self.diff_wald_p:.4f}   "
                    f"corr(L_obs,L_filt)={self.label_corr:.3f}  "
                    f"VIF=({self.vif_obs:.1f},{self.vif_filt:.1f})")
        if self.nonoverlap is not None:
            if self.nonoverlap.degenerate:
                line += (f"\n          non-overlap[n={self.nonoverlap.n_obs}]: "
                        f"DEGENERATE -- l_obs and l_filt are collinear (most often "
                        f"exactly identical) over these {self.nonoverlap.n_obs} rows, "
                        f"so b1/b2 are not separately identified here at all")
            else:
                line += (f"\n          non-overlap[n={self.nonoverlap.n_obs}]: "
                        f"b1={self.nonoverlap.b1:+.5f}(p={self.nonoverlap.p_b1:.4f})  "
                        f"b2={self.nonoverlap.b2:+.5f}(p={self.nonoverlap.p_b2:.4f})")
                if not self.hac.degenerate:
                    hac_sign = (np.sign(self.hac.b2), self.hac.b2_significant)
                    non_sign = (np.sign(self.nonoverlap.b2), self.nonoverlap.p_b2 < 0.05)
                    if hac_sign != non_sign:
                        line += "  [HAC vs non-overlap DIVERGE on b2 -- trust non-overlap]"
        return line


def run_horizon_robustness(p: TwoFactorParams, x_filt: np.ndarray, log_f: np.ndarray,
                           tau: np.ndarray,
                           horizons: tuple[int, ...] = EXTENDED_HORIZONS,
                           nonoverlap_horizons: tuple[int, ...] = NONOVERLAP_HORIZONS,
                           ) -> list[HorizonRow]:
    """The decisive cross-maturity spec (Sec. 6: labels from C1/C2, target
    from C3/C4) at horizons chosen against the fitted model's own
    half-life. Uses encompassing_regression's maxlags override (Sec. 3) --
    the original battery's default behaviour (h-1) is untouched."""
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)

    rows = []
    for h in horizons:
        y = cross_maturity_spread_change(log_f, h)
        maxlags = int(np.floor(1.5 * h))
        r_hac = encompassing_regression(y, l_obs, l_filt, h,
                                        target="cross_maturity_spread_change",
                                        subset="full", maxlags=maxlags)

        valid = np.isfinite(y) & np.isfinite(l_obs) & np.isfinite(l_filt)
        t_over_h = float(np.sum(valid)) / h

        if r_hac.degenerate:
            joint_stat = joint_p = diff_stat = diff_p = float("nan")
            corr = vif_o = vif_f = float("nan")
        else:
            joint_stat, joint_p, diff_stat, diff_p = _wald_tests(r_hac.fit)
            corr, vif_o, vif_f = _label_corr_and_vif(l_obs, l_filt, valid)

        r_nonoverlap = None
        if h in nonoverlap_horizons:
            r_nonoverlap = _nonoverlapping_regression(
                y, l_obs, l_filt, h, target="cross_maturity_spread_change")

        rows.append(HorizonRow(
            joint_wald_stat=joint_stat, joint_wald_p=joint_p,
            diff_wald_stat=diff_stat, diff_wald_p=diff_p,
            label_corr=corr, vif_obs=vif_o, vif_filt=vif_f,
            h=h, n_obs=r_hac.n_obs, t_over_h=t_over_h,
            reversion_frac=reversion_fraction(p.kappa, h),
            hac_maxlags=maxlags, hac=r_hac,
            nonoverlap=r_nonoverlap,
        ))
    return rows


def print_horizon_table(rows: list[HorizonRow]) -> None:
    print(f"\n{'=' * 78}\nHORIZON ROBUSTNESS -- cross-maturity spec, "
         f"h in {[r.h for r in rows]}\n{'=' * 78}")
    for r in rows:
        print(r)
    b2s = [r.hac.b2 for r in rows]
    increasing = all(b2s[i] <= b2s[i + 1] for i in range(len(b2s) - 1))
    decreasing = all(b2s[i] >= b2s[i + 1] for i in range(len(b2s) - 1))
    print(f"\nb2 monotone across horizons: "
         f"{'increasing' if increasing else 'decreasing' if decreasing else 'no'}")
    print("(HAC standard errors at h=252 rest on ~14 independent blocks -- "
         "report the number, do not lean on the p-value it implies.)")

    print("\nreversion-normalised coefficients (b/phi(h), x1e3) -- should be "
         "roughly flat under the signal-accumulation story:")
    print(f"  {'h':>5}  {'phi(h)':>8}  {'b1/phi (x1e3)':>15}  {'b2/phi (x1e3)':>15}")
    for r in rows:
        phi = r.reversion_frac
        b1_norm = 1e3 * r.hac.b1 / phi if phi > 0 else float("nan")
        b2_norm = 1e3 * r.hac.b2 / phi if phi > 0 else float("nan")
        print(f"  {r.h:5d}  {phi:8.3f}  {b1_norm:15.2f}  {b2_norm:15.2f}")


# =============================== 8. Label agreement and its consequences ===
#
# Study A's spec was written assuming ~85% naive/filtered agreement. The
# non-overlap degeneracy at h=66/132 (Sec. 7) is a symptom of the actual
# rate being far higher: the encompassing regression identifies b1 and b2
# only from DISAGREEMENT days, so a higher true agreement rate means less
# information for the regression to separate the two labels with at every
# horizon, not just the sparsely-sampled ones -- which is the more
# consequential reading of the degenerate rows, not a sampling curiosity.


def label_agreement(l_obs: np.ndarray, l_filt: np.ndarray) -> dict:
    """Rate and count of naive/filtered agreement over valid (non-NaN in
    both) rows -- the single number that determines how much information
    the encompassing regression has to separate b1 from b2 with at any
    horizon, since near-collinearity is a direct function of this rate."""
    valid = np.isfinite(l_obs) & np.isfinite(l_filt)
    disagree = (l_obs != l_filt) & valid
    n_valid = int(valid.sum())
    n_disagree = int(disagree.sum())
    return {
        "n_valid": n_valid,
        "n_disagree": n_disagree,
        "agreement_rate": 1.0 - n_disagree / n_valid if n_valid else float("nan"),
    }


def disagreement_by_year(l_obs: np.ndarray, l_filt: np.ndarray,
                         dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Disagreement count and rate per calendar year -- whether the ~93.7%
    overall rate is uniform or concentrated in specific (volatile) years."""
    valid = np.isfinite(l_obs) & np.isfinite(l_filt)
    disagree = (l_obs != l_filt) & valid
    df = pd.DataFrame({"year": pd.DatetimeIndex(dates).year[valid],
                       "disagree": disagree[valid]})
    out = df.groupby("year")["disagree"].agg(n_disagree="sum", n_days="count")
    out["rate"] = out["n_disagree"] / out["n_days"]
    return out


def disagreement_clustering(l_obs: np.ndarray, l_filt: np.ndarray,
                            log_f: np.ndarray) -> dict:
    """Do disagreements cluster near sign flips (small |s_t|), as the
    model predicts, or are they scattered through time regardless of the
    spread's own level? Also reports the run-length structure (contiguous
    disagreement streaks), since a sparse periodic sampler's chance of
    hitting a disagreement depends on how clustered they are, not just the
    marginal rate."""
    s = front_spread(log_f)
    valid = np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(s)
    disagree = (l_obs != l_filt) & valid
    agree = valid & ~disagree

    d = disagree.astype(int)
    change = np.diff(np.concatenate([[0], d, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    run_lengths = ends - starts

    return {
        "median_abs_s_disagree": float(np.median(np.abs(s[disagree]))) if disagree.any() else float("nan"),
        "median_abs_s_agree": float(np.median(np.abs(s[agree]))) if agree.any() else float("nan"),
        "mean_abs_s_disagree": float(np.mean(np.abs(s[disagree]))) if disagree.any() else float("nan"),
        "mean_abs_s_agree": float(np.mean(np.abs(s[agree]))) if agree.any() else float("nan"),
        "n_runs": int(len(run_lengths)),
        "run_length_median": float(np.median(run_lengths)) if len(run_lengths) else float("nan"),
        "run_length_max": int(run_lengths.max()) if len(run_lengths) else 0,
        "n_disagree": int(disagree.sum()),
    }


MIN_BOOTSTRAP_BLOCKS = 4


def _block_length_ok(n: int, block_length: int, min_blocks: int = MIN_BOOTSTRAP_BLOCKS) -> bool:
    """Live-bug guard. As block_length -> n, n_blocks -> 1: every circular
    resample becomes a single rotation of the whole series, which has the
    SAME mean as the original every time (a rotation permutes WHICH day is
    first, not the multiset of values) -- the null distribution collapses
    to a point and whatever z/p/fraction gets computed from it comes back
    0 or 1 regardless of the data, silently. Fires exactly at the small-n,
    long-h combinations this study lives at (e.g. h=252 on a panel with
    a few thousand rows already leaves few independent 252-day blocks).
    Every circular-block-bootstrap caller must check this BEFORE its loop
    and return NaN outright if it fails -- a test that cannot respect the
    requested dependence length has not been run, not run badly.
    """
    return n // block_length >= min_blocks


def _circular_block_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    """One circular-block-bootstrap resample of indices 0..n-1: draw block
    start points uniformly, wrap each block around the end of the series
    circularly, concatenate to length n. Preserves each series' own serial
    dependence up to ~block_length, unlike an i.i.d. row permutation.

    Callers must check _block_length_ok(n, block_length) themselves before
    looping -- this function does not, since the check is invariant across
    replicates within one call and belongs outside the loop, not repeated
    inside it every draw.
    """
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(block_length) + s) % n for s in starts])
    return idx[:n]


def circular_block_bootstrap_monotonicity(
    p: TwoFactorParams, x_filt: np.ndarray, log_f: np.ndarray, tau: np.ndarray,
    horizons: tuple[int, ...] = EXTENDED_HORIZONS,
    block_length: int = 252, n_boot: int = 500, seed: int = 0,
) -> dict:
    """Null distribution for "|t(b2)| monotone increasing across all five
    horizons" -- the pre-stated prediction under the signal-accumulation
    story. The two labels (paired, same resample indices, so their own
    cross-dependence is preserved) are circular-block-bootstrapped against
    the REAL, unresampled target series: this breaks any true temporal
    alignment between labels and y while preserving each label series'
    own serial dependence structure (block_length ~252), which is what
    makes a naive i.i.d. permutation test invalid for overlapping-window
    data. Replicates where any horizon's regression is degenerate (labels
    collinear in that particular resample) are skipped, not counted as
    non-monotone or monotone -- reported explicitly.
    """
    rng = np.random.default_rng(seed)
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)
    n = len(l_obs)
    if not _block_length_ok(n, block_length):
        return {"n_boot": n_boot, "n_skipped": n_boot, "n_used": 0,
               "n_monotone": 0, "fraction_monotone": float("nan"),
               "block_length_rejected": True,
               "reason": f"n={n}, block_length={block_length} gives fewer than "
                        f"{MIN_BOOTSTRAP_BLOCKS} blocks -- not run"}

    ys = {h: cross_maturity_spread_change(log_f, h) for h in horizons}
    maxlags = {h: int(np.floor(1.5 * h)) for h in horizons}

    n_monotone = 0
    n_skipped = 0
    for _ in range(n_boot):
        idx = _circular_block_indices(rng, n, block_length)
        l_obs_b, l_filt_b = l_obs[idx], l_filt[idx]

        t_abs = []
        skip = False
        for h in horizons:
            r = encompassing_regression(ys[h], l_obs_b, l_filt_b, h,
                                        maxlags=maxlags[h])
            if r.degenerate or not np.isfinite(r.se_b2) or r.se_b2 == 0:
                skip = True
                break
            t_abs.append(abs(r.b2 / r.se_b2))
        if skip:
            n_skipped += 1
            continue
        if all(t_abs[i] <= t_abs[i + 1] for i in range(len(t_abs) - 1)):
            n_monotone += 1

    n_used = n_boot - n_skipped
    return {
        "n_boot": n_boot, "n_skipped": n_skipped, "n_used": n_used,
        "n_monotone": n_monotone,
        "fraction_monotone": n_monotone / n_used if n_used else float("nan"),
    }


# ==== 9. Testing the filter as more than a classifier (disagreement days,
#         confidence, and the permanent/transitory monopoly claim) ========
#
# sign(f1 - f2) throws away the two things the filter uniquely produces --
# magnitude and calibrated uncertainty -- then asks it to beat the spread
# at the one job the spread does natively. The full-sample encompassing
# regression also dilutes the 225 genuinely informative (disagreement)
# days with 3,372 that contribute nothing to discrimination, since b1 and
# b2 are identified only from where the labels differ (Sec. 8).


def disagreement_event_study(l_obs: np.ndarray, l_filt: np.ndarray,
                             log_f: np.ndarray, h: int) -> dict:
    """On days where L_obs != L_filt, which one correctly signed the
    subsequent h-day cross-maturity spread change? A clean binary event
    study restricted to exactly the days that carry discrimination
    information -- McNemar's test on the paired correct/incorrect outcomes,
    not diluted by the ~94% of days where the labels agree and neither
    side can be credited or blamed differently from the other.
    """
    y = cross_maturity_spread_change(log_f, h)
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


def confidence_split_regression(p: TwoFactorParams, x_filt: np.ndarray,
                                P_filt: np.ndarray, log_f: np.ndarray,
                                tau: np.ndarray, h: int
                                ) -> tuple[RegressionResult, RegressionResult]:
    """Splits the sample at the median filtered chi variance (P_filt[:,0,0])
    and reruns the decisive cross-maturity encompassing regression on each
    half. The naive label cannot enter this comparison at all -- it has no
    notion of its own confidence -- which is exactly the point: if the
    filtered label wins anywhere, the model predicts it wins on the
    high-confidence (low P_filt) half specifically.

    Returns (low_confidence_result, high_confidence_result) -- i.e. (high
    P_filt / high uncertainty, low P_filt / high confidence), in that
    order so the "expected to win" half is returned second.
    """
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)
    y = cross_maturity_spread_change(log_f, h)
    p_chi = P_filt[:, 0, 0]

    valid = np.isfinite(y) & np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(p_chi)
    median_p = np.median(p_chi[valid])
    high_uncertainty = valid & (p_chi >= median_p)
    high_confidence = valid & (p_chi < median_p)

    maxlags = max(h - 1, 0)
    r_low_conf = encompassing_regression(y, l_obs, l_filt, h,
                                         target="cross_maturity_spread_change",
                                         subset="high_uncertainty",
                                         extra_mask=high_uncertainty, maxlags=maxlags)
    r_high_conf = encompassing_regression(y, l_obs, l_filt, h,
                                          target="cross_maturity_spread_change",
                                          subset="high_confidence",
                                          extra_mask=high_confidence, maxlags=maxlags)
    return r_low_conf, r_high_conf


def xi_monopoly_test(x_filt: np.ndarray, log_f: np.ndarray, h: int) -> RegressionResult:
    """Does xi_hat predict long-horizon front-contract LEVEL changes --
    something the spread structurally cannot address at all, since it has
    no permanent/transitory decomposition? A monopoly claim, not a horse
    race: single-regressor OLS (xi_hat only, no naive comparator to
    encompass) with HAC standard errors, maxlags = h - 1.
    """
    xi_hat = x_filt[:, 1]
    y = front_return(log_f, h)
    valid = np.isfinite(y) & np.isfinite(xi_hat)
    X = sm.add_constant(xi_hat, has_constant="add")
    y_masked = np.where(valid, y, np.nan)
    res = sm.OLS(y_masked, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
    alpha, b = res.params
    degenerate = res.model.rank < X.shape[1]
    return RegressionResult(
        target="xi_monopoly_front_return", horizon=h, subset="full",
        n_obs=int(res.nobs), alpha=float(alpha), b1=float("nan"), b2=float(b),
        se_b1=float("nan"), se_b2=float(res.bse[1]),
        p_b1=float("nan"), p_b2=float(res.pvalues[1]),
        r_squared=float(res.rsquared), fit=res, degenerate=degenerate,
    )


def _restricted_null_residuals(y: np.ndarray, l_obs: np.ndarray, l_filt: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Fit y ~ 1 + (l_obs + l_filt) -- the b1=b2 RESTRICTED model -- and
    return (fitted values, residuals), both full-length with NaN where y
    was NaN. Point estimates and residuals from OLS don't depend on the
    covariance estimator, so this doesn't need HAC; only used to generate
    a bootstrap DGP under H0: b1 = b2."""
    combined = l_obs + l_filt
    X = sm.add_constant(combined, has_constant="add")
    res = sm.OLS(y, X, missing="drop").fit()
    fitted_full = np.full_like(y, np.nan)
    resid_full = np.full_like(y, np.nan)
    valid = np.isfinite(y) & np.isfinite(combined)
    fitted_full[valid] = res.fittedvalues
    resid_full[valid] = res.resid
    return fitted_full, resid_full


def circular_block_bootstrap_restricted_null(
    p: TwoFactorParams, x_filt: np.ndarray, log_f: np.ndarray, tau: np.ndarray,
    horizons: tuple[int, ...] = EXTENDED_HORIZONS,
    block_length: int = 252, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """The correctly-specified null for the monotonicity question. The
    original bootstrap (Sec. 7) resampled under H0: "labels carry no
    information" -- but the joint Wald test (Sec. 8) already rejects that
    null hard at h >= 66, so a test against it says nothing about b2
    specifically: once the pair carries real, growing-with-h signal, a
    near-collinear split hands a growing share to WHICHEVER coefficient,
    and a monotone |t| sequence can follow mechanically with no
    discrimination behind it at all.

    The null that actually matches the discrimination question -- and
    that the diff-Wald test (Sec. 8) fails to reject at every horizon,
    p in [0.46, 0.97] -- is H0: b1 = b2. This resamples RESIDUALS from the
    b1=b2-restricted fit (circular block, preserving their own serial
    dependence) and adds them back onto the restricted fitted values, with
    the REAL labels held fixed throughout -- unlike Sec. 7's bootstrap,
    which resampled the labels instead. Each replicate's y is then refit
    with the unrestricted (b1, b2 free) model to see how often a monotone
    |t(b2)| pattern emerges purely from the collinear split of a signal
    that, by construction here, does not distinguish the two labels.
    """
    rng = np.random.default_rng(seed)
    l_obs = naive_label(log_f)
    l_filt = filtered_label(p, x_filt, tau)
    n = len(l_obs)
    if not _block_length_ok(n, block_length):
        return {"n_boot": n_boot, "n_skipped": n_boot, "n_used": 0,
               "n_monotone": 0, "fraction_monotone": float("nan"),
               "block_length_rejected": True,
               "reason": f"n={n}, block_length={block_length} gives fewer than "
                        f"{MIN_BOOTSTRAP_BLOCKS} blocks -- not run"}

    ys = {h: cross_maturity_spread_change(log_f, h) for h in horizons}
    maxlags = {h: int(np.floor(1.5 * h)) for h in horizons}
    fitted_r = {}
    resid_r = {}
    for h in horizons:
        fitted_r[h], resid_r[h] = _restricted_null_residuals(ys[h], l_obs, l_filt)

    n_monotone = 0
    n_skipped = 0
    for _ in range(n_boot):
        idx = _circular_block_indices(rng, n, block_length)

        t_abs = []
        skip = False
        for h in horizons:
            y_boot = fitted_r[h] + resid_r[h][idx]
            r = encompassing_regression(y_boot, l_obs, l_filt, h, maxlags=maxlags[h])
            if r.degenerate or not np.isfinite(r.se_b2) or r.se_b2 == 0:
                skip = True
                break
            t_abs.append(abs(r.b2 / r.se_b2))
        if skip:
            n_skipped += 1
            continue
        if all(t_abs[i] <= t_abs[i + 1] for i in range(len(t_abs) - 1)):
            n_monotone += 1

    n_used = n_boot - n_skipped
    return {
        "n_boot": n_boot, "n_skipped": n_skipped, "n_used": n_used,
        "n_monotone": n_monotone,
        "fraction_monotone": n_monotone / n_used if n_used else float("nan"),
    }


# ===== 10. Three checks before scoping a fix ================================
#
# The s-hat profile ([54.9, 6.5, 5.1, 16.6] bp) reframed the whole batch: not
# a noise-level estimate but front-contract misspecification the state has
# nowhere else to put. Two of the results built on top of that reframing
# can still be overturned by their own weaknesses, so each gets checked
# before anyone writes about it.


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def disagreement_event_study_block_bootstrap(
    l_obs: np.ndarray, l_filt: np.ndarray, log_f: np.ndarray, h: int,
    block_length: int | None = None, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """McNemar's test assumes the 225 disagreement days are independent.
    They arrive in 72 runs (median length 2, max 20, Sec. 8), and at
    large h the forward windows from days inside the same run -- or even
    nearby runs -- overlap almost entirely, so day-level McNemar overstates
    significance, likely badly. This resamples the (l_obs, l_filt, y)
    triple jointly via circular blocks (default block_length =
    max(h, 20), the observed max run length, so a block is long enough to
    keep an entire disagreement run and its h-day forward window together
    rather than splitting it), recomputes the disagreement-restricted
    win-rate difference on each replicate, and reports a bootstrap SE/z/p
    for the ACTUAL observed difference -- a corrected significance
    assessment of the same statistic, not a different null hypothesis.
    """
    if block_length is None:
        block_length = max(h, 20)
    rng = np.random.default_rng(seed)
    y = cross_maturity_spread_change(log_f, h)
    n = len(l_obs)

    valid0 = np.isfinite(l_obs) & np.isfinite(l_filt) & np.isfinite(y)
    disagree0 = (l_obs != l_filt) & valid0
    obs_correct0 = np.sign(l_obs[disagree0]) == np.sign(y[disagree0])
    filt_correct0 = np.sign(l_filt[disagree0]) == np.sign(y[disagree0])
    observed_diff = float(filt_correct0.mean() - obs_correct0.mean()) if disagree0.any() else float("nan")

    if not _block_length_ok(n, block_length):
        return {"h": h, "n_disagree": int(disagree0.sum()), "block_length": block_length,
               "n_boot_used": 0, "observed_diff": observed_diff,
               "block_bootstrap_se": float("nan"), "z": float("nan"), "p": float("nan"),
               "block_length_rejected": True,
               "reason": f"n={n}, block_length={block_length} gives fewer than "
                        f"{MIN_BOOTSTRAP_BLOCKS} blocks -- not run"}

    diffs = []
    for _ in range(n_boot):
        idx = _circular_block_indices(rng, n, block_length)
        lo_b, lf_b, y_b = l_obs[idx], l_filt[idx], y[idx]
        valid = np.isfinite(lo_b) & np.isfinite(lf_b) & np.isfinite(y_b)
        disagree = (lo_b != lf_b) & valid
        if disagree.sum() < 5:
            continue
        obs_c = np.sign(lo_b[disagree]) == np.sign(y_b[disagree])
        filt_c = np.sign(lf_b[disagree]) == np.sign(y_b[disagree])
        diffs.append(filt_c.mean() - obs_c.mean())

    diffs = np.array(diffs)
    se = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    z = observed_diff / se if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(2.0 * (1.0 - _norm_cdf(abs(z)))) if np.isfinite(z) else float("nan")

    return {
        "h": h, "n_disagree": int(disagree0.sum()), "block_length": block_length,
        "n_boot_used": len(diffs),
        "observed_diff": observed_diff, "block_bootstrap_se": se,
        "z": z, "p": p, "block_length_rejected": False,
    }


def xi_encompasses_log_price(x_filt: np.ndarray, log_f: np.ndarray, h: int
                             ) -> RegressionResult:
    """The decisive check on the xi monopoly claim: xi_hat is the long-run
    factor and will itself be close to the log front price, so "xi_hat
    predicts long-horizon level changes" could just be repackaged mean
    reversion in log(price) -- a well-known result the filter adds nothing
    to. Encompassing form, reusing the same machinery and reasoning as the
    naive/filtered spread labels (Sec. 3): y_t^(h) = a + c1*log_f1_t +
    c2*xi_hat_t + e_t. If xi_hat subsumes log_f1 (c2 significant, c1 not),
    the monopoly claim is genuine. If log_f1 does the work instead, the
    result is real but not the filter's -- the same class of error as the
    original Study A flaw, a predictor that is an affine function of the
    thing it predicts.
    """
    y = front_return(log_f, h)
    log_f1 = log_f[:, 0]
    xi_hat = x_filt[:, 1]
    return encompassing_regression(y, log_f1, xi_hat, h,
                                   target="xi_encompasses_log_price", subset="full")


def xi_monopoly_bootstrap_null(x_filt: np.ndarray, log_f: np.ndarray, h: int,
                               block_length: int | None = None,
                               n_boot: int = 2000, seed: int = 0) -> dict:
    """Block-bootstrap null distribution for the xi monopoly test's R^2,
    complementing the HAC p-value with a clustering-aware one. xi_hat is
    circular-block-resampled (block_length defaults to h) against the
    REAL, unresampled target -- breaking the true temporal alignment while
    preserving xi_hat's own serial dependence -- and the single-regressor
    HAC fit is rerun on each replicate. Reports the fraction of null
    replicates whose R^2 meets or exceeds the observed one, which does not
    inherit the small-T/h inflation that raw (non-adjusted) R^2 carries at
    the low end of the horizon range on its own.
    """
    if block_length is None:
        block_length = h
    rng = np.random.default_rng(seed)
    xi_hat = x_filt[:, 1]
    y = front_return(log_f, h)
    n = len(xi_hat)

    observed = xi_monopoly_test(x_filt, log_f, h)
    if not _block_length_ok(n, block_length):
        return {"h": h, "n_boot_used": 0, "observed_r2": observed.r_squared,
               "observed_r2_adj": observed.r_squared_adj,
               "null_r2_mean": float("nan"), "null_r2_p95": float("nan"),
               "fraction_null_r2_exceeds_observed": float("nan"),
               "block_length_rejected": True,
               "reason": f"n={n}, block_length={block_length} gives fewer than "
                        f"{MIN_BOOTSTRAP_BLOCKS} blocks -- not run"}

    r2_null = []
    for _ in range(n_boot):
        idx = _circular_block_indices(rng, n, block_length)
        xi_b = xi_hat[idx]
        valid = np.isfinite(y) & np.isfinite(xi_b)
        if valid.sum() < 10:
            continue
        X = sm.add_constant(xi_b, has_constant="add")
        res = sm.OLS(np.where(valid, y, np.nan), X, missing="drop").fit()
        if res.model.rank < X.shape[1]:
            continue
        r2_null.append(res.rsquared)

    r2_null = np.array(r2_null)
    frac_exceed = float(np.mean(r2_null >= observed.r_squared)) if len(r2_null) else float("nan")
    return {
        "h": h, "n_boot_used": len(r2_null),
        "observed_r2": observed.r_squared, "observed_r2_adj": observed.r_squared_adj,
        "null_r2_mean": float(r2_null.mean()) if len(r2_null) else float("nan"),
        "null_r2_p95": float(np.percentile(r2_null, 95)) if len(r2_null) else float("nan"),
        "fraction_null_r2_exceeds_observed": frac_exceed,
        "block_length_rejected": False,
    }
