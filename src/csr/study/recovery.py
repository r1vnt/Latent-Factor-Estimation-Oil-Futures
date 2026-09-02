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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

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

    @property
    def b1_significant(self) -> bool:
        return self.p_b1 < 0.05

    @property
    def b2_significant(self) -> bool:
        return self.p_b2 < 0.05

    def pattern(self) -> str:
        """Which of the four sig/insig combinations this regression landed
        in -- the building block classify_gate aggregates across horizons."""
        if self.b2_significant and not self.b1_significant:
            return "b2_only"
        if self.b2_significant and self.b1_significant:
            return "both"
        if self.b1_significant and not self.b2_significant:
            return "b1_only"
        return "neither"

    def __str__(self) -> str:
        return (f"[{self.target} h={self.horizon} {self.subset}] n={self.n_obs}  "
               f"b1(naive)={self.b1:+.4f} (p={self.p_b1:.4f})  "
               f"b2(filt)={self.b2:+.4f} (p={self.p_b2:.4f})  "
               f"R2={self.r_squared:.4f}  [{self.pattern()}]")


def encompassing_regression(y: np.ndarray, l_obs: np.ndarray, l_filt: np.ndarray,
                            h: int, target: str = "", subset: str = "full",
                            extra_mask: np.ndarray | None = None) -> RegressionResult:
    """OLS of y on [1, L_obs, L_filt] with HAC (Newey-West) standard errors,
    maxlags = h - 1 (the minimum implied by h-day overlap). Rows with any
    NaN among y/l_obs/l_filt, or excluded by `extra_mask` (e.g. April 2020,
    or outside a volatility tercile), are dropped listwise -- statsmodels'
    missing='drop' handles this uniformly rather than needing separate
    masking logic per caller.
    """
    y = np.asarray(y, float).copy()
    if extra_mask is not None:
        y = np.where(extra_mask, y, np.nan)
    X = sm.add_constant(np.column_stack([l_obs, l_filt]), has_constant="add")
    model = sm.OLS(y, X, missing="drop")
    maxlags = max(h - 1, 0)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha, b1, b2 = res.params
    se_b1, se_b2 = res.bse[1], res.bse[2]
    p_b1, p_b2 = res.pvalues[1], res.pvalues[2]
    return RegressionResult(
        target=target, horizon=h, subset=subset, n_obs=int(res.nobs),
        alpha=float(alpha), b1=float(b1), b2=float(b2),
        se_b1=float(se_b1), se_b2=float(se_b2),
        p_b1=float(p_b1), p_b2=float(p_b2),
        r_squared=float(res.rsquared), fit=res,
    )


# =================================================== 4. Passing the gate ===

GateVerdict = str   # "pass" | "partial" | "fail" | "inconclusive"


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
    """
    patterns = {h: r.pattern() for h, r in results_by_horizon.items()}

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
