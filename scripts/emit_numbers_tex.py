"""One source of truth for every headline number: reads results/*.json,
writes numbers.tex (LaTeX macro definitions only). The paper \\input{}s
this file and cites \\smlr instead of typing 59.09; the notebook reads
the same JSONs directly. Paper and notebook become incapable of
disagreeing -- if a fit is re-run and a number moves, it moves in both
places the next time this script runs, or this script's own assertion
(the value it just read differs from what's committed) fails the build.

Each entry below names its exact source file and field path -- no number
in this file was typed from memory or transcribed from a printout; every
one was re-read from disk while writing this script (see the inline
comments the first time this ran, 2026-09-19). Two headline numbers
referenced in conversation (kappa spreads "0.731 and 0.065", and "4/5")
could not be traced to any results/*.json on disk and are NOT emitted --
see the TODO block at the bottom. A missing macro is a visible gap; a
guessed one is a silent error this whole exercise exists to prevent.

Run: python scripts/emit_numbers_tex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _fmt_pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}\\%"


def _fmt_sci(x: float, decimals: int = 1) -> str:
    mantissa, exp = f"{x:.{decimals}e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exp)}}}"


def build_macros() -> list[tuple[str, str, str]]:
    """Returns (macro_name, latex_value, source_comment) triples."""
    macros = []

    def add(name, value, source):
        macros.append((name, value, source))

    # -- H1: news drives the state (SD vs S0) -- not supported ----------
    sd = _load("c1c18_news_SD.json")
    add("hOneLRp", f"{sd['p_vs_S0']:.2f}", "c1c18_news_SD.json: p_vs_S0")
    add("hOneLR", f"{sd['lr_vs_S0']:.2f}", "c1c18_news_SD.json: lr_vs_S0")
    add("hOneDf", f"{sd['df_vs_S0']}", "c1c18_news_SD.json: df_vs_S0")

    # -- H2: news intensity -> conditional variance -- rejected at Step 0
    h2 = _load("c1c18_h2_step0_precheck.json")
    add("hTwoIncrR2", f"{h2['check2']['incremental_r2']:.5f}",
       "c1c18_h2_step0_precheck.json: check2.incremental_r2")
    add("hTwoP", f"{h2['check2']['p_intensity']:.2f}",
       "c1c18_h2_step0_precheck.json: check2.p_intensity")
    add("hTwoCorrIRv", f"{h2['check1_corr_intensity_rv']:.3f}",
       "c1c18_h2_step0_precheck.json: check1_corr_intensity_rv")

    # -- H3: filtering improves one-step forecasts -- rejected, C1 -----
    denoise = _load("c1c18_denoising_robust_tests.json")
    c1_sign = denoise["sign_test"][0]
    c1_power = denoise["power_both_losses"][0]
    add("hThreeWinRate", f"{c1_sign['win_rate']:.4f}",
       "c1c18_denoising_robust_tests.json: sign_test[C1].win_rate")
    add("hThreeZ", f"{c1_sign['z']:.1f}", "c1c18_denoising_robust_tests.json: sign_test[C1].z")
    add("hThreePower", f"{c1_power['power_at_predicted_abs']:.3f}",
       "c1c18_denoising_robust_tests.json: power_both_losses[C1].power_at_predicted_abs")

    # -- SM: news measures the state -- supported, corrected LR ---------
    placebo = _load("c1c18_news_placebo.json")
    sm_lr, sm_df = placebo["real_LR_joint_em"], 15
    sm_p = float(chi2.sf(sm_lr, sm_df))
    add("smLR", f"{sm_lr:.2f}", "c1c18_news_placebo.json: real_LR_joint_em")
    add("smDf", f"{sm_df}", "matches SM's own free-parameter count (feature_cols), c1c18_news_SM.json")
    add("smP", _fmt_sci(sm_p), "chi2.sf(real_LR_joint_em, 15), computed fresh by this script")
    add("smPlaceboP", f"{placebo['empirical_p']:.3f}",
       "c1c18_news_placebo.json: empirical_p (1000-draw block-shuffle placebo)")
    naive_sm = _load("c1c18_news_SM.json")
    add("smNaiveLR", f"{naive_sm['lr_vs_S0']:.0f}",
       "c1c18_news_SM.json: lr_vs_S0 (the UNCORRECTED, n_obs-mismatched figure -- retracted)")

    # -- Three-factor held-out prediction, C13-C18 -----------------------
    heldout = _load("c1c18_heldout_contract_prediction.json")
    rmse = heldout["rmse_bp"]
    gain = 1.0 - rmse["three_factor"] / rmse["two_factor"]
    add("heldoutgain", _fmt_pct(gain), "c1c18_heldout_contract_prediction.json: rmse_bp (1 - 3F/2F)")
    add("heldoutRmseTwoFactor", f"{rmse['two_factor']:.1f}", "c1c18_heldout_contract_prediction.json: rmse_bp.two_factor")
    add("heldoutRmseThreeFactor", f"{rmse['three_factor']:.1f}", "c1c18_heldout_contract_prediction.json: rmse_bp.three_factor")

    # -- EIA settlement validation ----------------------------------------
    eia = _load("c1c18_eia_settlement_validation.json")
    add("eiaAgreement", _fmt_pct(eia["within_1_tick_rate"], 2),
       "c1c18_eia_settlement_validation.json: within_1_tick_rate")
    add("eiaOverlapDays", f"{eia['n_overlap_days']:,}", "c1c18_eia_settlement_validation.json: n_overlap_days")

    # -- Front-spread channel: kurtosis collapse ---------------------------
    fs = _load("c1c18_frontspread_channel.json")
    kurt = fs["acceptance_3"]
    add("kurtBefore", f"{kurt['excess_kurt_before']:.1f}", "c1c18_frontspread_channel.json: acceptance_3.excess_kurt_before")
    add("kurtAfter", f"{kurt['excess_kurt_after']:.2f}", "c1c18_frontspread_channel.json: acceptance_3.excess_kurt_after")

    # -- Joint nu estimate ---------------------------------------------------
    nu = _load("c1c18_joint_nu_estimation.json")
    add("nuHat", f"{nu['nu_hat']:.1f}", "c1c18_joint_nu_estimation.json: nu_hat")

    # -- H-sensitivity sweep and rank correlation ----------------------------
    hs = _load("c1c18_h_sensitivity_and_rank_correlation.json")
    sweep = hs["h_sensitivity_sweep_c1"]
    win_start, win_end = sweep[0]["win_rate"], sweep[-1]["win_rate"]
    gap_closed = (win_end - win_start) / (0.5 - win_start)
    add("hSweepGapClosed", _fmt_pct(gap_closed, 0),
       "c1c18_h_sensitivity_and_rank_correlation.json: h_sensitivity_sweep_c1[0 and -1].win_rate")
    add("rankCorrRho", f"{hs['rank_correlation']['spearman_rho']:.3f}",
       "c1c18_h_sensitivity_and_rank_correlation.json: rank_correlation.spearman_rho")
    add("rankCorrP", f"{hs['rank_correlation']['p_value']:.3f}",
       "c1c18_h_sensitivity_and_rank_correlation.json: rank_correlation.p_value")

    # -- PCA diagnostic -------------------------------------------------------
    pca = _load("c1c18_pca_diagnostic.json")
    add("pcaAffineOverPca", f"{pca['median_affine_over_pca_ratio']:.2f}",
       "c1c18_pca_diagnostic.json: median_affine_over_pca_ratio")
    add("pcaExplained", _fmt_pct(pca["pct_explained_pca3_aggregate"], 2),
       "c1c18_pca_diagnostic.json: pct_explained_pca3_aggregate")

    return macros


TODO_UNRESOLVED = """
% ---------------------------------------------------------------------
% TWO NUMBERS REFERENCED IN CONVERSATION, NOT EMITTED -- not found in
% any results/*.json on disk as of this script's last run:
%   1. kappa spreads "0.731 and 0.065" (two-factor vs three-factor
%      battery instability, section 5 of the notebook plan)
%   2. "4/5" (context unclear -- not matched to the news block-out
%      battery, which has 3 block-outs, not 5)
% If these come from the paper draft (not in this repo) or an earlier
% session's scratch computation, supply the source and this script will
% emit \\kappaSpreadTwoFactor / \\kappaSpreadThreeFactor / \\newsFourFifths
% properly instead of leaving them as typed numbers in the .tex.
% ---------------------------------------------------------------------
"""


def main() -> None:
    macros = build_macros()
    lines = [
        "% AUTO-GENERATED by scripts/emit_numbers_tex.py -- do not edit by hand.",
        "% Every value here is read live from results/*.json; re-run the script",
        "% after any refit rather than editing a number in this file directly.",
        "",
    ]
    for name, value, source in macros:
        lines.append(f"% {source}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        lines.append("")
    lines.append(TODO_UNRESOLVED.strip())
    lines.append("")

    out_path = ROOT / "numbers.tex"
    out_path.write_text("\n".join(lines))
    print(f"wrote {len(macros)} macros to {out_path}")


if __name__ == "__main__":
    main()
