# News-Augmented Oil Futures Modelling

### Latent factor estimation with financial news and state-space models

**Bradley Wright** · WTI futures · Commodity pricing · Kalman filtering · NLP

[Read the paper](article5.pdf) · [LaTeX source](article5.tex) · [Results](#results) · [Build instructions](#build-the-paper)

## Overview

Can financial news reveal information about the oil futures curve that prices alone do not capture?

This project explores that question through a state-space extension of the Gabillon two-factor framework. It combines structural commodity pricing, latent-factor estimation and text-derived news features, distinguishing two possible roles for news:

- **Driver:** lagged news predicts subsequent factor transitions.
- **Measurement:** news provides an additional observation of factor innovations.

The reported results favour the measurement interpretation within the tested specification. They also show why improved model fit and same-day curve reconstruction should not be treated as evidence of better temporal forecasting.

This is a research manuscript release. Analysis code, datasets and figure-generation scripts are not included in the materials documented here.

## Model

The two-factor foundation separates temporary price deviations from a persistent equilibrium level:

$$
\log S_t = \chi_t + \xi_t.
$$

The corresponding futures pricing relation is

$$
\log F(t,T)=e^{-\kappa(T-t)}\chi_t+\xi_t+A(T-t).
$$

The empirical news comparison extends the price model to **three factors**: fast and slow mean-reverting components plus the equilibrium-level factor. The pricing relation supplies the observation equation, while factor dynamics supply the transition equation.

| Specification | News channel |
| --- | --- |
| **S0** | No news contribution to state inference. |
| **SD** | Lagged features enter transient-factor transitions. |
| **SM** | Features measure exposed transient-factor innovations. |
| **SDM** | Both channels are included. |

Likelihood comparisons retain compatible news observations under the restricted baseline. Both fitted news channels concern the transient factors; effects on permanent equilibrium-level innovations are outside these tests.

## Data

- **Market:** NYMEX WTI crude oil futures, eighteen nearby contract positions.
- **Price sample:** 2010–2022, ending in November 2022.
- **Joint news–price sample:** approximately 1,950 trading observations from 2015 onwards.
- **Text features:** sentiment, article intensity, sentiment dispersion and topics covering supply disruption, OPEC policy, geopolitics and demand.
- **Panel construction:** individual expiring contracts, exchange-calendar expiry rules and time-varying maturities. Missing price observations are handled within the filter.

The underlying datasets are not distributed with this manuscript release. Replication requires the original data sources, scoring model, feature-construction settings and publication-to-settlement alignment.

## Results

The following figures are reported in the manuscript.

| Test | Result |
| --- | --- |
| News measurement channel | LR = **59.09**, 15 degrees of freedom; nominal **p = 3.6 × 10⁻⁷**. |
| Standalone driver channel | LR = **2.42**, 10 degrees of freedom; **p = 0.99**. |
| Block-shuffled measurement placebo | Reported permutation **p = 0.001**. |
| Three-factor versus two-factor pricing | **44.6% lower overall RMSE** on excluded deferred maturities. |
| Same-day pricing versus Nelson–Siegel | Lower RMSE at **14 of 18 maturities**, but larger errors at the first four. |
| Front-contract forecast versus random walk | The filter wins on **37.5% of days**. |

**Interpretation:** the news measurement channel exhibits an alignment-specific association with transient innovations. This does not establish causality or a tradable signal. Cross-sectional pricing gains are concentrated away from the front; pooled same-day RMSE favours Nelson–Siegel, and the temporal comparisons do not establish forecast superiority.

The leave-one-out state-update exercise retains fitted parameters. It is distinct from a test that also re-estimates all parameters without the excluded contract. News-intensity and content–persistence screens provide no support at the reported measures, horizons and thresholds.

## Validation

The study evaluates the estimator and its empirical interpretation through:

- Monte Carlo parameter recovery, interval coverage and driver-test size/power.
- Multi-start estimation and maturity-panel perturbations.
- Common-observation likelihood comparisons and numerical nesting checks.
- Update-order diagnostics separating same-day residuals from temporal forecasts.
- Excluded-maturity pricing and random-walk forecasting benchmarks.
- Block-shuffled news controls and residual-dependence diagnostics.

A smooth maturity-dependent observation-variance curve restricts unstable near-exact pricing configurations. Variance extensions improve tail calibration but leave serial dependence in squared innovations unresolved.

## Limitations and project status

- **Combined-channel optimisation remains incomplete.** The achieved SDM likelihood is below its nested SM fit, preventing a definitive conditional driver comparison.
- **The measurement result is exploratory** relative to the original Part I hypotheses.
- **Timing and endogeneity matter.** News may describe price movements; shuffling does not distinguish that from independent information about fundamentals.
- **Front-end fit and variance calibration remain problematic.** Fast-factor estimates are sensitive to maturity coverage, and squared innovations retain serial dependence.
- **Economic value has not been evaluated.** The trading study was not executed, and the proposed H4/H5 performance tests have no completed results reported in the manuscript.

Future work should complete optimisation, document the data and NLP pipeline, validate recursive variance dynamics and assess front-end alternatives on genuinely held-out data before evaluating trading performance.

## Build the paper

Use a LaTeX distribution containing **IEEEtran**, **pdfLaTeX** and the packages declared in `article5.tex`. Place the manuscript at the repository root and preserve its `figs/` paths.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error article5.tex
```

Without `latexmk`, run pdfLaTeX until cross-references settle:

```bash
pdflatex -interaction=nonstopmode -halt-on-error article5.tex
pdflatex -interaction=nonstopmode -halt-on-error article5.tex
pdflatex -interaction=nonstopmode -halt-on-error article5.tex
```

The bibliography is embedded in the manuscript; no separate BibTeX pass is required. Building the PDF does not reproduce the empirical analysis.

<details>
<summary><strong>Required figure assets</strong></summary>

Four images are included directly and must be present for a normal build:

- `figs/Figure_1.png`
- `figs/figure_2.png`
- `figs/Figure_3.png`
- `figs/Figure_4.png`

Other figure references use `\figplaceholder`, which displays a labelled box when an image is missing. All original assets are needed to reproduce the complete illustrated paper. Preserve filename case on case-sensitive systems.

</details>

## Attribution

**Paper:** *News-Augmented Latent Factor Estimation in Oil Futures Pricing: A State-Space Extension of the Gabillon Two-Factor Model*  
**Author:** Bradley Wright

No reuse licence is specified in the accompanying materials. Manuscript and underlying data permissions must be established separately.
