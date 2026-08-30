# Latent-Factor-Estimation-Oil-Futures

News-Augmented Latent Factor Estimation in Oil Futures Pricing: A State-Space
Extension of the Gabillon Two-Factor Model

Latent state of the oil futures curve, recovered from two measurement channels:
prices and news.

Prices are not the state of the market — they are a noisy instrument pointed at
it. News is a second instrument pointed at the same thing. A Kalman filter is
the machinery for combining noisy readings of one hidden quantity, so the
question becomes *how much state does each channel recover, and what is each
increment worth.*

Design document: see `docs/` (the published plan) and the accompanying paper.

---

## Status — Phase 3 machinery, day 1

Built, no data required:

- **General-`k` JIT filter** (`state/fast.py`) — univariate sequential updating,
  time-varying `Q_t` and `H_t`, exact treatment of missing observations.
- **State augmentation** (`state/augment.py`) — the four-dimensional
  `z = (χ, ξ, wχ, wξ)` that exposes innovations so the news channel can measure
  a flow rather than a stock.
- **Pre-registered spec lattice** (`pricing/specs.py`) — `S0 / SD / SM / SDM`
  committed before estimation, plus `SD_ASYM`. Degrees of freedom are derived
  from the registry, never hand-maintained.
- **Nesting assertions** (`tests/test_nesting.py`) — 12 tests, the hard gate.
- **Timestamp forensics** (`explore/01_timestamps.py`) — run on day one.
- **Expiry calendar** (`curve/expiry.py`) — CME NYMEX CL contract chain,
  time-to-maturity, roll indicator.

Not built: curve panel assembly, news pipeline, the three studies. Those need
data.

## Day-1 findings

**Curve depth.** No Bloomberg/LSEG/Datastream access via KCL. Proceeding on
the free EIA API (`petroleum/pri/fut`), which gives CL1–CL4 only — thin but
workable with the Bayesian prior on κ from §XII-A. `EIA_API_KEY` goes in a
local `.env` (gitignored, see `.env.example`), never committed or pasted into
chat.

**Timestamp check.** Ran `explore/01_timestamps.py` against a GDELT GKG sample
(one Wednesday window, 2024-01-10, ~30h buffer rather than a full week —
bandwidth-scoped; the check only uses documents timestamped on the release
day anyway). Raw script verdict: **FAIL** (37.5% published >5min before
release, IQR 465min). Forensic follow-up traced this to keyword-filter
imprecision (GKG has no full article text) rather than genuine crawl-time
contamination: most "early" documents report the American Petroleum
Institute's Tuesday-evening estimate — a real, separate, legitimately-earlier
release — not the EIA's Wednesday 10:30 ET report. Documents unambiguously
about the actual EIA release (n=9, thin) show zero early mass, lags 30–465min
post-release. **Needs corroboration across more weeks/releases before SM is
either cleared or ruled out** — one release is not enough to decide. See
`scripts/pull_gdelt_sample.py` and `figures/01_timestamps_sample.png`.

## The nesting assertion earned its place immediately

The lattice is only meaningful if each restricted spec reproduces its parent
exactly. The tests caught a real design error within minutes of being written.

Asymmetric news loadings — separate responses to good and bad news — seemed to
belong naturally in the measurement channel, splitting the feature at zero into
`u⁺` and `u⁻`. It does not nest, and the reason is not obvious:

> Splitting an **observation** at zero produces two rows, one of which reports a
> structural zero on any given day. The filter cannot distinguish that zero from
> a genuine reading of "the shock was zero", and updates the state on it. So
> `G⁺ = G⁻` does not recover the parent — the likelihoods differed by 6221.

Splitting a **driver** is fine, because there `u` is an input and `u⁺ + u⁻ = u`
exactly. Asymmetry therefore lives in `SD_ASYM`, which nests `SD` to `1e-9`.

Measurement-side asymmetry would have to condition on the sign of the *latent*
shock, which is nonlinear in the state and needs a particle filter. Out of scope
by choice, not oversight.

Without the assertion this would have shipped: the LR test still runs, still
returns a p-value, and nothing raises an error.

## Layout

```
src/csr/
  state/
    kalman.py       reference linear-Gaussian filter + smoother
    fast.py         JIT kernel, general k, univariate updating
    augment.py      innovation-exposing augmentation; news block; R_t
  pricing/
    twofactor.py    A(τ), state-space mapping, variance scalings
    specs.py        the pre-registered lattice
  curve/            [phase 1]  expiry calendar, panel assembly
  news/             [phase 2]  ingest, dedup, scoring, features
  study/            [phase 4+] recovery, increment, economics
explore/            # %% scripts — interactive, outputs never committed
scripts/            reusable pull/build scripts (GDELT sample, EIA curve)
tests/
```

`.py` throughout with `# %%` cell markers rather than `.ipynb`: hidden state and
out-of-order execution are the same failure mode as look-ahead bias, which is
the project's top risk. Exploration decides *what* to compute; the package
computes it. If a number reaches the paper, it lives in `src/` with a test.

## Design decisions worth knowing before editing

**`Q4` is singular by construction.** The augmented disturbance is
`(wχ, wξ, wχ, wξ)` — the same draw in both blocks — so `Q4` has rank 2. Harmless
for a Kalman filter, which needs only PSD, but never hand it to code assuming
invertibility.

**`R_t = ς²_t / n_t` is a plug-in, not a posit.** Daily sentiment is a mean over
that day's documents, so this is the sampling variance of a mean — both inputs
observed. `news_variance()` raises on negative counts, because passing the
*standardised* intensity here instead of raw counts is an easy and silent error.

**Realised volatility must be lagged before it reaches `scale_variances`.**
Contemporaneous RV makes the state prior a function of the data being explained,
and the prediction-error decomposition stops being the likelihood.

**News lag is enforced in code, not convention.** `append_news_block(lag=...)`
places news at the period whose shock it describes. The estimation grid may
include zero or negative lags purely as a timestamp diagnostic; anything used
for inference is `lag >= 1`.

**API keys never live in the repo.** `EIA_API_KEY` (and any future key) goes
in a local `.env`, which is gitignored. `.env.example` documents the expected
variable names with empty values.

## Usage

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[fast,dev,data]"
pytest tests/ -q                     # expect 32 passed
python explore/01_timestamps.py      # run from inside explore/ (relative data path)
```

## Next

1. `EIA_API_KEY` registered — wire up `scripts/pull_eia_curve.py` for CL1–CL4,
   daily, back to 2010.
2. `curve/build.py` — panel assembly, outlier screens, the Panel contract
   every study consumes.
3. Re-run the timestamp check across more weeks once the news pipeline can
   pull at scale, to settle SM viability with more than one release's worth
   of evidence.
4. `specs.py` is already committed. Do not edit the lattice after seeing
   results.
