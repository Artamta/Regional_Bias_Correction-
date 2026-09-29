# Frozen categorical comparison protocol — 2026-08-22

## Scientific status

This is a post-hoc, retrospective comparison on reused 2020–2021 development
data. It is not an untouched test, a 2025 evaluation, or a reproduction of the
rolling protocol in Guan et al. No model is retrained or selected here.

## Immutable inputs

- PBC V2 full: `resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json`
  with SHA-256 `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6`
  and a passed `pbc_postrun_v2` Slurm receipt.
- Frozen neural full: `resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z/manifest.json`
  with SHA-256 `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91`.
- The two inputs must identify the same 51-member FuXi cache, IMD truth,
  2020–2021 initialization dates, six leads, grid, and scoring weights.
- The evaluator must reproduce the PBC observation-bundle content hash and
  stream-verify the member-cache hash before scoring.

## Fair comparison set

All methods are evaluated on exactly 208 initializations and W1–W6 using the
persisted PBC training-only thresholds and identical threshold-derived masks:

1. raw FuXi categorical ensemble;
2. frozen train-only moment calibration;
3. frozen neural `summary_only`;
4. frozen neural `location_spread`;
5. Debias++;
6. Persistence++;
7. combined PBC.

The neural adjustment artifacts for seeds 42, 43, and 44 are reconstructed and
scored independently. Only the two proper-score values and descriptive
probability-bias summaries are averaged across seeds for each identical
case/lead. Forecasts, probabilities, adjustments, and parameters are never
averaged or written to the comparison output. Continuous CRPS is reproduced
against the frozen neural run before each reconstructed ensemble is accepted.

## Probability and score geometry

The strict convention is `F(q) = P(Y < q)`. Identical persisted physical
thresholds must receive exactly identical CDF values. With nonnegative rainfall,
`F(0)=0`; zero cuts are deterministic and excluded from the primary score.

The predeclared headline is
`normalized_informative_positive_cut_v2`: mean squared CDF error over distinct,
strictly positive persisted quintile cuts, with all-zero rows excluded and the
spatial denominator recomputed from the same threshold-only mask for every
method and reference. Skill is relative to the training empirical strict-CDF
climatology. The projected, zero-anchored nominal CDF is a representation
sensitivity only.

The secondary Guan-formula sensitivity sums squared CDF errors over all four
nominal quintile slots on common persisted-`q80 > 0` support. Its denominator is
the literal `[0.2, 0.4, 0.6, 0.8]` formula reference. Aggregates and bootstrap
effects pool score numerators and denominators with retained q80 scoring weight.
This is formula-aligned sensitivity evidence under equality-aware forecasts,
not a rolling-paper reproduction.

## Extremes and probability bias

The positive-q95 upper-tail event `Y >= q95` receives descriptive Brier score
and Brier skill on common `q95 > 0` support. It is secondary evidence, not the
primary promotion endpoint. Separate quintile and semidecile probability-bias
curves are generic CDF-bias diagnostics on their ordinary supported geometry;
they are not positive-q95 event-probability bias. The lower 5% tail is not
scored or promoted because zero-tied q05 thresholds are partially degenerate
and heterogeneous.

## Paired uncertainty

For both the primary and secondary score contracts, comparisons include each
neural/moment method versus raw, Debias++, Persistence++, and combined PBC, plus
combined PBC versus raw and its components. The bootstrap resamples the two test
years first and then circular 13-initialization blocks within each sampled year,
using 2,000 paired replicates and fixed seed `20260822`. Pooled primary W1–W6
inference first averages the six lead scores within each initialization. Pooled
secondary inference resamples retained-weight numerators and denominators. Only
two year clusters are available, so these intervals are exploratory and weak for
interannual uncertainty; they are not independent-test confirmation.

## Integrity and publication language

- The completed PBC V2 full receipt, immutable neural manifest, every consumed
  artifact, selected reconstruction helpers, scoring core, cache, and IMD
  observation bundle are content-bound.
- Output publication is atomic to a fresh directory and contains scores and
  receipts only—no forecast or learned-parameter arrays.
- Slurm postflight independently recomputes seed-score means, pooled/weekwise
  primary and retained-weight paper scores, upper-tail and probability-bias
  neural aggregation, and every bootstrap point effect before writing a gate
  receipt.
- Dates and observation stores are checked to exclude 2025. Any 2025 result
  requires a separate, explicitly authorized protocol.
- Claims must say “post-hoc 2020–2021 reused-development evidence.” Confidence
  intervals quantify paired resampling uncertainty; they do not turn this into
  independent test evidence.
