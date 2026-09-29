# Frozen operational-era categorical comparison

Date frozen: 2026-08-23  
Scientific status: conditional protocol; it becomes evidence only after a
successful source-locked smoke run and a successful source-identical full run.

## Question

On the already frozen 2022--2024 operational-era retrospective cases, how does
the validation-selected 42,434-parameter member adapter compare with raw FuXi
and the already fitted PBC V2 components when all methods use the same cases,
thresholds, verifying observations, and dynamic spatial weights?

This is a post-hoc transfer audit. It is not model selection, an operational
deployment claim, or an untouched final test.

## Immutable inputs

1. PBC V2 full manifest SHA-256:
   `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6`.
2. PBC V2 Slurm receipt SHA-256:
   `1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2`.
3. Frozen PBC fit SHA-256:
   `0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36`.
4. Accepted operational-era V3 full manifest at
   `resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/manifest.json`,
   SHA-256
   `7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad`.
   Its Slurm receipt is fixed at the same root as `slurm_gate_receipt.json`,
   SHA-256
   `65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd`.
   Neither path nor hash is an operator-supplied argument.
5. The operational full must contain all 296 eligible cases: 104 in 2022, 104
   in 2023, and 88 in 2024, with 50 members and seeds 42/43/44.

Every consumed artifact is checked against its parent manifest. Live reused
source must match the source snapshots in the completed PBC and operational
runs. The new driver, launcher, protocol, contract test, and reused cores are
copied into each output and content-hashed.

## Data firewall

- Forecast/member stores: literal `2022.zarr`, `2023.zarr`, and `2024.zarr`.
- Training observations used only to rebuild the already frozen operational
  context: literal 2002--2017 IMD stores.
- The parent-bound neural member-cache NPY covers 2,080 initializations from
  2002--2021. The whole 1.86 GB NPY is read-only integrity-scanned by the
  accepted capacity gate and canonical verified loader. Its initialization and
  grid metadata reconstruct the frozen split/context; its hindcast member
  values are not forecasts scored in this operational-era comparison. The NPY,
  metadata JSON, manifest JSON, and checksum sidecar paths and hashes are
  directly receipted.
- Verification observations: literal 2022--2024 IMD stores.
- Persistence lag exception: only the exact late-2021 dates needed for the
  earliest 2022 issue's completed `issue-14` through `issue-1` windows are
  selected from literal `2021.zarr`. No other 2018--2021 observation values
  are loaded.
- No forecast or observation store for 2025 may be constructed, discovered,
  globbed, opened, or scored. Every retained 42-day verification window ends
  by 2024-12-31.

## Methods and frozen transformations

The comparison has exactly five headline methods:

1. raw 50-member FuXi categorical forecast;
2. locked `base_42k` neural location-and-spread adapter, reported as the
   arithmetic mean of independently computed seed-42/43/44 scores;
3. frozen projected Debias++;
4. frozen projected Persistence++;
5. the frozen equal-weight PBC component combination.

No fit, fine-tuning, span selection, architecture selection, or blend-weight
selection is permitted. Neural adjustment fields are never averaged. PBC
parameters are deserialized from `pbc_fit.npz`:

- calendar thresholds and strict training empirical CDFs have shapes
  `[366,4,27,27]` and `[366,19,27,27]`;
- Debias++ corrections have shape `[366,6,K,27,27]` for the frozen candidate
  spans 14, 28, and 35 days; the persisted selected span is 14 for every lead;
- Persistence++ coefficients have shape `[6,K,27,27,5]`, ridge 0.001, and
  feature order intercept, training empirical climatology CDF, one-week lag
  indicator, two-week lag indicator, raw FuXi CDF;
- training fit indices are exactly 0--1651 and usable Persistence++ indices are
  exactly 4--1651 (1,648 cases);
- the component combination is the persisted 0.5/0.5 rule, not a new blend.

For each operational case, thresholds and climatological CDFs are selected by
the verification-week midpoint. Forecast CDFs use strict `P(Y < q)`. Exact
threshold ties receive bit-identical probabilities; zero thresholds are forced
to CDF zero and excluded from informative-positive-cut scoring.

## Persistence lag contract

At every issue, lag 1 is the completed seven-day observation window
`issue-7` through `issue-1`; lag 2 is `issue-14` through `issue-8`. All fourteen
daily dates must exist. There is no fallback or climatological imputation.

Each lag cell is an observation-fraction-weighted seven-day mean, not an
unweighted mean. Daily fractions are included in the lag content receipt. The
known 2023-10-12 IMD coverage anomaly must appear in exactly these full-run lag
blocks:

- 2023-10-16 lag 1, day 4;
- 2023-10-19 lag 1, day 1;
- 2023-10-23 lag 2, day 4;
- 2023-10-26 lag 2, day 1.

The fixed 12-case smoke subset does not place 2023-10-12 in a lag window and
must therefore report no affected lag block. It still consumes the already
audited operational full as its immutable parent.

## Common scoring support

Truth and weights are rebuilt through the operational-era audit V3 code and
must reproduce its exact input-content receipts and continuous case scores.
The frozen 171-cell India support must match the PBC support exactly.

For each case, lead, and grid cell, the score weight is

`India cell area x seven-day mean IMD observation fraction`.

Weekly truth and persistence lag means both use the corresponding daily
observation fractions. Every method shares the same finite-threshold mask and
dynamic weight denominator. The 2023-10-12 anomaly is therefore neither
silently dropped nor assigned the frozen full-week weight.

## Metrics

Primary categorical diagnostics are reported separately for K=4 quintiles and
K=19 semideciles:

- normalized squared CDF error over distinct, strictly positive persisted
  thresholds (equality-aware RPS in [0,1]);
- RPSS against the persisted training empirical strict-CDF climatology;
- a projected nominal-climatology representation sensitivity.

Secondary diagnostics are:

- the four-slot summed q20/q40/q60/q80 formula on cells with persisted q80 > 0,
  with its literal nominal reference;
- q95 > 0 upper-tail exceedance Brier score and Brier skill;
- signed CDF probability bias and mean absolute case probability bias at every
  nominal quintile and semidecile cut.

The q80 and q95 masks are threshold-only and common to every method. A method
is not allowed to change its own score denominator.

## Neural reconstruction identity

For every seed, the evaluator loads the operational full's receipted
`audit_adjustments.npz`, checks initialization, seed, checkpoint, member-count,
and log-spread/spread identities, reconstructs the corrected 50-member
ensemble, and recomputes continuous CRPS using the operational dynamic weights.
The result must match every corresponding operational `seed_case_metrics.csv`
row. Raw CRPS must likewise match `case_metrics.csv`. Forecast arrays are then
discarded; only per-seed scores are retained. Headline neural values are the
arithmetic mean of scores, never parameters, adjustments, probabilities, or
members.

## Paired uncertainty

The full run uses 10,000 deterministic paired resamples. Stage one samples the
three years with replacement. Stage two samples circular blocks of 13
initializations inside each sampled year. The initialization keeps all six
leads, all members, truth, weights, and every compared method paired.

Intervals are produced for K=4 RPS, K=19 RPS, q80 formula RPS, and q95 Brier,
for W1--W6 and each lead separately. Predeclared comparisons are every corrected
method versus raw, neural versus each PBC method, and combined PBC versus its
two components. Positive effect means lower score for the named method.

Only three year clusters are available, so intervals are exploratory and the
small-cluster limitation must accompany every inference.

## Smoke/full publication gate

- Smoke: fixed four cases per year, 200 bootstrap draws, non-scientific. It
  proves artifact deserialization, exact lag construction, neural CRPS
  identity, scoring geometry, output semantics, and receipts.
- Full: all 296 cases and exactly 10,000 draws. The launcher requires a passed
  smoke manifest/receipt generated from byte-identical source and identical
  PBC/operational parent hashes.
- Both modes publish atomically from an `.incomplete-*` staging directory.
- The driver first publishes only to a hidden launcher-owned gate-staging
  directory. The requested output path remains absent until the same-source
  score-table semantic reconstruction and Slurm gate receipt both pass; one
  same-filesystem atomic rename then publishes the complete directory. Failed
  gate staging is retained for diagnosis.
- The score-table reconstruction rebuilds seed-score averages, aggregates,
  bootstrap intervals, and lag/support receipts from the emitted case-score
  tables. It is not an independent per-cell CDF/RPS/Brier implementation.
- Full postflight requires its complete source-snapshot map to equal the
  accepted smoke map exactly. In both modes, every source digest must agree in
  the source map, artifact inventory, copied snapshot bytes, and live file.

## Claim boundary

If the full gate passes, the result may support a claim about frozen transfer
to a later operational-era retrospective over the India box. It does not prove
global skill, real-time deployment, independence from all prior analysis,
event-specific drought/flood performance, or performance in 2025.
