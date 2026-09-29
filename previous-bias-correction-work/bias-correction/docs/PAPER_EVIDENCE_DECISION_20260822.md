# Paper evidence decision: compact FuXi rainfall post-processing

Updated: 22 August 2026

Status: venue-neutral scientific decision record. This is not submission
prose. The 2025 IMD target remains unopened.

## Decision

Use the frozen **raw-identity compact adapter** as the main neural method. Use
raw FuXi, the training-only log-bias baseline, and the legacy anchored adapter
as fixed controls. Present raw-mean preservation as a target-sensitivity
ablation, not as the selected method.

The current evidence supports this statement:

> A three-seed ensemble of compact deterministic residual adapters, each with
> 144,689 parameters, improves weekly FuXi rainfall errors and spatial anomaly
> skill over India without requiring a
> fitted log-bias reconstruction anchor. The improvement persists under both
> gridded IMD and external gauge-derived 1.5-degree-cell verification. A
> constraint that preserves FuXi's regional rainfall amount repairs IMD mean
> bias but fails to transfer
> to the gauge target, showing that calibration choices are observation-target
> dependent.

The matched global-pretraining experiment completed and rejected the
pretraining component. Global initialization was 0.493% worse in pooled RMSE
than scratch, ACC was lower, and zero of three pretrained seeds achieved a
lower best India validation composite loss than matched scratch. This negative
result supports keeping the raw-identity India adapter as the core story.

## Canonical evidence

### Fixed 2022--2024 IMD audit

Artifact:
`../resultsv2/fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z`

The audit contains 100 frozen JJAS initializations and all six lead weeks. No
model was retrained or retuned. Intervals use 10,000 paired,
year-stratified, circular moving-block resamples of 13 initializations.

| Method | RMSE | MAE | Signed bias | ACC |
|---|---:|---:|---:|---:|
| Raw FuXi | 5.723 | 3.822 | -0.225 | 0.276 |
| Training-only log-bias | 5.365 | 3.543 | -0.824 | 0.337 |
| Legacy anchored adapter | 5.275 | 3.478 | -0.842 | 0.358 |
| Raw-identity adapter | 5.239 | **3.453** | -0.858 | **0.364** |
| Raw-mean-preserving raw identity | **5.227** | 3.538 | **-0.216** | 0.360 |

Raw identity versus raw FuXi:

- RMSE reduction: 0.484 mm/day, 95% interval 0.383 to 0.587;
  8.45% relative skill.
- MAE reduction: 0.369 mm/day, interval 0.305 to 0.432;
  9.65% relative skill.
- ACC increase: 0.0889, interval 0.0622 to 0.1160.
- Negative result: absolute pooled bias worsens by about 0.633 mm/day.

Raw identity versus the legacy anchored adapter:

- RMSE reduction: 0.0360 mm/day, interval 0.0136 to 0.0593.
- MAE reduction: 0.0244 mm/day, interval 0.0076 to 0.0413.
- ACC point increase: 0.0065, but its interval crosses zero.
- Absolute-bias difference is unresolved and its interval crosses zero.

The mean-preserving diagnostic restores IMD bias and keeps 8.65% RMSE skill
versus raw FuXi. Relative to raw identity, however, its RMSE advantage is
uncertain, MAE worsens by 2.46%, and ACC is slightly lower. It is a Pareto
trade-off rather than an overall winner.

### Frozen 2024 external rain-gauge sensitivity

Artifact:
`../resultsv2/fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z`

This evaluation uses the exact same 30 initializations as the E2 2024 subset,
all six leads, and 180 paired cases. It fits nothing to station observations. The primary
interval uses 2,000 circular moving-block draws of length 13, with lengths 4
and 8 retained as predeclared sensitivities.

| Method | RMSE | MAE | Signed bias | Mean absolute case bias | ACC |
|---|---:|---:|---:|---:|---:|
| Raw FuXi | 7.942 | 5.284 | +2.383 | 2.462 | 0.403 |
| Training-only log-bias | 7.530 | 4.949 | +1.911 | 2.060 | **0.435** |
| Legacy anchored adapter | 7.570 | 4.970 | +1.902 | 2.077 | 0.431 |
| Raw-identity adapter | **7.493** | **4.893** | **+1.790** | **1.978** | 0.431 |
| Raw-mean-preserving raw identity | 7.825 | 5.286 | +2.465 | 2.539 | 0.410 |

Raw identity improves raw FuXi by 0.449 mm/day RMSE (5.66%), 0.391 mm/day MAE
(7.40%), and +0.0276 ACC. The corresponding block-13 intervals are 0.389 to
0.507 for RMSE, 0.360 to 0.424 for MAE, and 0.0150 to 0.0390 for ACC.

Raw identity has the best pooled RMSE, MAE, and bias point estimates among the
five methods. A direct raw-identity-versus-log-bias or versus-legacy bootstrap
was not predeclared in E3, so those rankings are descriptive rather than a
formal superiority claim.

The E3 primary estimand was the legacy selected adapter versus raw FuXi. The
raw-identity-versus-raw contrast was frozen in advance as a secondary
comparison. It is useful external-target sensitivity, not an independent
confirmatory replication of the E2 primary story.

The raw-mean projection is decisively worse than raw identity at stations:
RMSE effect -0.332 mm/day, MAE effect -0.393 mm/day, and ACC effect -0.0204,
with all block-13 intervals below zero. Raw FuXi is slightly dry against IMD
but strongly wet against the gauge target; preserving its mean therefore
preserves opposite calibration errors. This is the clearest explanation for
the cross-target failure.

## What works

- A small shared six-lead residual network; a larger model is not required by
  the current evidence.
- Raw-FuXi reconstruction as the neural identity baseline; the fitted
  log-bias reconstruction anchor is not necessary for skill.
- Learned spatial and lead-dependent redistribution: error and ACC gains
  survive both IMD-grid and gauge-derived 1.5-degree-cell verification.
- Exact no-op initialization, immutable checkpoints, train-only climatology,
  common support, paired case scoring, and dependence-aware intervals.
- Explicit negative-result accounting: the model's IMD dry bias and the
  projection's station failure are part of the result.

## What does not work or is not yet supported

- Calling the system a universally successful “bias correction.” Bias sign
  and magnitude depend on the observation target.
- Promoting raw-mean preservation as the main method.
- Further loss, Box--Cox, tail, regime, or affine searches; existing variants
  failed their guards or shifted error elsewhere.
- A KL-divergence or probabilistic retrofit. This deterministic weekly-mean
  task would need a separate predictive-distribution contract and proper
  scores such as CRPS or RPS.
- Claiming benefit from whole-world patch pretraining. The completed matched
  E0 gate selected scratch; this specific log-bias-anchored initialization
  transfer did not work.
- Treating 2022--2024 or the station analysis as the untouched final test.
- Claiming improved extremes. For >=20 mm/day rainfall the raw-identity RMSE
  interval crosses zero, MAE is not improved, and dry bias remains large.

## Recommended paper structure

Working title:

**Compact Neural Post-processing of FuXi S2S Rainfall: Skill Gains and
Observation-Dependent Calibration over India**

Core contributions:

1. A compact, exact-no-op, six-lead deterministic neural post-processor for
   data-constrained regional S2S rainfall.
2. A matched anchor ablation showing that the learned correction does not
   require a fitted log-bias reconstruction baseline.
3. Frozen 2022--2024 gridded verification with dependence-aware paired
   uncertainty and transparent error/bias trade-offs.
4. Frozen secondary external-target sensitivity showing consistent neural
   gains versus raw FuXi and observation-dependent failure of a seemingly
   sensible mean constraint.

Main visuals:

1. Exact three-member raw-identity architecture and evidence timeline.
2. E2 lead- and rainfall-intensity effects, including the heavy/extreme
   limitations.
3. Cross-target projection failure, with IMD `|pooled signed bias|` and station
   `mean |case bias|` explicitly labeled as different estimands.

## Claim boundaries

- Deterministic weekly mean rainfall, six lead weeks, JJAS initializations.
- India 27×27 grid and frozen 171-cell support; not district-scale prediction.
- The station study is an independent observation target, not an untouched
  time period. Station rain days use 03:00--03:00 UTC while the gridded
  forecast convention is 00:00--00:00 UTC.
- The mixed station container was streamed through dates extending into 2025,
  but filtering occurred before rainfall conversion/materialization. The
  scored snapshot contains only 2024 values; no 2025 prediction or metric
  target was opened.
- The failed global experiment tested patch-based initialization into a
  log-bias-anchored India model. It is neither global forecast verification
  nor proof that all possible remote-context pretraining must fail.
- E2 absolute bias is the absolute pooled signed bias; E3 absolute bias is the
  mean absolute case bias. They answer different questions and must not share
  an unlabeled calibration axis.
- Bootstrap intervals are descriptive conditional on the frozen seasons and
  are not multiplicity-adjusted confirmatory tests.

## Final decision boundary

1. Retain completed E0/E1 as negative component evidence without retuning.
2. Completed: freeze the 2025-compatible raw-identity-versus-raw hierarchy and
   hash every live dependency (selection SHA-256
   `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6`).
3. Completed: storage-incapable synthetic CUDA preflight job 109981 opened no
   storage path and wrote receipt SHA-256
   `c2484d8d9e5a93c782e23b4419b5363e04e254a3f8e88e2d37dd37ba9f43db3b`.
4. Remaining: obtain explicit approval bound to those exact hashes before the
   one-time 2025 IMD target access.

Until step 4 is complete, do not open or score 2025.
