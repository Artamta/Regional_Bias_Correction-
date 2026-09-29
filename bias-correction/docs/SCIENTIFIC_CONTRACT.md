# Scientific contract

> **Scope warning (26 August 2026):** This file describes the legacy
> deterministic TP/T2M adapter line. Its day-range wording below is historical
> and does not define the aligned `resultsv3/` probabilistic experiment. For
> that branch, the calendar-label contract is unambiguously end-labelled IMD
> `init+1...+42`, with W1 `+1...+7` and W6 `+36...+42`; use
> `plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md` and its bound
> manifests. Never infer `resultsv3/` target offsets from this legacy summary.

This document summarizes the frozen FuXi–IMD control contract. The
machine-readable source of truth is
`../results/fuxi_imd_full_context_compact_allweeks/full_20260811T152024Z/selection.json`.
If this summary and the frozen selection disagree, stop and resolve the
discrepancy before running later-year evaluation.

## Forecast and target

- Forecast: six non-overlapping FuXi weekly means.
- Target: IMD weekly mean precipitation in mm/day.
- Historical deterministic W1 notation: initialization day through day +6.
- Historical deterministic W6 notation: day +35 through day +41.
- Regional grid: 27 × 27, 0–39°N and 60–99°E.
- Target, loss, and reported India metrics: the fixed 171-cell positive-weight
  IMD support.
- FuXi TP mean, TP ensemble spread, and T2M retain full regional context.

No target-derived field may supply information outside the fixed support.
Missing operational physical variables must not be zero-filled.

## Split and access

| Role | Years |
|---|---|
| Fit model and all target-derived preprocessing | 2002–2017 |
| Select the complete system | 2018–2019 |
| Reused exploratory diagnostics | 2020–2021 |
| Frozen development audit | 2022–2024 |
| One-time final initialization year | 2025 |

All normalization, IMD climatology, log-bias fields, target scales, and any
calibration coefficients must be fit without later-year targets. Opening a
later evidence tier never reopens selection.

## Frozen control

- Selected model: `normal_climo_model`.
- Residual gate: α = 1.0.
- Seeds: 42, 43, and 44.
- Active leads: W1–W6 with equal weights.
- Primary validation metric: equal-case, W1–W6 area-weighted RMSE.
- Attention-climatology candidate: rejected because it did not beat the
  matched normal-climatology model in both validation years.

Checkpoint hashes, feature/support metadata, and the selection timestamp are
recorded in the frozen selection JSON. Do not replace a checkpoint while
retaining the same selection file.

## Metrics

- RMSE, MAE, and signed bias are computed in physical mm/day space with the
  declared area/support weighting.
- ACC uses the same fixed 2002–2017 IMD climatology for every forecast method.
- Case is the independent reporting unit; all six leads remain attached to an
  initialization in dependence-aware resampling.
- Pixel-wise maps are descriptive unless a separately declared
  multiplicity-aware protocol says otherwise.

## Promotion and final test

A development candidate must pass every predeclared pooled, year, lead,
baseline, MAE, ACC, bias, and seed-robustness guard. Lowest pooled RMSE alone
is not enough. If no candidate qualifies, the frozen reference remains the
decision.

The 2025 evaluator requires a separately frozen 2025-compatible selection,
hash verification, a successful preflight that does not open 2025 data, and a
one-time access ledger. Follow
`RAW_IDENTITY_2025_SEALED_WORKFLOW.md` exactly. The legacy
`INDEPENDENT_2025_CONTROL_WORKFLOW.md` does not implement the selected
raw-identity reconstruction contract.
