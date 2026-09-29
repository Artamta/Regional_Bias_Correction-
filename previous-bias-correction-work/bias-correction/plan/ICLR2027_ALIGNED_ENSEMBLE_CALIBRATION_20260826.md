# ICLR 2027 corrected-alignment FuXi calibration contract

Status: frozen pre-run implementation contract. This document authorizes
retrospective training and evaluation through 2024 only. It does not authorize
access to a 2025 target.

## Withdrawal boundary

The immutable v1 publication snapshot at
`resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z`
constructed IMD targets with offsets 0--41 and purged outcomes through
`initialization+41`. FuXi native leads and the benchmark require end-labelled
IMD days 1--42. Consequently, all v1 checkpoints, fitted baselines, downstream
categorical products, metrics, and figures are historical diagnostics only and
must not support paper claims.

The frozen v1 source SHA-256 is
`0a46bb78266fdbf15b4e9cd95f1efd7553eaa933b9e1d73b535e8bc4eacea111`;
its manifest SHA-256 is
`94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91`.
Immutable v1 artifacts must remain untouched.

## Corrected experiment

- Experiment: `fuxi_allseason_ensemble_calibration_v2_aligned`.
- Output root: `resultsv3/fuxi_allseason_ensemble_calibration/`, always a fresh
  immutable directory published by atomic rename.
- Forecast cache: the verified 51-member 2002--2021 FuXi member cache; forecast
  fields may be reused because the defect affected target construction.
- Target: IMD weekly mean rainfall with W1 `init+1...+7` through W6
  `init+36...+42`.
- Splits: train 2002--2017, validation 2018--2019, reused development test
  2020--2021; purge left-split starts through `init+42`.
- Architecture: only the frozen 42,434-parameter `location_spread` adapter;
  no architecture or objective search.
- Seeds: 42, 43, and 44. Select one deployable checkpoint using minimum
  validation CRPS; an exact tie selects the lowest seed. Main predictions and
  scores use that checkpoint only. Other seeds are sensitivity evidence.
- Baselines: raw FuXi and a newly fitted training-only moment calibration.
  Nothing fitted or derived from v1 may be loaded.
- The full run requires a completed, source-matched CUDA smoke manifest.

## Acceptance gates

The smoke and full manifests must record offsets 1--42, the 42-day outcome
guard, exact split IDs, `sealed_2025_target_opened=false`, source/input hashes,
all candidate validation CRPS values, and the selected checkpoint hash. Tests
must reject offsets 0--41, v1 smoke manifests, non-`resultsv3` output paths,
seed-score averaging as the main forecast, and any 2025 path or initialization.
