# India S2S probabilistic bridge contract

Status: frozen preflight contract. The bridge may consume only a complete
`fuxi_allseason_ensemble_calibration_v2_aligned` parent under `resultsv3/`.
It must reject v1/resultsv2 parents, smoke runs, changed checkpoint hashes,
zero-based targets, seed-score averaging, and every opened 2025 target.

Three cohorts are distinct:

1. **Forecast-only climatology cohort:** all 517 paired FuXi starts from
   2020--2024. Corrected forecasts for late-2024 starts may be generated
   because inference uses no verification truth.
2. **All-season scoring cohort:** 505 starts whose last end-labelled IMD target
   (`initialization+42`) is no later than 2024-12-31.
3. **Views:** 170 June--September initialization cases and the primary
   valid-midpoint JJAS view, which contains 169 cases independently at each
   lead. These labels must never be interchanged.

Corrected-FuXi forecast climatology uses all 517 forecast-only starts with the
benchmark's fixed-366-day, centered 31-day, equal-year four-year LOYO
estimator. Scores use the 505-case truth firewall, IMD 1991--2019 observation
normal, benchmark weekly observation coverage, and all-India plus four
fractional homogeneous-region weights. Main neural rows use the single
validation-selected checkpoint. Seeds not selected for deployment are
sensitivity evidence only.

The bridge must first publish an immutable preflight containing exact cohort
dates/hashes and a hash-gated parent receipt. Inference and scoring may start
only after that preflight and the corrected full training manifest pass.
