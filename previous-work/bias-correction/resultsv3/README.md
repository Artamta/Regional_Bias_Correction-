# Corrected-alignment experiment index

This root is reserved for immutable experiments that use the FuXi/IMD
end-labelled verification contract: native lead days 1--42 pair with IMD
labels `initialization+1` through `initialization+42`. Every run must preserve
its source snapshot, input and checkpoint hashes, exact split membership,
Slurm metadata, case-level metrics, and a complete manifest.

The `resultsv2/fuxi_allseason_*` probabilistic calibration family is historical
only. Its frozen training snapshot used IMD offsets 0--41 and cannot support
scientific or manuscript claims. Do not copy its checkpoints, fitted moment
calibration, categorical thresholds, metrics, or figures into this root.

The first authorized experiment is
`fuxi_allseason_ensemble_calibration_v2_aligned`. It trains only the frozen
42,434-parameter `location_spread` architecture, uses seeds 42, 43, and 44,
selects one deployable checkpoint by 2018--2019 validation CRPS (lowest seed
breaks an exact tie), and keeps other seeds as sensitivity evidence. The 2025
target remains sealed and is out of scope.
