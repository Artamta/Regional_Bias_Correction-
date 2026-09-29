# Legacy independent operational 2025 control workflow (superseded)

> This older compact-control path is legacy and must not be used for the final
> untouched 2025 evaluation. It is superseded by
> `RAW_IDENTITY_2025_SEALED_WORKFLOW.md`. The legacy evaluator remains in place
> only for provenance and has not been modified into the new workflow.

This was the proposed one-time final-test path for the compact TP/T2M-only control. It
does not make the physical-variable candidates 2025-compatible: the saved 2025
operational archive has TP and T2M, but not TCWV, Q850, U/V850, Z500, MSL, or
OLR. A selected physical/member architecture therefore fails preflight before
any 2025 store is opened; missing channels are never filled with zeros.

## Scientific sequence

1. Complete and inspect the 2002–2017 train / 2018–2019 blocked-validation
   three-seed run.
2. Before opening 2025, hash-lock the independent-compatible control:

   ```bash
   PYTHONPATH=src:evaluate:hpc_compat:../neural_adapter/src \
   python evaluate/freeze_independent_2025_control_selection.py \
     --validation-run /absolute/path/to/completed_validation_run \
     --output /absolute/path/to/frozen_2025_control_selection.json \
     --configuration physical_control \
     --attest-no-2025-access
   ```

3. Run a GPU allocation/checkpoint preflight. `--preflight-only` validates only
   frozen metadata, code, checkpoints, normalization, and the checked local
   boundary. It exits before creating the access ledger and opens neither 2025
   predictors nor IMD targets:

   ```bash
   sbatch slurm/evaluate_independent_2025_control.sbatch \
     --validation-run /absolute/path/to/completed_validation_run \
     --selection-manifest /absolute/path/to/frozen_2025_control_selection.json \
     --output /absolute/path/not_created_by_preflight \
     --preflight-only
   ```

4. After that preflight passes, submit exactly one final evaluation to a fresh
   output directory:

   ```bash
   sbatch slurm/evaluate_independent_2025_control.sbatch \
     --validation-run /absolute/path/to/completed_validation_run \
     --selection-manifest /absolute/path/to/frozen_2025_control_selection.json \
     --output /absolute/path/to/fresh_independent_2025_result
   ```

The evaluator atomically writes a one-time access ledger beside the selection
manifest. A repeat launch with that manifest fails. The checkpoint ensemble,
log-bias anchor, normalization statistics, evaluation code, and complete
predictor/target contract are hash-locked at freeze time and checked again just
before publication. No 2025 target or predictor is used to fit, calibrate,
tune, select, or normalize anything.

The final result is built in a hidden sibling staging directory. A complete
manifest is written only after all tables, fields, figures, and live-versus-
frozen hash checks succeed; the staging directory is then published to the
requested path with one same-filesystem atomic rename. A failed run leaves no
apparently complete final directory. The Slurm wrapper requires CUDA, logs the
assigned GPU, and excludes the known MIG/unreliable nodes.

## Final outputs

- `metric_summary_block_bootstrap.csv`: W1–W6 and pooled ACC, RMSE, bias,
  paired improvement, 95% percentile intervals, and descriptive
  `bootstrap-supported` flags.
- `case_metrics.csv`: the auditable case × lead table for raw and corrected
  forecasts.
- `figures/metrics_independent_2025_by_lead.{png,pdf}`: presentation curves
  with 95% moving-block percentile intervals and bootstrap-support markers.
- `figures/spatial_independent_2025_W1...W6.{png,pdf}`: one 2×3 figure per
  lead. Top: IMD, raw FuXi, corrected. Bottom: raw error, corrected error, and
  local RMSE improvement percentage.
- `figures/spatial_rmse_skill_contact_sheet_2025.{png,pdf}`: all six local
  RMSE-improvement maps.
- `independent_2025_fields.npz`: fields, fixed climatology, support, local
  skill, local percentile bounds, and unadjusted bootstrap-support masks used
  by the figures.
- `manifest.json`: checkpoint/input/output hashes, source Zarr metadata hashes,
  boundary provenance, frozen-versus-live code and data-contract hashes,
  software, and the full uncertainty contract.

## Uncertainty-language guard

Initialization dependence is handled with 2,000 deterministic circular
moving-block draws of length 13; each sampled initialization carries all six
lead weeks. A result is labelled `bootstrap-supported` only when its paired 95%
percentile interval for improvement is wholly above zero. This is a descriptive
effect interval: the bootstrap distribution is not recentered under a null.
The workflow therefore makes no null-hypothesis or multiplicity-adjusted claim.
Spatial circles apply the same rule cell by cell without multiplicity
adjustment and are explicitly exploratory because one test year contains only
about three 13-start blocks.

## Synthetic verification (does not open 2025)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/src:/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/evaluate:/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/hpc_compat:/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src \
python -m pytest -q tests/test_independent_2025_control.py
```
