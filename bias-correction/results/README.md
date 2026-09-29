# Result artifact index

Everything below this directory is generated and Git-ignored. This file names
the canonical evidence that must be retained as whole, immutable run
directories.

## Canonical India results

- `fuxi_imd_full_context_compact_allweeks/full_20260811T152024Z`:
  frozen compact TP/T2M control selected on 2018–2019.
- `fuxi_imd_compact_validation_sweep/physical_confirm3_a100_20260811T233520Z`:
  three-seed physical-variable validation qualification.
- `fuxi_imd_locked_hindcast_evaluation/physical_full_compact_exploratory_2020_2021_20260812T010224Z`:
  reused 2020–2021 physical-candidate diagnostic; exploratory only.
- `fuxi_imd_bias_aware_validation_sweep/full_20260813T135600Z`:
  completed negative 2×2 anchor/loss ablation.
- `fuxi_imd_tail_weight_validation_sweep/screen_seed42_cn14_20260813T152500Z`:
  completed negative heavy-tail screen.
- `fuxi_imd_intensity_loss_validation_sweep/screen_seed42_cn13_20260814T_loss_v1`:
  completed negative intensity/regime-loss screen.
- `fuxi_imd_target_transform_validation_sweep/screen_seed42_cn14_20260814T_transform_v1`:
  fixed Box–Cox screen; exact log target retained.
- `fuxi_imd_train_affine_calibration/full_cn14_20260813T155500Z`:
  train-only affine diagnostic; not promoted.

The prior anchored-only 2022--2024 source audit lives outside this directory at
`../../studies/fuxi_imd_adapter_benchmark_v1/results/full_context_jjas_2022_2024_job91439`.

The completed raw-identity/no-fitted-log-bias ablation also lives in the
parallel generated-results tree at
`../resultsv2/fuxi_imd_no_log_bias_ablation/full_20260822T010749Z`. It is a
three-seed development candidate selected on 2018--2019 and evaluated only on
the reused exploratory 2020--2021 cohort; retain the directory whole.

The strongest canonical no-retraining 2022--2024 comparison lives at
`../resultsv2/fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z`.
It contains the five-method predictions, paired circular-block intervals, and
the raw-mean-preserving diagnostic. Manifest SHA-256:
`bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`.

The fixed external station-target sensitivity lives at
`../resultsv2/fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z`.
It scores the exact 30 common 2024 starts without fitting to station data.
Manifest SHA-256:
`5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`.

The completed global-source runs are retained under
`fuxi_global_patch_pretraining/execution_full_parallel_20260822a`. The
canonical matched India comparison is
`fuxi_imd_global_pretraining_comparison/execution_full_parallel_20260822a/full_20260822T040953Z`.
Its frozen gate selected scratch: global initialization worsened pooled RMSE
by 0.493%, and zero of three pretrained seeds achieved a lower best India
validation composite loss than matched scratch. Manifest SHA-256:
`37b6dac32a9fc2be799651b8e8577cac247d144caec379d1a3fdfce0417c1f5d`.
This was initialization transfer into the matched log-bias-anchored India
comparator, not a raw-identity global-pretraining experiment.

The artifact-only post-hoc projection diagnostic is retained at
`fuxi_imd_global_pretraining_followups/full_parallel_20260822a`. It rejected
mean preservation for both scratch and pretrained ensembles and is not
scientifically eligible for promotion. Manifest SHA-256:
`e4748243bf8a57b34dc4460af97ce57cbdcfd63d39daebe36e9458356b0510bc`.

## Canonical older IMERG evidence

- `fuxi_imerg_jjas_5yr/full_20260809T233638Z`
- `fuxi_imerg_spatiotemporal/full_scratch_20260810T003927Z`

These are a separate target/evidence line and are not current IMD selection
results.

## Interpretation rules

- Smoke directories are execution checks, not evidence.
- `full` in a directory name does not imply success. Require a complete
  manifest and inspect the recorded selection status.
- Seed-42 screens are hypothesis generation unless the protocol explicitly
  promotes and confirms them.
- Reused 2020–2021 outputs are exploratory.
- Do not cite the failed blocked-OOF run. Its partial artifacts are archived at
  `../archive/blocked_oof_failure_20260813.tar.gz`.
- The independent-2025 selection now lives under `../resultsv2/` and passed a
  storage-incapable synthetic CUDA preflight. No 2025 access ledger or result
  exists.

Never edit or prune a completed canonical directory in place. Its manifest may
hash code, checkpoints, tables, predictions, and figures. Supersede a complete
run with a new immutable directory and update
`../docs/EXPERIMENT_LEDGER.md`.
