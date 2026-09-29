# Repository map

## Scope

This directory contains the experiment orchestration, cache builders,
evaluation, tests, and presentation code for a six-lead deterministic
FuXi-to-IMD rainfall post-processor. The reusable neural layers and training
helpers live in the sibling `../../neural_adapter/src/fuxi_adapter` package.
Large forecast and observation archives live under `/storage`.

## Active data flow

```text
native FuXi shards + IMD daily Zarr
             |
             v
   aligned six-week fields and fixed support
             |
             v
 training-only climatology and normalization
             |
             v
 matched raw-identity and anchored residual adapters
             |
             v
 2018--2019 guards and frozen three-seed selection
             |
             v
 2022--2024 fixed audit + 2024 station sensitivity
             |
             v
 negative global component result -> sealed preflight -> one-time 2025 evaluation
```

A separate completed component branch trained the same compact backbone on
worldwide FuXi-to-IMERG patches (2002--2015 fit, 2016--2017 validation), then
compared that initialization with matched scratch training over India. The
frozen gate selected scratch (global RMSE 5.508 versus 5.481); the branch does
not change the E2/E3 method sets. It tested a log-bias-anchored India
comparator, not raw-identity pretraining, and is not a claim of global forecast
skill.

The key scientific boundary is that target-derived preprocessing is fit on
training years only. Validation chooses a complete system. Later years must
not change the anchor, normalization, architecture, checkpoint, residual gate,
or reporting contract.

## Dependency chain

The active source is a flat module graph under `src/`:

```text
loss / tail / target-transform screens
    -> fuxi_imd_bias_aware_validation_sweep
       -> fuxi_imd_compact_validation_sweep
          -> fuxi_imd_attention_climatology
          -> fuxi_imerg_a100_big_temporal
             -> fuxi_imerg_full_archive_latelead
                -> fuxi_imerg_experiment
                -> fuxi_imerg_spatiotemporal
                   -> spatiotemporal_model
```

This is why the older IMERG-named modules remain under `src/`. They own
shared loading, architecture, metric, and plotting behavior used by current
IMD experiments. Removing them now breaks imports, source snapshots, tests,
and Slurm jobs.

## Ownership

| Area | Owner files |
|---|---|
| Native FuXi/IMERG base I/O and contracts | `src/fuxi_imerg_experiment.py` |
| Temporal/shared model plumbing | `src/fuxi_imerg_spatiotemporal.py`, `src/fuxi_imerg_full_archive_latelead.py`, `src/fuxi_imerg_a100_big_temporal.py` |
| IMD loading, support, climatology, full-context adapter | `src/fuxi_imd_attention_climatology.py` |
| Compact architecture and physical-feature selection | `src/fuxi_imd_compact_validation_sweep.py` |
| Physical cache construction | `src/fuxi_physical_feature_cache.py`, `src/fuxi_physical_postselection_cache.py` |
| Anchor and bias-aware loss ablation | `src/fuxi_imd_bias_aware_validation_sweep.py` |
| Later negative screens | `src/fuxi_imd_tail_weight_validation_sweep.py`, `src/fuxi_imd_intensity_loss_validation_sweep.py`, `src/fuxi_imd_target_transform_validation_sweep.py` |
| OOF and affine calibration | `src/fuxi_imd_train_affine_calibration.py`, `src/fuxi_imd_blocked_oof_affine.py` |
| Raw-identity/no-fitted-log-bias experiment | `src/fuxi_imd_no_log_bias_validation.py` |
| Global patch pretraining and matched India comparison | `src/fuxi_global_patch_pretraining.py`, `src/fuxi_imd_global_pretraining_comparison.py` |
| Frozen later-year evaluation | `evaluate/evaluate_locked_physical_hindcast.py` |
| Canonical 2022--2024 matched audit | `evaluate/evaluate_raw_identity_2022_2024_audit.py` |
| External station-target sensitivity | `evaluate/evaluate_adapter_station_external_target.py` |
| Artifact-only paper evidence package | `evaluate/build_paper_evidence_package.py` |
| One-time 2025 raw-identity boundary | `evaluate/raw_identity_2025_contract.py`, `evaluate/freeze_raw_identity_2025_selection.py`, `evaluate/preflight_raw_identity_2025.py`, `evaluate/evaluate_raw_identity_2025.py`, `evaluate/raw_identity_2025_runtime.py`, `evaluate/raw_identity_2025_assets.py` |
| Superseded physical-control 2025 design | `evaluate/freeze_independent_2025_control_selection.py`, `evaluate/evaluate_independent_2025_control.py` |
| Figures, verification, and packages | `evaluate/` |

## State and side effects

- `cache/` stores finalized NPZ feature products. Cache builders use
  atomic per-initialization parts and a finalization step.
- `results/` stores immutable run directories with code snapshots,
  checkpoints, metrics, figures, and manifests.
- `resultsv2/` stores new immutable post-cleanup experiment products. Its
  index names the canonical E2 and E3 artifacts.
- `resultsv3/` is reserved for corrected FuXi/IMD +1...+42 calibration runs;
  v1 checkpoints or fitted products may not enter this root.
- `reports/` contains one checksum-verified, artifact-only paper evidence
  package; generated payloads are ignored while `reports/README.md` indexes
  the retained package.
- `logs/` is the Slurm output sink.
- `presentation/` contains generated/static payloads; their generators live in
  `evaluate/`.
- `trained_model/` contains curated older IMERG model bundles and must
  not be discarded until that API is intentionally deprecated.

Completed run directories are provenance units. Moving or pruning files inside
them invalidates manifest paths and hashes.

## Tests

The full lightweight suite is:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:evaluate:hpc_compat:../neural_adapter/src \
python -m pytest -q -p no:cacheprovider
```

Current baseline: 267 passed after hardening the sealed raw-identity 2025
boundary and the paper-evidence builders.

The historical blocked-OOF path still lacks a repaired real-data fold
truth-invariance test. It is deliberately outside the current paper evidence
line; existing tests do not make its failed partial run scientific evidence.

## Refactor boundary

The root-to-`src`/`evaluate` move is complete. The next structural refactor
should remain incremental:

1. introduce an immutable experiment contract containing paths, split years,
   target units, support, feature order, seeds, and evidence role;
2. parameterize loaders instead of mutating module globals;
3. extract data/features/training/selection/evaluation modules into a real
   `src/fuxi_bias/` package;
4. retain thin compatibility CLIs until Slurm files and result snapshot logic
   use the package;
5. add contract hashes and staged atomic publication for every new run.

The current flat basenames are compatibility interfaces. Extract tested
behavior first, then remove those interfaces only after import, snapshot, and
Slurm coverage exists.
