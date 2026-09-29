# Sealed raw-identity independent-2025 workflow

This is the sole current workflow for the untouched 2025 comparison. Its
primary hierarchy is fixed before access: the `normal_climo_model`, alpha 1,
three-seed (42/43/44) arithmetic residual ensemble named `raw_identity` versus
uncorrected `raw_fuxi`. Projection, legacy, physical, and global candidates
cannot be represented by this contract.

No 2025 access has been authorized or performed by implementing or testing
this workflow. The selection was frozen after independent review at the
canonical path on 22 August 2026 with SHA-256
`cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6`.
Storage-incapable synthetic CUDA preflight job 109981 then passed on an NVIDIA
A30; its receipt SHA-256 is
`c2484d8d9e5a93c782e23b4419b5363e04e254a3f8e88e2d37dd37ba9f43db3b`.
The access ledger, attempt-status record, and final result remain absent.

## Boundary and state sequence

1. `freeze_raw_identity_2025_selection.py` authenticates the canonical E2 and
   raw-run artifacts, copies the exact model source, normalization, raw anchor,
   and three checkpoints into `sealed/`, and builds the support/daily
   climatology bundle from 2002-2017 data only. It never probes a 2025 path.
2. `preflight_raw_identity_2025.py` accepts only `--selection-manifest`. It
   rechecks every local seal, loads all checkpoints from authenticated bytes,
   and runs synthetic 29-channel CUDA inference. It cannot accept a store,
   output, ledger, or receipt destination. Its receipt location is canonical.
3. A new user approval receipt must bind the selection SHA, preflight receipt
   SHA, frozen locator SHA, canonical-path SHA, and exact result/ledger/status
   paths. A receipt is not approval unless all v2 fields and the exact
   authorization sentence validate.
4. `evaluate_raw_identity_2025.py` accepts only the selection, approval, and
   mandatory CUDA flag. It creates a durable status record, commits the one
   global attempt ledger with `O_EXCL` and `fsync`, then executes the exact
   runtime and asset source bytes authenticated while loading the selection.
   The attempt stays consumed after success, failure, or interruption.
5. The runtime opens only the three exact frozen locators through
   descriptor-anchored, no-symlink Zarr mappings. It records the anchored root
   identity and every consumed key hash, rechecks those bytes before closing,
   writes a hidden sibling staging directory, fsyncs all artifacts and the
   complete manifest, and publishes with Linux
   `renameat2(RENAME_NOREPLACE)`.

The stable attempt-status record distinguishes preparation, ledger committed
before runtime execution, runtime loaded before store open, stores loaded,
artifacts staged, complete staging before publication, rename completion, and
parent-directory fsync completion. A failed run preserves its last stage and
staging path. If rename succeeds but the parent fsync fails, the record says
that publication occurred and authenticates the published result instead of
misreporting it as unpublished.

## Frozen science

- Initializations: every Monday and Thursday at 00Z from 2025-06-01 through
  2025-09-30, exactly 35 starts.
- Leads: W1 `init+0..6` through W6 `init+35..41`.
- Truth: ordinary float64 arithmetic mean of the seven daily IMD observations,
  cast to float32. Coverage never weights truth construction.
- Score weights: frozen India area times the minimum of the seven daily IMD
  coverage fractions; every fraction must be finite and in `[0,1]`.
- Primary scores: case/lead spatial RMSE, MAE, signed bias, and ACC, then equal
  averaging across starts/leads. Each score needs at least three cells and all
  primary metrics must be finite.
- Uncertainty: 10,000 paired circular moving-block draws, block length 13,
  seed 20260822, retaining all six leads.
- Secondary diagnostics: frozen E2 intensity strata `<1`, `1-5`, `5-10`,
  `10-20`, and `>=20` mm/day, with pooled area-times-coverage RMSE, MAE, and
  bias plus the same descriptive block effects. They are exploratory within
  the final evaluation, were not used for selection, and make no multiplicity
  claim. Empty strata produce explicit `insufficient_weight/no_estimate` rows
  with no interval and do not fail the primary attempt.

Intervals are descriptive and conditional on one JJAS season. They are not
null-recentered tests and do not measure interannual uncertainty.

## Commands (review first; do not run yet)

Freeze to the one canonical experiment directory (the command rejects every
other destination):

```bash
PYTHONPATH=src:evaluate:hpc_compat:../neural_adapter/src \
python evaluate/freeze_raw_identity_2025_selection.py \
  --output-directory /home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/raw_identity_independent_2025_sealed \
  --attest-no-2025-access
```

Synthetic CUDA preflight, after reviewing `selection.json`:

```bash
sbatch slurm/preflight_raw_identity_2025.sbatch \
  --selection-manifest /home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/raw_identity_independent_2025_sealed/selection.json
```

Only after independently verifying the canonical preflight receipt and after
the user explicitly approves access, create a receipt from this exact template.
Placeholders must be replaced from the already-frozen selection and canonical
preflight receipt; neither this template nor filling it speculatively is user
approval.

```json
{
  "schema_version": "raw_identity_independent_2025_user_approval_v2",
  "decision": "approve_exactly_one_independent_2025_access",
  "approved_by": "raj.ayush",
  "approved_utc": "<timezone-aware timestamp after freeze and preflight>",
  "test_year": 2025,
  "selection_manifest_sha256": "<exact selection.json SHA-256>",
  "preflight_receipt_sha256": "<exact canonical preflight receipt SHA-256>",
  "test_data_locators_sha256": "<frozen test_data_locators SHA-256>",
  "canonical_execution_paths_sha256": "<frozen canonical path-contract SHA-256>",
  "final_output": "<exact frozen canonical final_output>",
  "access_ledger": "<exact frozen canonical access_ledger>",
  "failure_record": "<exact frozen canonical failure_record>",
  "allowed_methods": ["raw_fuxi", "raw_identity"],
  "authorization": "I authorize one access attempt for the frozen raw_identity versus raw_fuxi independent 2025 evaluation."
}
```

After that receipt validates, the one permitted final command is:

```bash
sbatch slurm/evaluate_raw_identity_2025.sbatch \
  --selection-manifest /home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/raw_identity_independent_2025_sealed/selection.json \
  --approval-receipt /absolute/local/path/user_approval.json
```

There is intentionally no CLI override for forecast stores, IMD, preflight
receipt, result output, ledger, or status record.

The one-attempt ledger is deliberately outside the movable selection tree at
`/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/raw_identity_independent_2025_access_ledger.json`.
Deleting, moving, or refreezing the experiment directory therefore cannot
create another permitted attempt. The exact final output, ledger, failure
record, and preflight-receipt paths are embedded in the selection and rebound
by their canonical-path SHA in the approval receipt.

## Final assets

The canonical result directory contains the complete manifest, fields,
case/lead metrics, primary paired block effects, frozen secondary intensity
metrics/effects, bootstrap indices, compact summary, results note, and exact
authenticated workflow source snapshot. The fields artifact includes every
evaluated source array: raw precipitation, spread, T2M, both member-count
arrays, coordinates, requested daily IMD observations and coverage, weekly
truth/coverage, and fixed climatology. Its manifest binds these array hashes as
well as the consumed Zarr keys and anchored root identities. Only the sorted
unique W1-W6 verification dates are loaded from the IMD observation and
time-varying coverage arrays; unrelated 2025 truth chunks are not materialized.
Publication is all-or-nothing and an existing result is never replaced.
