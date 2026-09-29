# Global pretraining, local IMD adaptation

Status: complete negative component experiment. Global array job `109944`
completed seeds 42, 43, and 44 under
`results/fuxi_global_patch_pretraining/execution_full_parallel_20260822a`.
Matched India comparison job `109947` completed at
`results/fuxi_imd_global_pretraining_comparison/execution_full_parallel_20260822a/full_20260822T040953Z`;
its frozen gate selected scratch. Artifact-only diagnostic job `109959` also
completed and rejected mean preservation for both arms. The 2025
initialization year remains sealed.

## Question

Does pretraining the existing compact six-lead FuXi residual adapter on
worldwide FuXi-to-IMERG rainfall-error patches improve a matched FuXi-to-IMD
adapter trained and validated over India?

Only the initialization changes in the primary comparison. The India data,
features, architecture, optimizer, training schedule, seeds, reconstruction,
and metrics must otherwise be identical for the from-scratch and globally
pretrained candidates.

## Evidence and date contract

| Stage | Fit | Select/validate | Forbidden |
|---|---|---|---|
| Global FuXi-to-IMERG pretraining | 2002--2015 | 2016--2017 | Any target date in 2018 or later |
| India FuXi-to-IMD comparison | 2002--2017 | 2018--2019 | Any 2020--2025 predictor or target |

The split is enforced by verification date, not only by cache filename. A
forecast initialized on date `t` uses truth through `t + 41 days`. Schedules
are resolved from metadata before truth arrays are indexed and the fit and
validation target windows must not overlap. In the completed run the latest
fit target is 2016-01-01 and the earliest validation target is 2016-01-03;
the validation target ends on 2017-12-28. Late-2017 cases that would reach
2018 are removed before truth is read. This corrects the earlier, overly
strict wording that every fit case reaching calendar 2016 was removed.

All climatology, log-bias anchors, feature normalization, and target scales are
fit within their stage's fit years. Global IMERG preprocessing is provenance
only and is never reused as India IMD preprocessing.

## Global cache and patch contract

The immutable input is the completed annual cache under
`/storage/raj.ayush/s2s_final_data/final_iteration/global_tp_adapter/cache/annual`.
Every accepted year must contain:

- 104 initialization cases and six lead weeks;
- `dynamic` with shape `[104, 6, 8, 121, 240]` and ordered statistics
  `mean, std, q10, q25, q50, q75, q90, wet_fraction`;
- `truth` and `observation_fraction` with shape `[104, 6, 121, 240]`;
- latitude `90, 88.5, ..., -90` and longitude `0, 1.5, ..., 358.5`;
- weekly mean rainfall-rate units of mm/day and a complete status marker.

An example is one 27 x 27 patch containing all six leads. Patch schedules are
created from a fixed sampler seed and are identical across model seeds.
Latitude never wraps: valid center rows are 13 through 107 inclusive. Longitude
wraps modulo 240. Thus a patch can cross 0 degrees longitude, while north- and
south-pole rows can occur only at a patch edge. Loss weights combine valid
IMERG coverage with nonnegative cosine-latitude area weights; the exact pole
row has zero area weight.

Coordinates inside a patch use the same local convention as the India grid:
latitude runs from +1 in the north to -1 in the south, and longitude runs from
-1 in the west to +1 in the east, including a dateline-wrapping patch.

## Shared model and transfer contract

The source and destination use `FixedClimatologyAllLeadUNet` with the same
16/32/64 spatial widths, six-lead temporal attention, and 11-channel backbone.
The source cache has no T2M. Its T2M slot is therefore explicitly zero during
global pretraining and cannot be claimed as pretrained.

Transfer is strict and transactional:

1. Verify the checkpoint schema, feature order, architecture, patch contract,
   split contract, and their hashes before changing the India model.
2. Copy every compatible learned tensor.
3. Restore the first-convolution T2M slice from the matched India
   initialization because that slice saw only zeros globally.
4. Reset only `backbone.residual_head.weight` and
   `backbone.residual_head.bias` to exact zero.
5. Create a fresh optimizer, scheduler, scaler, IMD normalization, IMD anchor,
   and IMD target scale.

The reset makes the transferred model an exact epoch-zero no-op relative to
the India log-bias anchor. Missing or unexpected tensors, a contract mismatch,
or a nonzero epoch-zero residual is a hard failure.

## Comparison and promotion

The matched India candidates are:

- `scratch`: the shared model initialized normally;
- `global_pretrained`: the same model initialized by the verified transfer
  above.

Full evidence uses seeds 42, 43, and 44. Report equal-case, India-area-weighted
RMSE, MAE, signed bias, and climatological anomaly correlation for W1--W6,
including year, lead, and seed breakdowns. Any promotion decision must be made
only from 2018--2019 IMD validation and must compare paired initialization
cases.

The globally pretrained candidate is promoted only if its pooled RMSE improves
over the matched scratch candidate, the improvement is not confined to one
validation year or one seed, MAE and ACC do not regress materially, and the
absolute rainfall bias does not worsen materially. A failed guard is a valid
negative result; it is not permission for a new post-hoc sweep.

## Completed result

All three global source runs learned beyond their identical epoch-zero loss
and passed checkpoint, cache, schedule, preprocessing, source-code, and zero
T2M-kernel verification. Their best epochs were 20, 23, and 25 for seeds 42,
43, and 44. The matched India result was nevertheless negative:

| India ensemble | RMSE | MAE | Signed bias | ACC |
|---|---:|---:|---:|---:|
| Scratch | **5.4809** | **3.5466** | **-1.2120** | **0.3207** |
| Global pretrained | 5.5080 | 3.5555 | -1.2425 | 0.3119 |

Global initialization worsened pooled RMSE by 0.493%, worsened both validation
years, and zero of three pretrained seeds achieved a lower best India
validation composite loss than matched scratch. The promotion manifest
therefore records `selected_system: scratch` and
`global_pretraining_qualifies: false`. Comparison manifest SHA-256:
`37b6dac32a9fc2be799651b8e8577cac247d144caec379d1a3fdfce0417c1f5d`.
The artifact-only 10,000-draw circular-block diagnostic gives a
direction-normalized global-versus-scratch RMSE effect of -0.0270 mm/day (95%
interval -0.0467 to -0.0084) and ACC effect of -0.00886 (-0.01402 to
-0.00352), where positive would favor global. Both consistently favor scratch.

The fixed, post-hoc area-mean projection also failed for both ensembles. It
increased scratch RMSE by 0.0392 mm/day and pretrained RMSE by 0.0355, while
improving zero leads, years, or seeds. Follow-up manifest SHA-256:
`e4748243bf8a57b34dc4460af97ce57cbdcfd63d39daebe36e9458356b0510bc`.
It is explicitly scientifically ineligible for source promotion.

This experiment tests initialization transfer into the log-bias-anchored
India comparison. It is not evidence that the model has global forecast
skill, nor a universal rejection of every possible raw-identity or remote-
context pretraining design.

## Execution boundary

A smoke run uses one seed and hard-capped training indices, patches, and
epochs. The India stage still reads the complete 2002--2019 inputs and fits the
real train-only preprocessing before selecting its small smoke subset; this is
an execution smoke, not a cheap synthetic-data test. It must exercise CUDA
initialization, global cache reads, a longitude-wrapping patch,
both latitude edges, checkpoint creation, strict transfer, head/T2M reset, one
India training path, validation reconstruction, and manifest publication.

Every smoke manifest must state `smoke: true`, `scientific_eligible: false`,
and `test_predictions_created: false`. It records the Slurm
job/node/GPU, exact date bounds, cache and sampler fingerprints, source and
destination checkpoint hashes, transferred/reset tensors, epoch-zero no-op
check, artifact hashes, and stage completion. A full job may start only after
the smoke manifest is complete and manually inspected. Smoke metrics must
never appear in a paper table.

Use `slurm/run_global_pretrain_india_smoke.sbatch` first with two fresh output
roots. After inspecting the completed India smoke manifest, pass that manifest
and two new output roots to `slurm/run_global_pretrain_india_full.sbatch`.
That launcher is the simple single-GPU route and runs seeds 42, 43, and 44
sequentially.

For lower wall-clock time with the identical scientific contract, submit the
three seeds as an array and make the India comparison depend on the successful
completion of the complete array:

```bash
ARRAY_JOB_ID=$(sbatch --parsable \
  slurm/run_global_pretrain_seed_array.sbatch \
  COMPLETED_SMOKE_MANIFEST FRESH_PRETRAIN_ROOT)

sbatch --dependency=afterok:${ARRAY_JOB_ID} \
  slurm/run_global_pretrain_india_compare.sbatch \
  FRESH_PRETRAIN_ROOT FRESH_COMPARISON_ROOT
```

The array performs three simultaneous read-only cache scans, so storage
contention can make the speedup smaller than threefold. Each task writes an
isolated seed directory, log, and temporary directory. The comparison starts
only when every array task exits successfully and independently validates all
three manifests and checkpoints before opening India data. Both output roots
must be absent before submission. No launcher overwrites a completed result
directory or submits another job automatically.
