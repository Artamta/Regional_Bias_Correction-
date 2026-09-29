# Neural adapter experiment status

Updated: 15 August 2026

## Bottom line

There are three different projects in this workspace, and their scores must not
be mixed:

1. the older India FuXi–IMERG deterministic `neural_adapter`;
2. the 20-year India FuXi–IMD deterministic bias-correction study; and
3. this global FuXi–ERA5 probabilistic AI-Quest adapter.

The strongest current India generalization evidence is the frozen
144,689-parameter FuXi–IMD adapter evaluated on 100 JJAS starts from 2022–2024. It
improves raw FuXi RMSE, MAE, and ACC at every lead week, but it makes the mean
dry bias materially worse. This is a development generalization audit, not an
untouched final test. The 2025 evaluation has not been run.

The best validation-qualified feature extension is the 323,116-parameter
compact physical-variable adapter. It performed well on the already reused
2020–2021 hindcast, but has no independent test result and cannot currently be
used in the 2025 operational path unless the nine additional FuXi variables
are available there. A separate 2.54M-parameter model has slightly better
reused-test RMSE/ACC, but its capacity gain over the compact model is not
robust. There is therefore no single universal “best” model independent of the
metric and evidence level.

`ai-quest-global` has no real-data model result yet. Its code and synthetic
tests are ready, but the configured cache, checkpoints, histories, evaluation
tables, figures, and run directory do not exist. Therefore its present status
is **implementation-ready prototype; best model: none yet**.

## Scope and metric boundary

| Project | Target and output | Split used | Primary metrics |
|---|---|---|---|
| Older India adapter | IMERG; six deterministic weekly means | 2020–2022 train, 2023 validation, 2024 opened test | ACC, RMSE, MAE, bias |
| 20-year India adapter | IMD; six deterministic weekly means | 2002–2017 train, 2018–2019 validation | ACC, RMSE, MAE, bias |
| Global AI-Quest adapter | ERA5 climatological categories; probabilities for D19–25 and D26–32 | 2017–2018 train, 2019 validation, 2020–2021 sealed test | RPS, RPSS, reliability |

ACC/RMSE values from the deterministic India experiments cannot be presented
as evidence for the global probabilistic model. Likewise, a lower RPS in the
global experiment will not mean that deterministic IMD RMSE or bias improved.
For AI Quest, “bias” should primarily mean probability calibration and observed
category-frequency error; signed mm/day bias requires a separately defined
continuous rainfall diagnostic.

## Best defensible India result

The frozen three-seed full-context normal-climatology adapter was selected
without using 2022–2024 targets. Scores below are means of case-wise,
area-weighted spatial scores over 100 JJAS starts and W1–W6.

It predicts all six IMD weekly-mean rainfall fields in mm/day jointly, not
daily rainfall or weekly accumulation. FuXi retains the full 27×27 regional
context, while target/loss evaluation uses 171 IMD-supported India cells. Its
11 effective channels contain weekly FuXi TP mean/spread and T2M plus
training-only climatological, anomaly, coordinate, seasonal, lead, and support
features. One shared output head is used across all weeks.

| Forecast | RMSE (mm/day) | MAE (mm/day) | Bias (mm/day) | ACC |
|---|---:|---:|---:|---:|
| Raw FuXi | 5.723 | 3.822 | -0.225 | 0.276 |
| Training-only log-bias anchor | 5.365 | 3.543 | -0.824 | 0.337 |
| Frozen neural corrected forecast | **5.275** | **3.478** | -0.842 | **0.358** |

Relative to raw FuXi, the corrected forecast reduces RMSE by **7.82%** and MAE
by **9.01%**, and raises ACC by **+0.082**. It improves mean ACC and RMSE in
every lead, all three audit years, and all four reported regions. Relative to
the log-bias anchor alone, the neural residual reduces RMSE by **1.69%**, MAE
by **1.83%**, and raises ACC by **+0.021**.

The bias result is negative: absolute pooled bias worsens by 0.617 mm/day
relative to raw FuXi. Mean absolute case bias also rises from 1.122 to 1.324
mm/day. Approximately 80% of the raw-to-corrected RMSE gain comes from the
statistical log-bias anchor and 20% from the neural residual. The model should
therefore be described as improving error and spatial anomaly skill, not as
uniformly improving bias.

The numbers also locate the dry-bias source. Bias changes from -0.225 for raw
FuXi to -0.824 for the log-bias anchor, then only to -0.842 after the neural
residual. Thus nearly all of the added dry tendency is already present in the
anchor; the neural stage does not repair it. The experiment uses train-only
input/target normalization and GroupNorm, not BatchNorm. Fixed target-transform
and loss screens show a log/intensity-objective trade-off, but do not prove
that `log1p` alone causes the problem.

Primary evidence:

- [Technical audit](../studies/fuxi_imd_adapter_benchmark_v1/reports/imd_technical_audit_20260812/IMD_TECHNICAL_AUDIT_REPORT.md)
- [Frozen 2022–2024 result](../studies/fuxi_imd_adapter_benchmark_v1/results/full_context_jjas_2022_2024_job91439/RESULTS.md)
- [Audit manifest](../studies/fuxi_imd_adapter_benchmark_v1/results/full_context_jjas_2022_2024_job91439/manifest.json)

### Best validation-qualified physical-variable candidate

The `physical_full_compact` model adds TCWV, q850, u850, v850, two moisture
fluxes, z500, MSL, and OLR summaries. Against its matched compact control on
blocked 2018–2019 validation, it changes RMSE from 5.4563 to 5.4358, ACC from
0.3271 to 0.3302, and bias from -1.1726 to -1.1431 mm/day. The gain is real
under the declared validation guards, but small: 0.38% RMSE and +0.0031 ACC.

On the reused 2020–2021 exploratory hindcast it obtained RMSE 5.155, MAE
3.350, bias -0.716, and ACC 0.359, compared with raw FuXi at 5.828, 3.778,
-0.101, and 0.253. These are among the best completed numerical India results,
but they are not independent confirmation. The separate 2.54M model reaches
RMSE 5.151 and ACC 0.365 on reused 2020–2021, while the physical model has
slightly lower MAE (3.350 versus 3.356). Neither comparison is a final test.

- [Physical-variable validation result](../bias-correction/results/fuxi_imd_compact_validation_sweep/physical_confirm3_a100_20260811T233520Z/README.md)
- [Exploratory 2020–2021 manifest](../bias-correction/results/fuxi_imd_locked_hindcast_evaluation/physical_full_compact_exploratory_2020_2021_20260812T010224Z/manifest.json)
- [Regularized 2.54M model result](../bias-correction/results/fuxi_imd_attention_climatology_big_allweeks_regularized/full_20260810T042657Z/RESULTS.md)

## Experiment inventory

“Rejected” below means that the controlled experiment failed its declared
promotion guards. It does not prove that the general idea can never work.

### Older standalone `neural_adapter`

| Experiment | Status | Main finding |
|---|---|---|
| Residual U-Net v1 | **Invalid; never cite** | FuXi and IMERG targets were shifted by one day. The saved scores are scientifically invalid. |
| Aligned residual U-Net v2 | Complete corrected re-analysis | On opened 2024 data, raw to U-Net: ACC 0.205 to 0.313, RMSE 3.336 to 2.937, MAE 1.977 to 1.733; bias worsens from -0.047 to -0.545 mm/day. Added value beyond log-bias is mainly MAE. |
| Temporal v3 Huber/hybrid | Complete; failed development gate | Temporal attention improved raw FuXi but did not robustly beat log-bias. |
| Temporal v3 plus member summaries | Complete; branch stopped | W3–W6 ACC gain over log-bias is only +0.0030 and its interval crosses zero; bias becomes drier. |

Sources: [invalid v1](../neural_adapter/RESULTS_V1.md), [aligned v2](../neural_adapter/RESULTS_V2.md), and [v3 development result](../neural_adapter/RESULTS_V3.md).

### Twenty-year FuXi–IMD experiments

| Experiment | Status | What was learned |
|---|---|---|
| Training-only log-bias anchor | Works; essential baseline | Supplies about four-fifths of the total RMSE gain, but creates much of the dry bias. |
| Compact temporal U-Net, shared head | Works; selected | Full regional FuXi context, residual anchoring, GroupNorm, skip connections, and one bottleneck Transformer give consistent RMSE/ACC improvement. |
| Width-24, batch-16 refinement | Validation-qualified | Only compact refinement that passed every declared two-year/lead guard. |
| Full compact physical-variable bank | Validation-qualified | Small but consistent gain over the matched TP/T2M compact control; independent confirmation pending. |
| Shifted-climatology attention | Rejected | Did not beat the fixed normal-climatology control in both validation years. |
| Six independent weekly output heads | Rejected | Tiny pooled RMSE change, but worse ACC/bias, late-lead regressions, and only one of three seeds won. Keep the shared head. |
| Extra member summaries and spread gate | Rejected | Compact quantiles/extreme fraction did not add robust ACC; raw 51-member tensors were not tested. |
| 2.54M-parameter deeper model | Mixed; exploratory only | Numerically a little better, but the increment over the compact model was small and not robust. Capacity alone is not the bottleneck. |
| Factorized 3-D model | Rejected | Low validation objective did not translate into robust physical metrics across years/leads. |
| Input noise and stronger regularization | Rejected | Noise 0.03, more dropout/weight decay, and lower learning rate did not pass the physical guards. |
| Bias-aware/recentered objective | Rejected | Improved pooled RMSE and bias versus the current loss, but worsened MAE and still did not beat raw absolute bias. |
| Heavy-rain weighting | Rejected | Improved heavy-rain errors but badly damaged dry/light RMSE, pooled MAE, and ACC. |
| Equal-regime and wet-event losses | Rejected | Reduced some light-rain errors but worsened heavy rain and pooled skill. |
| Fixed Box–Cox target | Useful diagnostic; not promoted | Power 0.25 modestly improved pooled metrics, but missed the RMSE guard and its bias gain partly came from cancellation between intensity regimes. |
| Learnable Box–Cox target | Implemented; **not run** | Code/tests exist, but there is no GPU experiment result. |
| Train-only affine recalibration | Complete diagnostic; rejected | Did not pass promotion guards. |
| Blocked-OOF affine recalibration | Incomplete/failed | Full job failed in fold 2 with an IMD truth consistency error. Pooled OOF fit values must not be used as independent evidence. |
| Independent 2025 evaluation | Implemented; **untouched** | No frozen selection, access ledger, or result exists. This remains the only final untouched test. |

The validation years 2018–2019 have now supported many screens, so small gains
there carry selection-overfitting risk. Seed-42-only screens are hypothesis
generation. The repeatedly inspected 2020–2021 hindcast is exploratory, and
the already inspected 2022–2024 audit is the strongest generalization evidence
but not a final test.

## What is actually implemented in `ai-quest-global`

The current model is a 111,909-parameter probabilistic residual U-Net:

- input shape: `[batch, 2 periods, 18 channels, 121 latitude, 240 longitude]`;
- periods: D19–25 and D26–32 seven-day total precipitation;
- output: five ERA5 climatological-category probabilities at each grid cell;
- anchor: all-member category counts with Jeffreys smoothing;
- features: five `log(p0)` maps, five `log1p` TP member quantiles, latitude and
  longitude harmonics, valid-date harmonics, period flag, and land fraction;
- architecture: shared-period weights, residual U-Net, two skip levels,
  GroupNorm rather than BatchNorm, SiLU, Dropout2d, circular longitude, and
  bounded latitude padding;
- correction head: zero initialized, so epoch zero exactly reproduces the
  FuXi probability anchor;
- objective/selection: area/land-weighted RPS plus a small correction penalty;
  retain `p0` if the trained network does not beat it on 2019 validation RPS.

The audit verified that all 40 synthetic/unit tests pass. Source inspection
also sees the 13 TB FuXi archive with 2,080 initializations, 51 members, 42 lead
days, and 26 variables. However, its `full_data_verification` attribute is
false, and the configured directory below does not exist:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global
```

Consequently, real ERA5 preprocessing, training, validation, test evaluation,
multi-seed analysis, and calibration evidence are all pending.

## What should be added to AI Quest

Additions are ordered by expected scientific value, not novelty.

1. **Create a real baseline result first.** Build and verify the ERA5 cache,
   train seed 42, and compare uniform climatology, raw `p0`, and the neural
   model on 2019. Do not open 2020–2021 unless the configuration and decision
   gate are frozen.
2. **Add a non-neural calibration baseline.** Fit per-period temperature,
   vector/Dirichlet, or multinomial-logit scaling using training data only. The
   U-Net should demonstrate value over calibrated `p0`, not only raw `p0`.
3. **Use multiple seeds and dependence-aware uncertainty.** Confirm a candidate
   with at least seeds 42–44 and paired bootstrap intervals blocked by
   initialization/year. Grid cells are not independent samples.
4. **Expand probabilistic diagnostics.** Report category-wise reliability,
   sharpness, observed category frequency, RPS/RPSS by period, and Brier score
   by category. A pooled reliability curve can hide category-specific failure.
5. **Run controlled feature ablations.** Compare `p0` only, then TP quantiles,
   calendar/coordinates, and land fraction. This will show where skill comes
   from rather than crediting the complete network.
6. **Add compact ensemble diagnostics before raw member channels.** The current
   `p0` already uses all members. Test entropy, zero-rain fraction, mean,
   standard deviation, and IQR. Feeding 51 ordered member maps is much more
   expensive and imposes an artificial member order.
7. **Test compact physical predictors only after TP is established.** The India
   experiment supports a small ablation using train-normalized weekly summaries
   such as TCWV, q850, u/v850, z500, MSL, OLR, and T2M. Its observed increment
   was modest, so this should be an ablation rather than the new default.
8. **Test light period conditioning.** With only two periods, compare the shared
   model to a small FiLM or period-specific output head. The six-head India
   result warns against immediately duplicating the full head/backbone.
9. **Add regional and regime reporting.** Include tropics/extratropics,
   land/ocean, India, season, wetness, and category-frequency strata. Select on
   global RPS; use strata to diagnose harm rather than to tune repeatedly.
10. **Maintain a machine-readable result ledger.** Every run should record data
    hashes, exact cases, train-only climatology/normalization years, seed,
    checkpoint hash, validation decision, test-access state, metrics, and
    artifact path. Stale or failed Slurm runs must not look like evidence.

Before launching, also correct the Slurm contract: cache preparation is mainly
CPU/I/O work, while current wrappers request GPU resources and do not encode
the workspace's unreliable-node exclusions. This is execution hygiene rather
than a model experiment.

The first real study has only 208 training initializations from two years.
Spatial grid cells do not turn this into hundreds of thousands of independent
forecast cases. More blocked years or a carefully declared cross-validation
analysis are more valuable than a Swin Transformer, diffusion model, much
greater depth, all 26 variables, or all 51 raw member maps at this stage.

Do not switch the target to daily rainfall inside this experiment. The Quest
task is probabilistic weekly totals, and RPS is aligned with that objective. A
daily-to-daily model would be a separate experiment with a separate target,
architecture, aggregation rule, and evaluation protocol.

## Minimum promotion gate for the first real AI-Quest result

A neural candidate should advance only if it:

- beats raw `p0` and the simple train-only calibration baseline on 2019 RPS;
- improves both forecast periods rather than winning by pooled cancellation;
- shows no severe regional or category reliability failure;
- is stable across at least two of three fixed seeds; and
- is frozen, hashed, and documented before one evaluation on 2020–2021.

Until those conditions are met, the correct AI-Quest result statement is:

> The global probability adapter is implemented and unit-tested, but it has
> not yet produced a real validation or test result.

## Permission boundary

FuXi's published terms prohibit competition use without written permission
from the authors. This repository must remain offline research code until
written permission explicitly covers competition use. No result, probability
file, or trained weight should be submitted before that boundary is resolved.
