# All-season FuXi ensemble calibration experiment

Status: implementation contract, written before any experiment job is
submitted. This plan defines a new probabilistic study. It does not modify the
frozen deterministic adapter results, and it does not authorize or read the
sealed 2025 target.

This plan supersedes the probabilistic-stop rule in
`plan/RESEARCH_SPRINT_20260822.md` only for this separately versioned
experiment. The earlier deterministic evidence and its method-selection
decisions remain frozen.

## Question and claim boundary

Can a compact, exchangeable neural calibrator use the complete FuXi member set
to improve the calibration and sharpness of six-week rainfall forecasts over
India while retaining the deterministic spatial-skill gain?

The experimental unit is one initialization, not one ensemble member. The 51
members provide 51 related realizations of the same forecast case; they do
not turn one observed weather event into 51 independent observations. Members,
all six leads, and every grid cell from an initialization must remain in the
same data split and bootstrap block.

The intended paper claim is conditional on the fixed comparisons below. A
negative result for location or spread learning is an admissible result. No
architecture, loss, threshold, or split will be changed after development-test
metrics are opened.

## Data contract

Forecast source:

`/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_global_2002_2021.zarr`

Required source properties:

- root status is `complete`;
- 2,080 initializations: 104 per year for 2002--2021;
- 51 exchangeable members per initialization;
- 42 daily leads and native `tp` mean-rate values;
- six non-overlapping seven-day means after the source unit is converted to
  mm/day;
- the established 27 x 27 India grid and fixed land/validity mask;
- IMD rainfall is the training and primary verification target.

The versioned member cache has shape `(2080, 51, 6, 27, 27)`, dtype
`float32`, and units mm/day. The cache builder must inventory the Zarr before
extraction, publish per-initialization parts atomically, finalize only after all
2,080 records are present, and write metadata plus SHA-256 provenance. A model
run must refuse an incomplete or mismatched cache.

This experiment uses all seasons. There is no JJAS-only training arm in the
primary experiment because it would discard most independent initializations.
Season and day-of-year information may be deterministic covariates, but no
observation from validation or development-test years may be used to fit
normalization, climatology, or model parameters.

## Fixed temporal split and leakage control

| Role | Initialization years | Permitted use |
|---|---|---|
| Training | 2002--2017 | Fit normalization and all learned parameters |
| Validation | 2018--2019 | Early stopping and selection among declared configurations |
| Reused development test | 2020--2021 | One fixed comparison after validation choices are frozen |
| Untouched final | 2025 | Out of scope and sealed |

The 42-day outcome-window purge is exact:

- assign a case by its initialization year;
- retain a training initialization only when `initialization + 42 days` is
  before `2018-01-01`;
- retain a validation initialization only when `initialization + 42 days` is
  before `2020-01-01`;
- retain the declared 2020--2021 development-test initializations; their truth
  window may extend into 2022.

For the audited 104-start-per-year archive this yields 1,652 training cases,
196 validation cases, 208 development-test cases, and 24 purged boundary
cases. These counts are executable assertions, not estimates.

Thus, a six-week target window from a left split cannot cross into the next
split. This is a one-sided outcome-window purge/embargo, not a symmetric
plus/minus-42-day exclusion.

FuXi `lead_day` is one-based and IMD rainfall is end-date labelled. Therefore
the weekly verification contract is W1 = initialization+1 through +7 and W6 =
initialization+36 through +42. Initialization day is never included in W1.

The 2020--2021 period has been used by earlier exploratory work, so it must be
labelled a reused development test rather than independent confirmation.

## Forecast representation

For member rainfall `x_m >= 0`, work in stabilized rainfall space

```text
u_m = log(1 + x_m)
u_bar = mean_m(u_m)
d_m = u_m - u_bar.
```

The full model predicts a location correction `delta_mu` and a positive spread
multiplier `s` and reconstructs every member:

```text
u_tilde_m = u_bar + delta_mu + s * d_m
x_tilde_m = max(0, exp(u_tilde_m) - 1).
```

The same member encoder is used for every member and aggregation over members
is symmetric. Consequently, permuting input member order must only permute the
corresponding outputs; ensemble probabilities and summary fields must remain
unchanged. The location and log-spread heads are zero-initialized, so every
neural configuration is the exact raw-ensemble identity at epoch zero. Unit
tests must enforce both properties.

Training may randomly draw 16 of the 51 members at each optimization step as
set-valued data augmentation and to reduce GPU cost. Validation, early-stopping
scores, development-test evaluation, and exported forecasts always use all 51
members. Member subsampling never changes the number of independent cases.

## Fixed methods and ablations

All methods receive identical cases, targets, masks, area weights, and full-set
evaluation.

| Method | Purpose |
|---|---|
| `raw_fuxi` | Unchanged 51-member probabilistic reference |
| `moment_calibration` | Training-only, month/lead/grid location plus regional spread calibration |
| `summary_only` | Location--spread correction from fixed ensemble summaries, without a learned member-set encoder |
| `location_only` | Permutation-invariant set model learns location; spread multiplier is fixed to one |
| `location_spread` | Main set model jointly learns location and positive spread |

The baseline methods are evaluated automatically; the three neural
configurations are the only trained ablation arms. Neural seeds are fixed at
42, 43, and 44. They measure optimization variability and are not ensemble
members. The full run uses at most 100 epochs, validation early stopping with
patience 15, batch size 8, and 16 randomly selected training members per step.

No JJAS-only fit, independent-member MSE fit, KL loss, larger architecture,
extra meteorological variable, or post-test affine correction belongs to this
experiment. Those require a new hypothesis and output root.

The `moment_calibration` baseline is frozen as follows. For every lead and
verification-midpoint calendar month, calculate each training case's ensemble
mean in log1p space. At each grid cell, fit the mean residual between IMD
log1p truth and that raw ensemble log mean. Shrink a cell with `n` valid
training cases toward the area-weighted regional residual using strength 10:

```text
delta_grid = (n * mean_residual_grid + 10 * mean_residual_region) / (n + 10).
```

After applying that location correction, fit one regional spread factor for
the same lead/month group:

```text
s = sqrt(
      area-and-case-weighted corrected squared error
      / area-and-case-weighted raw ensemble log variance
    ),
s = clip(s, 0.25, 4.0).
```

Apply it memberwise without changing ranks:

```text
u_tilde_m = u_bar + delta_grid + s * (u_m - u_bar).
```

Only 2002--2017 training cases may determine `delta_grid`, regional residuals,
or `s`; validation and development-test targets are forbidden during fitting.

## Objective

The primary training score is area-weighted finite-ensemble CRPS:

```text
CRPS(F, y) = mean_m |x_m - y|
             - (1 / (2 M^2)) sum_m sum_n |x_m - x_n|.
```

This simultaneously rewards closeness to the observation and an appropriate
ensemble distribution. It replaces independent member-wise MSE, which would
encourage all members to collapse toward the conditional mean. KL divergence
is not used because each case has one verifying realization rather than an
observed target distribution.

The loss is averaged over valid grid cells with the fixed India-area weights,
then over leads and initialization cases. The training history must store both
training and full-51-member validation CRPS for every epoch, seed, and neural
configuration. Validation CRPS determines early stopping. No development-test
quantity may influence stopping or selection.

## Fixed evaluation matrix

The primary comparison is `location_spread` versus `raw_fuxi` on all retained
2020--2021 development-test cases. Report every method both pooled and
separately for weeks 1--6. Also report a secondary DJF/MAM/JJA/SON-by-week
table so an apparent all-season gain cannot be driven by one wet season alone.

Probabilistic metrics:

- CRPS and CRPSS, where
  `CRPSS = 1 - CRPS_method / CRPS_raw_fuxi` and positive is favorable;
- ensemble spread and spread/RMSE relationship;
- Brier score and forecast reliability for rainfall thresholds 1, 5, 10, and
  20 mm/day;
- empirical 50%, 80%, and 90% interval coverage and mean interval width;
- randomized rank histogram diagnostics for tied and zero rainfall.

Deterministic safeguards use the 51-member ensemble mean:

- RMSE;
- MAE;
- signed bias and absolute bias;
- anomaly correlation coefficient (ACC).

ACC anomalies use an equal-year, centred 31-day daily IMD climatology fitted
only on 2002--2017. The daily climatology is mapped to each verification date
and averaged into the same six forecast weeks before spatial anomaly
correlation is computed. Validation and development-test observations never
enter this reference climatology.

All metrics use the same observation mask and fixed area weights across
methods. Threshold probabilities are member fractions, not probabilities from
a separately fitted classifier. Missing observations change only the scoring
mask and must never change a forecast.

The mandatory weekwise table contains at least:

```text
split, method, seed, week, n_initializations, n_valid_cells,
crps, crpss_vs_raw, rmse, mae, bias, absolute_bias, acc,
ensemble_spread
```

Threshold and coverage tables may be long-form, keyed respectively by
`threshold_mm_day` and `central_interval`. Report each seed and a clearly
labelled across-seed summary; do not treat three seeds as independent weather
samples.

Paired uncertainty must resample initialization cases with all six weeks,
members, and grids attached. If moving blocks are used, order by initialization
date and keep blocks within a year. Do not bootstrap members or grid cells as
independent samples.

## Output and figure contract

Each run uses a fresh output directory. Smoke output is non-scientific and
cannot be promoted or merged into a full result. The trainer owns the following
mandatory artifacts:

```text
manifest.json
history/training_history.csv
metrics/weekwise_metrics.csv
metrics/case_metrics.csv
figures/training_loss_curves.png
figures/weekwise_metrics.png
```

`training_loss_curves.png` shows train and validation CRPS by epoch, with one
trace per configuration and seed. `weekwise_metrics.png` compares every method
with raw FuXi for weeks 1--6 and includes CRPS/CRPSS, RMSE, MAE, bias, and ACC;
Brier and interval-coverage panels or companion files must retain their
threshold/nominal-level labels.

The manifest must record at least:

- `experiment = fuxi_allseason_ensemble_calibration_v1`;
- mode, smoke/scientific status, completion status, timestamp, and output path;
- input cache metadata and manifest hashes;
- exact retained initialization dates and split counts;
- 42-day purge rule;
- units, grid, mask, area weighting, thresholds, and interval levels;
- model/configuration, seed, parameter count, optimizer, stopping epoch, and
  checkpoint hashes;
- Python executable and package versions, CUDA/GPU identity, command line,
  source-file hashes, and Slurm identifiers;
- confirmation that all 51 members were used for every reported metric;
- confirmation that no 2025 target was opened.

## Execution sequence

No job was submitted when this contract was written.

1. Inventory the native Zarr and run a bounded cache pilot.
2. Launch the resumable 260-task CPU extraction array at conservative
   concurrency six.
3. After the entire array succeeds, run one dependency-gated finalizer and
   verify the NPY plus both provenance sidecars.
4. Launch a fresh GPU smoke output. It must initialize CUDA, exercise all three
   configurations, use the 51-member evaluation path, write the mandatory
   artifacts, and finish with a `complete` smoke manifest.
5. Only after that manifest passes the launcher's preflight, launch the full
   three-seed experiment to another fresh output path.
6. Validate artifact existence, manifest fields, split dates, metric
   finiteness, member-permutation invariance, and the no-2025 assertion before
   interpreting results.

Example commands are intentionally separated so job identifiers are inspected
before being used as dependencies:

```bash
sbatch --array=0 slurm/build_fuxi_allseason_member_cache.sbatch pilot

sbatch slurm/build_fuxi_allseason_member_cache.sbatch build
```

After the pilot succeeds, launch the full array. After that array succeeds,
override its array to one task for dependency-gated finalization:

```bash
sbatch --array=0 --dependency=afterok:<BUILD_ARRAY_JOB_ID> \
  slurm/build_fuxi_allseason_member_cache.sbatch finalize
```

Then launch smoke and full training with absolute cache/output paths. The full
launcher additionally requires the completed smoke manifest:

```bash
sbatch slurm/run_fuxi_allseason_ensemble_calibration.sbatch \
  smoke /absolute/path/cache.npy /absolute/path/fresh-smoke-output

sbatch slurm/run_fuxi_allseason_ensemble_calibration.sbatch \
  full /absolute/path/cache.npy /absolute/path/fresh-full-output \
  /absolute/path/fresh-smoke-output/manifest.json
```

The exact job IDs, output roots,
manifest hash, runtime, selected validation configuration, and numerical result
table must be appended here only after verified execution.
