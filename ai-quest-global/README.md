# Global TP probability adapter

## Frozen 20260813 precipitation workflow

The current competition workflow is documented in
[`PRECIP_QUEST_20260813.md`](PRECIP_QUEST_20260813.md). It is separate from
the older exploratory adapters below. Its primary entry points are:

- `build_precip_cache.py`: two-case smoke, 64-task atomic historical parts,
  WeatherBench2/official cross-check, and all-2,080-parts finalization;
- `train_precip.py`: target-only, six-week TP-only, and full physical-context
  neural ablations with an epoch-zero `p0` fallback;
- `select_precip_candidate.py`: Uniform, raw/calibrated `p0`, raw/calibrated
  Debias++, neural promotion gates, frozen selection, and one-time test open;
- `predict_precip_issue.py`: strict 51-member 20260813 operational inference
  or a separately labelled no-live-FuXi emergency Uniform fallback; and
- `submit_precip_issue.py`: AI-WQ-package 3.29-native files and permission-,
  credential-, and literal-confirmation-gated upload.

The compact model interface is:

```text
context_x [B,6,23,121,240]
target_x  [B,2,18,121,240]
p0        [B,2,5,121,240]
output    [B,2,5,121,240]
```

Width 16 has 151,573 parameters. The six context and two target tokens share
a spatial encoder and one bottleneck Transformer; only target tokens and
target skips enter the decoder. All required node blacklists are embedded in
the new Slurm files under `slurm/`.

The current run record is intentionally blocked from upload until official
ERA5/land inputs, AI-WQ-package 3.29, registered credentials, and written FuXi
competition permission are present. See `run_records/20260813/`.

## Legacy exploratory workflow

This is a small global rainfall-calibration experiment. It converts the FuXi
ensemble into five ERA5 climatological categories and lets a 111,909-parameter
U-Net make a limited spatial correction to those probabilities.

It predicts two seven-day totals:

- D19–25
- D26–32

The output for each period is `probability(5, 121, 240)` on the global
1.5-degree grid.

Storage stays simple: the 13 TB hindcast and the compact prepared cache are
Zarr stores, the selected weights are one PyTorch `.pt` file, evaluation
tables are CSV, figures are PNG, and each local forecast period is NetCDF.

## What the model does

For each period, every FuXi member is summed over seven days and placed into
one of five ERA5 rainfall categories. The member counts receive a small
Jeffreys correction, `(count + 0.5) / (members + 2.5)`, so every category has
non-zero probability. This gives the FuXi ensemble anchor `p0`. The U-Net
predicts only a correction:

```text
FuXi weekly ensemble ──> smoothed category anchor p0
                                      │
18 input maps ──> small U-Net ──> correction logits
                                      │
              softmax(log(p0) + correction) ──> 5 probabilities
```

The final layer starts at zero, so the new network initially gives exactly
the FuXi `p0` anchor. If training does not beat that anchor on 2019 validation
RPS, `train.py` keeps the zero-correction checkpoint.

The 18 inputs are five `log(p0)` maps, five `log1p` FuXi-member rainfall
quantiles, latitude and longitude sine/cosine, valid-date sine/cosine, a period
flag, and land fraction. There is no VQ-VAE and no extra atmospheric variable
in this first experiment.

## Three-variable temporal prototype

`prototype_multivariable.py` is a separate, runnable extension for the three
gridded Quest targets: precipitation (`pr`), near-surface temperature (`tas`),
and mean sea-level pressure (`mslp`). It does not change the existing
precipitation cache, trainer, or checkpoint format.

The joint prototype contract is:

```text
features  [case, period=2, channel=38, latitude=121, longitude=240]
p0        [case, variable=3, period=2, quintile=5, latitude, longitude]
target    [case, variable=3, period=2, latitude, longitude]
```

For each variable, the input has five `log(p0)` maps and five ensemble-member
quantiles. Precipitation quantiles use `log1p`; temperature and pressure stay
in K and Pa before training-only standardization. Eight shared static maps
(latitude/longitude cycles, valid-date cycle, period flag, and land fraction)
bring the total to 38 channels. The nine-field physical predictor bank from
the India bias-correction ablation is intentionally not in this minimal
contract; adding it would require a named 47-channel cache and an ablation
showing that it helps globally.

The U-Net keeps circular longitude padding and bounded latitude padding. At
the bottleneck, one small Transformer mixes the two periods independently at
each spatial location. Three zero-initialized heads then add residual logits
to the three anchors:

```text
three ensemble anchors + 38 maps
              │
       shared global U-Net
              │
  two-period bottleneck attention
              │
       pr / tas / mslp heads
              │
 softmax(log(p0) + correction)
```

This is temporal mixing between D19–25 and D26–32. It is not an
autoregressive `x_t -> x_t+1 -> x_t+2` forecast and it does not see all six
forecast weeks. True all-week context needs a new source/cache contract.

Run the end-to-end synthetic contract on CPU:

```bash
python prototype_multivariable.py \
  --output-dir /tmp/quest-multivariable-prototype \
  --init-date 2026-08-15 \
  --device cpu
```

It writes six local NetCDFs (`pr`, `tas`, and `mslp`, two periods each), an
untrained checkpoint, and `prototype_manifest.json`. The inputs are synthetic
and the zero heads deliberately reproduce `p0`; these files prove shapes,
period mixing, probability normalization, and export only. They are not skill
evidence or submission-ready forecasts.

`multivariable.py` contains the real reusable contract helpers: smoothed
anchors, observation categories, the 38-channel feature builder, official
variable-specific masks, and an RPS loss that normalizes each variable before
averaging them equally. A trained version still requires a new three-variable
cache, training-split-only normalization, validation checkpoint selection,
and an untouched test comparison against both uniform climatology and `p0`.

### Synthetic optimization smoke and curves

The separate smoke trainer verifies actual optimization on independent
synthetic train and validation cases:

```bash
python train_multivariable_smoke.py \
  --device cpu \
  --epochs 30
```

It selects the lowest-validation-RPS checkpoint with epoch zero retained as
the `p0` fallback. If the required improvement is missed, artifacts are still
written but `minimum_improvement_passed` is false and the selected system is
identified explicitly. The run directory contains `history.csv`,
`summary.json`, a strictly reloadable `best.pt`, and these figures:

- `training_history.png`: objective, train/validation RPS, per-variable RPS,
  and weighted modal-quintile accuracy;
- `modal_quintile_accuracy.png`: the expanded diagnostic hit-rate curve;
- `validation_rps_by_variable.png`: uniform, `p0`, and selected-model RPS;
- `validation_reliability.png`: `p0` versus model calibration for every
  variable and lead.

Every plot is labelled synthetic. Modal-quintile accuracy is not weather ACC:
meteorological anomaly correlation requires continuous weekly forecasts,
observations, and climatological means. RPS/RPSS remain the primary metrics for
this five-category probability model.

## Real FuXi + IMERG integration smoke

`prepare_imerg_smoke.py` provides a small real-data vertical slice for
precipitation while the official Quest ERA5 files are unavailable. It reads the
native 51-member global FuXi archive, forms D19–25 and D26–32 accumulations, and
uses IMERG Final V07B with a fixed 2002–2016 calendar climatology. Selection is
by whole initialization and blocked year: 2017–2018 train, 2019 validation, and
2020 test. This path is exploratory pretraining only; IMERG is not the official
Quest ERA5 target. Its final static input is IMERG/climatology support frequency,
stored separately as `spatial_support_fraction`; it is not called or scored as
an official land mask.

Build the 16-case native-grid cache:

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python prepare_imerg_smoke.py \
  --output runs/imerg_real_smoke_20260815_v2/cache/fuxi_imerg_exploratory_2017_2020.npz \
  --cases-per-year 4
```

Train the spatial control selected by validation RPS:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/raj.ayush/.conda/envs/fuxi/bin/python train.py \
  --cache runs/imerg_real_smoke_20260815_v2/cache/fuxi_imerg_exploratory_2017_2020.npz \
  --run-dir runs/imerg_real_smoke_20260815_v2/training_spatial_control_40ep \
  --device cpu --batch-size 2 --max-epochs 40 --patience 8 --seed 42
```

An otherwise identical run with `--period-attention --attention-dropout 0.0`
tests two-period bottleneck mixing. On this deliberately tiny smoke, the simpler
spatial control was slightly better and is therefore the selected architecture:

| System | 2019 validation RPS | Skill vs `p0` | 2020 test RPS | Test skill vs `p0` |
|---|---:|---:|---:|---:|
| Raw FuXi `p0` | 1.032078 | 0.00% | 1.025068 | 0.00% |
| Spatial control | **0.891531** | **13.62%** | **0.881230** | **14.03%** |
| Two-period attention | 0.896901 | 13.10% | 0.883121 | 13.85% |
| Uniform climatology | — | — | **0.830543** | — |
| Validation-calibrated `p0` | 0.815875 | 20.95% | 0.815803 | 20.41% |
| Validation-calibrated spatial | **0.805398** | **21.96%** | **0.802819** | **21.68%** |

The uncalibrated selected model improves the raw FuXi anchor but remains worse
than uniform climatology (RPSS `-0.0610`). A single coefficient fitted on 2019,
`P = 0.6664 * uniform + 0.3336 * spatial`, gives retrospective 2020 RPSS
`+0.0334`, positive on all four dates and both leads. This calibration was
proposed after 2020 had already been inspected, so it is a promising ablation,
not untouched test evidence. Reproduce it with `calibrate_uniform_blend.py`.
These IMERG scores use all-grid cosine-area weighting because the
official Quest land mask is absent; they are ocean-dominated and not comparable
to official precipitation scoring. This proves real I/O, optimization, checkpoint selection, strict
reload, and evaluation; it is not evidence of competitive rank or official
Quest skill. The run contains training/validation RPS curves, explicitly named
modal-quintile diagnostic accuracy (not meteorological ACC), lead-wise RPS/RPSS,
reliability, spatial-improvement maps, CSVs, and checkpoints under
`runs/imerg_real_smoke_20260815_v2/`.

A validation-only continuation selected epoch 67 at RPS `0.878554`, another
`1.46%` below the 40-epoch checkpoint. Validation then plateaued while training
continued to improve, so additional epochs alone are unlikely to help. The
uncalibrated continuation still trails uniform (`0.826983`). Its command,
curve, checkpoint, and reload result are recorded in
`runs/imerg_real_smoke_20260815_v3/VALIDATION_ONLY_RESULTS.md`.

## Full ERA5 pipeline split (17/2/1)

- Train: 2002–2018 (17 years; 1,768 cases)
- Validation/model selection: 2019–2020 (2 years; 208 cases)
- Frozen one-time test: 2021 (1 year; 104 cases)

The versioned cache contract fits normalization only on 2002–2018. Model and
calibration choices use 2019–2020; 2021 remains sealed until that choice is
hashed and frozen. The evaluation land fraction is the public WeatherBench2
ERA5 field on the exact 1.5-degree grid, not the password-gated Quest file, so
results must be labelled as a reproducible ERA5-mask experiment rather than an
official competition score. The separate IMERG smoke above does not include
2021.

The archive contains 104 fixed calendar initialization dates per year. All-date
hindcasts are useful for model fitting, but they do not reproduce the Quest's
Thursday issue-date population and must stay labelled exploratory. A final
competition-like evaluation should use `evaluate.py --thursday-only` (or an
explicit official schedule) and report it separately. Using 2017 onward also
avoids evaluating on years included in the published FuXi base-model training
period, which ended in 2016.

ERA5 categories use 100 historical samples: the preceding 20 years and date
offsets `[-4, -2, 0, 2, 4]`. Equality with a boundary enters the upper
category. Cells with four identical precipitation boundaries are excluded.

## Environment

The existing environment with the compatible Zarr v2 reader is:

```bash
PYTHON=/home/raj.ayush/.conda/envs/fuxi/bin/python
```

For official ERA5 retrieval, install the current Quest package and ECBox
client if they are not already present:

```bash
$PYTHON -m pip install 'AI-WQ-package==3.29'
$PYTHON -m pip install sites-toolkit \
  -i https://get.ecmwf.int/repository/pypi-all/simple
```

Do not put the ECBox token in a script. Load it for the current shell:

```bash
read -s AI_WQ_PASSWORD
export AI_WQ_PASSWORD
```

## Run

First verify the existing FuXi archive:

```bash
$PYTHON prepare_data.py inspect-fuxi
```

Download all three official weekly variables for the target years and the
preceding climatology years:

```bash
$PYTHON prepare_data.py download-era5 \
  --variables pr tas mslp \
  --start-year 1997 \
  --end-year 2025
```

The current production cache builder remains precipitation-only. Retrieving
`tas` and `mslp` stages the official files, but a real joint three-variable
cache/trainer must still be implemented and validated before those variables
can be submitted.

Build the compact cache. The job is resumable by initialization and is the
slow part because it reads the large source archive:

```bash
sbatch slurm/prepare.sbatch
```

Train one model on an available non-MIG GPU:

```bash
sbatch slurm/train.sbatch
```

For a direct foreground run:

```bash
$PYTHON train.py --device cuda
```

`train.py` writes `best.pt`, `history.csv`, `figures/loss_curve.png`, and the
complete run configuration under:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/runs/
```

Evaluate uniform climatology, the FuXi `p0` anchor and the selected checkpoint:

```bash
$PYTHON evaluate.py \
  --cases /storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/fuxi_tp_2017_2021.zarr \
  --years 2020 2021 \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/test \
  --device cuda \
  --batch-size 8 \
  --thursday-only
```

Or submit the same evaluation as a GPU job:

```bash
sbatch slurm/evaluate.sbatch RUN_DIRECTORY/best.pt RUN_DIRECTORY/test
```

The test directory contains `evaluation_summary.csv`, a lead-wise RPS/RPSS
figure, pooled reliability curves, two spatial maps of `p0 RPS - model RPS`,
and `india_summary.csv` for the 5–40°N, 65–100°E India box. Positive values on
the spatial maps mean the neural correction improved on `p0`.

Create two local NetCDF files for one prepared initialization:

```bash
$PYTHON predict.py \
  --case /storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/fuxi_tp_2017_2021.zarr \
  --init-date 2020-06-02 \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/local_forecast
```

For a new global FuXi run, first convert its raw member files into one small
prediction case using the training-only normalization saved in the cache:

```bash
$PYTHON prepare_data.py prepare-case \
  --raw-directory /path/to/global_run/raw \
  --init-date 2026-07-28 \
  --output /tmp/fuxi_2026-07-28_quest_case.npz

$PYTHON predict.py \
  --case /tmp/fuxi_2026-07-28_quest_case.npz \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/local_forecast
```

### Offline official-format precipitation preview

`quest_submission.py` builds the two official-shaped precipitation DataArrays
and a provenance manifest, then reopens and strictly validates the serialized
files. It contains no upload, authentication, FTP, ECBox, or HTTP code. It
requires a literal Thursday case, matching cache/checkpoint/calibration
fingerprints, and the current Thursday–Sunday UTC window unless
`--allow-historical` is explicitly used.

```bash
$PYTHON quest_submission.py \
  --case /path/to/matching_prepared_case.npz \
  --checkpoint RUN_DIRECTORY/best.pt \
  --init-date 20260813 \
  --team REGISTERED_TEAM \
  --model REGISTERED_MODEL \
  --originating-centre PACKAGE_ORIGIN \
  --expver PACKAGE_EXPVER \
  --calibration-json RUN_DIRECTORY/uniform_blend_calibration.json \
  --output-dir RUN_DIRECTORY/offline_submission_preview
```

The checked historical example under
`runs/imerg_real_smoke_20260815_v2/offline_submission_preview_20190718/`
uses an unmodified Thursday date and passes the AI-WQ-package 3.29 structural
checks. Its manifest says `submission_ready=false`: the identities are
placeholders, the target is exploratory IMERG rather than official ERA5, the
date is outside the live window, and FuXi competition permission is unresolved.
For a real upload, populate the credentialed package-created template and use
the official API only after project-lead approval.

`prepare-case` reads only TP from leads 19–32, uses every available member,
and checks the run metadata date when it is present.

Run all synthetic checks without reading the large archive:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python train.py --smoke --device cpu --max-epochs 1 --run-dir /tmp/quest-tp-smoke
```

## Files

- `prepare_data.py`: ERA5 retrieval and FuXi-to-Zarr preprocessing
- `prepare_imerg_smoke.py`: real FuXi/IMERG exploratory precipitation cache
- `data.py`: weekly/category/feature rules
- `model.py`: the complete neural network
- `train.py`: RPS training and validation selection
- `metrics.py`: official-style RPS and RPSS
- `evaluate.py`: baseline comparison and plots
- `predict.py`: offline local NetCDF export
- `quest_submission.py`: strict offline official-format precipitation preview
- `multivariable.py`: three-variable anchors, features, masks, and RPS loss
- `prototype_multivariable.py`: native-grid synthetic temporal prototype
- `train_multivariable_smoke.py`: synthetic optimization, validation, and plots
- `slurm/`: small preprocessing, training, and evaluation launch scripts

IMERG is used only by the explicitly labelled exploratory smoke above. The
official pipeline keeps the ERA5 category definition and must not substitute
IMERG scores for Quest validation.

## Permission boundary

FuXi's published terms prohibit competition use without written permission
from the authors. All code here is therefore offline research code: it contains
no upload or submission function. Do not submit FuXi-derived probabilities or
weights until that written permission explicitly covers the competition.
