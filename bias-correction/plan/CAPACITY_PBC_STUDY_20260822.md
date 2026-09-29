# All-season capacity and FuXi-PBC study

Status: **PBC V2 amended before launch; V1 smoke invalidated, 2026-08-22**

This document governs two separate follow-ups to the completed all-season
ensemble-calibration V1/V2 experiments:

1. a validation-only width ablation of the exact exchangeable
   location--spread architecture; and
2. a static-hindcast FuXi adaptation of the categorical PBC baseline.

They share the canonical FuXi member cache, IMD support, case partitions, and
2025 quarantine, but they answer different questions. Their results must not
be combined into one selection score.

## Scientific questions

### A. Exact-model capacity

Does widening the existing permutation-invariant location--spread calibrator
produce a material, temporally robust validation-CRPS improvement over the
42k-parameter control?

Only the member-encoder and Conv3D backbone widths change. The correction
family, depth, inputs, CRPS loss, member subsampling, optimizer, splits, and
seeds remain fixed. This is a **width ablation**, not a depth or architecture
search.

### B. FuXi-PBC baseline

How much categorical probability skill is obtained from a transparent
train-fitted combination of probability debiasing and issuance-available
rainfall persistence, relative to the raw FuXi empirical CDF and its component
methods?

This implementation is a static-hindcast adaptation, not an exact claim of
reproducing a rolling/prequential PBC system. That divergence must remain in
the manifest and paper text.

## Immutable data boundary

- Forecast members: canonical 51-member, six-week, 27 x 27 all-season cache.
- Full-cache SHA-256:
  `2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70`.
- Native-source fingerprint:
  `655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae`.
- Fit years: 2002--2017, after the existing 42-day boundary purge.
- Validation years: 2018--2019, after the same purge.
- Retrospective development years: 2020--2021. The capacity experiment must
  not slice or score these years. PBC may score them only after all categorical
  definitions, window choices, ridge strength, and blend weight are frozen.
- Archive counts: 1,652 train, 196 validation, 208 development, 24 embargo.
- Smoke counts: 32 train, 16 validation, and, for PBC only, 16 development.
- 2025 is sealed. Neither workflow may open a 2025 forecast or observation
  store, directly or through a helper.
- One initialization, containing its complete member set and six lead weeks,
  remains the statistical resampling unit.

The 2018--2019 and 2020--2021 periods have already been used during model
development. These studies are therefore post-hoc/retrospective evidence and
cannot create an untouched-test claim.

## Frozen capacity candidates

| Candidate | Member width | Backbone width | Parameters | Role |
|---|---:|---:|---:|---|
| `small_20k` | 4 | 16 | 19,618 | smaller-capacity control |
| `base_42k` | 8 | 24 | 42,434 | exact V1 control |
| `medium_158k` | 16 | 48 | 157,570 | widened candidate |
| `large_294k` | 32 | 64 | 293,762 | largest candidate |

One always-on diagnostic arm is trained beside those four candidates:
`summary_matched_43k` uses `mode="summary_only"`, backbone width 26, and
43,058 parameters. Its constructor member width is 8 but is unused because
the member encoder is absent. This approximately parameter-matched control
tests whether member-set information helps independently of parameter count;
it is never eligible for capacity selection.

Candidate order is fixed as shown. Every candidate uses:

- `EnsembleLocationSpreadCalibrator`, `mode="location_spread"`;
- the same single residual Conv3D block and two-layer pointwise member encoder;
- seven unchanged context channels;
- area-weighted finite-ensemble CRPS;
- batch size 8 and 16 randomly sampled training members;
- all 51 members for checkpoint validation;
- AdamW at `2e-4`, weight decay `1e-4`, dropout `0.05`;
- at most 100 epochs, early-stopping patience 15, and seeds 42/43/44.

The full run writes validation artifacts and a frozen selection before any
later-period evaluation exists. No candidate may be selected from training
loss, parameter count alone, or 2020--2021 behavior.

## Frozen FuXi-PBC definition

All thresholds and target-derived quantities are fitted from 2002--2017 IMD
only.

- Quintile cumulative thresholds are the train-only calendar-conditioned
  0.2/0.4/0.6/0.8 quantiles. They define the headline RPS/RPSS evaluation.
- Semidecile thresholds are 0.05, 0.10, ..., 0.95. Because IMD zero ties make
  the nominal lower 5% event degenerate, only the upper 5% extreme is a valid
  headline extreme score; degeneracy must be reported rather than hidden.
- Preserve two RPSS references. The **scientific headline** is tie-aware
  train-empirical-climatology RPSS under
  `normalized_informative_positive_cut_v2`. Exact-equal persisted float32
  thresholds define one physical cut; q=0 is the deterministic empty strict-CDF
  event and is excluded. Each grid row is the mean squared CDF error over its
  distinct q>0 cuts. An all-zero row is unscored, the spatial denominator is
  recomputed from the common threshold-only mask, and every case/lead must
  retain positive scoring weight.
- A projected, zero-anchored nominal-reference RPSS remains a representation
  sensitivity. It is not called equiprobable or paper-compatible when rainfall
  thresholds tie.
- A separate secondary Guan-Methods formula sensitivity sums squared errors
  over all four nominal quintile slots on the common persisted-q80>0 mask. Its
  RPSS denominator is the literal `[0.2,0.4,0.6,0.8]` vector. That vector is a
  formula reference, not a physically valid tied-threshold forecast CDF. Its
  aggregate numerator and denominator are pooled with retained q80 area weight;
  it is neither the headline nor a claim of numerical paper reproduction.
- `raw_fuxi_categorical` is the member empirical CDF, using the frozen strict
  `<` event convention.
- `debias_plus_plus` adds a train-only calendar-window mean of observed-event
  indicator minus raw CDF, then clips and projects cumulative probabilities to
  the physical threshold geometry. Within exact-tie runs, nominal-slot values
  are averaged; q=0 is forced to zero and excluded; equal-weight PAVA is applied
  only to distinct positive cuts; and the result is expanded with bit-identical
  values at tied thresholds. Candidate half-spans are exactly 14, 28, and 35
  days and are selected separately by lead using 2018--2019 primary V2
  validation RPS only.
- `persistence_plus_plus` is a train-fitted ridge model using an intercept,
  train empirical climatology, raw CDF, and exact W1-rainfall lag indicators
  from initialization minus 7 and 14 days. Those lag observations must be
  timestamped strictly before forecast issuance. Ridge is fixed at `1e-3`.
- `pbc_combined` is the fixed 0.5/0.5 blend of the separately projected
  Debias++ and Persistence++ CDFs, followed by the same equality-aware
  positive-cut projection.
- The centered climatology window is fixed at 31 days.

The four methods above are all reported. `pbc_combined` is the predeclared
primary PBC method; component results may not be suppressed if the combination
is weaker.

## Launch protocol

Both workflows use a fresh output path and the same two-stage sequence:

1. submit `smoke` with a fresh output directory;
2. inspect the assigned GPU and log;
3. require a complete, hash-consistent smoke manifest and required artifacts;
4. submit `full` with a second fresh output directory and the completed smoke
   manifest as an explicit gate;
5. retain failed staging/log artifacts and never reuse their output path.

Launchers must exclude `cn2,cn3,cn4,cn15,cn16,cn17` on generic GPU
partitions. If moved to `GPU-AI` or `GPU-AI_prio`, they must additionally
exclude `gpu2`. Every allocation must record `nvidia-smi`, CUDA availability,
the Python executable, and package versions before scientific code starts.

The PBC computation is analytic/CPU-oriented; its GPU request exists only to
keep this audited launch path under the requested common CUDA/node preflight.
It must not be described as GPU-dependent.

Example commands, with deliberately user-chosen fresh paths:

```bash
sbatch slurm/run_allseason_capacity_ablation.sbatch \
  smoke /absolute/path/smoke-or-full-cache.npy /absolute/path/fresh-capacity-smoke

sbatch slurm/run_allseason_capacity_ablation.sbatch \
  full /absolute/path/full-cache.npy /absolute/path/fresh-capacity-full \
  /absolute/path/fresh-capacity-smoke/manifest.json

sbatch slurm/run_allseason_pbc_baseline.sbatch \
  smoke /absolute/path/full-cache.npy \
  /absolute/path/resultsv2/fuxi_allseason_pbc_baseline_v2/fresh-pbc-v2-smoke

sbatch slurm/run_allseason_pbc_baseline.sbatch \
  full /absolute/path/full-cache.npy /absolute/path/fresh-pbc-full \
  /absolute/path/resultsv2/fuxi_allseason_pbc_baseline_v2/fresh-pbc-v2-smoke/manifest.json
```

Submitting a full job without the matching completed smoke receipt is a
contract error. A smoke result is non-scientific even if its metrics look
favorable.

All artifacts and receipts under the former
`resultsv2/fuxi_allseason_pbc_baseline/` V1 namespace are invalid for promotion:
they predate the equality-aware probability geometry, q=0 anchor, normalized
informative-positive scoring, q80/q95 masks, and revised bootstrap. A V2 full
must be gated by a newly completed V2 smoke under
`resultsv2/fuxi_allseason_pbc_baseline_v2/`; the launcher must reject every V1
manifest or receipt.

## Capacity integrity and promotion gates

The launcher must first establish:

- complete full manifest, canonical candidates and seeds, and exact cache
  identity;
- 1,652/196 selected train/validation cases;
- no test metrics consulted and no 2025 target opened;
- required artifacts and source snapshots agree with their streamed SHA-256
  hashes;
- `base_42k` has 42,434 parameters and reproduces the frozen V1 validation
  control closely enough to rule out a changed implementation.
- the nonselectable `summary_matched_43k` control is present for every seed,
  has 43,058 parameters, and remains outside the width-candidate selector.

The experiment's validation selector is the authoritative within-run choice.
For **paper promotion beyond the 42k control**, the selected wider candidate
must additionally satisfy all of these predeclared interpretation gates:

1. mean three-seed validation CRPS improves on `base_42k` by at least 0.5%;
2. neither 2018 nor 2019 validation CRPS is worse than `base_42k`;
3. at least two of three matched seeds improve on their `base_42k` counterpart;
4. among candidates within 0.25% relative CRPS of the best eligible score,
   prefer the smaller parameter count;
5. all checkpoints and adjustment fields are finite and the full-member
   validation path is used.

If any gate fails, retain `base_42k` and report the capacity experiment as a
negative or inconclusive result. A qualifying wider model is only a frozen
candidate for later independent evaluation; this reused validation screen
alone cannot replace the published V1 evidence.

## PBC integrity, reporting, and promotion gates

The launcher must establish:

- experiment `fuxi_allseason_pbc_baseline_v2`, scoring contract
  `normalized_informative_positive_cut_v2`, complete full manifest, exact
  four-method order, exact cache identity, and
  1,652/196/208 selected split counts;
- selected Debias++ span for every lead comes only from the declared
  14/28/35 validation candidates;
- persistence lag timestamps are strictly pre-issuance and no verification
  truth enters a predictor;
- all CDFs are finite, bounded in [0,1], monotone, bit-identical at exact-equal
  thresholds, and exactly zero at q=0 after projection;
- tie/zero diagnostics and the degenerate lower-tail decision are retained;
- diagnostics enumerate duplicate quintile and semidecile thresholds by
  lead/grid support rather than treating repeated zero thresholds as distinct
  effective categories;
- normalized informative-positive-cut RPSS against training-empirical
  climatology is the scientific headline; projected-nominal and summed
  nominal-cut/q80-positive paper-formula sensitivities are present under
  unambiguous names and cannot replace it;
- every case/lead has positive retained area weight for primary q>0 scoring,
  q80-positive paper sensitivity, and q95-positive upper-tail scoring;
- paired intervals use two-stage test-year resampling followed by circular
  13-initialization blocks within each sampled year;
- the complete derived Persistence++ lag bundle (values, source indices, and
  window bounds), plus weekly IMD truth, observation fraction, and scoring
  weights, are content-hashed with named dtype/shape/C-order-byte receipts; a
  full run must match both successful V2 smoke hashes so daily or weekly IMD
  drift is rejected;
- no 2025 target is opened;
- case-level quintile RPS and paired initialization-block uncertainty are
  available, in addition to aggregate tables.

All four methods are valid baselines and must be reported regardless of sign.
Use the following language gates:

- call `pbc_combined` **competitive with raw FuXi** only if its pooled RPSS is
  positive and the paired 95% block-bootstrap interval excludes zero;
- call the combination **better than both components** only if its paired RPS
  improvement over both Debias++ and Persistence++ has intervals excluding
  zero;
- otherwise call it a transparent baseline or negative component result, not
  a promoted method;
- never describe 2020--2021 as independent or untouched;
- never compare its categorical RPS numerically with neural CRPS. A paper
  claim against the neural adapter requires scoring frozen neural members on
  these exact stored categorical thresholds, cases, support, and RPS contract.

The static-hindcast divergence from rolling/prequential PBC remains a named
limitation even when the result is positive.

## Stop rules

- Do not vary capacity widths, network depth, dropout, learning rate, member
  subsampling, PBC spans, ridge, climatology window, blend weight, event
  convention, or category definitions after seeing full-run metrics.
- Do not open 2022--2025 to rescue either result under this protocol.
- Do not launch a replacement full run into an existing directory.
- A scheduler/node failure may be rerun unchanged to a new output path; a
  scientific-contract failure requires a new dated plan rather than an
  unrecorded patch.

## Minimum paper artifacts

Capacity:

- validation case metrics and by-year summary;
- per-seed training histories and selected checkpoints;
- frozen `selection.json`;
- manifest, cache provenance, source snapshot, and artifact hashes.

PBC:

- one row per method/initialization/lead for normalized informative-positive-cut
  RPS, projected-nominal sensitivity, and retained-weight-aware paper-formula
  q80-positive sensitivity;
- weekwise and seasonal metrics;
- component-ablation metrics;
- upper-tail semidecile and probability-bias diagnostics;
- date/lead/grid threshold-equality, q=0, and retained-support diagnostics;
- paired two-stage year/circular-block effects;
- content-bound daily-IMD-derived Persistence++ lag and weekly-IMD/scoring
  provenance;
- fitted baseline bundle and exact scoring support;
- manifest, cache provenance, source snapshot, and artifact hashes.
