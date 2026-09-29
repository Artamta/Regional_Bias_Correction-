# Locked 2022--2024 operational-era ensemble audit

Status: **frozen before launch, 2026-08-22**

This protocol governs one post-hoc, all-season retrospective audit of the
already validation-selected `base_42k` FuXi neural ensemble adapter. It is an
evaluation only. It may not train, fine-tune, select, blend, or promote a
different model after the 2022--2024 results are seen.

## Fixed scientific question

Does the compact adapter selected using 2002--2017 training and 2018--2019
validation retain probabilistic and deterministic skill when it is applied,
without alteration, to the later 50-member operational-era FuXi archive?

The only methods are:

1. `raw_fuxi`: the exact 50 operational-era members; and
2. `base_42k`: each of the locked seed 42/43/44 checkpoints, with headline
   values formed by averaging **scores** across seeds.

Parameters, adjustment fields, and predictions are never averaged across
seeds. No other capacity, loss, baseline, blend, or checkpoint can enter this
audit.

## Immutable model receipt

- Capacity run:
  `resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z`.
- Capacity manifest SHA-256:
  `2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c`.
- Frozen validation winner: `base_42k`, 42,434 parameters.
- Optimization seeds: 42, 43, and 44.
- Training years: 2002--2017.
- Selection years: 2018--2019 only.
- The full capacity artifact inventory, selection, architecture/configuration,
  source snapshots, and checkpoint hashes must pass before forecast data are
  opened.

## Exact data whitelist and 2025 quarantine

Forecast access is limited to the three direct paths ending in:

- `tp/common_1p5/2022.zarr`;
- `tp/common_1p5/2023.zarr`; and
- `tp/common_1p5/2024.zarr`.

The evaluator must construct those paths from the literal whitelist
`{2022, 2023, 2024}`. It may not glob, list, discover, probe, stat, open, or
otherwise access a `2025.zarr` forecast or IMD store. The fact that the parent
model-run label contains the text `2020_2025` does not authorize opening the
2025 year store.

Training-only IMD access is limited to 2002--2017. Verification IMD access is
limited to 2022--2024. A 2024 initialization is eligible only when
`initialization + 41 days <= 2024-12-31`; this makes 2025 verification truth
unnecessary and forbidden. Cross-year verification for 2022 and 2023 is
allowed because the required truth remains inside the 2022--2024 whitelist.

Every eligible initialization from the three forecast stores is retained in a
full run. There is no seasonal filter or arbitrary scientific subsample.
Smoke-only thinning is non-scientific and must be identified in the manifest.

## Alignment and distribution-shift checks

- Operational members have shape `[init, 50, 6, 27, 27]` and units
  `mm day-1`.
- The stored `ensemble_mean_weekly` product must be reproduced exactly using
  the archive writer's canonical operation,
  `np.nanmean(members, axis=member, dtype=float64).astype(float32)`. The
  post-cast float32 arrays must be bitwise identical at every retained
  case/lead/grid point. The receipt records the exact mismatch count and
  checked count, plus the pre-round absolute residual and its maximum fraction
  of one stored float32 ULP as diagnostics; no fixed absolute tolerance is an
  acceptance criterion for this product.
- Lead weeks must be exactly 1--6, with each seven-day mean verified against
  initialization-day offsets 0--41. Every selected case/member/lead is checked
  by streaming native daily fields in bounded eight-case chunks; the receipt
  records the checked counts, streamed daily content hash, and exact maximum
  weekly-versus-daily difference.
- Latitude and longitude values and their ordering must exactly match the
  canonical hindcast cache and scoring support.
- All 50 members must be available for every retained initialization.
- Raw and calibrated forecasts use identical cases, members, leads, truth,
  climatology, grid, and area/support weights.
- The operational ensemble has 50 members while the training/validation
  hindcast ensemble has 51. This member-count/domain shift is part of the
  audit question and must remain explicit in every manifest/readout.
- The 2002--2017 IMD climatology and normalization are reconstructed using the
  frozen training contract and applied unchanged to 2022--2024. Before any
  forecast score, the reconstructed float32 lead means must equal
  `[0.9928932189941406, 0.9928548336029053, 0.9927545189857483,
  0.992607057094574, 0.9922419190406799, 0.9919546246528625]` with SHA-256
  `0fdceee6daa9bf0469967f9ac3e7df5a9ffb0bedfc53930a4dcf8184b89fc911`,
  and the lead standard deviations must equal
  `[0.8253104090690613, 0.8253679871559143, 0.8254558444023132,
  0.8255149126052856, 0.8256960511207581, 0.825825035572052]` with SHA-256
  `051a34ffe3842bdd6089db1d270b4a8dc1d403afa7b43ea917e4a4c5f9a19926`.
  These values are independently reproduced from the earlier frozen neural
  normalization receipt; both exact values and dtype/shape/content hashes are
  postflight-gated.
- Content hashes are recorded for every loaded forecast, observation, date,
  coordinate, availability, support, and derived scoring array.

### Prelaunch observation-coverage amendment (2026-08-22)

The read-only integration preflight, before any raw or adapter score was
computed, found that IMD 2023-10-12 has reduced remapping coverage at five
coastal cells; two of those cells have no observation that day. All other
training/audit days retain the frozen support. The original assumption that
`observation_fraction` is time-invariant therefore cannot be used for this
later-period audit.

The v3 audit contract retains the v2 coverage amendment: it derives every
daily deviation from the loaded 2022--2024 fraction arrays and rejects the run
unless the complete identity
set is exactly 2023-10-12 at grid indices `(10,6)`, `(10,7)`, `(11,6)`,
`(11,7)`, and `(12,7)` (coordinates 24N/69E, 24N/70.5E, 22.5N/69E,
22.5N/70.5E, and 21N/70.5E), with exactly two zero-coverage cells. The full
296-case audit must derive exactly 12 affected case/lead blocks and a minimum
dynamic-to-frozen weight ratio of `6/7`. The fixed 12-case plumbing smoke
must include initialization 2023-10-12, derive its affected W1/day-1 block,
and obtain the same minimum ratio. Thus the smoke exercises the real coverage
exception.

The model context and normalization remain on the frozen 171-cell
2002--2017 support. Evaluation uses, for each case/lead/cell, the seven-day
mean of the actual daily observation fractions multiplied by cell area. The
weekly truth is the observation-fraction-weighted mean of available daily
values. This reduces to the canonical weekly mean and frozen spatial weight
whenever daily coverage is constant. Raw and calibrated forecasts receive the
identical dynamic case/lead weights. Case/lead support counts and weight sums
must be retained, and the manifest must report the minimum/maximum support.

This amendment handles missing verification data; it does not change the
model, dates, methods, seeds, training climatology, score definitions, or
selection. It was frozen in response to a failed data preflight, not after
viewing forecast performance.

### Pre-score stored ensemble-mean amendment (2026-08-23)

The first source-locked full attempt (Slurm job 110577) stopped safely while
loading 2022 forecast inputs, before model application or any forecast score,
because the audit compared an unrounded float64 member mean against the
archive's stored float32 mean under a fixed `2e-6 mm day-1` tolerance. The
benchmark writer actually computes a float64 `nanmean` and then casts to
float32. Since half a float32 ULP grows with field magnitude, the absolute
tolerance can reject a correct heavy-rain product.

A read-only, bounded eight-case scan of all 296 eligible initializations
checked 1,294,704 case/lead/grid means. Reproducing the writer's float64
`nanmean` followed by a float32 cast matched the stored product bit for bit at
all 1,294,704 values (zero mismatches). Five unrounded residuals exceeded the
old `2e-6` cutoff; the largest was `3.2806396461637632e-6 mm day-1` at
initialization 2022-05-09, W1, 27N/97.5E, and every pre-round residual was at
most one-half of a stored float32 ULP. A direct float32 reduction is not an
acceptable substitute.

The v3 contract therefore gates exact post-cast bitwise identity and rejects
even a one-ULP corruption. It retains the pre-round maximum and maximum
fraction-of-ULP only as provenance diagnostics. This amendment changes no
forecast, case, model, score, bootstrap, or selection rule, and requires a
fresh v3 smoke before another full launch.

## Frozen metrics and uncertainty

Report finite-ensemble CRPS, ensemble-mean RMSE, MAE, centred spatial ACC,
signed bias, central 50/80/90% coverage, ensemble spread, and pooled
spread--skill ratio. Preserve case/lead rows, W1--W6 summaries, pooled
summaries, per-year summaries, verification-season summaries, per-seed
scores, ranks, and event-reliability diagnostics.

For CRPS, RMSE, MAE, ACC, and signed bias, use a paired two-stage bootstrap:

1. resample the three years with replacement; and
2. within each sampled year, resample circular 13-initialization blocks.

One initialization is the statistical unit. Its 50 members and all six lead
weeks stay grouped. Use 10,000 draws in a full run. The small number of year
clusters is a limitation and must not be hidden.

## Evidence label and relationship to existing evidence

The only permitted label is **post-hoc 2022--2024 operational-era
retrospective audit; no retraining or selection**. It is later-period
generalization evidence, not an untouched final test and not a prospective
operational verification.

This audit complements rather than overwrites the existing frozen 100-start
2022--2024 deterministic/raw-identity audit. The earlier audit used a selected
case set, ensemble summaries, additional deterministic comparators, and a
different adapter lineage. This protocol uses every eligible all-season
initialization, the individual 50-member distribution, probabilistic scores,
and the capacity-selected `base_42k` adapter. Their metrics must not be pooled
or described as repetitions of the same experiment.

The 2025 forecast/target remains quarantined for the separately frozen
one-time final-control workflow. Neither favorable nor unfavorable results
from this audit authorize opening it.

## Launch and stop rules

1. Run a fresh smoke output first and issue an atomic Slurm gate receipt only
   after its manifest and required artifacts pass the v3 post-run audit. That
   audit checks exact case/method/seed/lead identities, raw--calibrated pairing
   and finiteness, recomputes headline seed-score arithmetic and all principal
   summaries from case rows, and deterministically recomputes the paired
   bootstrap table.
2. A full job requires that exact smoke manifest and gate receipt. It writes to
   a second fresh path and receives its own atomic receipt after validation.
   The smoke-to-full gate binds every source snapshot hash, including this
   protocol, all imported evaluation sources, the driver, and the launcher;
   any live-source change requires a new smoke.
3. Generic GPU jobs exclude `cn2,cn3,cn4,cn15,cn16,cn17` and log GPU identity.
4. Scheduler failure may be rerun unchanged to a fresh path. A scientific
   contract change requires a new dated protocol and experiment identifier.
5. Do not change a metric, date rule, model, seed, threshold, support, or
   reporting method after inspecting audit results.
