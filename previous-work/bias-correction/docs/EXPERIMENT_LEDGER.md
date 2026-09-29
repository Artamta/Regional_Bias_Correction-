# FuXi–IMD experiment ledger

Updated: 2 September 2026

This is the authoritative human-readable index for model status and evidence
level. A low metric in a run directory does not override the promotion
decision recorded here and in that run's selection manifest.

## Current decision

The frozen raw-identity TP/T2M adapter is now the strongest deterministic
neural candidate. It was selected on 2018--2019 without a fitted log-bias
reconstruction anchor, improved on the legacy anchored adapter in the fixed
2022--2024 audit, and retained favorable error and ACC effects against a
separate 2024 rain-gauge target. This is still development evidence rather
than an untouched temporal final.

The raw-mean-preserving diagnostic is not the primary method. It repaired the
large dry-bias trade-off under IMD verification and slightly improved RMSE,
but worsened MAE there and degraded RMSE, MAE, ACC, and absolute bias against
the station target. This target dependence is scientifically useful, but it
rules out describing the constraint as a universal calibration fix.

No later loss, target-transform, affine-calibration, or global-initialization
candidate passed all predeclared guards. The matched global-pretraining
experiment completed and selected scratch: global initialization was worse on
pooled RMSE, MAE, signed bias, and ACC, and zero of three pretrained seeds
achieved a lower best India validation composite loss than matched scratch.
This negative result does not change the completed E2/E3 method sets.

The 2025 final initialization year remains sealed. Its raw-identity-versus-raw
selection is frozen and its storage-incapable synthetic CUDA preflight passed;
no access ledger or final result exists.

### Parallel all-season probabilistic branch

**Withdrawn from scientific use on 26 August 2026.** The immutable v1 training
and capacity-selection snapshots constructed end-labelled IMD targets with
offsets 0--41, while FuXi native lead days and the benchmark require offsets
1--42. Every checkpoint, fitted moment/PBC product, downstream operational
audit, metric, figure, and paper package that depends on that branch is
historical only. The files remain immutable for provenance, but none of their
numerical results may support a manuscript claim. Corrected retraining uses
the frozen `location_spread` architecture under
`fuxi_allseason_ensemble_calibration_v2_aligned` and writes only to
`resultsv3/`; the 2025 target remains sealed.

The corrected branch now has accepted **retrospective** evidence. Seed 43 was
selected only by 2018--2019 validation CRPS, then scored on the operational
2020--2024 bridge. Across six lead-specific 2022--2024 all-India
valid-midpoint JJAS cohorts (100 cases per lead; 600 case-lead rows), with
bootstrap draws synchronized by within-year ordinal, the neural adapter has
CRPS 2.645294 versus 3.156954 for raw FuXi and 2.762956 for train-only moment
calibration. This is 16.2074% CRPS skill versus raw (paired 95% interval
14.8087--17.6945%) and 4.2585% versus moment (2.9866--5.7035%). ACC is
.321966 versus .295873 for raw, a .026093 increase (.008680--.043603). The
signed-bias delta versus raw is .020124 mm/day (-.207024--.246136), so no
signed-bias improvement is established. The lead-specific date sets have 70
dates in their intersection and 130 in their union; this is not one shared
100-date cohort.

The independent training-alignment audit passed. It reconstructed all 2,080
weekly IMD tensors from `+1...+42`, exactly reproduced the
train/validation/reused-development/embargo IDs and counts
(1,652/196/208/24), verified the 171-cell support, and matched the native
2017-11-17 FuXi lead-day 1--42 row to the cache byte-for-byte. Its weekly-target
hash is `31f371c2...cca4`. The frozen run did not preserve a logical hash or
copy of the target tensor it consumed, so the audit explicitly limits its
claim: it proves the source/current-source reconstruction and contracts, not
historical equality to an unrecorded tensor.

An independent table-level audit passed after reconstructing summaries,
paired intervals, and the story gate, verifying 276 files and all 45,450
case-metric rows. Raw FuXi matches the benchmark evaluator on 15,150 rows
within the frozen tolerances. A separate prediction-level audit then reopened
raw members and daily IMD, independently rebuilt aligned targets, all three
50-member forecasts, LOYO climatologies, and every metric without importing
the production scoring implementations. It reproduced all 45,450 rows with
zero failures; maximum absolute differences were (1.78\times10^{-15}) for
CRPS, (1.11\times10^{-16}) for ACC, and (2.27\times10^{-13}) for ensemble
variance. The retrospective component of the gate passes,
but the final headline/prospective gate remains pending the explicitly
authorized one-time sealed-2025 point-direction check. Neither scoring nor
audit opened a 2025 target; the latest target label opened was 2024-12-30.

The frozen adapter also passed the planned **unfitted IMERG sensitivity** on
the same 505 starts. Across the corresponding six lead-specific 2022--2024
all-India valid-midpoint JJAS cohorts, neural CRPS is 2.766890 versus 3.151507
for raw FuXi (12.2042%
skill; block-16 interval 10.5092--14.0175%) and 2.825593 for train-only moment
calibration (2.0775%; 0.6697--3.6625%). Its ACC increase versus raw is
.013586, but the interval crosses zero (-.005676--.033500); this is reference
robustness for probabilistic improvement, not a second confirmed ACC gain.
Signed-bias change versus raw is also unresolved. All 15,150 common raw-FuXi
deterministic rows reproduce the benchmark, and no 2025 observation was
opened; the latest IMERG target label was 29 December 2024.

The exact-common-midpoint sensitivity retains 85 dates per lead (35/35/15 in
2022/2023/2024) and also passes: 15.8651% CRPS skill versus raw
(14.3550--17.5204%), 4.0606% versus moment (2.6139--5.7070%), and ACC delta
.031306 (.012323--.050610). On the full 169-case-per-lead IMD cohort, the
calibrated FuXi mean has lower RMSE than the fixed six-system MME at every lead;
paired intervals exclude zero at W1--W6. Its ACC is higher with intervals
excluding zero at W1 and W3--W6, while W2 is unresolved.

The table-only IMD homogeneous-region seasonal extension uses the same frozen
2020--2024 bridge cases and compares raw FuXi with the location-and-spread
adapter at W1--W6. CRPS improvement is resolved in 14/24 JF, 24/24 MAM,
24/24 JJAS, and 20/24 OND region--lead cells, with no resolved CRPS
degradations. RMSE is uniformly favorable only in JJAS; ACC is mixed outside
JJAS. These are pointwise retrospective intervals without multiplicity
correction, not a new confirmatory test. No 2025 target was opened.

A separate following-Thursday ERPAS sensitivity is complete for 33
valid-midpoint JJAS periods per lead in 2023--2024. Under one common IMD
1991--2019 anomaly normal, INDRA-S2S v1 ACC is
.671/.372/.285/.164 at W1--W4, versus .570/.309/.203/.080 for raw FuXi and
.472/.203/.043/.012 for ERPAS. Conditional paired block intervals for INDRA
minus each baseline exclude zero at all four leads. On the same cases and
support, INDRA RMSE is 4.406/5.701/6.010/6.142 mm/day, versus
5.118/6.286/6.579/6.736 for raw FuXi and 5.629/6.548/7.084/7.094 for ERPAS;
all paired INDRA-minus-baseline RMSE intervals are below zero. This is
limited-period retrospective evidence only: FuXi is initialized 24 hours
after ERPAS, the cohort contains only two seasons, and the
common-observation-normal ACC contract differs from the main benchmark's
method-specific forecast climatologies.

## Evidence inventory

| Experiment | Canonical artifact | Evidence role | Status |
|---|---|---|---|
| All-season 51-member location-and-spread calibration | `../resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z` | Misaligned v1 historical artifact | Withdrawn: targets used offsets 0--41 |
| All-season capacity selection | `../resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z` | Misaligned v1 historical artifact | Withdrawn: training imported the 0--41 target builder |
| All-season CRPS--MSE loss sensitivity | `../resultsv2/fuxi_allseason_hybrid_loss_ablation/full_final_20260822T141844Z` | Misaligned v1 historical artifact | Withdrawn |
| Static India/IMD PBC V2 | `../resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z` | Misaligned v1 historical artifact | Withdrawn; thresholds and fits must be rebuilt |
| Seven-method common-support categorical comparison | `../resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z` | Misaligned v1 historical artifact | Withdrawn |
| All-season operational-era ensemble audit | `../resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z` | Audit of a misaligned v1 checkpoint | Withdrawn |
| All-season operational-era categorical comparison | `../resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z` | Audit of misaligned v1 products | Withdrawn |
| Operational categorical paper supplement | `../presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z` | Package derived from misaligned v1 products | Withdrawn |
| CCAI NeurIPS 2026 anonymous source package | `../presentation/deliverables/ccai_neurips2026_submission_source_20260823` | Package derived from misaligned v1 products | Withdrawn; do not submit |
| Corrected all-season ensemble calibration v2 | `../resultsv3/fuxi_allseason_ensemble_calibration/full_20260825T224711Z` | Correct +1...+42 training/validation/development contract | Complete; seed 43 selected by validation CRPS, manifest `a98bf491a41c7245ef2ee0904e4882eb0e3ca3081db2b0291b1f0fa1669fd5df` |
| Independent training-alignment audit | `../resultsv3/fuxi_allseason_training_alignment_audit/audit_full_20260826T005845Z` | Independent targets, splits, support, cache, and native-lead reconstruction | Passed with explicit unpersisted historical-target-hash limitation; receipt `6188197bc2679bcce602e5d5fe9a2adb1e60b23d1f23d5a8ae1939126aba3a07` |
| Corrected operational bridge inference | `../resultsv3/india_s2s_probabilistic_bridge/inference_full_20260825T233121Z` | Frozen predictions for 517 forecast starts; 505 scoreable without 2025 truth | Complete; inference manifest `c3d4fb825e8549a00ee393e3b522d8b8e9bb289338bbea383ea693bceb797eca` |
| Corrected operational bridge scoring | `../resultsv3/india_s2s_probabilistic_bridge/scoring_full_20260825T234808Z` | Retrospective 2020--2024 IMD benchmark and synchronized-ordinal 2022--2024 story gate | Complete accepted retrospective evidence; manifest `4ab303a3fa0025bfb2b8654bf34770091f117df18f236c0c724d8e1078450943` |
| Independent table-level corrected-score audit | `../resultsv3/india_s2s_probabilistic_bridge/audit_full_20260826T000133Z` | Independent table, interval, provenance, and safety reconstruction | Passed; audit receipt `d7c822e0c09528b443e1dddd83872aab2894d82da5dedce6b4d0f1fbc3807bcc` |
| Independent prediction-level audit | `../resultsv3/india_s2s_probabilistic_bridge/prediction_audit_full_20260826T002645Z` | Independent truth/member/climatology/metric reconstruction for all saved rows | Passed; zero failures on 45,450 rows, receipt `0f176fd18fef1387be32e1d96a045beaa5e6c313f8deb65ab0f80d1484b67168` |
| Unfitted IMERG observation sensitivity | `../resultsv3/india_s2s_probabilistic_bridge/imerg_sensitivity_full_20260826T002644Z` | Same frozen forecasts and 505 starts against period-start IMERG | Complete; CRPS improvement robust, ACC increase unresolved; manifest `0d127b1b75b31179bc48a6a1755978da43085891afbae478016a5a04ee1efe7d` |
| Calibrated FuXi regional sensitivity | `../resultsv3/india_s2s_calibrated_regional_sensitivity/full_20260826T012159Z` | Four IMD regions, three methods, six leads; exact frozen interval rows | Complete; CRPS/RMSE skill resolved in 24/24 region--lead cells, ACC in 13/24; pointwise intervals, manifest `7ec5c68d0cfd1778388d3fbe8dca65d855ae621b53f3ff748cfb659f171a73f2` |
| IMD regional and seasonal comparison | `../resultsv3/india_s2s_regional_seasonal_comparison/full_20260902T000001Z` | Four IMD homogeneous regions, JF/MAM/JJAS/OND, W1--W6; table-only sensitivity from the accepted bridge | Complete retrospective sensitivity; CRPS broadly favorable, RMSE and ACC mixed outside JJAS, no 2025, manifest SHA-256 `099037881e3b58548dc9ca5fcf5b825aecfcce09a7f37787ccebd2778cb6e003` |
| Corrected categorical PBC-inspired baseline | `../resultsv3/fuxi_allseason_pbc_baseline_v3_aligned/full_20260826T011400Z` | Static train-only categorical thresholds, Debias++, Persistence++, and projected combination on reused all-season 2020--2021 development data; not a rolling-PBC reproduction | Complete and supplementary only; nonconventional normalized informative-positive-cut quintile RPS, explicit +1...+42 guard, no 2025, manifest `040945005740180fdd8e1e5fffe685585ab3da689d25c1111960fc42f3f729c7` |
| Exact-common and calibrated-versus-MME comparisons | `../resultsv3/india_s2s_paper_comparisons/full_v2_20260826T001721Z` | Exact-midpoint sensitivity and paired deterministic paper payoff | Complete; provenance-fixed manifest `3b754c7024537e56691f56fa8f6932dd8ae40f6b358716cb3fffa8e9051c85da` |
| Paper release bundle | `../resultsv3/india_s2s_paper_release_bundle/full_v3_20260826T012700Z` | Contract-complete Tables 1, 2A, and 2B plus comparison, audit, limitation, and safety tables in CSV/Markdown/LaTeX | Complete; 47 outputs and 66 upstream bindings verified, manifest `ac7c01633044abf33bf6d27c7b7e0b12bb03b8fdf98ee71c91c713f5f9bced98` |
| Paper figure package | `../resultsv3/india_s2s_paper_figures/full_v4_20260826T011100Z` | Hash-bound data and PDF/PNG Figures 1--4 plus all-season rainfall and temperature Figures S1--S2 | Complete and visually inspected; manifest `60c0d8589adba1ff109a761ec37391415950e89b52052b04025d09a77f054ce1`, visual receipt `edc0b45469555d44379470e46829de9141292a82cefb8250a289dd66885fb1c4` |
| ERPAS limited-period supplement | `../../studies/india_s2s_verification_v2/results/conference_figures/Figure_4_erpas_limited_period_extension.pdf` | Separate 31-case 2023--2024 June--September-initialization matched-valid-time FuXi--ERPAS sensitivity at W1--W4 on a distinct 22x22, 169-cell IMD support | Complete and supplementary only; not numerically comparable to the main valid-midpoint curves; method audit `63743db88079f68ccf564a68fa09314321e061707074061ca93069f704434cdb`, validated figure `fda837f49c2de4e48114b3a103b68e99345b5bf4eca2ebcdede91ec550434d8d`, paired intervals `026968a75e0ff6d843479f660dd83eb9b53a4af12d4dcda0cb3f26eff16fe1cf` |
| ERPAS--raw FuXi--INDRA matched-valid-time ACC/RMSE sensitivity | `../resultsv3/india_s2s_erpas_corrected_acc_rmse_sensitivity/full_v1_20260827T024543Z` | Following-Thursday, 33 valid-midpoint JJAS cases per lead in 2023--2024; aligned +1...+28 IMD targets, one common 1991--2019 IMD normal for ACC, and common weighted support for RMSE | Complete limited-period retrospective evidence; calibrated FuXi improves ACC and RMSE against both comparators at W1--W4 under paired intervals, FuXi is 24 h newer, and ERPAS ends at W4 |
| Full-context compact TP/T2M adapter | `../results/fuxi_imd_full_context_compact_allweeks/full_20260811T152024Z` | 2018–2019 selection | Frozen current control |
| Compact physical-variable bank | `../results/fuxi_imd_compact_validation_sweep/physical_confirm3_a100_20260811T233520Z` | 2018–2019 validation, three seeds | Qualified, not independently confirmed |
| Physical-variable locked hindcast | `../results/fuxi_imd_locked_hindcast_evaluation/physical_full_compact_exploratory_2020_2021_20260812T010224Z` | Reused 2020–2021 | Exploratory only |
| Frozen anchored-only source audit | `../../studies/fuxi_imd_adapter_benchmark_v1/results/full_context_jjas_2022_2024_job91439` | Prior frozen 2022–2024 development audit and source for the matched E2 evaluation | Legacy source evidence; superseded as the headline comparison by canonical E2 |
| Raw-identity/no-fitted-log-bias adapter | `../resultsv2/fuxi_imd_no_log_bias_ablation/full_20260822T010749Z` | 2018–2019 three-seed selection; reused 2020–2021 evaluation | Qualified and carried into the completed fixed audit |
| Canonical raw-identity matched audit | `../resultsv2/fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z` | Fixed 2022--2024, 100 starts, no retraining | Complete canonical development audit; all 16 artifact hashes verified |
| Frozen station-target sensitivity | `../resultsv2/fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z` | Fixed 2024 gauge-derived cell target, 30 starts × 6 leads | Complete frozen secondary external-target sensitivity; all 14 artifact hashes verified |
| Global patch pretraining versus scratch | `../results/fuxi_imd_global_pretraining_comparison/execution_full_parallel_20260822a/full_20260822T040953Z` | 2002–2015 global fit, 2016–2017 global validation; matched 2018–2019 India comparison | Complete negative component result; scratch selected, manifest SHA-256 `37b6dac32a9fc2be799651b8e8577cac247d144caec379d1a3fdfce0417c1f5d` |
| Global-comparison mean-preservation follow-up | `../results/fuxi_imd_global_pretraining_followups/full_parallel_20260822a` | Artifact-only post-hoc 2018–2019 diagnostic | Both scratch and pretrained projections rejected; scientifically ineligible for promotion, manifest SHA-256 `e4748243bf8a57b34dc4460af97ce57cbdcfd63d39daebe36e9458356b0510bc` |
| Bias-aware anchor/loss 2×2 | `../results/fuxi_imd_bias_aware_validation_sweep/full_20260813T135600Z` | Reused 2018–2019 validation | No candidate qualified; reference retained |
| Heavy-tail weighting | `../results/fuxi_imd_tail_weight_validation_sweep/screen_seed42_cn14_20260813T152500Z` | Seed-42 screen | Rejected |
| Intensity/regime losses | `../results/fuxi_imd_intensity_loss_validation_sweep/screen_seed42_cn13_20260814T_loss_v1` | Seed-42 screen | Rejected |
| Fixed Box–Cox targets | `../results/fuxi_imd_target_transform_validation_sweep/screen_seed42_cn14_20260814T_transform_v1` | Seed-42 screen | Exact log target retained |
| Learnable Box–Cox target | No result directory | Implemented only | Not run; no evidence |
| Train-only affine calibration | `../results/fuxi_imd_train_affine_calibration/full_cn14_20260813T155500Z` | Development diagnostic | Not promoted |
| Blocked-OOF affine calibration | `../archive/blocked_oof_failure_20260813.tar.gz` | Intended leakage-safe development | Failed entering fold 2; not evidence |
| Independent 2025 evaluation | `../resultsv2/raw_identity_independent_2025_sealed/selection.json` | Final untouched test; selection SHA-256 `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6` | Frozen and synthetic CUDA preflight passed; no access ledger/result; not opened |

The physical-variable model is not the default for 2025 because its nine
additional FuXi variables are absent from the current operational archive.
Do not fill missing variables with zeros.

## Best defensible result

Five frozen forecasts were evaluated on the exact same 100 JJAS starts from
2022--2024, with no retraining or retuning:

| Forecast | RMSE (mm/day) | MAE (mm/day) | Bias (mm/day) | ACC |
|---|---:|---:|---:|---:|
| Raw FuXi | 5.723 | 3.822 | -0.225 | 0.276 |
| Training-only log-bias anchor | 5.365 | 3.543 | -0.824 | 0.337 |
| Frozen anchored adapter | 5.275 | 3.478 | -0.842 | 0.358 |
| Frozen raw-identity adapter | 5.239 | **3.453** | -0.858 | **0.364** |
| Raw-mean-preserving raw identity | **5.227** | 3.538 | **-0.216** | 0.360 |

Relative to raw FuXi, raw identity reduces RMSE by 8.45%, reduces MAE by
9.65%, and increases ACC by 0.0889. The paired year-stratified circular-block
interval for its RMSE improvement is 0.383 to 0.587 mm/day. Relative to the
legacy anchored adapter, it improves RMSE by 0.036 mm/day (interval 0.014 to
0.059) and MAE by 0.024 mm/day (interval 0.008 to 0.041).

The raw-identity bias claim is negative: its pooled signed bias changes from
-0.225 to -0.858 mm/day. Raw-mean preservation restores bias to -0.216 and
has the best IMD RMSE, but its MAE is 2.46% worse than unprojected raw identity.
Describe the model as an error- and spatial-skill-improving post-processor,
not as a uniformly successful bias correction.

The independent-target sensitivity strengthens and sharpens that conclusion.
Across 180 paired 2024 station-verification cases, raw identity improves raw
FuXi RMSE by 0.449 mm/day (95% circular-block interval 0.389 to 0.507), MAE by
0.391 mm/day, and ACC by 0.0276. Its pooled station RMSE/MAE/ACC are
7.493/4.893/0.431 versus 7.942/5.284/0.403 for raw FuXi. The station target is
wetter-biased for raw FuXi rather than dry-biased; consequently raw-mean
preservation degrades RMSE to 7.825 and is rejected as the main method.
Raw identity versus raw was a frozen secondary E3 comparison; the sole
predeclared E3 primary estimand was the legacy selected adapter versus raw.
The raw-identity station result is therefore consistent external-target
sensitivity, not a second confirmatory test.

Rainfall-intensity results bound the claim further. Raw identity improves
pooled dry and moderate errors, but for >=20 mm/day rainfall its RMSE
improvement interval crosses zero, its MAE point effect is unfavorable, and
its signed dry bias remains large. The current evidence does not establish an
extreme-rainfall improvement.

## Later-screen decisions

- `physical_full_compact` passed the declared validation guards, changing
  matched-control RMSE from 5.4563 to 5.4358 and ACC from 0.3271 to 0.3302.
  The gain is small and lacks untouched confirmation.
- The bias-aware/recentered candidate improved pooled RMSE and signed bias but
  worsened MAE and did not beat raw FuXi absolute bias.
- Heavy-rain weighting reduced tail errors at the cost of much larger
  dry/light errors, pooled MAE, and ACC.
- Equal-regime and wet-occurrence objectives moved gradients in the requested
  direction but worsened heavy-rain and pooled skill.
- Fixed Box–Cox power 0.25 was a weak Pareto direction, not a promoted model.
  Its apparent pooled bias gain partly came from cancellation between
  low-intensity overprediction and heavy-rain underprediction.
- Train-only affine recalibration failed its promotion rules.
- The full blocked-OOF run stopped before fold 2 training with
  `weekly IMD truth changed between folds`. The one-fold smoke result
  was also not promoted.
- The raw-identity ablation used raw FuXi, rather than fitted log-bias, as the
  neural reconstruction anchor. Its frozen three-seed `normal_climo_model`
  improved raw-FuXi RMSE by 10.67% in 2018 and 6.22% in 2019. On the reused
  2020--2021 cohort it reached RMSE 5.1503, MAE 3.3452, bias -0.7526 mm/day,
  and ACC 0.3634. It slightly exceeded the anchored adapter on the same cases
  but retained a large dry bias; these are exploratory development results,
  not independent confirmation.
- Global patch pretraining failed its frozen matched gate. On 2018--2019 the
  scratch ensemble had RMSE/MAE/bias/ACC
  5.4809/3.5466/-1.2120/0.3207, while the pretrained ensemble had
  5.5080/3.5555/-1.2425/0.3119. Global RMSE was 0.493% worse, both validation
  years regressed, only three of six leads improved the composite score, and
  zero of three pretrained seeds achieved a lower best India validation
  composite loss than its matched scratch seed. This tests a
  log-bias-anchored transfer initialization; it does not prove that every
  possible global or raw-identity pretraining scheme must fail.
- The artifact-only regional-mean follow-up also failed: projection increased
  scratch RMSE by 0.0392 mm/day and pretrained RMSE by 0.0355, with zero
  improving leads, years, or seeds. It remains a post-hoc diagnostic.

Detailed loss and transform protocols are preserved under
`experiments/`.

## Publication route

1. Retain the completed global-pretraining-versus-scratch comparison as a
   negative component result; do not tune a replacement from 2018--2019.
2. Retain the completed no-retraining 2022--2024 matched audit and frozen
   station-target sensitivity as immutable evidence; do not retune from them.
3. Treat raw identity as the primary model and raw-mean preservation as a
   target-dependent Pareto diagnostic, not a promoted universal fix.
4. Keep the failed blocked-OOF affine branch out of the main evidence line.
   Repair it only if a later calibration study explicitly requires it.
5. Freeze the raw-identity-versus-raw 2025 hierarchy using
   `RAW_IDENTITY_2025_SEALED_WORKFLOW.md`. The older
   `INDEPENDENT_2025_CONTROL_WORKFLOW.md` is a superseded physical-control
   design and must not be used for the selected raw-identity model.
6. Perform the GPU preflight, then open 2025 exactly once after explicit user
   approval.
7. Report case-level paired effects with initialization-block uncertainty,
   all six leads, years/regions, rainfall regimes, and the negative bias result.
8. For the parallel probabilistic paper, use only the corrected `resultsv3/`
   bridge scoring and passed independent audit as retrospective calibration
   evidence. The 2022--2024 synchronized pooled component passes against raw
   and moment calibration, while the final headline/prospective gate remains
   pending sealed 2025. Do not restore any v1 categorical or continuous
   result, and do not select a new blend or network from the retrospective
   evaluation.

Repeatedly tuning against 2018–2019 is now a larger publication risk than
retaining the current model. A larger network is not the missing evidence.

## Separate AI-Quest result

`../../ai-quest-global` is a global probabilistic ERA5-category project
scored with RPS/RPSS, not this deterministic IMD experiment. Its current
validation result is also not publication-ready: the calibrated neural model
improves the calibrated Debias++ control by only about 0.33%, three-seed
confirmation is incomplete, and its untouched 2021 test is unopened.

Use its provenance and test-access patterns, not its targets, metrics, or
claims.
