# FuXi--IMD research sprint: 22--29 August 2026

Status: timestamped frozen analysis plan with documented pre-launch revisions.
This is a venue-neutral scientific plan, not a submission draft or a public
preregistration. The 2025 IMD target year remains sealed.

## Fixed research question

Can a compact, deterministic six-lead neural post-processor improve weekly
FuXi rainfall over India beyond transparent statistical and from-scratch
controls, and which part of the system provides that improvement?

The paper-worthy result is the answer to this question, including a negative
answer for any proposed component. The sprint will not optimize a new metric
or architecture after seeing each result.

## Evidence tiers

| Tier | Years | Permitted use |
|---|---|---|
| Training | 2002--2017 | Fit India models, normalization, climatology, and anchors |
| Selection | 2018--2019 | Select frozen India candidates and evaluate predeclared hypotheses |
| Reused exploration | 2020--2021 | Context only; never described as independent confirmation |
| Frozen development audit | 2022--2024 | No-retuning robustness audit of already frozen systems |
| Untouched final | 2025 | One hash-locked access only, after explicit user approval |

Global FuXi-to-IMERG pretraining has its own earlier boundary: 2002--2015 fit,
2016--2017 validation, and no target date in 2018 or later may be read.

## E0: matched global-pretraining experiment

Question: does global rainfall-error pretraining improve the exact same India
adapter relative to a matched random initialization?

- Architecture: `FixedClimatologyAllLeadUNet`, 144,689 parameters.
- Seeds: 42, 43, and 44.
- Only the initialization differs between `scratch` and `global_pretrained`.
- The India residual head is reset to zero in both arms.
- The unavailable global T2M input slice is restored from the matched India
  initialization rather than transferred.
- Primary comparison: globally pretrained versus scratch on 2018--2019.
- Reporting controls: raw FuXi and the training-only log-bias anchor.
- Promotion is determined only by the already encoded multi-seed, year, lead,
  RMSE, MAE, ACC, bias, and epoch-zero gates.
- A failed gate is a valid negative result. There will be no patch-size,
  architecture, learning-rate, or loss sweep afterward.

Completed execution:

- global array job: `109944` (three seed tasks);
- dependent India comparison: `109947`;
- global output root:
  `results/fuxi_global_patch_pretraining/execution_full_parallel_20260822a`;
- comparison output root:
  `results/fuxi_imd_global_pretraining_comparison/execution_full_parallel_20260822a/full_20260822T040953Z`.

All three array tasks and the dependent comparison completed successfully.
The frozen gate selected scratch. Scratch RMSE/MAE/bias/ACC were
5.4809/3.5466/-1.2120/0.3207; global-pretrained values were
5.5080/3.5555/-1.2425/0.3119. Global initialization was 0.493% worse in RMSE,
both validation years regressed, and zero of three pretrained seeds achieved
a lower best India validation composite loss than matched scratch. Comparison
manifest SHA-256:
`37b6dac32a9fc2be799651b8e8577cac247d144caec379d1a3fdfce0417c1f5d`.

The earlier smoke run is non-scientific and must not appear in a result table.

## E1: fixed regional-amount-preserving diagnostic

Question: can the neural model retain its learned spatial redistribution while
preventing it from changing the trusted India-area rainfall amount?

- This is evaluation-only: no training and no fitted coefficient.
- For each case and lead, solve one spatially uniform offset in log space so
  the corrected nonnegative field has exactly the reference anchor's weighted
  India-area physical mean.
- Preserve zero residuals as the exact anchor identity.
- Apply the same deterministic rule to every seed and to the stored ensemble.
- Primary reporting is paired against the corresponding unprojected method.
- Because the rule was conceived after inspecting earlier development bias,
  label it a post-hoc development hypothesis. It cannot change E0's promotion
  decision.

Uncertainty is fixed before execution: year-stratified, circular moving blocks
of 13 initializations, 10,000 paired draws, with all six leads kept
together and identical draw indices for all methods. Report descriptive 95%
intervals and probability of a favorable paired effect for RMSE, MAE, ACC,
and absolute bias.

Completed execution: CPU-only job `109959`, dependency `afterok:109947`, output
`results/fuxi_imd_global_pretraining_followups/full_parallel_20260822a`.
Pending job `109953` was canceled before execution and superseded after the
pre-launch circular-bootstrap correction. Pending replacement `109955` was
also canceled before execution because the live `iiser` partition does not
permit this job's account; `109959` uses the accessible `gpu` partition
without requesting a GPU and excludes the known unsafe nodes.

Both exploratory gates failed. Projection worsened scratch RMSE by 0.0392
mm/day and global-pretrained RMSE by 0.0355, with zero improving leads, years,
or seeds. Manifest SHA-256:
`e4748243bf8a57b34dc4460af97ce57cbdcfd63d39daebe36e9458356b0510bc`.

## E2: no-retraining 2022--2024 matched audit

Question: does the raw-identity model's improvement persist on the already
defined 100-start development audit, and does amount preservation reduce its
dry-bias trade-off?

Freeze before loading 2022--2024:

- raw FuXi;
- training-only log-bias;
- legacy frozen anchored adapter;
- frozen raw-identity/no-fitted-log-bias adapter;
- raw-amount-preserving raw-identity adapter;
- E0 globally pretrained and matched scratch models are excluded from this
  already hash-bound evaluator because E0 was incomplete when E2 was frozen.
  E0 subsequently failed its gate, so no later-year E0 audit was added.

No model, alpha, normalization, climatology, or method list may be changed
after inference begins. Report pooled, year, lead, region, and rainfall-regime
RMSE, MAE, signed bias, absolute bias, and ACC, plus paired block uncertainty.
This remains a retrospective development audit because earlier anchored
2022--2024 outcomes were already known.

Uncertainty is fixed at 10,000 paired, year-stratified, circular moving-block
draws of length 13 ordered initializations, retaining all six leads and using
one shared draw matrix for every method and metric. Every initialization must
have equal marginal inclusion within its year. Intervals are descriptive
development-audit intervals, not multiplicity-adjusted tests.

Pre-launch revision: the initial non-circular design was rejected during code
review because truncated boundary blocks strongly under-sampled the beginning
and end of each season. No production result had been run. The circular design
corrects the estimand mismatch; earlier 1,000-draw `/tmp` integrations are
noncanonical execution checks only.

The amount-preserving transformation uses only the fixed 171-cell adapter
support and India-area weights. Case/lead IMD observation coverage is used for
verification metrics only and must not alter a forecast; changing truth or
coverage must leave every projected field unchanged.

Completed execution: canonical GPU job `109956`, fresh output
`resultsv2/fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z`.
The launcher and evaluator were independently reviewed before submission; the
full repository suite passed 199 tests. The job completed in 55 seconds on an
NVIDIA A30. All 16 declared output hashes and the pinned provenance checks
passed; manifest SHA-256 is
`bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`.

Canonical pooled result: raw identity improves raw-FuXi RMSE by 8.45% and MAE
by 9.65%, with ACC +0.0889, but its signed dry bias worsens from -0.225 to
-0.858 mm/day. Raw-mean preservation improves RMSE by 8.65% and restores bias
to -0.216 mm/day, but loses 2.46% MAE relative to unprojected raw identity.
It is therefore a Pareto diagnostic, not an overall winner.

## E3: independent rain-gauge target sensitivity

Question: do the frozen gridded-adapter improvements survive verification
against the cleaned station observations in `/home/raj.ayush/saptarishi_stuff`?

- Use only the common 30 JJAS 2024 initializations and all six leads.
- Use already-frozen predictions; do not train, calibrate, select, or blend on
  station observations.
- Compare raw FuXi, training-only log-bias, the frozen anchored adapter, the
  frozen raw-identity adapter, and its fixed amount-preserving diagnostic.
- The completed evaluator contains exactly the five methods frozen in E2. E0
  was incomplete when E3 was frozen and is excluded. E0 subsequently failed
  its gate; the immutable E3 method set was not changed.
- Reuse one frozen station cleaning and grid-mapping contract. Hash its inputs
  and reject a mismatch in dates, coordinates, methods, or station counts.
- Treat independent 1.5-degree cells, not duplicated gauges in a cell, as the
  primary spatial units. Resample paired initialization blocks rather than
  individual station records.
- Report ACC, RMSE, MAE, signed/absolute bias, coverage, and results by lead.

Primary estimand: the equal-case All-India mean RMSE difference
`raw_fuxi - selected_adapter` over the fixed 180 initialization-by-lead cases;
positive values favor the adapter. Secondary estimands are adapter-minus-raw
ACC, raw-minus-adapter MAE, signed bias, and absolute-bias change, with
`log_bias` as the fixed secondary baseline. Per-lead values are descriptive.

Station aggregation is frozen before scoring: require at least 6 of 7 daily
values, median-combine stations inside a 1.5-degree cell, subtract the common
2018--2023 station climatology for ACC, use India land-area weights, require at
least 20 common cells per case, and use identical cells for every method.
Station availability is a scoring mask only; it must not alter any gridded
forecast or amount-preserving transformation.
Primary uncertainty uses 2,000 paired circular moving-block draws of length 13
ordered initializations with all six leads attached. Block lengths 4 and 8 are
reported only as predeclared sensitivity analyses and cannot replace the
primary interval.

The evaluator must assert the exact 30-date list, 92--99 represented cells and
295--339 gauges per case, the frozen 171-cell adapter support, and all input
hashes before computing metrics. It must index the adapter grid from parsed
`grid_II_JJ`, not the older compressed `grid_position` column.

This is an external-observation robustness check, not a second training result
and not an untouched temporal final: 2024 IMD performance was already known.
The separate Saptarishi multi-model XGBoost result is context only because its
best ensemble and best individual model are nearly tied and its documented
purged split does not match the actual frozen run lineage.

Completed execution: CPU-only job `109958`, submitted with
`afterok:109956`, fresh output
`resultsv2/fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z`.
It started only after the canonical E2 launcher exited successfully and
completed in 9 seconds. All 14 declared output hashes passed; manifest SHA-256
is `5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`.
Pending job `109957` was canceled before execution because the live `iiser`
partition does not permit this job's account; the replacement requested no
GPU and used the accessible `gpu` partition with unsafe nodes excluded.

External-target result: against the fixed 2024 gauge network, the raw-identity
adapter improves raw FuXi by 0.449 mm/day RMSE (95% block interval 0.389 to
0.507), 0.391 mm/day MAE, and +0.0276 ACC. The raw-mean-preserving diagnostic
is worse than raw identity on RMSE, MAE, ACC, and absolute bias. This target
dependence rejects amount preservation as the main method and supports raw
identity as the robust neural candidate.

## Known result added to the ledger

The completed raw-identity/no-fitted-log-bias ablation is no longer future
work. Its three-seed selection chose `normal_climo_model`, alpha 1.0.

- 2018 RMSE improvement versus raw FuXi: 10.67%.
- 2019 RMSE improvement versus raw FuXi: 6.22%.
- Reused exploratory 2020--2021: RMSE 5.1503, MAE 3.3452, signed bias
  -0.7526 mm/day, and ACC 0.3634.
- On the same reused cases it slightly exceeded the anchored adapter's RMSE,
  MAE, and ACC, but retained a substantial dry bias.

This supports testing a simpler mechanistic story: the network learns useful
spatial and lead-dependent correction without requiring an explicit fitted
log-bias anchor.

## Stop rules for this sprint

Do not run:

- KL divergence or a probabilistic retrofit to this deterministic model;
- larger networks or another attention/capacity sweep;
- another loss, target-transform, affine-calibration, or bias-anchor grid;
- a physical-variable operational model whose inputs are unavailable in 2025;
- naive blocked OOF using globally pretrained checkpoints whose IMERG years
  overlap the proposed India folds;
- any 2025 target read.

A genuine probabilistic forecast requires a separate distributional contract,
proper scores such as CRPS/RPS or log score, and calibration diagnostics. It is
a separate study rather than a one-week add-on.

## Final boundary

With E0--E3 complete, choose the final method hierarchy without viewing 2025;
snapshot and hash every checkpoint, preprocessing artifact, evaluator, input metadata,
and the access-ledger path and implementation contract. Run a sealed preflight
that creates no ledger and cannot read 2025 truth. Only after a separate
explicit user decision may the final dispatcher atomically create and hash
the one-time ledger before importing the 2025 runtime. One access produces one
immutable final result; it is not a new tuning set.
