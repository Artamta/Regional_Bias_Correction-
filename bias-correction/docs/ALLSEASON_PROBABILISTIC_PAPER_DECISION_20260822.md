# Paper evidence decision: lightweight probabilistic FuXi calibration

Updated: 23 August 2026

Status: live venue-neutral decision record. The completed evidence below is
immutable. The reused-development categorical comparison, no-retraining
2022--2024 continuous audit, and post-hoc 2022--2024 categorical retrospective
have passed their independent gates. The 2025 forecast and target remain
sealed and unopened.

## Current decision

Use the 42,434-parameter `location_spread` ensemble adapter as the main neural
method. It is the validation-retained capacity, directly optimizes empirical
ensemble CRPS, preserves FuXi's weather-member distribution, and jointly learns
log-rainfall location and spread corrections. Use the summary-only neural model,
train-only moment calibration, raw FuXi, and a frozen-split PBC adaptation as
fixed controls.

The paper should be about **lightweight probabilistic calibration under regional
data and compute constraints**, not network scaling, universal bias removal, or
whole-world training.

Working title:

> **Lightweight Probabilistic Calibration of FuXi-S2S Rainfall over India**

## Completed headline evidence

Canonical neural run:
`../resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z`

This experiment trains on 2002--2017, selects checkpoints on 2018--2019, and
reports a reused 208-initialization 2020--2021 development evaluation. Each
initialization retains all 51 FuXi members and six weekly leads.

| Method | CRPS | RMSE | MAE | ACC | Bias | RMS spread / pooled RMS error |
|---|---:|---:|---:|---:|---:|---:|
| Raw FuXi | 1.6590 | 3.5452 | 2.1080 | 0.2312 | -0.1206 | 0.562 |
| Train-only moment calibration | 1.4920 | 3.3975 | 2.1055 | 0.2821 | +0.1197 | 1.230 |
| Summary-only neural | 1.3915 | 3.1241 | 1.9295 | 0.3312 | -0.1116 | 0.971 |
| Set neural location only | 1.4751 | 3.0958 | 1.8667 | **0.3584** | -0.3588 | 0.603 |
| **Set neural location + spread** | **1.3874** | **3.0994** | **1.9133** | 0.3521 | -0.1578 | **0.967** |

Location + spread versus raw FuXi:

- CRPS reduction: 16.37%, paired initialization-block 95% interval
  14.28% to 18.16%; positive at every lead W1--W6.
- RMSE reduction: 12.57%, interval 10.39% to 14.73%.
- MAE reduction: 9.24%, interval 7.53% to 10.74%.
- ACC increase: +0.121, interval +0.070 to +0.166.
- Signed-bias change: -0.037 mm/day, interval -0.167 to +0.091; this is not a
  bias-improvement result.
- Central 50/80/90% coverage changes from 0.190/0.347/0.427 to
  0.490/0.698/0.789.

The summary-only neural model is almost tied. The set encoder's pooled CRPS
increment is small and post-hoc uncertainty does not justify calling it
essential. The useful result is compact location/spread calibration; the exact
member representation is a secondary ablation.

## Completed capacity and loss decisions

Capacity screen:
`../resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z`

Validation retained the 42,434-parameter base model. Relative validation CRPS
changes were approximately -0.045% for the 157,570-parameter medium model and
-0.042% for the 43,058-parameter summary-matched control, far below the frozen
0.5% promotion threshold and not robust across validation years/seeds. The
19,618- and 293,762-parameter models were worse. A bigger network is not the
missing result.

Hybrid-loss screen:
`../resultsv2/fuxi_allseason_hybrid_loss_ablation/full_final_20260822T141844Z`

Pure CRPS is retained. Adding MSE with weights 0.10, 0.25, and 0.50 worsened
development CRPS from 1.3874 to 1.3913, 1.3981, and 1.4066; the MSE-only arm
reached 1.5611. The deterministic-loss mixture did not produce a useful
accuracy--calibration trade-off.

## Completed classical probability baseline

Canonical PBC adaptation:
`../resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z`

This is a frozen-split India/IMD adaptation inspired by Guan et al. (2026), not
a rolling/prequential reproduction. Its primary score handles duplicate dry
thresholds by scoring only distinct positive cuts.

| Method | Normalized tie-aware RPS | RPSS vs train-empirical climatology |
|---|---:|---:|
| Raw FuXi categorical | 0.24848 | -0.1719 |
| Debias++ | 0.20778 | +0.0201 |
| Persistence++ | **0.19617** | **+0.0748** |
| Combined PBC | 0.19703 | +0.0708 |

Combined PBC reduces the raw score by 20.71% (95% interval 18.45% to 23.42%)
and improves every lead. Persistence++ is numerically best; the combined model
does not significantly beat it. That negative component result must remain
visible.

## Completed common-support categorical comparison

Accepted run:
`../resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z`

All seven methods were rescored on the same 208 starts, IMD truth, thresholds,
spatial support, and equality-aware positive-cut RPS. Persistence++ has the
lowest pooled point estimate (0.19617), followed by combined PBC (0.19703),
while the selected neural adapter reaches 0.20628 and raw FuXi 0.24848. The
leadwise result is the useful scientific finding:

- W1: neural location+spread beats combined PBC by 8.32% (paired interval
  4.46% to 13.23%) and Persistence++ by 7.73% (4.39% to 12.38%).
- W2: neural versus combined PBC is unresolved (1.19%, interval -4.93% to
  6.53%).
- W3--W6: combined PBC and Persistence++ are stronger than the neural adapter.

This is reused-development evidence with only two year clusters. Say
"lead-dependent complementarity," not universal neural superiority.

## Completed no-retraining operational-era audit

Accepted run:
`../resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z`

The exact validation-selected 42,434-parameter checkpoints were applied
unchanged to every eligible 2022--2024 start: 104 in 2022, 104 in 2023, and 88
in 2024 (296 total), with 50 members and all six leads. No training,
fine-tuning, blending, checkpoint selection, parameter averaging, or forecast
averaging was permitted. Scores were computed per seed and then averaged.

| Metric | Raw FuXi | Frozen adapter | Paired effect (95% interval) |
|---|---:|---:|---:|
| CRPS | 1.5626 | **1.3164** | **15.76% skill [13.78, 17.76]** |
| RMSE | 3.4203 | **3.0777** | **10.02% skill [7.71, 12.34]** |
| MAE | 2.0355 | **1.8713** | **8.07% skill [5.25, 10.92]** |
| ACC | .2738 | **.3366** | **+.0628 [.0262, .0964]** |
| Signed bias | -.0164 | -.0368 | -.0204 [-.1372, .0932] |
| 50/80/90% coverage | .187/.338/.415 | **.519/.720/.800** | descriptive |
| Spread/error | .566 | **1.032** | descriptive |

CRPS skill is positive at all leads, from 29.02% at W1 to 10.50% at W6, with
every lead interval above zero. Yearwise point skills are 15.72% (2022),
14.23% (2023), and 17.48% (2024). Signed bias is not improved; W5 becomes
significantly more negative. The audit is strong later-era transfer evidence,
but remains retrospective, has only three year clusters, compares only against
raw FuXi, and does not establish a deployable operational system.

## Completed operational-era categorical comparison

Accepted run:
`../resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z`

The locked neural and projected classical methods were scored on the same 296
eligible 2022--2024 starts, training-derived thresholds, IMD truth, and support.
Nothing was retrained, selected, or fitted to this cohort. The neural headline
is the arithmetic mean of three per-seed scores on each case and lead; no
forecasts, probabilities, parameters, or corrections were averaged.

| Score family | Raw FuXi | Neural | Persistence++ | Combined PBC | Neural reduction vs raw (exploratory 95% interval) |
|---|---:|---:|---:|---:|---:|
| Quintile RPS | .239489 | .184063 | **.182154** | .182280 | **23.14% [15.87, 30.60]** |
| Semidecile RPS | .211836 | .161821 | **.160485** | .160797 | **23.61% [16.09, 31.41]** |
| Descriptive upper-q95 Brier | .064569 | **.056665** | .057935 | .058062 | **12.24% [9.01, 16.54]** |

At W1, neural quintile RPS beats Persistence++ by 10.68% [3.87, 17.06] and
combined PBC by 9.42% [3.80, 15.51]. Pooled ordinary quintile and semidecile
RPS are numerically slightly worse than both classical controls, but all paired
intervals cross zero; this is statistical indistinguishability, not a neural
win or classical win. The neural upper-q95 Brier score is lowest pooled, but
that threshold diagnostic is secondary and does not establish event or
extremes skill.

Only three year clusters support these exploratory intervals. The bootstrap
table contains 252 paired comparisons with no multiplicity adjustment, and all
neural contrasts condition on the mean of per-seed scores. Therefore this
result supports later-era categorical transfer and lead-dependent
complementarity, but not global training, prospective validation, deployment,
2025 performance, or drought/heavy-rain event claims.

The provenance anchors are manifest SHA-256
`cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d`,
semantic-audit SHA-256
`00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c`,
Slurm-receipt SHA-256
`cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c`,
and paired-bootstrap CSV SHA-256
`632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188`.

## Current paper decision

Use the operational-era continuous result as the main transfer evidence and
retain 2020--2021 as explicitly reused development evidence. Make the
categorical result equally visible: in reused development, neural is strongest
against the frozen classical pair at W1 and Persistence++/combined PBC lead
from W3 onward and pooled; in the later-era retrospective, W1 neural strength
persists but pooled ordinary neural--classical differences are unresolved.
The descriptive pooled q95 advantage is a calibration diagnostic, not an
extremes claim. The negative capacity and loss ablations support the compact
CRPS-only design. Do not select a new blend, architecture, loss, or checkpoint
from either evaluation period.

The immutable derived paper bundle is
`../presentation/deliverables/fuxi_allseason_probabilistic_paper_20260822T192948Z`
(manifest SHA-256
`cf9d17dd335a1f680d0a7fb2f5f7a45ac7b556bb5a621d0bf564157cdc4be0d9`).
The accepted later-era categorical tables and two-panel transfer figure are in
`../presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z`
(manifest SHA-256
`bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a`).
The anonymous official-style submission source is in
`../presentation/deliverables/ccai_neurips2026_submission_source_20260823`;
its `main.tex` SHA-256 is
`0538c8ffc253c854bc304d67e03940ee80954a1126703b3b341698a972a72116`.
Tectonic 0.16.9 compiled `main.pdf` with SHA-256
`ad335e16c38a041e2a7b1d108d0aeb438c190ff7b8d091d79e638ea1ec468258`.
The rendered file is five US-letter pages: pages 1--4 contain all main text and
page 5 contains references only, so the four-page main-text limit is met.

## Four-page contribution set

1. A tiny permutation-invariant residual location/spread adapter for weekly
   FuXi rainfall ensembles, with exact identity initialization and direct CRPS
   training.
2. All-season Indian evaluation with weather members preserved as the
   predictive distribution, rather than reducing the input to an ensemble mean.
3. A common-support comparison with a recent PBC-style baseline, including
   dry-threshold equality handling and dependence-aware paired intervals.
4. Later-era, different-member-count evaluation with no retraining, plus honest
   negative capacity and loss ablations.

## Main figures and tables

1. One compact method diagram: 50/51 unordered FuXi members -> log-location,
   RMS spread, set summary, and seven context channels -> residual 3-D
   convolution -> memberwise location/spread transform.
2. One main later-era categorical figure: raw, selected neural, Persistence++,
   and combined PBC under the shared quintile score, plus paired neural
   contrasts; show the W1 neural win and unresolved W2--W6 classical contrasts.
3. One appendix leadwise figure: raw-versus-neural continuous CRPS skill for
   development and operational-era cohorts, with paired uncertainty. Keep the
   descriptive q95 Brier result in text or an appendix table.
4. One small ablation table: parameter count, validation CRPS, development
   CRPS, and promotion result; pure CRPS versus hybrid losses.

## Claim boundaries

- Regional India-box calibration on a 27x27, 1.5-degree grid; not district-scale
  guidance.
- Weekly mean rainfall through week 6; not daily timing or local flash-flood
  prediction.
- The input archive is global FuXi output, but the adapter is fitted only over
  the India box. This is not whole-world training.
- The model improves proper scores and ensemble calibration; it does not
  establish universal signed-bias improvement.
- q95 diagnostics are descriptive unless paired event-focused uncertainty is
  added. The later-era q95 comparison has paired uncertainty but still does not
  define or verify drought, heavy-rain, flood, or impact events; do not claim
  improved extremes from it.
- 2020--2021 is reused development evidence; 2022--2024 is a post-hoc
  no-retraining operational-era retrospective audit; neither is an untouched
  prospective test.
- Later-era categorical intervals contain only three year clusters, cover 252
  unadjusted paired comparisons, and condition neural contrasts on the mean of
  per-seed scores rather than one deployable forecast.
- This is regional India-box fitting and verification, not global training or
  a prospective operational study.
- The 2025 target remains unopened.
