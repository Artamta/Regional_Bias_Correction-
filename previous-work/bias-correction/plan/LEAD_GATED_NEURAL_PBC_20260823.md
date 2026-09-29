# Frozen one-shot lead-gated neural--PBC diagnostic

Date: 23 August 2026

Status: frozen before the standalone evaluator and publication bundle are
implemented. A preliminary score-only calculation on the already completed
2020--2021 and 2022--2024 case-score tables was inspected before this plan was
written. Therefore every result governed by this plan is explicitly post-hoc,
exploratory evidence; it is not an independent test or a preregistered
superiority result.

## Question

Can the repeated lead-dependent crossover be turned into one transparent
forecast policy that has lower pooled equality-aware quintile RPS than both
Persistence++ and combined PBC?

## Single frozen policy

Use exactly one hard lead gate:

- W1: the accepted 42,434-parameter location--spread neural adapter;
- W2--W6: the accepted projected Persistence++ forecast.

The policy is named `neural_w1_persistence_w2_w6`. There is no fitted blend,
weight search, seasonal switch, regional switch, seed selection, alternate
lead cutoff, or rescue rule. The neural headline remains the arithmetic mean
of separately computed seed scores already present in the accepted comparison
artifacts. Forecasts, probabilities, members, parameters, and correction
fields are never averaged across neural seeds.

Because the policy selects one complete forecast method at each lead, its RPS
for a case/lead is exactly the accepted score of the selected method. The
score-only evaluator must not reconstruct, average, or persist probability
arrays.

## Immutable inputs

### Reused development cohort

- Cohort: 208 initializations from 2020--2021, six leads.
- Evidence label: reused development; post-hoc and not independent.
- Manifest:
  `resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z/manifest.json`
- Manifest SHA-256:
  `44ca7130e626bfadc71d311d0002dbb39b67aebbf16d0dfb2802999b6be4bdbc`
- Case scores:
  `resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z/metrics/quintile_case_scores.csv`
- Case-score SHA-256:
  `a6085cc530b2a9662e2b59a79f65f712186cd08467f8e88881c07ae359344047`

### Later retrospective cohort

- Cohort: 296 initializations from 2022--2024, six leads.
- Evidence label: post-hoc operational-era retrospective; no retraining or
  selection.
- Manifest:
  `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/manifest.json`
- Manifest SHA-256:
  `cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d`
- Case scores:
  `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/metrics/quintile_case_scores.csv`
- Case-score SHA-256:
  `b2bf3e748b37f2946ea4dbadd1b33720ebc20bc25930c5e9592a4fee1f017ee7`

Both accepted manifests must remain complete, canonical/scientifically
eligible as applicable, and must prove that the sealed 2025 target was not
opened. Any hash or contract mismatch stops the evaluator.

## Endpoints

Primary endpoint: pooled normalized informative-positive-cut quintile RPS.

Primary paired contrasts, in this fixed order:

1. hybrid versus Persistence++;
2. hybrid versus combined PBC;
3. hybrid versus the accepted neural adapter;
4. hybrid versus raw FuXi categorical.

Report score reduction as `1 - hybrid / baseline`. The descriptive phrase
"lower pooled RPS" is permitted only when the point effect is positive. The
phrase "paired interval above zero" is permitted only when the corresponding
95% interval is entirely positive. The phrase "independently beats PBC" is
forbidden for these cohorts regardless of the interval because the gate was
motivated after their leadwise results were known.

## Uncertainty

- 2020--2021: reproduce the accepted two-year circular 13-initialization block
  design with 2,000 draws and seed 20260823.
- 2022--2024: use two-stage year resampling followed by circular
  13-initialization blocks within sampled years, 10,000 draws, seed 20260824.
- One initialization retains all six leads; members and grid cells are never
  resampled independently.
- Intervals remain exploratory because there are only two or three year
  clusters and the policy itself is post-hoc.

## Acceptance and stop rules

The one attempt is considered descriptively successful only if the hybrid has
lower pooled RPS than both Persistence++ and combined PBC in both cohorts and
all four paired intervals are above zero. Otherwise retain the original paper
story and report the diagnostic as negative or unresolved.

No alternative gate, blend weight, probability average, seasonal rule,
region-specific rule, or new model training may follow from these scores under
this plan. Do not modify or overwrite either accepted input directory.

## 2025 boundary

This evaluator has no route to a 2025 forecast or observation. The 2025 target
remains sealed. A future confirmatory evaluation would require a separate
hash-bound protocol, storage-incapable preflight, and explicit approval before
one-time target access. This plan supplies only the frozen policy that such a
protocol could test.

