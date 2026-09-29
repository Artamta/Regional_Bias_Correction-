# Benchmarking and Calibrating Subseasonal Monsoon Rainfall Forecasts over India

## Abstract

Subseasonal rainfall prediction over India requires both realistic rainfall
amounts and useful spatial anomaly patterns, but these properties need not be
retained by the same forecast system. We evaluate seven individual AI and
dynamical forecast systems and a fixed six-system equal-weight multi-model mean
(MME) over the Indian summer monsoon season (June--September; JJAS) during
2020--2024. Forecasts and observations are placed on a common 1.5-degree grid,
with 169 cases for each available system--lead pair, selected by the midpoint
of its seven-day valid period. Against India Meteorological Department (IMD) rainfall, the raw
MME has lower root-mean-square error (RMSE) than raw FuXi-S2S at Weeks 1--5
under paired 95% intervals, whereas FuXi has higher anomaly correlation
coefficient (ACC) at Weeks 5--6. The intervening ACC differences at Weeks 3--4
are unresolved. This complementarity motivates a 42,434-parameter
location-and-spread adapter for the FuXi ensemble. The corrected experiment
uses observation labels at `init+1` through `init+42`, trains on 2002--2017,
and selects one checkpoint using 2018--2019 validation CRPS. In an audited
retrospective 2022--2024 JJAS comparison, the adapter improves pooled CRPS by
16.21% over raw FuXi (95% interval 14.81--17.69%) and by 4.26% over train-only
moment calibration (2.99--5.70%); ACC increases by 0.0261 over raw FuXi
(0.0087--0.0436). Against unfitted IMERG, CRPS skill remains 12.20% over raw
FuXi (10.51--14.02%) and 2.08% over moment calibration (0.67--3.66%), while the
ACC increase is unresolved. Across the full retrospective 2020--2024 benchmark, the
calibrated FuXi mean has lower RMSE than the fixed MME at every lead, while its
ACC is higher with intervals excluding zero at Weeks 1 and 3--6 and unresolved
at Week 2. Nominal 90% coverage improves at every lead but remains below 0.90.
These results are retrospective. The 2025 target remains sealed, and no
prospective claim is made.

<!-- Drafting evidence: claims B01--B04, A02--A04, P02--P09, S01, and G01;
canonical evidence bundle `india_s2s_paper_release_bundle/full_v3_20260826T012700Z`;
canonical comparison `india_s2s_paper_comparisons/full_v2_20260826T001721Z`;
canonical figure package `india_s2s_paper_figures/full_v4_20260826T011100Z`. -->

## 1. Introduction

Indian summer monsoon rainfall affects water management, agriculture, energy,
and disaster preparedness. Forecasts at Weeks 1--6 could extend decision
windows beyond conventional weather prediction, but monsoon rainfall combines
sharp spatial gradients, intermittency, organized synoptic disturbances, and
strong land--ocean contrasts. A useful evaluation must therefore separate at
least two questions: whether a forecast controls rainfall magnitude error and
whether it retains the spatial pattern of rainfall anomalies.

Comparisons across subseasonal systems are easily confounded by unmatched
initialization dates, changing model availability, different grids,
inconsistent weekly labels, and incompatible climatologies. These choices are
especially consequential over India, where observational support and regional
rainfall regimes are heterogeneous. We address this problem with a paired
five-year protocol: common dates, a common grid, one primary observation
reference, fixed lead windows, method-specific forecast climatologies, and
dependence-aware uncertainty.

The benchmark reveals a useful separation rather than uniform failure. A
fixed equal-system MME generally controls raw-field RMSE, while FuXi-S2S
retains stronger late-lead spatial anomaly skill. This result supplies the
scientific reason for the second part of the study. We ask whether lightweight
ensemble calibration can improve FuXi's rainfall distribution and magnitude
without discarding its pattern information. The experiment uses one frozen
location-and-spread architecture and one validation-selected checkpoint. Its
ensemble members are evaluated probabilistically with CRPS and its ensemble
mean is evaluated with the same ACC and RMSE protocol as the benchmark.

Our contributions are:

1. a common-date 2020--2024 JJAS comparison of seven individual systems and a
   fixed-composition six-system MME over India at Weeks 1--6;
2. paired uncertainty for the central MME--FuXi magnitude-versus-pattern
   contrast, together with regional and observation-reference sensitivity;
3. a corrected, manifest-bound FuXi calibration experiment with strict
   `+1...+42` target alignment, validation-only checkpoint selection, and
   identical 50-member retrospective cases; and
4. an independently reproduced evaluation connecting probabilistic CRPS
   gains to deterministic ACC and RMSE within the benchmark evaluator.

The calibration evidence is deliberately labelled retrospective. Forecast
years through 2024 had already informed development choices, and the sealed
2025 outcome has not been evaluated. The benchmark therefore remains the
paper's backbone even though the retrospective calibration gate passes.

<!-- Drafting evidence: claims B01--B03, P01--P08, and G01; `CLAIM_REGISTRY.md`;
figure-package manifest hard gates. -->

## 2. Related positioning

Prior work has evaluated multi-model subseasonal forecasts over India, so our
novelty claim concerns the aligned protocol and the particular AI--dynamical
comparison, not priority for India-focused S2S evaluation.

Malik and Mishra evaluated nine operational S2S systems over India using
heterogeneous reforecast archives, twice-monthly JJAS samples, and leads
through four weeks, including a 1999--2010 common period for extremes
\citep{MalikMishra2024IndiaS2S}. Our contribution is instead a common-date
2020--2024 AI--dynamical comparison spanning six weeks, homogeneous regions,
observation sensitivity, paired uncertainty, and aligned calibration.

Recent global AI weather systems have extended data-driven prediction toward
subseasonal ranges, while operational dynamical ensembles remain central to
forecast production. Our study places representatives of both families under
one India-specific verification contract.

CMA, ECMWF, NCEP, and UKMO are represented through the operational S2S
archive, while FuXi-S2S, NeuralGCM, and DLESyM represent recent learned or
hybrid model families
\citep{VitartEtAl2017S2SDatabase,ChenEtAl2024FuXiS2S,KochkovEtAl2024NeuralGCM,CresswellClayEtAl2025DLESyM}.
Exact archived experiments, ensemble treatment, availability, and upgrade
caveats are reported separately.

Statistical post-processing can correct ensemble location, dispersion, and
reliability at much lower computational cost than retraining a forecast model.
Here post-processing is tested as a constrained diagnostic: the architecture
is frozen, probabilistic and deterministic outputs come from the same
calibrated ensemble, and a train-only moment calibration provides a classical
baseline.

Ensemble post-processing commonly adjusts location and dispersion, with CRPS
providing a distribution-wide verification and estimation criterion. Neural
precipitation post-processing and probabilistic bias correction extend these
ideas to subseasonal forecasts
\citep{Hersbach2000CRPS,GneitingEtAl2005EMOS,ScheuererEtAl2020S2SPrecip,GuanEtAl2026PBC}.
Here the deployed output remains one continuous, member-preserving ensemble.

IMD is the primary reference because the study concerns rainfall over India;
IMERG supplies an observation-reference sensitivity rather than an assumed
interchangeable truth product.

The primary IMD product is gauge-based daily gridded rainfall, whereas IMERG
Final combines satellite retrievals and gauge information
\citep{PaiEtAl2014IMDRainfall,HuffmanEtAl2020IMERG,HuffmanEtAl2023IMERGDailyV07}.
Their distinct sampling and error characteristics motivate treating IMERG as
an independent sensitivity; the evaluator maps their different date labels
onto identical physical forecast windows.

<!-- Drafting evidence: claim B03, the positioning guard in
`MANUSCRIPT_OUTLINE.md`, and `references.bib`. -->

## 3. Data and verification protocol

### 3.1 Forecast systems and fixed MME

We evaluate CMA, DLESyM-v0, ECMWF, FuXi-S2S, NCEP, NeuralGCM, and UKMO as
individual systems. The benchmark uses archived ensemble-mean forecasts;
native ensemble sizes differ across systems. An additional equal-system MME
averages six fixed components: CMA, DLESyM-v0, raw FuXi-S2S, NCEP, NeuralGCM,
and UKMO. ECMWF is excluded from the MME at every lead. This fixed composition
avoids changing the identity of the MME when ECMWF precipitation is
unavailable at Week 3. ECMWF remains an individually evaluated system at the
other leads. The fixed MME therefore includes raw FuXi and is not independent
of it.

| Forecast entry | Individual evaluation | Fixed-MME member | Main availability note |
|---|---:|---:|---|
| CMA | yes | yes | Weeks 1--6 |
| DLESyM-v0 | yes | yes | Weeks 1--6 |
| ECMWF | yes | no | Week 3 precipitation unavailable |
| FuXi-S2S | yes | yes | raw ensemble mean enters the MME |
| NCEP | yes | yes | Weeks 1--6 |
| NeuralGCM | yes | yes | Weeks 1--6 |
| UKMO | yes | yes | Weeks 1--6 |
| Equal-system MME | derived | -- | same six components at every lead |

**Table 1:** Forecast entries and fixed-MME membership. The canonical generated
table adds experiment/archive identifiers, ensemble treatment, explicitly
recoverable native sizes, years, variables, and availability. Native sizes for
the upgraded operational physics systems are labelled unavailable rather than
inferred beyond the bound manifest.

<!-- Drafting evidence: claim B01; evidence-bundle table
`tables/main_deterministic_benchmark.csv`; benchmark `methods_manifest.json`. -->

### 3.2 Observations, domain, weeks, and cohorts

IMD gridded daily rainfall is the primary observation. IMERG is used only for
reference sensitivity. All main results use a 27 by 27 grid at 1.5-degree
spacing with 171 fixed forecast/calibration-support cells. The fractional IMD
region geometry overlaps 174 cells; three are shown separately because they
fall outside the fixed scoring/training support. We report all-India scores
and four fractionally masked homogeneous regions: northwest India, central
India, the south peninsula, and east--northeast India.

The two products encode daily time differently. End-labelled IMD uses offsets
`+1...+42`; UTC-period-start-labelled IMERG uses offsets `0...41` for the same
physical forecast windows. This IMERG convention is therefore not the
withdrawn one-day-early IMD alignment. IMD anomalies use the 1991--2019 normal,
whereas the IMERG sensitivity uses its independently frozen 2001--2019
climatology.

Weekly forecast windows are defined from initialization: Week 1 spans
`[init, init+7 days)` and is verified against IMD end labels `init+1...+7`;
Week 6 spans `[init+35, init+42 days)` and uses labels `init+36...+42`, with
the intervening weeks following the same convention. A case belongs to JJAS
when the midpoint of its seven-day valid period lies in June--September. This
produces 169 cases at every lead during 2020--2024.

For the calibrated bridge, 517 operational forecasts are available for
forecast-climatology construction and 505 have scoreable targets without
crossing the sealed boundary. Each valid-midpoint JJAS lead still contains 169
scored cases. The corrected workflow records 2024-12-30 as the latest target
label opened.

**[Figure 1 near here.]** Verification-grid domain, four fractional regional
masks, weekly alignment, and the train/validation/development/retrospective
timeline. The rendered panel is a grid-cell schematic rather than political
or coastline geometry; the 2025 interval is shown only as sealed.

<!-- Drafting evidence: claims B01, P01, and S01; figure contract
`figure1_data_contract.json`; scoring truth receipt. -->

### 3.3 Deterministic verification

ACC is computed for each case as an area-weighted spatial correlation between
forecast and observed anomalies. Each method uses its own four-year
leave-one-year-out forecast climatology over 2020--2024; observations use the
IMD 1991--2019 normal. A calibrated method is therefore not evaluated with the
raw FuXi climatology. RMSE, mean absolute error (MAE), and signed bias are
computed on weekly raw rainfall fields using area times observation-coverage
weights. Reported lead values are arithmetic means of per-case scores.

The primary uncertainty analysis uses 10,000 paired, year-stratified circular
block resamples with blocks of 16 forecast starts
\citep{PolitisRomano1992CircularBlock}. A block length of 13 starts is retained
as sensitivity. Pairing preserves the common dates and makes
intervals statements about within-case method differences.

<!-- Drafting evidence: claim B02; uncertainty manifest
`india_s2s_benchmark_uncertainty/full_v2_20260825T232000Z/manifest.json`; figure
package captions. -->

### 3.4 Probabilistic verification

Raw FuXi, train-only moment calibration, and the neural location-and-spread
adapter are scored on identical 50-member cases. We use empirical CRPS,
nominal 90% ensemble coverage, and a pooled spread--error ratio. CRPS is
computed from the ensemble members and spatially aggregated with the same
area-aware support. A spread--error ratio near one indicates that pooled
ensemble variance and pooled ensemble-mean squared error have similar scale;
coverage is reported directly rather than interpreted from this ratio alone.
The ensemble mean of each method is passed through the deterministic evaluator
for ACC, RMSE, MAE, and bias.

The primary calibration intervals use the same 10,000-resample,
year-stratified circular-block design. Lead-wise 2020--2024 comparisons use
169 cases per lead. The pooled 2022--2024 story gate uses six lead-specific
100-case cohorts, or 600 case--lead rows. The six date sets have 70 dates in
their intersection and 130 in their union; they are not described as one
common 100-date sample. Bootstrap indices are synchronized by within-year
ordinal across leads.

<!-- Drafting evidence: claims P02--P08; scoring
`tables/paired_block_intervals.csv`; figure-package caption and
`data/figure4_retrospective_gate.csv`. -->

## 4. Corrected FuXi calibration

The calibration experiment targets a specific deficiency exposed by the
benchmark: FuXi retains useful anomaly-pattern skill at later leads but has
larger raw-field magnitude error than the MME. The adapter is a frozen
42,434-parameter location-and-spread model. It transforms the FuXi ensemble
while retaining a 50-member operational forecast, allowing its members and
ensemble mean to support probabilistic and deterministic verification from
one method.

Targets are rebuilt from IMD offsets `+1...+42`, and split purging protects the
complete 42-day outcome window. The model is trained from scratch using
51-member FuXi reforecasts from 2002--2017. Years 2018--2019 are used for
validation and checkpoint selection. Three seeds are trained, and seed 43 is
the sole deployable checkpoint because it has the lowest validation CRPS
(1.3252039). Parameters and predictions are not averaged. No earlier
misaligned checkpoint or target-derived calibration is loaded.

The 2020--2021 period had already been used during development, so its
diagnostics are reserved for the supplement and explicitly labelled reused
development. The entire operational 2020--2024 bridge is treated as
retrospective evidence. The classical comparator is a location-and-spread
moment calibration fit from training data only.

An independent alignment auditor rebuilt the 2,080 weekly IMD target tensors
from explicit `+1...+42` dates, reproduced the train, validation,
reused-development, and embargo IDs and counts (1,652, 196, 208, and 24), and
verified the 171-cell support. For the final retained training initialization
(2017-11-17), native FuXi lead days 1--42 reproduced the cached weekly members
byte-for-byte. The reconstructed weekly-target tensor has SHA-256 prefix
`31f371c2`. The original training run did not, however, persist a logical hash
or copy of the target tensor it consumed. The audit therefore establishes the
frozen source/current-source contract and reconstruction, but cannot
retroactively prove byte equality to that unrecorded historical tensor.

The selected checkpoint, split definitions, target offsets, source snapshots,
input hashes, predictions, case tables, and bootstrap indices are bound by
immutable manifests. A separate table-level implementation reproduced the
all-season and JJAS summaries, both bootstrap cohorts, and the pooled story
gate. It verified 276 bound files, all 45,450 case-metric rows, and raw-FuXi
identity on 15,150 matched rows. A second, prediction-level implementation
independently reopened the raw members and daily IMD, rebuilt the aligned
targets, three 50-member forecasts from the bound compact inference shards,
method-specific climatologies, and every case metric without importing either
production scorer. It did not rerun training or neural-network inference. All
45,450 rows matched with zero failures; maximum absolute differences were
\(1.78\times10^{-15}\) for CRPS and \(1.11\times10^{-16}\) for ACC. Both audit
statuses are `passed`.

<!-- Drafting evidence: claims A02--A04 and P04--P05; corrected manifest
`fuxi_allseason_ensemble_calibration/full_20260825T224711Z/manifest.json`;
alignment-audit receipt
`fuxi_allseason_training_alignment_audit/audit_full_20260826T005845Z/audit_receipt.json`;
table-audit receipt `audit_full_20260826T000133Z/audit_receipt.json`;
prediction-audit receipt
`prediction_audit_full_20260826T002645Z/audit_receipt.json`. -->

## 5. Five-year raw benchmark results

### 5.1 Magnitude and pattern skill separate with lead

Both focal forecasts lose ACC with lead, but their relative strengths differ.
At Week 1, MME and FuXi ACC are 0.626 and 0.618, respectively, and their paired
difference is unresolved. MME RMSE is lower (4.690 versus 4.956 mm day\(^{-1}\)).
At Week 2, the MME has higher ACC (0.436 versus 0.404) and lower RMSE (5.488
versus 5.988), with both primary paired intervals excluding zero in the
favourable direction. At Weeks 3--4, FuXi has higher point ACC, while the ACC
differences are unresolved; MME maintains lower RMSE with intervals excluding
zero. At Week 5, FuXi's ACC advantage and MME's RMSE advantage both exclude
zero. At Week 6, FuXi's ACC advantage excludes zero, whereas the smaller MME
RMSE advantage is unresolved.

| Lead | MME ACC | FuXi ACC | MME RMSE | FuXi RMSE | Paired interpretation |
|---:|---:|---:|---:|---:|---|
| W1 | 0.626 | 0.618 | 4.690 | 4.956 | ACC unresolved; MME RMSE lower |
| W2 | 0.436 | 0.404 | 5.488 | 5.988 | MME ACC higher and RMSE lower |
| W3 | 0.281 | 0.306 | 5.922 | 6.237 | ACC unresolved; MME RMSE lower |
| W4 | 0.192 | 0.219 | 6.147 | 6.389 | ACC unresolved; MME RMSE lower |
| W5 | 0.099 | 0.174 | 6.307 | 6.486 | FuXi ACC higher; MME RMSE lower |
| W6 | 0.048 | 0.143 | 6.285 | 6.371 | FuXi ACC higher; RMSE unresolved |

**Table 2A:** Focal all-India IMD JJAS contrast, with 169 cases per available
system--lead pair. RMSE is in mm day\(^{-1}\). The canonical generated table
contains all seven individual systems and the fixed MME, with paired effect
and interval fields populated only for the supported MME--FuXi contrast.

**[Figure 2 near here.]** All-system IMD JJAS ACC and RMSE by lead, including
an explicit missing marker for ECMWF Week 3. The curves are descriptive and
contain no error bars; paired MME--FuXi intervals and their interpretation
appear in Table 2A and the surrounding text.

This is the benchmark's central diagnosis. The fixed MME provides a strong
magnitude baseline, whereas raw FuXi increasingly separates from it in
late-lead spatial anomaly skill. Neither metric alone captures the full
forecast behavior.

<!-- Drafting evidence: claims B01--B02; evidence-bundle
`tables/main_deterministic_benchmark.csv`; paired table
`mme_vs_fuxi_paired_intervals.csv`; figure data `data/figure2_values.csv`. -->

### 5.2 Region and observation-reference sensitivity

The regional evaluation applies the same case IDs and lead definitions with
fractional homogeneous-region masks. We retain the regional curves as a
heterogeneity diagnostic rather than reducing them to a universal ranking.
With IMERG replacing IMD, the qualitative focal contrast persists: the fixed
MME has lower point RMSE at Weeks 1--6, while FuXi has higher point ACC at
Weeks 3--6. These are descriptive reference-sensitivity results. They do not
establish numerical equivalence between IMD and IMERG or significance of the
reference difference.

**[Figure 3 near here.]** Regional MME and FuXi ACC under IMD, alongside
matched IMERG-minus-IMD reference sensitivity on identical JJAS case IDs.

<!-- Drafting evidence: claim B03; IMD and IMERG seasonal/case tables bound in
`ARTIFACTS.sha256`; figure data `data/figure3_values.csv`. -->

### 5.3 Limited-period systems and ranking boundary

ERPAS, CNRM, and Spire are not appended to the five-year common-date ranking.
The defensible ERPAS evidence is a separate matched-valid-time sensitivity:
31 cases whose ERPAS initialization falls in June--September 2023--2024,
complete only through Weeks 1--4. This initialization-month filter differs
from the main benchmark's valid-period-midpoint JJAS rule. ERPAS was issued on
Wednesday and is compared with the preceding-Monday FuXi forecast, so FuXi is
48 hours older although the verified weekly windows are identical.
Pooled FuXi-minus-ERPAS ACC is 0.050, 0.146, 0.166, and 0.133 at Weeks 1--4;
ERPAS-minus-FuXi MAE is 0.173, 0.552, 0.836, and 0.486 mm day\(^{-1}\), so
positive values favour FuXi. Across 3--5-start moving-block sensitivities,
ACC intervals exclude zero at Weeks 2--3 and MAE intervals at Weeks 2--4;
Week 1 is unresolved, and regional Week-1 signs differ. Because the comparison
uses system-specific forecast climatologies whose ERPAS provider baseline
years are not documented locally, it supports only limited-period
transferability, not a universal ranking. It is verified against IMD on a
distinct 22 by 22, 1.5-degree grid with 169 fixed-support cells and an IMD
1991--2020 climatology, rather than the main benchmark's 27 by 27 grid,
171-cell support, and 1991--2019 normal. Its absolute curves are therefore not
numerically comparable to Figure 2.

CNRM begins on 22 October 2020 and overlaps only 217 of the 517 main-cycle
starts, so it cannot enter the frozen five-year common cohort. The locally
available Spire archive is a JFM-2026 case study without an auditable hindcast
climatology, outside both the 2020--2024 benchmark and the sealed-2025
calibration gate. Neither system is scored or ranked in the main paper.

**[Supplementary Figure S3 near here.]** The separate ERPAS--FuXi
matched-valid-time comparison, including pooled paired ACC/MAE intervals and
descriptive regional ACC differences.

<!-- Drafting evidence: claim B04; validated ERPAS figure package, paired
initialization-overlap table, and limited-period integration note bound in
`ARTIFACTS.sha256`. -->

## 6. Retrospective calibrated benchmark

### 6.1 Probabilistic skill and ensemble diagnostics

The neural adapter reduces CRPS relative to raw FuXi at every lead in the
retrospective 2020--2024 JJAS cohort. CRPS skill is 29.23%, 22.45%, 15.14%,
12.12%, 10.23%, and 9.07% at Weeks 1--6, respectively. All corresponding
primary lead-wise intervals exclude zero. Neural CRPS point estimates are also
lower than train-only moment calibration at every lead.

| Lead | Raw CRPS | Moment CRPS | Neural CRPS | Neural skill vs raw | Raw coverage90 | Neural coverage90 |
|---:|---:|---:|---:|---:|---:|---:|
| W1 | 2.8480 | 2.1740 | 2.0156 | 29.23% | 0.156 | 0.686 |
| W2 | 3.2502 | 2.7205 | 2.5206 | 22.45% | 0.331 | 0.761 |
| W3 | 3.1888 | 2.8540 | 2.7061 | 15.14% | 0.543 | 0.798 |
| W4 | 3.1760 | 2.9148 | 2.7911 | 12.12% | 0.657 | 0.816 |
| W5 | 3.1633 | 2.9178 | 2.8397 | 10.23% | 0.726 | 0.838 |
| W6 | 3.0959 | 2.9001 | 2.8151 | 9.07% | 0.741 | 0.841 |

Coverage improves relative to raw FuXi at all six leads, but neural coverage
ranges from 0.686 to 0.841 and therefore remains below the nominal 0.90. The
pooled neural spread--error ratio is 0.745, 0.959, 1.163, 1.019, 0.965, and
0.927 from Weeks 1--6, compared with 0.140, 0.299, 0.492, 0.612, 0.681, and
0.690 for raw FuXi. The improvement is substantial, but the coverage shortfall
precludes a claim of complete reliability.

<!-- Drafting evidence: claim P03; evidence-bundle
`tables/calibrated_fuxi_by_lead.csv`; scoring JJAS summary; figure data
`data/figure4_values.csv`. -->

### 6.2 Pooled retrospective gate

Across six lead-specific 2022--2024 valid-midpoint cohorts of 100 cases each,
with bootstrap draws synchronized by within-year ordinal, pooled CRPS changes
from 3.1570 for raw FuXi to 2.6453 for the neural adapter. This is 16.21% skill
with a paired 95% interval of 14.81--17.69%. Against train-only moment
calibration (CRPS 2.7630), neural skill is 4.26% (2.99--5.70%). ACC changes from
0.2959 to 0.3220 relative to raw FuXi, a difference of 0.0261
(0.0087--0.0436).

Signed bias changes from -0.0606 to -0.0405 mm day\(^{-1}\), but the paired
difference, 0.0201, has an interval of -0.2070 to 0.2461. We therefore claim no
signed-bias improvement. The retrospective CRPS and ACC gates pass, while the
bias guardrail finds no significant change; this does not complete the
prospective headline gate.

A stricter sensitivity aligns exactly common valid-period midpoints across the
six leads. It retains 85 midpoints per lead (510 case--lead rows). Neural CRPS
skill remains 15.87% versus raw (14.35--17.52%) and 4.06% versus moment
calibration (2.61--5.71%); the ACC difference versus raw is 0.0313
(0.0123--0.0506). This sensitivity supports the direction of the pooled result
without replacing the pre-specified lead-specific cohort. Its 85 midpoints are
distributed 35, 35, and 15 across 2022, 2023, and 2024, respectively; the
reduced 2024 representation is an additional reason to keep it secondary.

<!-- Drafting evidence: claim P02; evidence-bundle `tables/story_gate.csv`;
canonical comparison `tables/exact_common_midpoint_sensitivity.csv`; figure
data `data/figure4_retrospective_gate.csv` and
`data/figure4_exact_common_sensitivity.csv`. -->

### 6.3 Deterministic comparison with the fixed MME

Calibration changes the deterministic conclusion. Across 2020--2024, the
neural ensemble mean has lower RMSE than the fixed MME at every lead, with all
primary paired intervals excluding zero. Relative RMSE skill is 9.64%, 4.15%,
4.92%, 6.21%, 6.84%, and 7.55% from Weeks 1--6. Neural ACC is higher with
intervals excluding zero at Weeks 1 and 3--6; the Week 2 difference is near
zero and unresolved.

| Lead | Neural RMSE | MME RMSE | RMSE skill vs MME (95% interval) | Neural ACC | MME ACC | ACC difference (95% interval) |
|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4.238 | 4.690 | 9.64% (7.63, 11.54) | 0.658 | 0.626 | 0.032 (0.019, 0.046) |
| W2 | 5.260 | 5.488 | 4.15% (1.84, 6.31) | 0.436 | 0.436 | -0.000 (-0.025, 0.024) |
| W3 | 5.631 | 5.922 | 4.92% (2.67, 7.12) | 0.334 | 0.281 | 0.053 (0.023, 0.084) |
| W4 | 5.766 | 6.147 | 6.21% (3.85, 8.34) | 0.246 | 0.192 | 0.053 (0.019, 0.088) |
| W5 | 5.875 | 6.307 | 6.84% (4.88, 8.79) | 0.177 | 0.099 | 0.078 (0.028, 0.127) |
| W6 | 5.810 | 6.285 | 7.55% (5.65, 9.36) | 0.179 | 0.048 | 0.131 (0.091, 0.170) |

This comparison is intentionally limited to deterministic scores: the
benchmark MME is an equal average of archived system ensemble means, whereas
raw, moment-calibrated, and neural FuXi retain 50 members for CRPS. Moreover,
the MME includes raw FuXi. The result therefore shows that a calibrated FuXi
mean can improve on this operationally simple magnitude baseline under paired
verification; it is not evidence of independence from, or universal
superiority over, multi-model forecasting.

**Table 2B:** Retrospective calibrated-FuXi probabilistic and deterministic
scores. The canonical 18-row table combines lead-wise raw, moment, and neural
scores with case count, 50-member count, selected-checkpoint hash, and
pointwise block-16 paired effects and intervals. The separate pooled story
gate is not collapsed into these lead-wise rows.

**[Figure 4 near here.]** Raw, moment-calibrated, and neural FuXi CRPS,
coverage, spread--error, paired CRPS skill, and ACC change; inset: synchronized
2022--2024 retrospective gate. Retrospective neural-versus-fixed-MME and
exact-common-midpoint sensitivities appear as annotations.

<!-- Drafting evidence: canonical comparison
`tables/calibrated_fuxi_vs_mme.csv`; comparison receipt confirms fixed MME
members and ECMWF exclusion; figure-package hard gates and
`data/figure4_calibrated_fuxi_vs_mme.csv`. -->

### 6.4 Unfitted IMERG sensitivity

The selected adapter was trained against IMD and was not refit to IMERG. On
the same six-lead 2022--2024 design, the neural forecast has CRPS 2.7669 versus
3.1515 for raw FuXi: 12.20% skill with a paired interval of 10.51--14.02%. Its
CRPS is also lower than train-only moment calibration (2.8256), giving 2.08%
skill (0.67--3.66%). The ACC difference relative to raw FuXi is positive at
0.0136 but unresolved (-0.0057--0.0335), and the signed-bias difference is
likewise unresolved at 0.0178 mm day\(^{-1}\) (-0.2134--0.2474). Thus the
probabilistic improvement is robust to an unfitted observation reference, but
IMERG does not independently confirm the ACC gain. The latest IMERG target is
2024-12-29 and the latest source label opened is 2024-12-31; no 2025
observation was accessed.

<!-- Drafting evidence: claim P08; IMERG manifest
`imerg_sensitivity_full_20260826T002644Z/manifest.json` and paired interval
table. -->

### 6.5 Regional calibration sensitivity

The calibrated effect is not confined to the all-India aggregate. Across the
four IMD homogeneous regions and six leads, neural CRPS and ensemble-mean RMSE
improve relative to raw FuXi in all 24 region--lead cells, with the frozen
pointwise block-16 intervals excluding zero. The ACC difference is positive
and resolved in 13 of 24 cells and unresolved in the remainder; MAE improves
with intervals excluding zero in 20 of 24. These are retrospective,
pointwise intervals without multiplicity adjustment. Bias, coverage, and
spread--error changes are reported as target-relative diagnostics rather than
monotonic improvement claims, and no regional IMERG inference is made.

<!-- Drafting evidence: claim P09; regional-sensitivity manifest
`india_s2s_calibrated_regional_sensitivity/full_20260826T012159Z/manifest.json`
and its exact frozen interval-lineage receipt. -->

### 6.6 Corrected categorical development baseline

For completeness, the supplement rebuilds a static PBC-inspired baseline--not
a numerical reproduction of rolling issue-wise PBC--with train-only quintile
thresholds, Debias++, Persistence++, and their equal-weight projected
combination under the corrected `+1...+42` target contract
\citep{GuanEtAl2026PBC}. Its score is normalized informative-positive-cut
quintile RPS (NIPC-RPS): mean squared cumulative-probability error over each
row's distinct strictly positive persisted cuts; rows with no such cut are
spatially excluded. The bounded score is not conventional summed RPS. On the
reused all-season 2020--2021 development period, combined PBC reduces pooled
NIPC-RPS by 20.37% relative to raw FuXi (13-start block interval
18.13--22.93%). It improves on Debias++ by 4.36% (3.40--5.19%) but is
statistically unresolved relative to Persistence++ (-0.75--0.67%). This is a
categorical development result, not a CRPS comparison with the continuous
adapter, and it supplies no independent test evidence.

<!-- Drafting evidence: claim C01; aligned PBC manifest
`fuxi_allseason_pbc_baseline_v3_aligned/full_20260826T011400Z/manifest.json`
and `metrics/paired_block_bootstrap.csv`. -->

## 7. Discussion

### 7.1 What the benchmark adds

The raw benchmark separates two forecast attributes that would be obscured by
a single ranking. Equal-system averaging reduces rainfall magnitude error,
while FuXi preserves more late-lead spatial anomaly structure. These are
complementary strengths. The observation-reference sensitivity retains this
qualitative contrast, although IMD and IMERG should not be treated as
numerically interchangeable.

This separation also makes the calibration experiment interpretable. The
adapter is not introduced as an unrelated model comparison: it tests whether
FuXi's retained pattern information can be combined with improved ensemble
distribution and raw-field error. Retrospectively, this occurs. CRPS improves
against both raw FuXi and moment calibration under IMD, and that CRPS direction
also transfers to unfitted IMERG. The calibrated mean lowers RMSE against the
fixed MME at every lead and retains an ACC advantage at five of six leads under
the primary IMD intervals.

### 7.2 What the calibration does not establish

The coverage result is an important qualification. Neural coverage is much
closer to nominal than raw FuXi, but it remains below 0.90 at every lead. The
adapter also does not establish a signed-bias improvement in the pooled gate.
Accordingly, we use “calibration” to name the experiment and report its
specific CRPS, coverage, spread, ACC, RMSE, and bias effects rather than
asserting blanket correction.

The MME comparison requires another qualification. Its six components are
fixed at every lead and include raw FuXi, while ECMWF is excluded throughout.
This supports a stable deterministic baseline, but it does not provide an
independent probabilistic multi-model ensemble. A future experiment could
construct a member-level MME with an explicitly equalized ensemble contract;
that experiment is outside the present evidence.

<!-- Drafting evidence: claims B02--B04 and P02--P09; canonical comparison v2;
evidence-bundle calibrated lead table. -->

## 8. Limitations

First, all calibration results are retrospective. The model was trained only
on 2002--2017 and selected on 2018--2019, but 2020--2021 was reused during
development and the broader 2020--2024 archive had already been examined. The
sealed 2025 target has not been opened. The independent audits verify the
frozen evaluation path and recorded contract within tolerance; they do not
turn retrospective evidence into prospective evidence.

Second, five monsoon seasons provide a limited number of independent seasonal
realizations, and neighboring twice-weekly starts overlap in their valid
windows. Year stratification and circular block resampling address serial
dependence operationally, but uncertainty remains conditional on the archive
and block design. We report the 13-start block sensitivity alongside the
16-start primary analysis. Lead-wise 95% intervals are pointwise and are not
adjusted for multiplicity across leads or metrics.

Third, the forecast systems have unequal native ensemble sizes, and the raw
benchmark compares archived ensemble means. ECMWF Week 3 precipitation is
unavailable. The fixed six-system construction avoids a lead-varying MME but
also excludes otherwise available ECMWF leads. CNRM has only 217 overlapping
main-cycle starts, ERPAS has a different issue cycle and only a 31-case
June--September-initialization 2023--2024 matched-valid-time sensitivity
through Week 4 on a distinct grid/support, and the local Spire archive is
outside the study period. The deterministic calibrated FuXi--MME comparison
should be read under that explicit composition.

Fourth, reference choice matters. IMD is the fitted primary target and IMERG
is an unfitted sensitivity with different measurement and support
characteristics. The evidence supports both the qualitative raw MME--FuXi
contrast and the calibrated CRPS direction across references, but not equality
between products or an independently resolved IMERG ACC gain. Regional masks
and a 1.5-degree grid also limit conclusions about localized extremes and
scales below the verification grid.

The independent alignment audit exactly reconstructs the frozen source,
splits, support, and weekly targets, but the training run did not persist a
logical hash or copy of the target tensor it consumed. Historical byte
identity to that unrecorded tensor therefore cannot be proven retroactively.

Finally, the main paper focuses on rainfall in JJAS. Temperature and
all-season rainfall diagnostics are supplementary and do not constitute a
second main storyline. The corrected categorical baselines are reused
2020--2021 development diagnostics scored with nonconventional NIPC-RPS, not
CRPS competitors in the main continuous comparison. A member-level
probabilistic MME remains outside the present evidence.

<!-- Drafting evidence: claims B01, B03, A02, P08, S01, and G01; figure/table
contracts; corrected and comparison manifests. -->

## 9. Conclusion

A paired five-year India JJAS benchmark shows that raw forecast strengths
separate with lead: the fixed six-system MME controls rainfall magnitude error
through most of the six-week range, while raw FuXi retains stronger late-lead
spatial anomaly skill. A small, corrected location-and-spread adapter converts
that complementarity into an audited retrospective improvement. It lowers
CRPS relative to raw FuXi and train-only moment calibration, increases ACC
relative to raw FuXi in the pooled gate, and gives a mean forecast with lower
RMSE than the fixed MME at every lead. Coverage improves markedly but remains
below nominal, and no signed-bias improvement is established.

The standalone contribution is therefore the connection between diagnosis
and intervention under one verification contract: benchmark where magnitude
and pattern skill reside, then test whether constrained post-processing can
combine them. This conclusion is limited to the retrospective 2020--2024
archive. The 2025 target remains sealed, and the final prospective headline
gate remains pending explicit authorization and a pre-specified one-time
direction check.

<!-- Drafting evidence: claims B02, P02--P09, S01, and G01. -->

## AI-use disclosure

Generative AI tools assisted code drafting and documentation. The authors
independently specified the scientific protocol, re-derived the data alignment
and evaluation metrics, audited all data splits and provenance, reproduced the
reported results from frozen code, and take full responsibility for the
methods and claims.

## Bibliography

The verified BibTeX bibliography is maintained in `references.bib` and is
resolved during manuscript typesetting.
