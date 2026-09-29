# Figure and Table Contracts

Every generated display must carry source artifact hashes in its manifest.
Captions must state reference, cohort definition, case count, spatial
weighting, metric aggregation, and uncertainty method. Plot scripts must read
tables directly; manual numerical transcription is not allowed.

The canonical table release is
`resultsv3/india_s2s_paper_release_bundle/full_v3_20260826T012700Z`
(manifest `ac7c01...ed98`). It contains contract-complete Tables 1, 2A, and
2B in CSV, Markdown, and LaTeX plus the audit/safety tables.

The canonical generated package is
`resultsv3/india_s2s_paper_figures/full_v4_20260826T011100Z` (manifest
`60c0d8...4ce1`). Figures 1--4 and supplementary Figures S1--S2 have PDF and
PNG outputs; all six were visually inspected under the bound receipt
`edc0b4...b1c4`.

## Main figures

### Figure 1: domain and temporal contract

- Panels: 27 x 27 verification grid with 171 fixed forecast/calibration-
  support cells; four fractional IMD homogeneous regions whose geometry
  overlaps 174 cells, including three hatched cells outside the fixed support;
  W1--W6 valid windows; train/validation/development/retrospective split
  timeline. This is a grid-cell schematic, not political/coastline geometry.
- Inputs: benchmark methods manifest, corrected manifest, independent
  alignment-audit receipt, and bridge cohort receipt.
- Guard: show 2025 as sealed, never as an observed result.

### Figure 2: five-year JJAS benchmark

- Panels: all-system IMD ACC and RMSE versus lead; emphasize the fixed MME and
  FuXi without hiding other systems.
- Inputs: IMD `seasonal_summary.csv` (`fe13e4...770`) and manifest-bound
  MME-versus-FuXi intervals (`de7680...9c18`), recomputed from bound case
  metrics (`000544...a14`) under the paper's frozen paired year-stratified
  circular-block protocol.
- Guards: valid-midpoint JJAS only; N=169 for each available system--lead
  pair; fixed six-system MME;
  explicit ECMWF W3 missing marker; significance language is limited to the
  paired MME-versus-FuXi contrasts actually present in the interval table.
  Figure 2 itself shows descriptive curves without error bars; paired
  intervals are reported in Table 2A and the text.

### Figure 3: regional and reference sensitivity

- Panels: region x lead ACC for the fixed MME and FuXi; matched IMD-versus-
  IMERG ACC contrast on the same valid-midpoint view.
- Inputs: IMD and IMERG seasonal/case tables listed in `ARTIFACTS.sha256`.
- Guards: keep all-India and four fractional regions distinct; do not compare
  RMSE magnitudes across references as if their support and daily products
  were identical.

### Figure 4: retrospective calibrated FuXi effect

- P02--P07 support this figure as retrospective evidence. Do not label it a
  prospective or final-gate result while G01 is pending.
- Required rows: raw FuXi, corrected train-only moment calibration, and the
  single selected neural checkpoint on identical 50-member cases.
- Required panels: paired CRPS skill and ACC delta by lead; coverage and
  spread/error diagnostic; bias guard. Include all-India plus regional
  sensitivity in the supplement.
- Inputs: scoring JJAS summary (`46afe6...f383`), paired interval table
  (`a29e6a...283e`), selected checkpoint hash
  `a9a71465e2399773d5968bc94e4a5d535826b9ec5d1c4d3ec555fd8449586fee`,
  passed table audit (`d7c822...7bcc`), prediction audit (`0f176f...7168`),
  alignment audit (`618819...3a07`), and comparison manifest
  (`3b754c...c85da`).
- Regional supplement input: calibrated regional-sensitivity manifest
  (`7ec5c6...73f2`), with 72 descriptive method--region--lead rows and the
  exact 240 primary block-16 interval rows copied from the frozen scorer.
- Primary gate inset: six lead-specific 2022--2024 valid-midpoint cohorts, 100
  cases per lead and 600 case-lead rows, with bootstrap draws synchronized by
  within-year ordinal. The lead-specific date sets have an
  intersection of 70 and union of 130; never describe them as one shared
  100-date cohort. Use the 16-start block interval as primary and 13 as
  sensitivity.
- Upstream traceability must retain case identity (`initialization`, serialized
  as `init` in the upstream case table), lead, region, method, metric,
  reference, and artifact provenance. The rendered v4 package intentionally
  contains aggregate display tables; those tables must carry `estimate`,
  `baseline`, `paired_effect`, `ci_lower`, `ci_upper`, `n_cases`, and
  `member_count` where applicable and point back to the manifest-bound case
  table rather than pretending to contain case rows themselves.
- Current state: rendered and visually checked as a retrospective figure; the
  prospective headline state remains pending.

## Main tables

### Table 1: systems and availability

Source the seven systems, experiments, ensemble treatment, MME membership,
variables, years, and unavailable leads from the benchmark methods manifest.
State that the comparison is between archived ensemble-mean systems with
unequal native ensemble sizes.

### Table 2A: deterministic JJAS benchmark

Use IMD ACC/RMSE/MAE/bias and N by system and lead. Populate the dedicated
MME-minus-FuXi contrast columns from the bound paired interval table; interval
cells for other system contrasts remain pending. Never silently substitute
naive independent-case intervals.

### Table 2B: retrospective calibrated FuXi

Populate from the bound scoring tables under P02--P08. Report raw, moment, and
neural CRPS; ensemble-mean ACC/RMSE; bias; coverage90; spread/error; paired
effects; interval; N; and selected checkpoint hash. State 169 cases per lead
for 2020--2024 and the separate six-cohort, 100-per-lead 2022--2024 gate.
Do not average seed scores into a forecast or collapse lead-specific dates
into a fictitious common cohort.

## Supplement

- Full regional system-by-lead tables and IMD--IMERG sensitivity.
- Unfitted calibrated-FuXi IMERG sensitivity from P08, preserving its
  resolved CRPS and unresolved ACC/bias distinction.
- Calibrated-FuXi IMD regional sensitivity from P09; intervals are pointwise
  and no regional IMERG inference is available.
- One all-season rainfall figure and one validated temperature figure.
- Supplementary Figure S3: the separately validated 31-case 2023--2024
  ERPAS--FuXi matched-valid-time extension through W4. Cases are selected by
  June--September ERPAS initialization, not by the main valid-period-midpoint
  rule. Use the archived PDF (`fda837...4d8d`), method audit
  (`63743d...cdb`), and 4,000-resample table (`026968...e1cf`). State that
  ERPAS is issued Wednesday, FuXi is the preceding-Monday forecast, the
  verified windows match, intervals are pointwise, and provider-climatology
  years are undocumented. State also that it uses IMD with a 1991--2020
  climatology on a distinct 22 x 22, 1.5-degree, 169-cell support; its values
  are not numerically comparable to Figure 2's 27 x 27, 171-cell main cohort.
  Do not append these curves to Figure 2.
- Corrected 2020--2021 development diagnostics, labelled reused development
  in the title, caption, table header, and prose.
- Corrected categorical Debias++/Persistence++/combined-PBC table from the
  aligned PBC manifest (`040945...29c7`). Call it a static PBC-inspired
  baseline, not a numerical reproduction. Report only normalized informative-
  positive-cut quintile RPS/RPSS (NIPC-RPS/NIPC-RPSS): distinct positive cuts
  count once, all-zero-cut rows are excluded, and the bounded score is not
  conventional summed RPS. Keep the 2020--2021 reused-development label and
  state that combined PBC does not resolve an advantage over Persistence++.
- Seed sensitivity, reliability, and rank histograms as diagnostics only.
- Withdrawal note listing every v1-derived artifact family excluded from the
  manuscript.

## Release checks

1. `sha256sum -c paper/ARTIFACTS.sha256` passes.
2. Every plotted row traces to a manifest-bound table and exact case IDs.
3. Raw FuXi bridge metrics match the benchmark evaluator on common cases.
4. The prediction-level audit receipt remains `passed`: its independent
   implementation reproduces CRPS, ACC, RMSE, MAE, bias, coverage, spread,
   spread/error, and regional aggregation for all 45,450 rows within the
   frozen tolerances.
5. Search the manuscript for `resultsv2`, `2025`, `first`, `state of the art`,
   and development metrics; each occurrence must satisfy the registry.
6. The IMD and IMERG target receipts still end at 2024-12-30 and 2024-12-29,
   respectively; the IMERG source-label receipt ends at 2024-12-31, and all
   audits report `sealed_2025_target_opened=false`.
7. The ERPAS supplement remains a separate 2023--2024, 31-case, W1--W4
   matched-valid-time sensitivity selected by ERPAS initialization month and
   scored on its distinct grid/support; CNRM and Spire remain outside the main
   ranking.
