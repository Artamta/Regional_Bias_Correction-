# Manuscript Outline

## Working title and thesis

**Benchmarking and Calibrating Subseasonal Monsoon Rainfall Forecasts over
India**

The standalone story is not that every forecast is uniformly bad. It is that
JJAS rainfall skill decays sharply with lead, while two useful properties
separate: a fixed multi-model mean controls magnitude error and FuXi-S2S
retains the strongest late-lead anomaly-pattern skill. This motivates a
lightweight, member-preserving calibration test rather than appending an
unrelated neural method.

## Abstract structure

1. State the India JJAS common-date verification gap without claiming the
   first India S2S benchmark.
2. Report the supported 169-case-per-lead IMD benchmark contrast (claims B01
   and B02).
3. Describe the corrected +1...+42 location-and-spread adapter and the
   validation-only single-checkpoint selection (claims A02 and A03).
4. Report the audited retrospective result: over six lead-specific
   2022--2024 valid-midpoint JJAS cohorts, with bootstrap draws synchronized by
   within-year ordinal, neural CRPS skill is 16.21% versus raw FuXi
   (95% interval 14.81--17.69%) and ACC increases by .0261
   (.0087--.0436). Do not imply that the sealed-2025 direction check has
   occurred.
5. Close with the deterministic payoff: the calibrated FuXi mean has lower
   RMSE than the fixed MME at all six leads. If space permits, add that the
   unfitted IMERG sensitivity retains the CRPS gain but not a resolved ACC
   gain.

## Main text

### 1. Introduction

- Operational value and difficulty of Weeks 1--6 Indian monsoon rainfall.
- Need for common dates, grid, truth, climatology, weights, and uncertainty
  across AI and dynamical systems.
- Contributions: paired five-year benchmark; regional/reference sensitivity;
  and audited retrospective FuXi ensemble calibration. Keep prospective
  generalization conditional on G01.

### 2. Data and verification

- Seven individual precipitation systems and a fixed six-system equal-weight
  MME; IMD primary and IMERG sensitivity.
- Valid-midpoint JJAS assignment, W1 `init+1...+7` through W6
  `init+36...+42`, 27 x 27 grid, 171 fixed scoring/training cells, and 174
  cells intersected by the fractional region geometry.
- Method-specific four-year leave-one-year-out forecast climatology, IMD
  1991--2019 normal, fractional regional masks, and paired block uncertainty.

### 3. Five-year benchmark

- Lead-time decay across all systems.
- Descriptive MME--FuXi contrast from B02.
- Regional heterogeneity and IMD--IMERG sensitivity from B03.
- Keep ERPAS as the separate 31-case 2023--2024 W1--W4 matched-valid-time
  sensitivity from B04. Its June--September initialization filter and 22 x 22,
  169-cell IMD support differ from the main valid-midpoint cohort and grid.
  Exclude CNRM, ERPAS, and Spire from the five-year ranking because their
  cohort, issue-cycle, lead, support, or climatology contracts do not match it.

### 4. Corrected FuXi calibration

- Train 2002--2017; validate/select on 2018--2019; treat 2020--2021 as reused
  development only.
- Freeze the 42,434-parameter location-and-spread model; select seed 43 by
  validation CRPS; compare raw, train-only moment calibration, and neural
  forecasts on identical members.
- Report the independent alignment reconstruction: exact `+1...+42` targets,
  1,652/196/208/24 train/validation/development/embargo cases, and a byte-exact
  native lead-day spot check. State that the original run did not preserve a
  logical target-tensor hash, so historical tensor equality is not provable.
- Keep D01--D02 out of the abstract and headline table.

### 5. Retrospective calibrated benchmark

- Report the 2020--2024 169-case-per-lead results from P03 and the
  synchronized-ordinal 2022--2024 pooled gate from P02. The latter has six
  lead-specific 100-case cohorts (600 case-lead rows), not one common 100-date
  cohort.
- Neural CRPS skill versus raw is 29.23, 22.45, 15.14, 12.12, 10.23, and
  9.07% at W1--W6. ACC gains exclude zero at W1--W4 and W6; W5 is unresolved.
- In the primary pooled 16-start-block analysis, neural CRPS skill is 16.21%
  (14.81--17.69%) versus raw and 4.26% (2.99--5.70%) versus moment
  calibration. The ACC delta versus raw is .0261 (.0087--.0436), while the
  signed-bias delta is unresolved at .0201 (-.2070--.2461) mm day-1.
- Report CRPS, ACC, RMSE, bias, coverage, spread/error, regions, lead-wise
  effects, and paired intervals from the single manifest-bound evaluation.
- Label every result retrospective. Both independent audits passed, but they
  do not replace the prospective check.
- Report the exact-common-midpoint result as a sensitivity: 85 midpoints per
  lead, with a deliberately disclosed 35/35/15 split across 2022/2023/2024.
- Compare the calibrated ensemble mean with the fixed MME on the 169 paired
  cases per lead. RMSE improves at W1--W6; ACC improves at W1 and W3--W6,
  while W2 is unresolved (P07).
- Report the unfitted IMERG transfer separately: CRPS skill is 12.20% versus
  raw and 2.08% versus moment, while the ACC delta is positive but unresolved
  (P08).
- State that the alignment, table-level, and prediction-level audits passed
  (A04 and P04--P05), with A04's historical-hash limitation intact.

### 6. Discussion and limitations

- Distinguish pattern skill from amount calibration.
- State archive upgrades, unequal ensemble sizes, overlapping starts,
  reference sensitivity, retrospective development exposure, and absence of
  prospective evidence.
- Temperature and all-season diagnostics remain supplementary.

## Story gate

**Headline inclusion:** corrected 2022--2024 CRPS skill and ACC change versus
raw FuXi are positive with paired 95% intervals excluding zero; no serious
bias degradation occurs; and, only after separately authorized one-time
evaluation, the 2025 points do not reverse either direction.

**Qualified inclusion:** CRPS passes, ACC is positive but its interval includes
zero, the bias guardrail shows no serious degradation, and any authorized 2025 points do not reverse the
direction. Claim only improved probabilistic calibration with unresolved
point-skill change.

**Rejection:** CRPS fails, ACC reverses, serious bias degradation appears, or
the adapter is clearly inferior to the corrected moment baseline. Remove the
calibration row, figure, and numbers.

The current state is **retrospective component passed; final headline gate
pending**. The six-cohort 2022--2024 analysis passes the CRPS and ACC gates,
beats the moment baseline in CRPS, and shows no significant signed-bias change.
No 2025 evaluation is
authorized or reported, so prospective/generalization language remains
prohibited. Until that one-time point-direction check is explicitly
authorized and passes, retain the benchmark-first framing; the fallback title
remains **A Paired Five-Year Benchmark of Subseasonal Monsoon Rainfall
Forecasts over India**.
