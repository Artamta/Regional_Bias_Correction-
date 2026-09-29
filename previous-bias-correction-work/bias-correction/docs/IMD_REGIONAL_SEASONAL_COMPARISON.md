# IMD regional and seasonal comparison

This sensitivity compares **Model v1** (`raw_fuxi`) with **Probabilistic
Correction v1** (`location_spread`) for four existing IMD homogeneous-region
aggregates, four valid-period-midpoint seasons, and lead weeks W1–W6.

## Frozen scope

- Verification: IMD, 2020–2024 only; no 2025 observation was opened.
- Regions: Northwest India, Central India, South Peninsula, and East &
  Northeast India.
- Seasons: JF, MAM, JJAS, and OND, assigned from the midpoint of each seven-day
  valid period.
- Source: accepted bridge scoring run
  `resultsv3/india_s2s_probabilistic_bridge/scoring_full_20260825T234808Z`.
- Uncertainty: paired, year-stratified, four-start circular-block bootstrap;
  10,000 replicates. Intervals are pointwise and have no multiplicity
  correction.

## Result

Resolved improvement / resolved degradation counts out of 24 region–lead
cells per season are:

| Season | CRPS | RMSE | ACC |
| --- | ---: | ---: | ---: |
| JF | 14 / 0 | 8 / 5 | 3 / 5 |
| MAM | 24 / 0 | 19 / 0 | 3 / 4 |
| JJAS | 24 / 0 | 24 / 0 | 12 / 0 |
| OND | 20 / 0 | 11 / 0 | 1 / 3 |

The defensible summary is that probabilistic correction improves CRPS broadly
across regions and seasons. RMSE is uniformly favorable only during JJAS, and
ACC is mixed outside JJAS. East & Northeast India has the largest average
CRPS improvement in every season. ERPAS and Deterministic Correction v1 are
not shown because comparable all-season forecasts are unavailable.

The canonical package is
`resultsv3/india_s2s_regional_seasonal_comparison/full_20260902T000001Z`.
It contains exact descriptive and interval tables, PNG/PDF figures, bootstrap
receipts, a copied compiler, and a hash manifest.
