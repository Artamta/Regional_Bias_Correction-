# Frozen FuXi Bridge: IMERG Sensitivity

## Status

Complete unfitted observation-reference sensitivity. Canonical result:
`resultsv3/india_s2s_probabilistic_bridge/imerg_sensitivity_full_20260826T002644Z`.
Manifest SHA-256:
`0d127b1b75b31179bc48a6a1755978da43085891afbae478016a5a04ee1efe7d`.

## Frozen Contract

- Forecasts: selected seed-43 bridge predictions; 517 forecast-only starts for
  each method-specific leave-one-year-out climatology.
- Verification: the same 505 starts used by the IMD bridge, scored against
  IMERG 2020--2024 only.
- Period labels: W1 `init..init+6`; W6 `init+35..init+41`.
- Normal: IMERG 2001--2019; weekly support is the minimum of seven daily
  observation fractions.
- Methods: raw FuXi, train-only moment calibration, and the selected neural
  location-spread adapter, each with the same 50 operational members.
- Regions: all India and four fractional IMD homogeneous regions.
- Uncertainty: paired year-stratified circular blocks of 16 starts, with 13 as
  sensitivity; 10,000 replicates.

## Main Result

For synchronized 2022--2024 all-India valid-midpoint JJAS (100 cases per lead,
600 case-lead rows), neural CRPS is 2.766890 versus 3.151507 raw and 2.825593
moment-calibrated. Skill is 12.2042% versus raw (95% interval
10.5092--14.0175%) and 2.0775% versus moment (0.6697--3.6625%).

The ACC increase versus raw is +0.013586 (-0.005676--0.033500), so the point
gain is not resolved under IMERG. Signed-bias change versus raw is +0.017761
mm/day (-0.213400--0.247407). The defensible sensitivity claim is therefore
probabilistic robustness, not independently confirmed ACC or bias improvement.

## Verification

All 15,150 common raw-FuXi case/lead/region rows reproduce the frozen IMERG
benchmark for ACC, RMSE, MAE, bias, valid-cell count, and effective area. All
15 result artifacts match the manifest. The receipt opens only 2020--2024;
the latest scored target is 2024-12-29. The 12 late-2024 starts whose weeks
would require 2025 truth are excluded.
