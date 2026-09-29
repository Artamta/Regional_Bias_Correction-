# Paper Implementation Status

Updated: 26 August 2026

## Completed and frozen

- Corrected `+1...+42` FuXi location-and-spread training, three seeds, and
  validation-only selection of seed 43.
- Forecast-only inference for 517 operational 2020--2024 starts and IMD
  scoring for the 505 starts that do not require 2025 truth.
- Ten-thousand-replicate paired year-stratified circular-block uncertainty,
  with 16-start primary and 13-start sensitivity intervals.
- Independent training-alignment, table-level, and prediction-level audits.
  The alignment audit reconstructed all 2,080 weekly targets and split IDs;
  the prediction audit rebuilt daily targets, members, climatologies, and all
  45,450 case-metric rows.
- Unfitted IMERG sensitivity, exact-common-midpoint sensitivity, and paired
  calibrated-FuXi-versus-fixed-MME comparison.
- Four-region calibrated-FuXi IMD sensitivity compiled directly from frozen
  case summaries and interval rows, without new resampling or array access.
- Corrected train-only categorical Debias++, Persistence++, and static
  PBC-inspired combination under the explicit `+1...+42` target guard. Their
  nonconventional normalized informative-positive-cut quintile RPS evaluation
  remains reused 2020--2021 development evidence only.
- Contract-complete, manifest-bound Tables 1, 2A, and 2B; Figures 1--4 and supplementary all-season
  rainfall/temperature Figures S1--S2 in PDF/PNG, and a full claim-controlled
  manuscript draft. All six canonical figures passed visual inspection.
- A separately validated, hash-bound ERPAS--FuXi Supplementary Figure S3 and
  paired 3--5-start block sensitivity are integrated as a 31-case 2023--2024
  June--September-initialization matched-valid-time addendum. Its distinct IMD
  climatology and 22 x 22, 169-cell support are explicit. CNRM, ERPAS, and
  Spire exclusion reasons are recorded; none enters the fixed five-year
  ranking.

The retrospective evidence supports the combined paper: corrected FuXi
improves CRPS versus raw and moment calibration; its deterministic mean beats
the fixed MME in RMSE at all six leads; and the CRPS direction transfers to
IMERG. Coverage remains sub-nominal, pooled signed-bias improvement is not
established, and the IMERG ACC increase is unresolved.

The alignment receipt also records a non-recoverable historical limitation:
the original run did not persist a logical hash or copy of the target tensor it
consumed. The audit proves the frozen source/current-source contract,
reconstructed arrays, splits, support, cache bytes, and native spot check, but
cannot prove byte equality to that unrecorded historical tensor.

## Deliberate remaining gates

- Typeset the submission/supplement from the verified BibTeX bibliography and
  generated LaTeX tables; a BibTeX executable is not installed in this workspace.
- Open the sealed 2025 target only after explicit authorization, once, under
  the pre-specified direction check. Until then, all calibration language must
  remain retrospective and no prospective headline is permitted.

The original `FINAL-PAPER-RESULTS/final-check.MD` planning file remains
unchanged.
