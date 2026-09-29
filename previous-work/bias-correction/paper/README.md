# Paper Workspace

This directory is the claim-controlled scaffold for the working paper
**Benchmarking and Calibrating Subseasonal Monsoon Rainfall Forecasts over
India**.

The paper now has a complete benchmark backbone and an audited corrected
2020--2024 retrospective calibration evaluation. Independent training-
alignment, table-level, and prediction-level audits passed. The six
lead-specific 2022--2024 IMD cohorts meet the CRPS and ACC gates, while the
bias guardrail finds no significant change; the unfitted IMERG sensitivity confirms
the CRPS direction but not a statistically resolved ACC increase. The final
headline/prospective gate is still pending the explicitly authorized one-time
sealed-2025 point-direction check. Keep the manuscript benchmark-first, label
calibration evidence retrospective, and do not promote reused 2020--2021
development scores.

## Files

- `MANUSCRIPT_OUTLINE.md`: section-level story and fallback route.
- `MANUSCRIPT_DRAFT.md`: claim-controlled full prose draft.
- `CLAIM_REGISTRY.md`: allowed, qualified, pending, and withdrawn statements.
- `FIGURE_TABLE_CONTRACTS.md`: required inputs and checks for every planned
  display.
- `IMPLEMENTATION_STATUS.md`: completed artifacts and deliberate remaining
  gates.
- `references.bib`: verified primary bibliography for the manuscript draft.
- `ARTIFACTS.sha256`: byte-level bindings for all evidence cited here.

Run the artifact check from the repository root:

```bash
sha256sum -c paper/ARTIFACTS.sha256
```

Do not hand-copy new scores into prose. First add the immutable result table
and manifest to `ARTIFACTS.sha256`, then update the claim registry, and only
then update the manuscript. The accepted retrospective evidence is frozen at
`resultsv3/india_s2s_probabilistic_bridge/scoring_full_20260825T234808Z`
(manifest `4ab303...0943`), with table audit `audit_full_20260826T000133Z`
(`d7c822...7bcc`) and prediction audit
`prediction_audit_full_20260826T002645Z` (`0f176f...7168`). The historical
training alignment is reconstructed in
`resultsv3/fuxi_allseason_training_alignment_audit/audit_full_20260826T005845Z`
(`618819...3a07`), with its unpersisted-target-hash limitation explicit. Paper tables and
figures are frozen under `india_s2s_paper_release_bundle/full_v3_20260826T012700Z` and
`india_s2s_paper_figures/full_v4_20260826T011100Z`. The `resultsv2/`
probabilistic calibration line is withdrawn because its IMD targets were
shifted one day early. All corrected receipts record
`sealed_2025_target_opened=false`; this workspace contains no 2025 result.
The separately validated ERPAS figure is bound as Supplementary Figure S3;
it is a 31-case 2023--2024 June--September-initialization matched-valid-time
sensitivity on a distinct IMD grid/support, not part of the five-year ranking.
CNRM and Spire exclusions are recorded in claim B04.
