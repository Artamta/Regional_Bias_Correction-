# FuXi–IMD six-week rainfall post-processing

This is the working directory for FuXi-to-IMD weekly rainfall post-processing
over India. It contains the established deterministic adapter line and a new
member-preserving probabilistic calibration line; both predict six weekly mean
rainfall fields in mm/day. It is separate from the probabilistic global
ERA5/AI-Quest project in `../ai-quest-global`; scores, targets, and evidence
from the two projects must not be mixed.

## Current scientific position

Updated 26 August 2026:

> **Alignment withdrawal (26 August 2026):** The immutable v1 probabilistic
> calibration snapshot used end-labelled IMD offsets 0--41, one day earlier
> than FuXi native leads and the India benchmark contract. Its checkpoints,
> fitted baselines, downstream categorical products, metrics, and figures are
> withdrawn from manuscript use. The numerical bullets below are retained only
> as historical context and are not valid evidence. Corrected retraining is
> versioned under `resultsv3/`; see
> `plan/ICLR2027_ALIGNED_ENSEMBLE_CALIBRATION_20260826.md`.

### Corrected `resultsv3/` probabilistic branch (current paper evidence)

- The corrected adapter uses end-labelled IMD offsets `+1...+42`, trains on
  2002--2017, selects seed 43 on 2018--2019 validation CRPS, and treats all
  2020--2024 results as retrospective.
- Across six lead-specific 2022--2024 JJAS cohorts, it improves CRPS by 16.21%
  versus raw FuXi and 4.26% versus train-only moment calibration; ACC increases
  by .0261, while the signed-bias change is unresolved.
- The unfitted IMERG sensitivity retains the CRPS direction (12.20% versus raw
  and 2.08% versus moment), but its ACC change is unresolved.
- The calibrated ensemble mean has lower RMSE than the fixed six-system MME at
  all six leads. Regional CRPS and RMSE improvements versus raw FuXi are
  resolved in all 24 IMD region--lead cells; regional ACC is resolved in
  13/24 and otherwise unresolved.
- Corrected train-only categorical Debias++, Persistence++, and a static
  PBC-inspired combination were rebuilt under the same `+1...+42` guard.
  Their nonconventional normalized informative-positive-cut quintile RPS
  evidence is limited to reused all-season 2020--2021 development data and is
  supplementary.
- ERPAS is retained only as a separate 31-case, 2023--2024 June--September-
  initialization matched-valid-time W1--W4 supplement on a distinct IMD
  grid/support. CNRM, ERPAS, and Spire do not enter the fixed five-year ranking
  because their cohort or verification contracts do not match it.
- Training-alignment, table-level, and prediction-level audits passed. The
  alignment receipt explicitly records the original run's missing logical
  target-tensor hash. The 2025 target remains sealed. Canonical claims and
  hashes are in `paper/CLAIM_REGISTRY.md` and `paper/ARTIFACTS.sha256`; the
  contract-complete table release is
  `resultsv3/india_s2s_paper_release_bundle/full_v3_20260826T012700Z`.

### Withdrawn v1 probabilistic history (do not cite)

Every numerical bullet in this subsection, through the categorical package
ending immediately before “Current deterministic adapter line,” is retained
only to explain prior decisions. It is superseded by the corrected branch
above and prohibited from manuscript use.

- The new all-season experiment retains all 51 FuXi members and calibrates
  their location and spread over the 39N--0N, 60E--99E India box. On the
  reused 208-start 2020--2021 development evaluation it improves pooled CRPS by
  16.37% (95% initialization-block interval 14.28% to 18.16%), RMSE by
  12.57%, MAE by 9.24%, and ACC by 0.121 versus raw FuXi. RMS spread / pooled
  RMS error improves from 0.562 to 0.967. This is promising probabilistic
  calibration evidence, not an independent final test and not a successful
  signed-bias claim. The frozen run and derived manuscript bundle are indexed
  in `resultsv2/README.md`.
- The exact frozen adapter also passed a no-retraining 2022--2024
  operational-era retrospective audit over all 296 eligible starts and 50
  members. Pooled CRPS improves from 1.5626 to 1.3164: 15.76% skill (paired
  95% interval 13.78% to 17.76%), with positive skill at W1--W6 and in each
  year. RMSE improves 10.02%, MAE 8.07%, ACC by 0.063, 90% coverage from
  0.415 to 0.800, and spread/error from 0.566 to 1.032. Signed-bias change is
  unresolved, W5 becomes significantly more negative, and only three year
  clusters support the interval. This is strong retrospective transfer
  evidence, not prospective operational validation.
- In the accepted reused-2020--2021 common-support categorical comparison,
  Persistence++ is best pooled; the neural location+spread adapter beats
  Persistence++ and combined PBC at W1, is unresolved against the strongest
  classical methods at W2, and loses to them at W3--W6. This lead
  crossover—not universal neural dominance—is the defensible development
  result.
- A separate accepted post-hoc operational-era categorical retrospective
  applied the frozen methods to all 296 eligible 2022--2024 starts without
  retraining, selection, or fitted blending. Relative to raw FuXi, the neural
  adapter reduces pooled quintile RPS by 23.14% (exploratory 95% interval
  15.87--30.60%), semidecile RPS by 23.61% (16.09--31.41%), and descriptive
  upper-q95 Brier score by 12.24% (9.01--16.54%). It again beats Persistence++
  and combined PBC at W1; for pooled ordinary RPS it is numerically slightly
  worse but statistically indistinguishable from both, while its pooled q95
  Brier score is lowest. These intervals have only three year clusters, the
  252 paired comparisons are unadjusted for multiplicity, and neural headline
  rows condition on the mean of per-seed scores rather than one deployable
  forecast. This is not global, prospective, 2025, event, or extremes
  evidence. The accepted manifest/semantic/Slurm/bootstrap SHA-256 values are
  `cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d`,
  `00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c`,
  `cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c`,
  and `632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188`.
  Its receipt-bound tables and publication figure are at
  `presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z`
  (manifest SHA-256
  `bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a`).

### Current deterministic adapter line

- The frozen raw-identity ensemble is the strongest current deterministic
  candidate. It averages the standardized residuals from three independently
  trained 144,689-parameter members. On the fixed 100-start 2022--2024 audit, it
  improves raw FuXi RMSE from 5.723 to 5.239 mm/day, MAE from 3.822 to 3.453,
  and ACC from 0.276 to 0.364. It also modestly improves the legacy anchored
  adapter's RMSE and MAE.
- This is not a complete bias correction. Raw identity worsens IMD pooled bias
  from -0.225 to -0.858 mm/day. A no-fit raw-mean constraint restores it to
  -0.216 and gives RMSE 5.227, but worsens MAE and is therefore a Pareto
  diagnostic rather than the main method.
- On a frozen external 2024 gauge-derived 1.5-degree-cell target (30 starts ×
  6 leads), raw
  identity improves raw FuXi RMSE by 0.449 mm/day, MAE by 0.391, and ACC by
  0.0276. The raw-mean constraint degrades all of those station metrics,
  showing that its calibration benefit is target-dependent. This raw-identity
  contrast was a frozen secondary comparison in the station analysis, not its
  predeclared primary estimand.
- Matched worldwide FuXi-to-IMERG patch pretraining did not transfer: its
  India RMSE was 5.508 versus 5.481 for the identical scratch ensemble, ACC
  was 0.312 versus 0.321, and zero of three pretrained seeds achieved a lower
  best India validation composite loss than its matched scratch seed.
  This log-bias-anchored transfer experiment is therefore a completed negative
  component result; it was not a raw-identity pretraining test.
- The compact physical-variable model passed its 2018–2019 validation guards,
  but its gain is small and it has no independent confirmation. Its extra
  variables are also unavailable in the current 2025 operational path.
- Bias-aware, heavy-tail, intensity-regime, affine, and fixed Box–Cox screens
  did not pass their promotion rules. These negative results are retained and
  should not be silently re-labelled as selected models.
- The full blocked out-of-fold affine run failed while entering fold 2. Its
  artifacts and logs are archived; it is not evidence.
- Heavy and extreme rainfall remain a limitation. Raw identity has favorable
  pooled skill, but its >=20 mm/day RMSE interval crosses zero and its extreme
  signed dry bias is large; do not claim an extremes improvement.
- The 2025 initialization-year evaluation remains untouched. The exact
  raw-identity-versus-raw selection is now frozen (selection SHA-256
  `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6`),
  and storage-incapable synthetic GPU preflight job 109981 passed. No access
  ledger or result exists; the one-time run still requires a new approval
  bound to the frozen selection and preflight hashes.

The authoritative status table and publication route are in
`docs/EXPERIMENT_LEDGER.md`. The repository dependency map is in
`docs/REPOSITORY_MAP.md`.

## CCAI NeurIPS 2026 submission source

The compact anonymous source package is at
`presentation/deliverables/ccai_neurips2026_submission_source_20260823`.
It uses the byte-exact official style and the receipt-bound operational
categorical figure. Run `bash validate.sh` inside that directory before upload.
The final `main.tex` SHA-256 is
`0538c8ffc253c854bc304d67e03940ee80954a1126703b3b341698a972a72116`.
Tectonic 0.16.9 compiled `main.pdf` with SHA-256
`ad335e16c38a041e2a7b1d108d0aeb438c190ff7b8d091d79e638ea1ec468258`.
The PDF is five US-letter pages: all main text occupies pages 1--4 and page 5
contains references only, satisfying the workshop's four-page main-text limit.

## Evidence boundary

| Years | Role | Allowed use |
|---|---|---|
| 2002–2017 | Training and train-only preprocessing | Fit models, anchors, normalization, and climatology |
| 2018–2019 | Reused blocked validation | Selection evidence; further tiny screens risk overfitting |
| 2020–2021 | Reused exploratory hindcast | Diagnostics only, never independent confirmation |
| 2022–2024 | Post-hoc frozen operational-era retrospective | Strong transfer evidence, but neither prospective nor a final test |
| 2025 | Untouched final initialization year | Open exactly once after the complete system is frozen |

## Environment and tests

The experiment code imports the sibling package at
`../neural_adapter/src/fuxi_adapter`. The documented
`s2s_core` and `fuxi` Conda environments currently do not include
pytest, so install the dependencies into a controlled environment before
relying on them:

```bash
python -m pip install -r requirements.txt
```

Run the synthetic and contract suite without creating bytecode or a pytest
cache:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:evaluate:hpc_compat:../neural_adapter/src \
python -m pytest -q -p no:cacheprovider
```

The current compact-layout baseline is 303 passing tests. These tests include
the frozen E2/E3 contracts and Slurm routing regression, but they do not turn
the failed historical fold-2 OOF run into evidence.

## Working layout

| Path | Purpose |
|---|---|
| `src/` | Active experiment, model, loader, and cache code |
| `evaluate/` | Evaluation, freeze, verification, plotting, and report generators |
| `tools/` | Convenience loaders and legacy model-bundle utilities |
| `slurm/` | Cluster launch wrappers |
| `tests/` | Synthetic, contract, leakage, and packaging tests |
| `plan/` | Timestamped, venue-neutral frozen analysis plans |
| `docs/` | Scientific status, protocols, and archived meeting notes |
| `results/` | Generated run artifacts; Git-ignored except for its index |
| `resultsv2/` | New immutable post-cleanup runs; Git-ignored except for its index |
| `resultsv3/` | Corrected +1...+42 alignment runs; Git-ignored except for its index |
| `reports/` | Hash-verified artifact-only evidence packages and their index |
| `cache/` | Finalized feature caches; Git-ignored |
| `logs/` | New Slurm logs; Git-ignored |
| `presentation/` | Generated/static figure and report payloads |
| `archive/` | Compressed failed-run/log evidence removed during cleanup |

The project root intentionally contains no Python files. Several older
`src/fuxi_imerg_*` modules remain because the active IMD workflow imports their
shared loading, model, and diagnostic logic. Do not delete them until those
common functions are extracted behind tested interfaces.

Run Python entry points from the project root, for example
`python src/fuxi_imd_blocked_oof_affine.py --help` or
`python evaluate/preflight_raw_identity_2025.py --help`, with the
`PYTHONPATH` shown above. Slurm launchers already set the required paths.

## Canonical workflow

1. Read `docs/EXPERIMENT_LEDGER.md` and choose an evidence role before
   running anything. Smoke runs check execution only.
2. Treat raw identity as the primary deterministic candidate and `base_42k`
   location+spread as the selected probabilistic adapter; keep their evidence
   lines, metrics, and scientific labels separate.
3. Retain the completed matched global-pretraining experiment as a negative
   component result: scratch was selected by the frozen promotion gate.
4. Preserve the completed E2 IMD audit and E3 station-target sensitivity as
   immutable evidence. Do not tune from either result; keep raw-mean
   preservation as a target-dependent diagnostic.
5. Keep blocked-OOF affine calibration outside the main evidence line unless
   a separate future calibration study first repairs and tests its loader.
6. Follow `docs/RAW_IDENTITY_2025_SEALED_WORKFLOW.md`: freeze the exact
   raw-identity selection, verify its storage-incapable synthetic GPU
   preflight, and request a new hash-bound approval before the one-time 2025
   evaluation. `docs/INDEPENDENT_2025_CONTROL_WORKFLOW.md` is retained only as
   the superseded physical-control design.
7. Build paper tables and figures only from the receipt-gated all-season paper
   bundle and frozen manifests, with dependence-aware intervals and explicit
   evidence labels.

The next useful work is a final manual source/PDF review and submission handoff;
do not start another broad neural search on reused years. The final 2025 control
stays sealed unless a separately authorized one-time protocol is invoked.

## Result and cache policy

- Complete canonical run directories are immutable units. Do not edit or prune
  files inside them because their manifests hash internal artifacts.
- A directory name such as `full` does not prove scientific completion.
  Inspect its manifest and selection record.
- Smoke runs are non-scientific and may be regenerated.
- Final NPZ caches are retained. Their resumable `.parts` build shards may
  be removed after the final archive passes an integrity check.
- New runs should stage into a temporary sibling and publish by atomic rename,
  leaving an explicit failed-run record instead of a half-complete directory.

See `results/README.md` for canonical paths and
`docs/CLEANUP_20260819.md` for exactly what was removed.

## What to reuse from ai-quest-global

The global project has useful engineering patterns: one hashable scientific
contract, atomic cache finalization, epoch-zero baseline preservation, strict
cache/checkpoint fingerprints, a frozen selection record, and an explicit
one-time test-access sentinel. Those patterns should be adopted here during
the package refactor.

Do not copy its categorical RPS objective, competition submission code, or
global result claims into this deterministic IMD study.
