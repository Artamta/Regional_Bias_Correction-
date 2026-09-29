# Related-work map: all-season probabilistic FuXi calibration

Updated: 22 August 2026

Status: venue-neutral research-positioning note. This is not submission prose.
The 2025 target remains sealed and unopened.

## Closest work and the actual distinction

| Work | Forecast/target | Method and output | What it establishes | Difference from this study |
|---|---|---|---|---|
| [FuXi-S2S (Chen et al., 2024)](https://doi.org/10.1038/s41467-024-50714-1) | Global FuXi-S2S, ERA5 | Native 42-day AI ensemble forecasts | FuXi-S2S is a strong global subseasonal forecasting system | This study does not replace FuXi. It calibrates FuXi's full rainfall ensemble to IMD over India and audits the correction out of era. |
| [Horat and Lerch (2024)](https://doi.org/10.1175/MWR-D-23-0150.1) | Global ECMWF, biweekly weeks 3--4 and 5--6 | CNN/UNet categorical probabilities from ensemble-mean fields | Spatial neural post-processing can produce calibrated global S2S tercile forecasts | This study retains the individual FuXi weather members, predicts weekly rainfall through week 6, and evaluates continuous ensemble quality as well as categorical probabilities. |
| [Scheuerer et al. (2020)](https://doi.org/10.1175/MWR-D-20-0096.1) | ECMWF precipitation over California, through week 4 | ANN/CNN categorical probabilities | Regional neural S2S precipitation post-processing can beat conventional baselines | This study targets India, FuXi, all seasons, six weekly leads, and a member-preserving continuous ensemble. |
| [Höhlein et al. (2024)](https://doi.org/10.1175/AIES-D-23-0070.1) | Short/medium-range temperature and wind gusts | Permutation-invariant ensemble post-processing | Set architectures can preserve member information, although a few ensemble degrees of freedom often carry most useful information | This study tests that idea for gridded S2S rainfall. Its summary-only near-tie is consistent with, rather than contradictory to, that finding and must be reported. |
| [Noh and Ahn (2026)](https://doi.org/10.1038/s41612-026-01430-8) | Global GEFSv12 and station precipitation, weeks 1--5 | Large geographically and seasonally conditioned deterministic deep network | Explicit spatial context improves S2S rainfall correction, especially early leads | This study asks a different question: can a very small probabilistic adapter calibrate a modern AI ensemble under regional data and compute constraints? |
| [Guan et al. (2026)](https://arxiv.org/abs/2604.16238) and [official code](https://github.com/mouatadid/pbc) | Global dynamical and AI S2S forecasts | Operational probabilistic bias correction using Debias++, Persistence++, projection, and blending | Carefully designed classical/ML probability correction is a very strong baseline | This study implements a frozen-split, India/IMD adaptation as a baseline. It is not a numerical reproduction: the rolling/prequential schedule, archive, target, calendar, and score support differ. |

## Defensible contribution

The paper should claim a compact, data-constrained calibration study, not a new
foundation model and not universal bias removal:

1. A 42,434-parameter permutation-invariant adapter transforms each FuXi
   rainfall member using learned location and spread fields, while retaining a
   51-member empirical predictive distribution.
2. It is optimized directly with area-weighted finite-ensemble CRPS on
   2002--2017, selected only on 2018--2019, and first reported on reused
   2020--2021 development years.
3. A no-retraining 2022--2024 operational-era audit tests the exact frozen
   checkpoints on later 50-member FuXi stores and is reported separately as
   retrospective generalization evidence.
4. A tie-aware categorical comparison places the neural forecast and a
   frozen-split PBC baseline on the exact same IMD thresholds, cases, support,
   references, and dependence-aware bootstrap.
5. Negative ablations show that larger networks and hybrid CRPS--MSE losses do
   not provide a robust gain. The summary-only neural method nearly matches the
   set encoder, so member-level representation is a modest increment rather
   than the headline discovery.

## Reviewer-safe boundaries

- Say **probabilistic calibration** or **post-processing**, not universally
  successful bias correction. Continuous pooled signed bias did not improve in
  the 2020--2021 development evaluation.
- Do not say that the PBC implementation reproduces Guan et al. It is a
  frozen-split adaptation with equality-aware dry-threshold handling.
- Do not call 2020--2021 independent. It is reused development evidence.
- Call 2022--2024 an out-of-era retrospective audit, not an untouched test.
- Do not make an extreme-rainfall claim from q95 point estimates without
  intervals and event-focused validation.
- Do not claim that the set encoder is essential; the summary-only ablation is
  nearly tied.
- Do not claim global training. The input FuXi archive is global, but neural
  fitting and IMD scoring are regional.
- Do not open or report 2025 until a separate, explicitly authorized one-time
  final-test protocol is executed.

## Four-page story

The cleanest short-paper sequence is:

1. **Problem:** raw AI ensembles can be accurate yet underdispersed and
   locally miscalibrated for actionable Indian S2S rainfall.
2. **Method:** a small permutation-invariant residual location/spread adapter,
   trained with CRPS and designed to work with a variable number of members.
3. **Evidence:** continuous skill and coverage calibration, later-era
   no-retraining audit, and a common-support neural-versus-PBC categorical
   comparison.
4. **Learning from ablations:** more capacity and deterministic-loss mixing do
   not help; most gain comes from learning distributional location/spread, and
   simple persistence remains a formidable categorical baseline.
5. **Impact and limits:** inexpensive regional calibration can make existing
   AI ensembles more usable for probabilistic rainfall guidance, but this
   experiment does not yet validate district decisions, extremes, or an
   untouched operational deployment.

