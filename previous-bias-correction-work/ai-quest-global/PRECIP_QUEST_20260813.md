# AI Weather Quest precipitation methodology — issue 20260813

## Scope and decision rule

This workflow produces only five precipitation-quintile probabilities on the
global 1.5° grid for D19–25 and D26–32. ERA5T is the ranking observation;
IMERG and MSWEP are supplementary diagnostics and do not enter selection.
Validation (2019–2020) chooses the complete system before the 2021 test is
opened once. A neural adapter is eligible only after passing the declared
gates against calibrated statistical controls.

The 2,080 FuXi initializations are split without overlap:

| Role | Years | Cases |
|---|---:|---:|
| Train/fitting | 2002–2018 | 1,768 |
| Validation/selection | 2019–2020 | 208 |
| Frozen test | 2021 | 104 |

## Data flow

For a week beginning at `S`, ERA5 precipitation is the sum of exactly 28
six-hour accumulations from `S+06h` through `S+7d 00h`, converted from metres
to millimetres. Each grid-cell boundary set is q20/q40/q60/q80 from the prior
20 years and day offsets `[-4,-2,0,+2,+4]`: exactly 100 prior-only samples.
Equality with a boundary enters the upper category.

Each FuXi case supplies six context weeks. TP is clipped at zero, converted
from its 24-hour mean rate to daily millimetres, summed by member for seven
days, transformed with `log1p`, and reduced to q10/q25/q50/q75/q90. TCWV,
q850, q500, q250, and T2M are weekly-meaned by member and reduced to ensemble
mean/std. This yields 15 dynamics maps per week. Only 2002–2018 forecasts over
the supplied ERA5 land-fraction cells fit predictor means and standard deviations.

Unsmoothed member category counts are retained. The strictly positive neural
anchor is

```text
p0_k = (count_k + 0.5) / (51 + 5×0.5) = (count_k + 0.5) / 53.5.
```

Observation coverage can invalidate `target_valid`; it is absent from every
feature-building interface and cannot change `context_x`, `target_x`, the 100
climatology samples, or `p0`.

## Controls and calibration

Debias++ stores training-only forecast-CDF errors by cyclic target-start day
of year, lead, boundary, and grid cell. Validation independently selects the
paper-declared half-window span `s` from `{14, 28, 35}` days for each lead.
At inference, the locally averaged correction is added to the raw cumulative
probabilities, clipped to `(1e-6, 1-1e-6)`, projected onto nondecreasing CDFs
by exact least-squares isotonic regression, and differenced back to five
categories.

Every calibratable model uses one validation-only model weight per lead:

```text
Pcal = gamma_lead Pmodel + (1 - gamma_lead) Uniform,
0 <= gamma_lead <= 1.
```

The code fits the exact RPS-optimal bounded coefficient rather than searching
a rounded grid. Evaluation uses cosine-area weights, official land fraction
`>=0.5`, and excludes invalid/persistently arid quintile targets.

## Neural architecture

The adapter is a probabilistic postprocessor; FuXi remains the weather
forecaster. Separate `23→16` context and `18→16` target stems produce eight
tokens. A shared `16→32→64` two-pool encoder maps the global grid to 30×60.
One width-64, four-head Transformer layer with FFN 128 mixes the six context
and two target tokens independently at each bottleneck cell. Only the two
target tokens continue through the decoder, with target-only skip connections.
A shared zero-initialized `16→5` head predicts correction logits:

```text
P = softmax(log(p0) + correction).
```

Width 16 has 151,573 trainable parameters and is exactly `p0` at epoch zero.
The fixed training defaults are AdamW (`3e-4`, weight decay `1e-4`), masked
RPS plus `1e-4 × mean(correction²)`, BF16, gradient clip 1.0, and effective
batch size 8. The screening/full limits are 12/30 epochs with patience 3/6.

The selection script enforces the declared physical-context, width, seed, and
ensemble promotion gates. It writes `comparison.csv`, configuration hashes,
training histories/RPS curves, a validation RPS plot, reliability plot, and a
frozen selection hash. The test flag creates a one-time sentinel and refuses a
second test opening in the same selection directory.

## Operational and submission safety

The operational branch uses a fresh 51-member, 42-day FuXi run for 20260813,
explicitly labelled as an experimental GFS-proxy initialization. The config
requires input offsets no later than Thursday 00 UTC and a dedicated global
output contract retaining TP, TCWV, q850, q500, q250, and T2M. Wrong-date,
regional, or TP/T2M-only fields are rejected. Uniform can be generated only
by the separate emergency command and only when the expected live FuXi file
does not exist.

Submission files are created by AI-WQ-package 3.29's
`AI_WQ_create_empty_dataarray`; this repository does not own the NetCDF schema.
Both serialized files are reopened and checked for exact coordinates, shape,
finiteness, range, and probability-sum error no larger than `1e-6`. Local
hashes and the manifest are written before any network call. Upload additionally
requires literal `--confirm-submit 20260813`, registered credentials, and a
hashed written-permission record explicitly covering FuXi competition use and
derived submissions.

## Limitations and disclosure

Operational FuXi is initialized through a GFS proxy rather than the ERA5
initialization used by the historical archive, creating a domain shift. The
neural sample is modest relative to the global grid, so statistical controls
remain first-class candidates. IMERG/MSWEP scores are contextual only.

An AI coding assistant helped implement and test the pipeline from a
human-specified scientific plan. Scientific choices, data permissions,
credential use, model registration, candidate approval, and submission remain
the responsibility of the research team.
