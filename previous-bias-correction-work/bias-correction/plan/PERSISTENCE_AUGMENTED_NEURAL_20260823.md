# Frozen plan: persistence-augmented all-season neural calibration

Status: **pre-launch protocol frozen on 23 August 2026 before any candidate
training or candidate score was computed**.

This is a new, explicitly post-hoc experiment motivated by the completed
neural--PBC lead crossover. It does not alter, replace, or reinterpret any
accepted all-season neural, capacity, PBC, categorical-comparison, or
operational-era artifact.

## Question

Does adding exactly two issuance-available IMD rainfall histories to the
existing compact FuXi ensemble adapter improve both continuous ensemble CRPS
and equality-aware categorical RPS, especially beyond week 1, without simply
benefiting from extra input parameters?

The candidate is not assumed to beat Persistence++ or combined PBC. A negative
or inconclusive result is valid and must remain visible.

## Evidence boundary

- Model fitting: purged 2002--2017 initializations only.
- Checkpoint and arm selection: purged 2018--2019 validation only.
- Reused development evaluation: 2020--2021, only after `selection.json` is
  durably written. This period is not independent confirmation.
- The already inspected 2022--2024 retrospective cohorts are not used for
  feature design, fitting, checkpoint choice, promotion, or rescue.
- The 2025 initialization year remains sealed. No code in this experiment may
  open a 2025 forecast or observation store.
- One initialization, including all weather members and all six leads, remains
  the sampling and resampling unit.

## Single changed factor

The accepted neural context has seven channels: training-only IMD climatology,
season sine/cosine, latitude, longitude, lead, and support mask. The candidate
adds four channels and changes nothing else:

1. completed seven-day mean IMD rainfall from issue minus 7 through issue minus
   1 day;
2. completed seven-day mean IMD rainfall from issue minus 14 through issue
   minus 8 days;
3. availability mask for the first lag;
4. availability mask for the second lag.

Every rainfall lag is transformed with `log1p`. Each lag channel receives one
area-weighted scalar mean and standard deviation fitted from available
2002--2017 values only. Unsupported cells and unavailable lag values are
encoded as normalized zero; the corresponding availability mask is zero.
Available masks are one on scoring support. The four lag fields are broadcast
unchanged across the six forecast-lead planes.

For an issue at date `t`, the two lag windows end at `t-1` and `t-8`.
`window_end < t` is a hard executable invariant. No current verification
observation, future observation, or outcome from the issue being predicted may
enter a feature.

## Frozen arms

| Arm | Context | Expected parameters | Role |
|---|---|---:|---|
| `base_42k` | accepted seven channels | 42,434 | exact architecture control |
| `zero_lag_45k` | seven channels plus four identically zero channels | 45,026 | parameter/input-shape control; nonselectable |
| `persistence_lag12_45k` | seven channels plus the two normalized lags and masks | 45,026 | only promotion-eligible candidate |

The 45k arms are initialized from the same-seed 42k state: every common tensor
is copied exactly, the shared part of the widened input convolution is copied,
and weights for the four added channels are initialized to exactly zero. Thus
`zero_lag_45k` is a direct control for input shape and parameter count rather
than a separately randomized network.

All arms otherwise use:

- `EnsembleLocationSpreadCalibrator`, `mode="location_spread"`;
- member width 8, backbone width 24, dropout 0.05;
- the unchanged location/spread member transformation;
- area-weighted finite-ensemble CRPS only;
- batch size 8 and 16 randomly sampled members during training;
- all 51 members for validation and evaluation;
- AdamW, learning rate `2e-4`, weight decay `1e-4`;
- at most 100 epochs, early-stopping patience 15;
- seeds 42, 43, and 44.

No MSE auxiliary loss, categorical training loss, extra network depth, wider
backbone, alternate lag, or learned blend is part of this experiment.

## Validation-only promotion

Each checkpoint is selected by full-51-member validation CRPS. Arm promotion
then uses case-paired validation scores only. Quintile thresholds for validation
RPS are the 0.2/0.4/0.6/0.8 quantiles fitted from the effective training split
only with a centered 31-day calendar window and minimum sample count 8, under
the same equality-aware distinct-positive-cut contract as PBC V2.

`persistence_lag12_45k` is promoted only if all of the following hold:

1. mean three-seed validation normalized informative-positive-cut RPS improves
   on `zero_lag_45k` by at least 0.5%;
2. mean W2--W6 validation RPS also improves on `zero_lag_45k` by at least
   0.5%, so a W1-only gain cannot promote the candidate;
3. mean validation CRPS is no worse than `zero_lag_45k`;
4. neither 2018 nor 2019 validation RPS is worse than `zero_lag_45k`;
5. at least two of three matched seeds improve validation RPS;
6. the same conclusions are not contradicted by the exact `base_42k` control;
7. every checkpoint, correction, probability, lag, and score is finite on
   support, and every categorical CDF satisfies the frozen equality/zero rules.

`zero_lag_45k` must also reproduce `base_42k` closely enough to exclude an
input-shape or initialization artifact: its mean validation CRPS and RPS must
each differ from `base_42k` by no more than 0.25% in relative magnitude. It is
never promotion-eligible.

If any gate fails, retain `base_42k` and report the persistence addition as a
negative or inconclusive feature ablation. Development scores cannot reverse
the validation decision.

## Development and PBC comparison

Only after the validation decision is written may the program reconstruct
2020--2021 ensembles. Continuous CRPS/RMSE/MAE/bias/ACC/coverage and
equality-aware categorical scores are reported for all arms.

The categorical comparison must use the accepted PBC V2 manifest and its exact
208 initializations, training-only thresholds, IMD truth, equality geometry,
support, weights, and normalized informative-positive-cut RPS. Neural scores
are computed per seed and only scores are averaged; forecasts, probabilities,
correction fields, and parameters are never averaged across optimization
seeds.

Use these language gates:

- call the candidate **better than combined PBC pooled** only if its paired
  pooled RPS reduction has a 95% initialization-block interval entirely above
  zero;
- call the candidate **better than the strongest classical baseline pooled**
  only if the corresponding interval is above zero against both combined PBC
  and Persistence++;
- otherwise report the point estimate and interval as a tie, loss, or
  unresolved comparison;
- never turn the reused 2020--2021 result into an independent-test,
  prospective, operational, bias-removal, or extremes claim.

## Uncertainty

Use the existing paired two-stage resampling contract: sample the two
development years, then circular 13-initialization blocks within each sampled
year, preserving all six leads for each initialization. Use 2,000 draws and a
fixed seed of 20260823. Report pooled and W1--W6 effects.

## Stop rules

- Do not change lag lengths, transforms, normalization, imputation, masks,
  context ordering, architecture, loss, optimizer, checkpoint metric,
  promotion thresholds, or PBC comparator after seeing a smoke or full score.
- Smoke metrics are plumbing only and cannot change the contract.
- Do not tune from 2020--2024 results or open 2025 to rescue the candidate.
- A scheduler/node failure may be rerun unchanged to a fresh directory. A
  scientific-contract failure requires a new dated plan.
- Do not overwrite or modify any accepted result directory.

## Launch sequence

1. Run focused synthetic/contract tests.
2. Submit a one-seed, two-epoch smoke run to a fresh directory.
3. Require a complete manifest, valid CUDA receipt, source snapshot, lag
   provenance, selection artifact, and finite validation outputs.
4. Submit the canonical three-seed full run to a second fresh directory gated
   by the successful smoke manifest.
5. Only after the full validation selection is locked, run a separately gated
   score-only 2020--2021/PBC evaluator and inspect its paired comparisons
   without launching an unplanned follow-up search.
