# Frozen plan: deterministic–probabilistic hybrid-loss ablation

Frozen: 22 August 2026, before any hybrid-loss 2020–2021 score is computed.

Validation-stage amendment: after live validation traces but still before any
V2 2020–2021 metric was computed, the CRPS-only control was added to the final
RMSE candidate comparison. The original wording compared RMSE only among
guard-eligible hybrids and could therefore select a hybrid that was dominated
by the control. This logical correction is explicitly post-hoc to validation;
all training arms, checkpoints, probabilistic guards, and development data
remain unchanged.

## Scientific status

This is a separate, explicitly post-hoc development study motivated by the
completed `fuxi_allseason_ensemble_calibration_v1` result. It does not modify
that immutable run and cannot turn the reused 2020–2021 period into an
independent test. The 2025 control remains sealed and must not be opened.

## Question

Can a deterministic ensemble-mean error term improve point-forecast accuracy
without materially degrading the probabilistic calibration obtained from
ensemble CRPS?

One calibrated 51-member ensemble supplies both products:

- probabilistic: members, quantiles, threshold probabilities, and intervals;
- deterministic: the physical ensemble mean used for RMSE, MAE, bias, and ACC.

No separately trained deterministic forecast is blended with the ensemble.

## Fixed data and architecture

- Forecast: FuXi native whole-world reforecast source, 2002–2021, all seasons.
- Target: IMD weekly mean precipitation in mm day-1.
- Scored region: 39N–0N, 60E–99E, 27x27 India box, 171 positive-weight cells.
- Members/leads: all 51 members and W1–W6 at validation and evaluation.
- Splits: train 2002–2017 (1,652), validation 2018–2019 (196), reused
  development evaluation 2020–2021 (208), outcome-purged embargo (24).
- Architecture: the exact `location_spread` permutation-invariant calibrator.
- Seeds: 42, 43, and 44 as optimization replicates, never weather samples.
- Training member subsample: 16 for every arm; validation/evaluation use 51.
- Optimizer, learning rate, regularization, batch size, AMP, maximum epochs,
  and patience remain identical to V1.

The verified canonical member cache is reused read-only. It has shape
`[2080, 51, 6, 27, 27]`, SHA-256
`2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70`,
and source fingerprint
`655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae`.

## Loss family

Let `C` be area-weighted empirical ensemble CRPS and `E` be area-weighted
MSE of the physical ensemble mean. Compute `C0` and `E0` once from raw FuXi
on the effective training split only. For fixed alpha,

`L_alpha = (1 - alpha) C + alpha (C0 / E0) E`.

The scale factor gives both terms CRPS units without using validation or
development-evaluation targets. `alpha=0` exactly recovers V1 CRPS training.
Memberwise MSE is forbidden because it would encourage ensemble collapse.

Fixed arms:

| ID | alpha | Role |
|---|---:|---|
| `crps_only` | 0.00 | exact loss control |
| `hybrid_010` | 0.10 | light deterministic auxiliary term |
| `hybrid_025` | 0.25 | balanced auxiliary term |
| `hybrid_050` | 0.50 | strong auxiliary term |
| `mse_only` | 1.00 | deterministic endpoint; diagnostic and ineligible for selection |

No alpha may be added after examining 2020–2021 results.

## Checkpointing and validation-only selection

Each seed/arm checkpoint minimizes its own full-51-member validation
`L_alpha`. Every epoch records validation CRPS, ensemble-mean MSE/RMSE,
50/80/90% central coverage, coverage error, and total objective.

The MSE term and its `E0` scale are pooled area-weighted mean squared errors.
For validation selection and every reported table, RMSE is instead the
arithmetic mean of case/lead spatial area-weighted RMSE values, exactly matching
the evaluation pipeline; it is not the square root of a pooled cross-case MSE.

Define coverage error as the mean absolute difference between empirical and
nominal 50/80/90% coverage. Aggregate checkpoint metrics arithmetically over
the three optimization seeds. A hybrid arm is eligible only if:

1. mean validation CRPS is at most 0.5% worse than `crps_only`; and
2. mean coverage error is no more than 0.01 worse than `crps_only`.

Compare `crps_only` with every eligible `hybrid_010`, `hybrid_025`, and
`hybrid_050`, then choose the lowest validation RMSE. Values within 0.25% of
the minimum are tied and the smaller alpha wins, so a hybrid cannot be selected
when its validation RMSE is dominated by (or practically tied with) the
simpler control. If no hybrid is eligible, retain `crps_only`. `mse_only` is
always diagnostic and cannot be selected. The selection record must be written
before any 2020–2021 metric is calculated.

## Evaluation and uncertainty

Report every arm, regardless of selection, on the reused development period:

- CRPS/CRPSS, Brier scores at 1/5/10/20 mm day-1, reliability, ranks;
- central 50/80/90% coverage and width, RMS spread / pooled RMS error;
- ensemble-mean RMSE, MAE, signed and absolute bias, and ACC;
- pooled, W1–W6, DJF/MAM/JJA/SON, and per-seed tables;
- 2,000-draw paired year plus 13-initialization circular-block intervals
  versus raw FuXi and versus `crps_only`.

The main conclusion is a Pareto conclusion, not simply the lowest single
metric. A hybrid is useful only if deterministic improvement survives the
predeclared probabilistic guards.

## Paper boundary

This study answers only the loss question. Existing V1 raw/moment/location,
summary, and set-encoder results answer the architecture question. A future
censored-shifted-gamma EMOS plus ECC baseline and an independent temporal or
external-target evaluation remain higher-value additions than a wider neural
hyperparameter search.
