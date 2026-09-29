# FuXi–IMERG A100 experiment: meeting readout

## One-sentence result

The 2.54-million-parameter multi-scale temporal adapter improved the frozen
2018–2019 validation score over the 145-thousand-parameter control, but it was
effectively tied with the small model on the exploratory 2020–2021 test.

## What was run

- Training: 2002–2017, 560 JJAS initializations
- Validation: 2018–2019, 70 initializations
- Exploratory test: 2020–2021, 70 initializations
- Three fixed seeds per model: 42, 43, 44
- Models: small temporal control, large spatial control, large multi-scale temporal
- Learned correction: Weeks 5–6 only; Weeks 1–4 remain exactly log-bias
- Hardware: three NVIDIA A100-SXM4-80GB GPUs
- Wall time: 212 seconds (3.5 minutes)

## Loss result

| Model | Parameters | Mean best train loss | Mean best validation loss | Mean gap |
|---|---:|---:|---:|---:|
| Small temporal | 144,689 | 0.4553 | 0.5144 | 0.0591 |
| Large spatial | 983,761 | 0.4540 | 0.5118 | 0.0578 |
| Large temporal | 2,544,049 | 0.4557 | 0.5114 | 0.0558 |

For the large temporal model, mean validation loss fell from 0.5262 at epoch 1
to 0.5114 at the selected checkpoints. Continuing training reduced train loss
to 0.4309 but worsened validation loss to 0.5380. Early stopping is therefore
doing necessary model selection; a monotonically falling validation curve is
not a realistic or scientifically useful target.

The large temporal model reduced best validation loss by 0.0030 (0.59%) and
the train–validation gap by 5.6% relative to the small control on the same
split. Its three selected epochs were 6, 8, and 9.

## Validation-only selection

The large temporal ensemble was selected before test prediction. Its W5–W6
RMSE improvement over log-bias was positive in both validation years:

- 2018: +1.69%
- 2019: +2.06%
- pooled W5–W6 validation RMSE: 5.273 mm/day
- ensemble validation composite loss: 0.5099

The validation residual gate selected alpha = 1.0, so no post-hoc shrinkage was
needed.

## Exploratory 2020–2021 test, Weeks 5–6

| Method | ACC | RMSE | MAE | Bias | RMSE skill vs raw | RMSE skill vs log-bias |
|---|---:|---:|---:|---:|---:|---:|
| Raw FuXi | 0.164 | 5.297 | 3.628 | -0.232 | 0.00% | -4.26% |
| Log-bias | 0.185 | 5.080 | 3.410 | -0.834 | +4.09% | 0.00% |
| Small temporal | 0.207 | 5.045 | 3.389 | -0.771 | +4.75% | +0.69% |
| Large spatial | 0.201 | 5.048 | 3.402 | -0.703 | +4.71% | +0.65% |
| Large temporal | 0.202 | 5.046 | 3.411 | -0.602 | +4.74% | +0.68% |

Paired uncertainty for large temporal versus log-bias:

- Delta ACC: +0.017, 95% CI [-0.010, +0.057]
- RMSE reduction: +0.68%, 95% CI [-0.64%, +2.21%]
- MAE reduction: -0.05%, 95% CI [-2.36%, +1.62%]

The neural increment over log-bias is therefore not statistically robust on
this two-year exploratory test. Most of the improvement over raw FuXi comes
from the training-only log-bias correction. The large temporal model improved
local RMSE over log-bias across 59.8% of the India-weighted area, but its mean
advantage over the small model was only 0.00035 mm/day and is negligible.

## Safe paper/meeting claim

“A regularized multi-scale temporal adapter improved the predeclared two-year
validation objective and produced a small exploratory late-lead RMSE gain over
log-bias. However, its 2020–2021 performance was statistically indistinguishable
from both log-bias and a much smaller temporal adapter, so larger capacity alone
did not yield a robust improvement.”

Do not claim that the large model decisively beats the small model or that the
neural increment over log-bias is significant.

## Artifacts

- Full result: `results/fuxi_imerg_a100_big_temporal/full_20260810T023253Z`
- Training components: `figures/00_training_components.png`
- Validation selection: `figures/01_validation_selection.png`
- Test skill by lead: `figures/02_test_skill_by_lead.png`
- Spatial improvement: `figures/03_test_spatial_improvement.png`
- Frozen selection: `selection.json`
- Test predictions: `predictions.zarr`
- Verification: `python evaluate/verify_a100_big_temporal.py --run <full-result>`
