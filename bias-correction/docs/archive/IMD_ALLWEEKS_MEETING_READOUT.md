# FuXi-S2S to IMD, Weeks 1–6: meeting readout

## One-sentence result

A validation-selected temporal residual model improves FuXi rainfall over India across Weeks 1–6 relative to both raw FuXi and log-bias correction; forecast-conditioned attention was tested but did not add a robust benefit.

## Experiment

- Train: 560 FuXi initializations from 2002–2017
- Validation and model selection: 70 initializations from 2018–2019
- Exploratory reused test: 70 initializations from 2020–2021
- Target: IMD daily rainfall aggregated into six 7-day lead weeks
- Domain: 171 IMD-supported cells on the 27 × 27 India grid
- Learned correction: all six weeks, with equal weight in the training loss
- Selected model: 144,689-parameter temporal U-Net, ensemble of seeds 42, 43, and 44
- Attention ablation: 145,115 parameters with nine training-only IMD climatology candidates

## Validation decision

The normal-climatology model was selected before test prediction. Its W1–W6 RMSE improvement over log-bias was positive in both validation years:

- 2018: +2.71%
- 2019: +2.99%

Attention improved over the normal model in 2018 but worsened in 2019, so it was not selected.

## Exploratory 2020–2021 test

| Method | Common-IMD ACC | PCC, no climatology | RMSE (mm/day) | MSESS vs IMD climatology |
|---|---:|---:|---:|---:|
| Raw FuXi | 0.253 | 0.545 | 5.828 | -0.040 |
| Log-bias | 0.305 | 0.591 | 5.389 | +0.096 |
| Selected W1–W6 model | **0.352** | **0.614** | **5.211** | **+0.156** |

Selected model versus log-bias, pooled over W1–W6:

- RMSE improvement: **+3.32%**, paired 95% CI **+2.41% to +4.00%**
- Common-IMD ACC change: **+0.047**, paired 95% CI **+0.025 to +0.073**
- PCC change: **+0.023**, paired 95% CI **+0.018 to +0.032**
- Local RMSE improves over approximately **70.3% of the IMD-weighted area**

The RMSE point estimate improves at every week: W1 +6.00%, W2 +3.77%, W3 +2.77%, W4 +2.65%, W5 +2.10%, and W6 +3.03%. The W4 individual confidence interval crosses zero; the predeclared pooled W1–W6 result is the primary inference.

The all-week model also improves W3–W6 RMSE by **+0.69%** over the separately trained W3–W6 model, paired 95% CI **+0.12% to +1.09%**.

## Correlation definition

- Common-IMD ACC subtracts the same fixed 2002–2017 IMD climatology from forecasts and observations. It is leakage-safe, but it is not an own-model-climatology ACC.
- PCC correlates absolute weekly rainfall fields and uses no climatology. It is supplementary because persistent wet and dry geography can increase it.

## What to say clearly

- The result supports learned temporal correction across all six lead weeks.
- Attention is an ablation, not the selected model and not the source of the headline gain.
- The 2020–2021 test is exploratory because it has been reused during development.
- Longer training alone is unlikely to help: early stopping caught the validation plateau while training loss continued downward.
