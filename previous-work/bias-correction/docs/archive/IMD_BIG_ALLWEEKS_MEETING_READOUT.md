# FuXi-S2S to IMD, 2.54M-parameter W1–W6 model

## Result

The validation-selected 2,544,049-parameter multi-scale temporal model improves W1–W6 rainfall over India by **11.06% versus raw FuXi** and **3.82% versus log-bias** on the reused 2020–2021 exploratory test.

| Method | Common-IMD ACC | PCC | RMSE (mm/day) | MSESS vs IMD climatology |
|---|---:|---:|---:|---:|
| Raw FuXi | 0.253 | 0.545 | 5.828 | -0.040 |
| Log-bias | 0.305 | 0.591 | 5.389 | +0.096 |
| Selected 145k model | 0.352 | 0.614 | 5.211 | +0.156 |
| Selected 2.54M model | **0.359** | **0.618** | **5.183** | **+0.163** |

Large model versus log-bias, pooled over W1–W6:

- RMSE improvement: **+3.82%**, paired 95% CI **+2.70% to +4.74%**
- ACC change: **+0.054**, paired 95% CI **+0.035 to +0.078**
- PCC change: **+0.028**, paired 95% CI **+0.024 to +0.035**
- MAE improvement: **+2.77%**, paired 95% CI **+1.13% to +3.13%**
- Local RMSE improves over **76.5%** of the IMD-weighted area

## Does extra capacity help?

Relative to the selected 145k all-week model, the 2.54M model changes pooled W1–W6 scores by:

- RMSE: **+0.52% improvement**, paired 95% CI **-0.02% to +0.97%**
- MAE: **+0.94% improvement**, paired 95% CI **+0.03% to +1.53%**
- ACC: **+0.0069**, paired 95% CI **-0.00005 to +0.0124**
- PCC: **+0.0042**, paired 95% CI **+0.0001 to +0.0076**

Capacity therefore gives a modest point improvement, especially in Weeks 1–2, but the pooled RMSE advantage over the 145k model narrowly crosses zero. The defensible conclusion is that the large model is best numerically, while most of the learned gain does not require 2.54M parameters.

## Validation and training

The normal-climatology large model was selected before test prediction. It improved validation RMSE over log-bias in both years:

- 2018: +3.46%
- 2019: +3.39%

The large attention model was not selected because it was worse than the normal model in both validation years. Selected epochs for the normal model were 6, 10, and 4 for seeds 42, 43, and 44. The full run used three A100 GPUs and completed in 3.2 minutes.

## Paper wording

“A 2.54M-parameter multi-scale temporal adapter improved the validation-selected W1–W6 FuXi post-processing result over raw and log-bias forecasts. Its incremental RMSE benefit over a 145k adapter was small and not significant at the paired 95% level, showing that added capacity alone provides limited additional value.”

The 2020–2021 evaluation has been reused during development and must be described as exploratory.
