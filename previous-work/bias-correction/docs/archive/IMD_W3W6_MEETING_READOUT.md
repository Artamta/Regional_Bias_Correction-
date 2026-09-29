# FuXi-S2S to IMD: meeting readout

## One-sentence result

A validation-selected, regularized temporal residual model improves FuXi Weeks 3–6 rainfall over India relative to both raw FuXi and a log-bias baseline, while a forecast-conditioned attention climatology adds no robust extra value.

## Experiment

- Train: 560 FuXi initializations, 2002–2017
- Validation and model selection: 70 initializations, 2018–2019
- Exploratory reused test: 70 initializations, 2020–2021
- Target: IMD daily rainfall aggregated to six 7-day lead weeks
- Grid: 27 × 27, with 171 IMD-supported India cells
- Learned correction: Weeks 3–6; Weeks 1–2 remain exactly log-bias
- Model: 144,689-parameter temporal U-Net, ensemble of seeds 42, 43 and 44
- Attention ablation: 145,115 parameters with nine training-only IMD climatology candidates

## Validation decision

The normal-climatology model was selected before test prediction. It improved W3–W6 RMSE over log-bias in both validation years:

- 2018: +1.04%
- 2019: +1.54%

Attention changed sign relative to the normal model across validation years, so it was not selected.

## Exploratory 2020–2021 test

| Method | Common-IMD ACC | PCC, no climatology | RMSE (mm/day) | MSESS vs IMD climatology |
|---|---:|---:|---:|---:|
| Raw FuXi | 0.163 | 0.498 | 5.933 | -0.117 |
| Log-bias | 0.207 | 0.542 | 5.548 | +0.009 |
| Selected W3–W6 model | **0.243** | **0.559** | **5.439** | **+0.053** |

Selected model versus log-bias, pooled over W3–W6:

- RMSE improvement: **+1.96%**, paired 95% CI **+0.77% to +2.91%**
- Common-IMD ACC change: **+0.035**, paired 95% CI **+0.005 to +0.065**
- PCC change: **+0.017**, paired 95% CI **+0.008 to +0.030**
- Local RMSE improves over approximately **62.4% of the IMD-weighted area**

RMSE improves at every active lead in the point estimates: W3 +1.71%, W4 +1.86%, W5 +1.60%, and W6 +2.70%. W3 and W6 are individually positive at the paired 95% interval; the primary inference is the predeclared pooled W3–W6 result.

## Correlation definition

- Common-IMD ACC: spatial correlation after subtracting the same fixed 2002–2017 IMD climatology from forecasts and observations. This is leakage-safe and does not require a separate FuXi climatology, but it should not be called an own-model-climatology ACC.
- PCC: spatial correlation of absolute weekly forecast and observed rainfall. It uses no climatology, but it rewards persistent wet/dry geography and is therefore supplementary to ACC.

## What not to claim

- Do not claim a robust attention-specific improvement.
- Do not call 2020–2021 a fresh untouched test; it has been reused during development.
- Do not say longer training solved the problem. Early stopping selected epochs 5–14; later epochs reduced training loss while validation stopped improving.

## Next defensible experiment

For rainfall-intensity underprediction, compare the current log-residual target against a hybrid objective that adds physical-space Huber loss on mm/day rainfall. Keep the log anchor for stability, validation-select the blend, and retain the current model as the control.
