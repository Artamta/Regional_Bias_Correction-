# FuXi–IMD loss experiments

## Objective

Improve physical IMD RMSE, MAE, bias, and spatial anomaly correlation without
allowing a reduction in heavy-rain underprediction to create artificial rain in
dry and light-rain cells.

The diagnosed validation trade-off is:

- the current log-anchor control has pooled RMSE 5.456, MAE 3.531, bias
  -1.173 mm/day, and ACC 0.3272;
- the physical-recentered bias-aware candidate has RMSE 5.431, MAE 3.570,
  bias -0.765 mm/day, and ACC 0.3284;
- the stronger global-bias loss therefore improves RMSE and absolute bias but
  worsens MAE;
- by observed intensity, it makes dry and light-rain bias more positive while
  reducing the large negative heavy-rain bias.

All figures above are blocked 2018–2019 validation diagnostics. They are not
independent test results.

## Fixed experimental contract

The following are held fixed in the new loss screen:

- train years: 2002–2017;
- blocked validation years: 2018–2019;
- quarantined years: 2020 onward;
- target: IMD weekly mean rainfall;
- anchor: the training-only physical-recentered log-bias correction;
- architecture: width-24 temporal-attention U-Net, 323,017 parameters;
- features and input normalization;
- anchored log-residual target and its lead-wise training-only RMS scale;
- optimizer, batch size, learning rate, weight decay, noise augmentation,
  maximum epochs, patience, and seed policy.

The one-seed screen is only hypothesis generation. A non-reference candidate
must pass seeds 42, 43, and 44 before it can be frozen for independent testing.

## Common terms

Let `p` be reconstructed corrected rainfall, `y` IMD rainfall, `c` the fixed
IMD climatology, and `r` the standardized anchored log-residual.

### Anchored Smooth-L1

`L_log = area_mean(SmoothL1(r_pred - r_target; beta=1))`

This robust term operates in standardized log-residual space. It stabilizes
training and treats relative rainfall errors more evenly, but compresses the
largest physical rainfall errors.

### Spatial anomaly-correlation loss

`L_ACC = 1 - corr_area(p - c, y - c)`

This protects spatial anomaly skill. It is insensitive to a spatially uniform
offset, so it cannot control bias by itself.

### Global physical-bias loss

`L_global = mean_case,lead[(area_mean(p - y) / s_lead)^2]`

`s_lead` is the training-only, area/case-weighted mean IMD rainfall for each
lead week. A single global mean permits positive dry-cell and negative
heavy-cell errors to cancel.

## New terms

### Rainfall-regime bias

IMD cells are separated using fixed thresholds:

1. dry: `[0, 1)` mm/day;
2. light: `[1, 5)` mm/day;
3. moderate: `[5, 10)` mm/day;
4. heavy: `[10, infinity)` mm/day.

For every case, lead, and available regime:

`b_k = area_mean_k(p - y) / s_lead,k`

`L_strata = equal_mean_case,lead,regime(b_k^2)`

`s_lead,k` is the training-only mean IMD rainfall in that lead/regime, floored
at 1 mm/day. Giving each available regime equal weight prevents numerous dry
cells or a few high-amplitude heavy cells from silently dominating the bias
objective. Validation targets never fit these scales.

### Soft wet-occurrence Brier loss

At the operational wet threshold `t = 1 mm/day`:

`q = sigmoid((p - t) / 0.5)`

`L_wet = area_mean[(q - 1[y >= t])^2]`

This differentiable occurrence term penalizes false drizzle as well as missed
wet cells. The 0.5 mm/day temperature avoids a discontinuous threshold during
optimization.

### Robust physical MAE

`z = (p - y) / s_lead`

`L_phys = area_mean[sqrt(z^2 + 0.01^2) - 0.01]`

This Charbonnier term is a smooth approximation to normalized physical MAE.
It aligns training more directly with the reported mm/day MAE while remaining
less sensitive to extremes than physical MSE.

## Predeclared candidates

| Candidate | Log Smooth-L1 | ACC | Global bias | Regime bias | Wet Brier | Physical MAE |
|---|---:|---:|---:|---:|---:|---:|
| Global bias 0.25 reference | 0.55 | 0.20 | 0.25 | 0 | 0 | 0 |
| Moderate global bias | 0.65 | 0.20 | 0.15 | 0 | 0 | 0 |
| Rainfall-regime bias | 0.60 | 0.20 | 0.05 | 0.15 | 0 | 0 |
| Regime bias + wet occurrence | 0.55 | 0.20 | 0.05 | 0.15 | 0.05 | 0 |
| Balanced physical | 0.45 | 0.20 | 0.05 | 0.15 | 0.05 | 0.10 |

Each row sums to one. Objective magnitudes are not compared between rows
because their definitions differ.

## Previous loss screen retained for completeness

A fixed heavy-rain Smooth-L1 multiplier screen tested weights 2, 3, and 5 for
IMD rainfall at least 10 mm/day. Weight 2 reduced heavy-rain underprediction
but increased dry/light-rain errors and did not pass all physical guards.
Weights 3 and 5 increasingly damaged MAE and/or ACC. Therefore the present
screen does not repeat one-sided heavy-rain weighting.

## Selection and reporting

Ranking uses physical IMD metrics only: pooled/year/lead RMSE, MAE, signed
bias, absolute bias, and ACC. Training objective values are shown only as
within-candidate convergence diagnostics.

A candidate is not promoted merely because it has the lowest pooled RMSE. It
must also satisfy the predeclared year, lead, raw-FuXi, MAE, ACC, absolute-bias,
and seed-robustness guards recorded in each run manifest. If none qualifies,
the reference is retained and the result is documented as a negative result.

The final independent evaluation is allowed only after the loss coefficients,
normalization constants, seed-ensemble rule, and stopping epoch are frozen.

## Completed seed-42 screen (2026-08-14)

The full one-seed screen completed successfully on two NVIDIA A30 GPUs. These
are blocked 2018–2019 validation results, not independent-test results.

| Loss | RMSE | MAE | Bias | ACC | Passed every guard |
|---|---:|---:|---:|---:|:---:|
| Moderate global bias 0.15 | 5.4558 | 3.5784 | -0.9005 | 0.3260 | No |
| Global bias 0.25 reference | 5.4565 | 3.5943 | -0.7701 | 0.3250 | No |
| Regime bias + wet Brier | 5.6696 | 3.6434 | -1.6702 | 0.3071 | No |
| Regime bias + wet Brier + physical MAE | 5.6726 | 3.6458 | -1.6847 | 0.3072 | No |
| Regime bias | 5.6772 | 3.6321 | -1.6944 | 0.2995 | No |

Relative to the global-bias-0.25 reference, global bias 0.15 changed:

- RMSE by -0.012% (essentially tied);
- MAE by -0.441% (better);
- ACC by +0.0010 (slightly better);
- absolute bias by +16.93% (worse).

It is therefore a useful Pareto point but not a replacement for the reference
when bias correction is the primary objective.

The equal-regime candidates did what their gradients requested but exposed a
bad weighting choice. The regime term strongly reduced dry bias (from +1.248
to roughly +0.76 to +0.79 mm/day) and light-rain bias (from +1.436 to roughly
+0.56 to +0.60 mm/day). However, it made heavy-rain bias much more negative
(from -6.511 to roughly -7.81 to -8.02 mm/day). Consequently pooled RMSE rose
by 3.91–4.04%, MAE rose by 1.05–1.43%, absolute bias more than doubled, and ACC
dropped by 0.018–0.026.

The wet-occurrence and physical-MAE terms were only tested on top of the
failing equal-regime objective, so this screen does **not** establish that
either auxiliary term is independently harmful. It establishes that neither
is strong enough at the predeclared coefficient to rescue the over-dry regime
loss.

### Decision

No candidate advances to three-seed confirmation. The global-bias-0.25
reference is retained for this experiment, even though it does not beat raw
FuXi absolute bias and therefore is not yet a fully satisfactory bias
correction. No independent test year was opened.

The next loss experiment, if performed, should be a small predeclared
one-factor screen around the reference: either a 0.20 global-bias coefficient,
or a moderate global-bias term plus a low-weight one-sided heavy-rain miss
penalty. It should not reuse the equal-regime normalization. Repeated tuning on
2018–2019 should stop after that screen and move to blocked out-of-fold
development estimates to control selection overfitting.
