# FuXi–IMD target-transform experiments

## Scientific question

Does the `log1p` anchored target cause the remaining heavy-rain
underprediction, and can a less-compressive or learned monotone transform
improve physical RMSE, MAE, bias, and ACC without creating drizzle?

The answer cannot be inferred from validation loss magnitudes because different
transforms have different numerical scales. All ranking therefore uses
reconstructed IMD-space physical metrics.

## Existing evidence

The log representation is probably one contributor, but not a hard model
capacity limit:

- compared with raw FuXi, the log anchor improves dry, light, and moderate
  stratum RMSE by approximately 13%, 22%, and 15%, respectively;
- it worsens heavy-rain RMSE by approximately 1.3% and makes heavy-rain bias
  more negative;
- while retaining `log1p`, a heavy-tail loss can reduce heavy-rain RMSE by
  roughly 10%, proving that the model can represent a stronger tail response;
- that tail improvement creates much larger dry/light errors, so the central
  problem is the objective/intensity trade-off as well as transformation
  curvature.

## Fixed Box–Cox-1p screen

The monotone transform is

`T_lambda(p) = expm1(lambda * log1p(p)) / lambda`

for `lambda > 0`, with the continuous limit

`T_0(p) = log1p(p)`.

Its inverse is

`p = expm1(log1p(lambda * z) / lambda)`

for `lambda > 0`, and `p = expm1(z)` for `lambda = 0`.

Therefore:

- `lambda = 0` is the exact current log target;
- intermediate values retain some compression while strengthening gradients
  for high rainfall;
- `lambda = 1` is the physical linear rainfall target.

Predeclared powers: `0`, `0.25`, `0.5`, `0.75`, and `1.0`.

For every power and lead week, the target is independently standardized using
the area-weighted RMS transformed anchor residual fitted on 2002–2017 only:

`r = [T_lambda(IMD) - T_lambda(anchor)] / s_lambda,lead`.

This prevents a transform from winning merely because it changes target
amplitude.

## Fixed contract

Only target/reconstruction curvature changes. The following remain fixed:

- IMD target and support mask;
- physical-recentered training-only anchor;
- width-24 temporal-attention U-Net;
- 11 effective input channels and their existing normalization;
- global-bias-aware objective weights 0.55 Smooth-L1, 0.20 ACC, and 0.25
  physical bias;
- model optimizer, regularization, batch size, early stopping, and seed;
- training years 2002–2017 and validation years 2018–2019;
- quarantine of every year from 2020 onward.

The screen starts with seed 42. A non-log candidate advances only if it passes
the existing physical year/lead/raw-FuXi guards. It then requires seeds 42, 43,
and 44 before any independent evaluation.

## Completed one-seed screen (seed 42)

The blocked 2018–2019 screen completed without producing any prediction or
metric for 2020 onward.  The pooled physical results were:

| Power | RMSE (mm/day) | MAE (mm/day) | Bias (mm/day) | ACC |
|---:|---:|---:|---:|---:|
| 0.00 (exact log) | 5.4569 | 3.5947 | -0.7681 | 0.3250 |
| 0.25 | **5.4327** | **3.5920** | -0.7196 | **0.3266** |
| 0.50 | 5.4385 | 3.5994 | -0.6932 | 0.3216 |
| 0.75 | 5.4364 | 3.6163 | -0.5999 | 0.3189 |
| 1.00 (physical linear) | 5.4334 | 3.6304 | **-0.5472** | 0.3146 |

Relative to exact log, power 0.25 reduced pooled RMSE by 0.44%, MAE by
0.08%, and absolute pooled bias by 6.32%, while increasing ACC by 0.0015.
This is a useful direction but falls just short of the predeclared 0.5% RMSE
threshold and does not beat raw FuXi absolute pooled bias.

The apparent pooled bias gain also needs careful interpretation.  At power
0.25, dry-rain bias increased by 0.095 mm/day, light-rain bias increased by
0.074 mm/day, and heavy-rain bias became 0.009 mm/day more negative.  The
pooled bias therefore moved toward zero partly by cancellation between
overprediction of low intensities and underprediction of high intensities.
The screen retained exact log as the locked reference because no non-log
candidate passed every predeclared guard.  Power 0.25 remains an exploratory
hypothesis, not a selected model or independent result.

## Learnable transform policy

A raw jointly learned transform is unsafe: it can alter target amplitude and
make transformed Smooth-L1 smaller without producing a better physical
forecast. Per-lead transform parameters also add unnecessary validation tuning.

A learnable transform will therefore be attempted only if the fixed screen
shows that an intermediate power beats both exact log and physical-linear
targets. The first learnable version must use:

- one bounded global curvature parameter, not six lead-specific parameters;
- initialization at the best fixed training-stage power;
- training-only, curvature-specific detached RMS normalization;
- transform state saved separately from the forecast model state so existing
  strict checkpoint loading remains compatible;
- physical-space checkpoint guards and a three-seed confirmation;
- no access to 2020 onward.

If no intermediate fixed power qualifies, a learned curvature parameter cannot
be promoted by this experiment and the log representation is retained.

Post-screen exploratory amendment: power 0.25 produced a weak pooled Pareto
direction but failed the promotion guards.  At the user's request, one bounded
global-power run may therefore be made as hypothesis generation only, with
exact-log and fixed-0.25 controls in the identical training code path.  This
does not change the locked selection rule above.  The learned result cannot be
called a new best model, used for an independent claim, or evaluated on 2020+
unless it first passes blocked inner-fold and multi-seed confirmation.

## Additional success guards

In addition to the existing pooled/year/lead guards, a transform candidate
should ideally achieve:

- at least 0.5% pooled RMSE improvement versus exact log;
- no more than 0.5% pooled MAE regression;
- ACC loss no larger than 0.005;
- absolute pooled bias no worse than exact log and preferably no worse than raw
  FuXi;
- equal-stratum mean absolute bias must improve, so opposite dry- and
  heavy-rain errors cannot cancel into a favourable pooled bias;
- at least 2% heavy-rain RMSE and MAE improvement;
- no more than 1% dry- or light-rain MAE regression;
- improvement in at least four of six lead weeks;
- robustness in at least two of three seeds before promotion.
