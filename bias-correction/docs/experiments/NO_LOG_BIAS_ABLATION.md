# No-fitted-log-bias neural ablation

## Question

How much of the frozen FuXi-to-IMD adapter's skill requires the explicit
training-only log-bias correction, rather than the neural adapter itself?

This is a one-factor development ablation. It is not a new independent test
and it must not access the 2025 initialization year.

## Primary contract

- Train and fit all target-derived preprocessing on 2002–2017 only.
- Select checkpoints and the residual gate on 2018–2019 only.
- Treat 2020–2021 as reused exploratory diagnostics.
- Keep 2022–2024 closed until the selection and checkpoint hashes are frozen;
  any later score is an exploratory development audit because the anchored
  outcomes are already known.
- Keep 2025 untouched.
- Use the frozen full-context normal-climatology architecture: 144,689
  parameters, batch size 32, learning rate 2e-4, weight decay 2e-3, dropout
  0.30, maximum 100 epochs, patience 15, and seeds 42/43/44.
- Keep the original 0.75 Smooth-L1 + 0.20 ACC + 0.05 bias objective, equal
  W1–W6 lead weights, IMD support, feature tensor, and normalization.
- Select the residual gate on the existing alpha grid from 0 to 1 in steps of
  0.025 using equal-case W1–W6 validation RMSE.

Only the reconstruction reference changes. The target is

`[log1p(IMD) - log1p(raw FuXi)] / s_lead`,

where `s_lead` is fitted on 2002–2017 only. Reconstruction is

`expm1(log1p(raw FuXi) + alpha * s_lead * neural_residual)`.

Therefore zero neural output reproduces raw FuXi. No fitted lead/month/grid
log-bias field enters neural training or reconstruction. The ordinary
training-only log-bias forecast is retained as a reporting comparator only.

The model still uses training-only IMD climatology features. Removing those
features would be a second, confounded ablation and is outside this protocol.
Call the result a **raw-identity neural adapter** or **no-fitted-log-bias
adapter**, not a target-free or purely forecast-only AI model.

## Selection

The primary causal comparison is the matched normal-climatology architecture.
It is selected only if it reduces RMSE relative to raw FuXi in both 2018 and
2019. The attention-climatology model may be trained in the same run as a
secondary descriptive screen, but it cannot replace the matched model in the
primary anchor attribution.

Report physical RMSE, MAE, signed bias, and common-training-climatology ACC for
raw FuXi, reporting-only log-bias, the matched raw-identity neural model, and
the frozen anchored model. Lowest transformed loss is not a performance claim.

## Output

Every smoke and full run uses a fresh immutable directory under
`resultsv2/fuxi_imd_no_log_bias_ablation/`. A complete run must preserve the
selection record, normalization, raw-reference target scale, three
checkpoints, histories, predictions, case metrics, source snapshots, hashes,
and evidence status.

