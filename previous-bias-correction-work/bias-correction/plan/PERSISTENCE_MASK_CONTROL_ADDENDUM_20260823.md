# Frozen addendum: active-mask control for persistence attribution

Status: **frozen on 23 August 2026 while the original three-arm full run was
still training `base_42k`, seed 42, and before any full-run persistence-arm
checkpoint, full validation RPS, selection, or development score existed**.

The completed plumbing smoke had already shown that `zero_lag_45k` numerically
reproduces `base_42k`. No smoke score is scientific. This addendum was prompted
by an independent design audit, not by a favorable or unfavorable full
candidate result.

## Attribution problem

The original parameter-matched control appends four identically zero channels.
The persistence candidate appends two lag-rainfall channels and two independent
availability masks. Almost every archive issue has both lag windows and every
2018--2019 validation issue does, so both availability masks are effectively
copies of the existing scoring-support channel throughout validation.

Consequently, weights connected to all four extra channels stay inactive in
`zero_lag_45k`, while the two mask pathways in the candidate can train. The
three-arm experiment remains a valid practical screen of the complete proposed
input package, but it cannot by itself attribute a gain specifically to recent
rainfall rather than duplicate-mask capacity or regularization geometry.

## Additional frozen arm

Add exactly one nonselectable control:

| Arm | Added channels in frozen order | Parameters | Role |
|---|---|---:|---|
| `mask_only_45k` | zero, zero, lag-1 availability, lag-2 availability | 45,026 | active-mask attribution control |

The two zero fields occupy the exact normalized-rainfall positions. The masks
are the same issue-safe, support-masked values supplied to
`persistence_lag12_45k`. The arm uses the same 11-channel architecture, copied
same-seed 42k state, exactly zero new input weights, post-construction RNG reset,
optimizer, loss, batches, member subsampling, checkpoint metric, seeds, and
early stopping as the original full experiment.

The control is trained on the exact purged 2002--2017 inventory and selected at
the checkpoint level by exact 2018--2019 full-member CRPS. It must reuse or
byte-verify the original full run's cache, observations, lags, support,
thresholds, training indices, validation indices, and categorical score
geometry. It cannot index or score 2020+ initialization cases during training,
checkpoint selection, or joint arm selection. The canonical observation loader
may materialize and hash its already-defined full archive bundle; receipts must
describe that honestly and prove that no 2020+ initialization index entered a
dataset, score, or selection.

Because this is a separately scheduled matched arm, its receipt must match the
original full run's GPU model, Python/library/CUDA software versions, AMP mode,
and deterministic settings. Prefer the same physical node when it remains
usable. A hardware or numerical-stack mismatch invalidates the joint selector
and requires an unchanged rerun on matching hardware.

## Immutable inputs and PBC binding

The original three-arm full output remains immutable. The addendum accepts it
only after its full external Slurm receipt and every artifact/source checksum
pass. It must also bind the following already accepted PBC V2 identities before
any later PBC comparison:

- PBC manifest SHA-256:
  `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6`;
- observation bundle SHA-256:
  `3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49`;
- issue-time lag bundle SHA-256:
  `9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba`;
- categorical threshold/fit artifact SHA-256:
  `0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36`.

Any mismatch stops the workflow; it cannot be repaired by refitting or changing
a path after scores are known.

## Superseding joint selector

The original `selection.json` remains historically valid for its frozen
three-arm question, but it is not sufficient for a persistence-attribution or
PBC-superiority claim. A new hash-bound joint selection combines score rows
from the immutable original full run and `mask_only_45k` without averaging
forecasts, parameters, adjustments, or checkpoints.

`persistence_lag12_45k` is jointly promoted only if:

1. the original zero-control reproduction gate passes;
2. every original promotion gate passes against both `base_42k` and
   `zero_lag_45k`;
3. pooled validation RPS improves on `mask_only_45k` by at least 0.5%;
4. W2--W6 validation RPS improves on `mask_only_45k` by at least 0.5%;
5. pooled validation CRPS is no worse than `mask_only_45k`;
6. neither 2018 nor 2019 validation RPS is worse than `mask_only_45k`;
7. at least two of three matched seeds have lower validation RPS than
   `mask_only_45k`;
8. all case keys, probabilities, CDF geometry, adjustments, checkpoints, and
   scores are finite, complete, and hash-bound.

`mask_only_45k` is never selectable. If any gate fails, `base_42k` remains the
selected arm. Development results cannot override the joint validation choice.

Even if every gate passes, the permitted interpretation is deliberately
limited: the complete issuance-available rainfall feature is useful beyond
duplicate availability-mask pathways under this architecture. Because the two
rainfall-channel kernels are active only in the candidate, this addendum does
not prove that an arbitrary active no-information field could never change
optimization. Do not claim a pure parameter-count causal effect or use the
phrase "not simply extra active parameters."

## Later comparison and language

Only the jointly selected arm may receive a formal "better than PBC" or
"better than the strongest classical baseline" label. A nonselected arm can be
reported only as a descriptive, reused-development sensitivity, even if its
point estimate or interval is favorable.

The downstream evaluator otherwise retains the original fixed contract: exact
2020--2021 PBC case inventory, per-seed scoring before score averaging, no
forecast averaging, paired two-stage year/circular 13-initialization blocks,
2,000 draws, seed 20260823, and pooled plus W1--W6 comparisons against both
combined PBC and Persistence++.

## Stop rules

- Do not edit the running/completed three-arm sources, plan, launcher, or
  artifacts.
- Do not change the mask-only fields, arm width, initialization, RNG stream,
  training recipe, joint gates, or PBC identities after an addendum smoke or
  score is seen.
- Run a fresh one-seed/two-epoch addendum smoke, then a fresh three-seed full
  mask-only run, each with an external Slurm receipt.
- Preserve negative, failed, and inconclusive outcomes.
- The 2022--2024 retrospective cohorts remain out of scope and 2025 remains
  sealed.
