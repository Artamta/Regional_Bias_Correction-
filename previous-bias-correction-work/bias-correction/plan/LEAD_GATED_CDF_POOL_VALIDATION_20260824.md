# Frozen one-shot validation of a deployable neural--PBC CDF hybrid

Date: 24 August 2026

Status: frozen before reconstructing or scoring the candidate probability
forecast described below.

This is a final, one-candidate validation screen, not an independent test. The
candidate was motivated after leadwise 2020--2024 results had already been
inspected. In addition, 2018--2019 was previously used for neural checkpoint
selection and PBC Debias++ span selection, and the standalone neural validation
scores are already known. The new information in this screen is the score of
the exact probability-level hybrid against frozen PBC on common PBC thresholds.

## One candidate

The only candidate is `neural_seed_pool_w1__persistence_w2_w6`:

- W1: the equal-weight linear pool of the three frozen 42,434-parameter neural
  seed CDFs, `(F_42 + F_43 + F_44) / 3`;
- W2--W6: the frozen projected Persistence++ CDF;
- neural seed weights are exactly one third and are never fitted;
- no alternate lead cutoff, seasonal or spatial gate, component weight,
  checkpoint, seed subset, model training, refit, or rescue search is allowed.

This is one probability forecast. A convex average of valid CDFs is a valid CDF
and the W1 pool is equivalent to an equally weighted 153-member mixture of the
three corrected 51-member ensembles. It is distinct from the earlier
score-only diagnostic, which averaged three seed scores and cannot define a
deployable forecast.

## Immutable inputs

- Canonical neural manifest:
  `resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z/manifest.json`
  with SHA-256
  `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91`.
- Frozen PBC manifest:
  `resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json`
  with SHA-256
  `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6`.
- Validation-adjustment parent manifest:
  `resultsv2/fuxi_allseason_persistence_augmented/full_20260822T233542Z/manifest.json`
  with SHA-256
  `da4be561bca3c0ffa1a14fcf8f28f8a77e05bb6a60240edc4a48aed65948d642`.
- Its frozen selection receipt has SHA-256
  `bdc284b0a27e1914c7fd9926a356086288e4edd1ffaf50f24df7cf17c51f1e21`.
- Canonical 2002--2021 member cache has data SHA-256
  `2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70`,
  metadata SHA-256
  `b2cd07dd540cd96ee1bc7ef5df2c59cefffac78122aa56bc42beeb4c3242580a`,
  and manifest SHA-256
  `4e05cdc8fcbe609e151beb627bd94ee21fd87a5a00efc3a14fdaf804b4ccd0d8`.
- PBC parameter archive `models/pbc_fit.npz` has SHA-256
  `0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36`.
- PBC scoring support `evaluation/scoring_support.npz` has SHA-256
  `6149d5d58a9e46b0b8faf4cd71a7bb633f1c3eba0d6fb1bb207f9f5edf590898`.
- Frozen base validation adjustments have SHA-256 values:
  seed 42 `cc72ba9a927a5ef69f6698ca4be9906182fbd859301fe384811c6e2026670647`,
  seed 43 `8164ebefc57567a7c2e919109bb2fcf5582cdfc242b4920af26dbeab8ce82f18`,
  and seed 44 `bbfb9f4a645108a536c550e1ed5ddfffddb3989ba932cd941195ddcc669b6a4b`.
- The IMD truth/weight array-bundle digest is
  `3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49`.
- The strictly pre-issuance lag array-bundle digest is
  `9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba`.

All three base checkpoint state dictionaries must be tensor-identical to the
corresponding accepted `location_spread` checkpoint even though their wrapper
checkpoint files have different hashes. Any input, state-tensor, split,
provenance, support, or sealed-year mismatch stops the run.

## Reconstruction and score

Use exactly the 196 purged 2018--2019 validation initializations and all six
leads. Reconstruct frozen PBC training thresholds, raw CDF, empirical
climatology, two exact pre-issuance lag indicators, projected Persistence++,
and projected equal-weight combined PBC from the persisted fit. Reconstruct
each neural seed from the raw 51 members and its saved affine log-location and
spread fields, evaluate it at the exact PBC thresholds, and average the three
W1 CDFs. Do not use the nearby but non-identical thresholds saved by the
persistence-feature experiment.

The primary endpoint is normalized informative-positive-cut quintile RPS:
mean squared CDF error over distinct strictly positive persisted thresholds,
area weighted on the common IMD support. Scores must be finite and in [0,1].
Every forecast and observation CDF must be bounded, nondecreasing, exactly
equal at tied thresholds, exactly zero at `q=0`, finite on support, and NaN off
support.

## Frozen decision rule

Promote the hybrid only if all of the following hold against both
Persistence++ and equal-weight combined PBC:

1. pooled relative RPS reduction `1 - hybrid / baseline` is at least 0.005;
2. hybrid mean RPS is lower separately in 2018 and in 2019;
3. the lower endpoint of a paired 95% interval for relative RPS reduction is
   strictly above zero;
4. every provenance, geometry, probability, support, and temporal check passes.

Uncertainty uses 2,000 deterministic draws with seed 20260824. The retained
validation support is 104 initializations in 2018 and 92 in 2019. Each draw
first resamples the two years, then uses circular 13-initialization blocks
within each sampled source year. Every sampled source year contributes its
native case count, so draw sizes can be 184, 196, or 208 initializations. This
retains natural case weighting instead of silently forcing equal year weights.
An initialization retains all six leads.

Write `selection.json` and its SHA-256 atomically before any downstream
2020--2024 reconstruction. If any gate fails, record `rejected`, retain
Persistence++ as the categorical baseline, and stop. No second candidate or
modified rule is allowed.

## Claim and 2025 boundary

Passing permits only: "the frozen lead-gated probability hybrid passed its
reused 2018--2019 validation screen and has lower retrospective RPS than both
frozen PBC comparators under this static India/IMD protocol." It does not prove
that the neural adapter alone beats PBC, does not reproduce the full rolling
Guan et al. protocol, and is not an independent NeurIPS superiority result.

This evaluator has no path to a 2025 forecast or observation. The sealed 2025
target remains unopened. Any one-time 2025 confirmation requires a separate
preflight and explicit user approval.
