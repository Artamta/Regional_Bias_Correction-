# Lead-gated neural--PBC hybrid: decision record

Date: 24 August 2026

## Outcome

The single frozen candidate was promoted on its reused 2018--2019 validation
screen. It is one deployable categorical probability forecast:

- W1: equal-weight pool of the frozen neural seed-42, seed-43, and seed-44 CDFs;
- W2--W6: frozen projected Persistence++;
- no fitted weights, alternate cutoff, seasonal/spatial router, retraining, or
  rescue search.

This is not an accuracy/ACC improvement. The endpoint is normalized
informative-positive-cut quintile RPS, where lower is better.

| Reused validation method | Pooled RPS |
|---|---:|
| Lead-gated hybrid | **0.182417** |
| Persistence++ | 0.185356 |
| Equal-weight combined PBC | 0.188733 |
| Neural CDF pool at every lead | 0.186499 |
| Raw FuXi categorical | 0.245967 |

The frozen paired results were:

- versus Persistence++: 1.585% lower RPS, 95% interval [0.566%, 2.535%];
- versus combined PBC: 3.346% lower RPS, 95% interval [2.046%, 4.794%].

Every gate passed: at least 0.5% pooled improvement against both comparators,
lower RPS in 2018 and 2019 separately, both paired intervals above zero, and
all provenance/CDF checks valid.

The mechanism is simple. At W1, the neural CDF pool scored 0.139497 versus
0.157129 for Persistence++ (11.22% lower) and 0.160517 for combined PBC
(13.09% lower). At W2--W6 the hybrid is exactly Persistence++, so all pooled
gain comes from replacing the weaker PBC W1 forecast.

## Repeated retrospective evidence

The separate hash-bound score diagnostic, which uses the arithmetic mean of
the three neural seed scores at W1, found:

| Cohort | Hybrid score upper bound | Reduction vs Persistence++ | Reduction vs combined PBC |
|---|---:|---:|---:|
| Reused 2020--2021 | 0.194050 | 1.08% [0.63%, 1.72%] | 1.51% [0.39%, 2.67%] |
| Post-hoc 2022--2024 | 0.179492 | 1.46% [0.56%, 2.36%] | 1.53% [0.22%, 2.76%] |

These are conservative for the realizable CDF pool: quadratic RPS is convex,
so scoring the equal-weight seed CDF pool cannot be worse than averaging the
three seed scores, up to negligible float32 roundoff. The reported lower
interval endpoints are therefore conservative lower endpoints under the same
paired draws; the router's upper endpoints are not confidence upper bounds for
the unscored real pool. All remain post-hoc bounds, not an independent test.

## What can be claimed

Permitted:

> A fixed lead-gated categorical hybrid passed its reused validation screen
> and consistently lowers retrospective RPS under the repository's static
> India/IMD PBC protocol. Neural correction is strongest at W1, while
> Persistence++ is retained at W2--W6.

Not permitted:

- "the neural adapter alone beats PBC";
- "independent superiority over PBC";
- "ACC improved" or "continuous CRPS improved" from this experiment;
- "full reproduction of the rolling Guan et al. protocol";
- any claim based on 2025, which remains sealed and unopened.

This materially strengthens the paper as a lead-dependent complementarity and
hybridization result. It is not by itself a clean NeurIPS confirmatory result:
the policy was motivated after 2020--2024 leadwise inspection, and 2018--2019
was already used by the component pipelines. The scientifically valuable next
step is a separately authorized, one-time sealed-year confirmation—not another
round of tuning.

## Receipts

- Frozen protocol:
  `plan/LEAD_GATED_CDF_POOL_VALIDATION_20260824.md`, SHA-256
  `8b3f7c5e5a35d896fe8084900e028c5c1a6c17a041e9370b227692146af79dec`.
- Publication-receipted real-CDF validation bundle:
  `resultsv2/fuxi_lead_gated_cdf_hybrid/publication_receipted_20260823T190247Z`,
  manifest
  SHA-256
  `ec77f1b48ab11b9cead7811d9a90b5f1feb559dd99750f8bd4cfa3242467081c`.
- Locked selection SHA-256:
  `930cb9b84d932ac11c64c300e25e619eb6e701cbd4dad61785168525adae66fb`.
- External post-run receipt SHA-256:
  `c0dd216c3b2d267bd660f915a2460cb01300624fc806f5cccb29ea63a8cdbc63`.
- Hardened post-hoc diagnostic bundle:
  `resultsv2/fuxi_lead_gated_neural_pbc/posthoc_score_only_hardened_20260823T185532Z`,
  manifest SHA-256
  `71b009b2fa748bca77e493a6233eca9fd0d9818b1611387976f951af06cc5d3c`.
