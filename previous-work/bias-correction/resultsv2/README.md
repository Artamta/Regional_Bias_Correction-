# Version-2 experiment results

This directory is reserved for new immutable experiment outputs that follow
the post-cleanup provenance contract. Generated run directories are not edited
in place.

The no-fitted-log-bias neural ablation is stored under
`fuxi_imd_no_log_bias_ablation/`. Smoke runs prove execution only. Full runs
are validation-selected development evidence, and any 2020–2024 diagnostics
remain exploratory. No run here may access 2025 unless a separate one-time
final-test protocol explicitly authorizes it.

Canonical follow-up artifacts:

- `fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z`:
  complete 51-member, all-season India-box probabilistic calibration run using
  2002--2017 training, 2018--2019 checkpoint selection, and the reused
  2020--2021 development test. All 224 non-manifest artifacts were independently
  verified; manifest SHA-256
  `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91`.
  Its manuscript-scale derived bundle is at
  `../presentation/deliverables/fuxi_allseason_ensemble_calibration_20260822`
  (manifest SHA-256
  `b1e72076a28f3abc2b7b8b89c3fdb7245968a7e960e87d4f4bcde019ccb9786c`).
- `fuxi_allseason_capacity_ablation/full_20260822T220000Z`: validation-only
  capacity screen that retained the 42,434-parameter `base_42k` model;
  manifest SHA-256
  `2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c`.
- `fuxi_allseason_hybrid_loss_ablation/full_final_20260822T141844Z`:
  CRPS--MSE objective sensitivity; pure CRPS remained best on reused
  development evidence; manifest SHA-256
  `a3e77e4cf4e6485a756d99b68fec102fece37ef096725e496fe4a27819828f5e`.
- `fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z`: accepted frozen-split
  PBC adaptation; Persistence++ has the lowest pooled categorical RPS;
  manifest SHA-256
  `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6`.
- `fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z`: accepted
  seven-method common-support comparison. Neural location+spread is the W1
  specialist, W2 is unresolved against combined PBC, and classical methods
  lead W3--W6 and pooled; manifest SHA-256
  `44ca7130e626bfadc71d311d0002dbb39b67aebbf16d0dfb2802999b6be4bdbc`.
- `fuxi_allseason_operational_era_audit/full_20260822T190829Z`: accepted
  post-selection 2022--2024 no-retraining audit over all 296 eligible starts
  and 50 members. Pooled CRPS skill is 15.76% (paired 95% interval
  13.78--17.76%) and is positive at every lead; manifest SHA-256
  `7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad`,
  Slurm receipt SHA-256
  `65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd`.
- `fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z`:
  accepted post-hoc 2022--2024 categorical retrospective over all 296 eligible
  starts, with no retraining, selection, or fitted blending. Against raw FuXi,
  the locked neural adapter reduces pooled quintile RPS by 23.14%
  (exploratory 95% interval 15.87--30.60%), semidecile RPS by 23.61%
  (16.09--31.41%), and descriptive upper-q95 Brier score by 12.24%
  (9.01--16.54%). Neural beats Persistence++ and combined PBC at W1; its
  pooled ordinary RPS is numerically slightly worse but statistically
  indistinguishable from both, while its pooled q95 Brier score is lowest.
  Only three year clusters support the exploratory intervals, all 252 paired
  comparisons are unadjusted for multiplicity, and neural results condition
  on the mean of per-seed scores. This is not global, prospective, 2025,
  event, or extremes evidence. Manifest SHA-256
  `cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d`,
  semantic-audit SHA-256
  `00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c`,
  Slurm-receipt SHA-256
  `cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c`,
  and paired-bootstrap CSV SHA-256
  `632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188`.

The receipt-gated paper tables, claim boundaries, source registry, and three
publication-scale PDF/PNG figures derived from the accepted runs are at
`../presentation/deliverables/fuxi_allseason_probabilistic_paper_20260822T192948Z`;
bundle manifest SHA-256
`cf9d17dd335a1f680d0a7fb2f5f7a45ac7b556bb5a621d0bf564157cdc4be0d9`.
The later-era categorical paper supplement, including pooled/leadwise
quintile and semidecile tables plus a two-panel transfer figure, is at
`../presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z`;
bundle manifest SHA-256
`bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a`.
The official-style anonymous manuscript handoff is at
`../presentation/deliverables/ccai_neurips2026_submission_source_20260823`;
`main.tex` SHA-256
`0538c8ffc253c854bc304d67e03940ee80954a1126703b3b341698a972a72116`.
Tectonic 0.16.9 compiled `main.pdf` with SHA-256
`ad335e16c38a041e2a7b1d108d0aeb438c190ff7b8d091d79e638ea1ec468258`;
the output is five US-letter pages, with main text on pages 1--4 and references
only on page 5.
- `fuxi_imd_raw_identity_2022_2024_audit/canonical_circular_20260822T0225Z`:
  fixed five-method 2022--2024 development audit, 10,000 paired circular-block
  draws, no retraining; manifest SHA-256
  `bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`.
- `fuxi_imd_adapter_station_external_target/canonical_five_method_20260822T0230Z`:
  frozen external 2024 station-target sensitivity over 30 starts and six
  leads; manifest SHA-256
  `5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`.
- `raw_identity_independent_2025_sealed/selection.json`: the frozen
  raw-identity-versus-raw final-test contract; selection SHA-256
  `cad4af2a7443ee57ccec29f45ce812fb08f7e78ab135e6fe6f4871245b4dd6b6`.
  Synthetic CUDA preflight job 109981 passed with receipt SHA-256
  `c2484d8d9e5a93c782e23b4419b5363e04e254a3f8e88e2d37dd37ba9f43db3b`.
  It created no storage access, access ledger, or result.

The two raw-identity audit directories immediately above are immutable
development evidence. The final raw-identity entry is only the sealed contract
and synthetic proof for the still-unopened 2025 final test.
