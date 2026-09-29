# CCAI NeurIPS 2026 workshop manuscript draft

**Evidence cutoff:** 23 August 2026  
**Draft status:** evidence-backed manuscript prose built only from completed, receipt-gated artifacts. The 2022–2024 operational categorical extension has passed independent reconstruction. References and the appendix outline are outside the four-page main-text budget; final typesetting and a template-level page check remain.

## Title options

1. **A 42k-Parameter Ensemble Adapter for Lead-Dependent Calibration of FuXi-S2S Rainfall over India** *(recommended)*
2. **Lightweight Distributional Calibration of FuXi-S2S Rainfall over India**
3. **Calibrating, Not Replacing, an AI Subseasonal Rainfall Ensemble over India**

---

<!-- Four-page main text begins. -->

# A 42k-Parameter Ensemble Adapter for Lead-Dependent Calibration of FuXi-S2S Rainfall over India

## Abstract

Calibrated subseasonal rainfall uncertainty over India matters for anticipatory planning, yet a strong global AI ensemble can remain regionally underdispersed. We introduce a 42,434-parameter, permutation-invariant residual adapter that transforms every FuXi-S2S rainfall member through learned log-location and spread fields and is trained with finite-ensemble continuous ranked probability score (CRPS). We fit on 2002–2017, select on 2018–2019, and label 2020–2021 as reused development evidence. Across 208 such starts, CRPS fell from 1.6590 to 1.3874, a 16.37% reduction (paired 95% interval 14.28–18.16%); 90% coverage rose from 0.427 to 0.789. Without retraining or selection, the frozen adapter retained 15.76% CRPS skill (13.78–17.76%) across 296 eligible 2022–2024 starts. Later-era quintile RPS improved 23.14% over raw FuXi; at week 1 the adapter beat Persistence++ by 10.68%, while their pooled difference was unresolved. This post-hoc retrospective evidence supports lightweight, lead-dependent calibration, not prospective validation or deployment readiness.

## 1. Motivation and contributions

Subseasonal rainfall forecasts can inform preparedness across agriculture, water management, and disaster-risk planning, but those uses need a predictive distribution rather than a single corrected mean. FuXi-S2S provides a global 42-day AI weather ensemble [1]. Our question is narrower: can a small regional post-processor improve the reliability of its weekly rainfall ensemble over India without training another global forecasting model?

We make four contributions. First, we introduce a compact residual adapter that is permutation invariant in the ensemble-member dimension, preserves every member, and is exactly the identity at initialization. Second, we train the adapter directly with an area-weighted proper score over six weekly leads and a regional IMD target. Third, we compare the frozen neural forecast with strong persistence-based probability corrections on identical cases, thresholds, observations, support, and paired resamples. Fourth, we test simple explanations for the remaining error: increasing capacity from 19,618 to 293,762 parameters and mixing a deterministic MSE term into CRPS do not pass the frozen selection rules. A source-locked, no-retraining 2022–2024 retrospective then checks whether the continuous improvement survives a later forecast era and a change from 51 to 50 members.

Our claim is deliberately limited. This is regional probabilistic calibration, not global training, universal signed-bias removal, event-specific drought or flood prediction, prospective validation, or an operational deployment study.

## 2. Related work and positioning

FuXi-S2S establishes the underlying global AI ensemble [1]; we calibrate its rainfall members regionally rather than replace the forecasting system. Neural S2S precipitation post-processing has produced categorical probabilities or deterministic fields [2,3,5], whereas our endpoint remains a continuous memberwise empirical distribution through week 6. Permutation-invariant post-processing motivates the set representation [4], although our summary-only near-tie prevents a claim that member-level encoding is essential. Recent probabilistic bias correction (PBC) provides a strong classical benchmark [6]. Our Debias++/Persistence++/combined implementation is a frozen-split India/IMD adaptation with equality-aware dry thresholds, not a reproduction of its rolling protocol.

## 3. Data and method

### 3.1 Data and temporal contract

The regional target is weekly mean rainfall in mm day\(^{-1}\) on a 27 × 27, 1.5° India box, with 171 supported cells. Hindcast fitting uses 51-member FuXi-S2S forecasts and IMD observations from all seasons. The purged temporal split is 2002–2017 for fitting and 2018–2019 for selection. Results on 208 starts in 2020–2021 are reused development evidence, not an untouched test.

The later audit loads the already selected checkpoints without retraining, tuning, blending, or method selection. It retains every protocol-eligible 2022–2024 start whose full 42-day verification window ends by 31 December 2024: 104 starts in 2022, 104 in 2023, and 88 in 2024 (296 total), each with 50 members. We call this a **post-hoc 2022–2024 operational-era retrospective audit**. The 2025 forecast and target remain sealed and supply no evidence in this paper.

### 3.2 Member-preserving residual adapter

For initialization \(i\), member \(m=1,\ldots,M\), week \(\ell=1,\ldots,6\), and cell \(g\), let nonnegative raw rainfall be \(x_{im\ell g}\). We first stabilize rainfall and separate its ensemble location and deviations:

\[
u_{im\ell g}=\log(1+x_{im\ell g}),\qquad
\mu_{i\ell g}=M^{-1}\sum_m u_{im\ell g},\qquad
d_{im\ell g}=u_{im\ell g}-\mu_{i\ell g}.
\]

A shared pointwise encoder \(\phi\) and symmetric mean produce a set representation, while \(r\) measures log-space spread:

\[
z_{i\ell g}=M^{-1}\sum_m\phi(u_{im\ell g},d_{im\ell g}),\qquad
r_{i\ell g}=\sqrt{M^{-1}\sum_m d_{im\ell g}^{2}}.
\]

A compact residual Conv3D acts jointly over lead, latitude, and longitude:

\[
(\delta_{i\ell g},a_{i\ell g})
=h_\theta[z_{i\ell g},\mu_{i\ell g},r_{i\ell g},c_{i\ell g}],
\qquad s_{i\ell g}=\exp(2\tanh a_{i\ell g}).
\]

The seven context channels \(c\) are training-normalized IMD climatology, day-of-year sine and cosine, latitude, longitude, lead, and the fixed support mask. Zero-initialized output heads make the original ensemble an exact starting point. The corrected members are

\[
\widetilde u_{im\ell g}=\max\{0,\,
u_{im\ell g}+\delta_{i\ell g}+(s_{i\ell g}-1)d_{im\ell g}\},
\qquad \widetilde x_{im\ell g}=\exp(\widetilde u_{im\ell g})-1.
\]

Thus \(\delta=0,s=1\) gives the raw forecast, and permuting input members only permutes output members. The selected adapter has 42,434 trainable parameters and can evaluate either 51 or 50 members without changing its architecture.

**Figure 1 (single-column schematic; artwork to typeset).** Unordered FuXi members → log members and deviations → shared member encoder and symmetric pooling → log-location, RMS spread, and seven context channels → lead × latitude × longitude residual Conv3D → location and positive-spread fields → corrected members. Annotate: 42,434 parameters; exact identity at initialization; all members retained.

### 3.3 Proper-score objective and categorical endpoint

For one verifying value \(y\), the retained training loss is finite-ensemble CRPS,

\[
\operatorname{CRPS}(\widetilde x,y)=
M^{-1}\sum_m|\widetilde x_m-y|
-\frac{1}{2M^2}\sum_{m,n}|\widetilde x_m-\widetilde x_n|.
\]

We average over valid cells using latitude-area weights, then over leads and initializations. CRPS rewards both accuracy and distributional sharpness; memberwise MSE instead encourages collapse toward a conditional mean. Each case has one verifying realization, so no target distribution is available for a KL-divergence loss.

For the common categorical comparison, training-only rainfall quantiles define thresholds. We use strict empirical CDFs, \(\widehat F(q)=M^{-1}\sum_m\mathbf 1(\widetilde x_m<q)\), and \(O(q)=\mathbf 1(y<q)\). If \(Q^+_{i\ell g}\) contains each distinct positive physical threshold once, the equality-aware score is

\[
R_{i\ell g}=|Q^+_{i\ell g}|^{-1}
\sum_{q\in Q^+_{i\ell g}}[\widehat F_{i\ell g}(q)-O_{i\ell g}(q)]^2.
\]

Cells without a positive threshold are excluded, duplicate physical cuts are collapsed, and the common area-weight denominator is recomputed for every method. This avoids giving duplicated zero-rainfall quantiles artificial weight.

## 4. Evaluation protocol

Continuous evaluation reports CRPS, ensemble-mean RMSE and MAE, anomaly correlation coefficient (ACC), signed bias, central 50/80/90% empirical coverage, and spread/error. Raw and calibrated forecasts always use the same cases, members, truth, grid support, and observation-coverage weights. Neural seeds 42, 43, and 44 are optimization replicates: we score each checkpoint separately and average scores for the same case and lead; we never average parameters, adjustment fields, or predictions.

Uncertainty uses paired two-stage resampling: sample years with replacement, then circular blocks of 13 initializations within each selected year, keeping all members and six leads attached to an initialization. The two development years and three operational-era years provide weak interannual uncertainty, so intervals are exploratory rather than evidence of independent replication.

The 2022–2024 audit is receipt gated: it whitelists only the three allowed yearly forecast stores, verifies normalization and daily-to-weekly products, and binds checkpoints, configuration, source, dates, grid, and loaded arrays by content hashes. These integrity checks prevent the later period from becoming another model-selection loop; they do not turn a retrospective audit into a prospective trial.

## 5. Results and ablations

### 5.1 Continuous calibration improves accuracy and reliability

Table 1 shows the main continuous result. On reused 2020–2021 development data, the adapter reduces CRPS by 16.37% (paired 95% interval 14.28–18.16%), RMSE by 12.57%, and MAE by 9.24%, while ACC increases by 0.121. Raw FuXi is sharply underdispersed: 90% coverage is 0.427 and spread/error is 0.562. The adapter moves these to 0.789 and 0.967. It does not establish signed-bias improvement: the bias change is −0.037 mm day\(^{-1}\) (−0.167 to 0.091).

**Table 1. Pooled continuous performance.** “Development” is reused evidence; “operational era” is a post-hoc retrospective with frozen checkpoints and no retraining or selection. CRPS-skill intervals are paired 95% intervals versus raw FuXi.

| Cohort | Members / starts | Method | CRPS | CRPS skill | RMSE | MAE | ACC | Signed bias | 90% coverage | Spread/error |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020–2021 reused development | 51 / 208 | Raw FuXi | 1.6590 | 0 | 3.5452 | 2.1080 | 0.2312 | −0.1206 | 0.427 | 0.562 |
| 2020–2021 reused development | 51 / 208 | Neural location + spread | **1.3874** | **16.37% [14.28, 18.16]** | **3.0994** | **1.9133** | **0.3521** | −0.1578 | **0.789** | **0.967** |
| 2022–2024 operational-era retrospective | 50 / 296 | Raw FuXi | 1.5626 | 0 | 3.4203 | 2.0355 | 0.2738 | −0.0164 | 0.415 | 0.566 |
| 2022–2024 operational-era retrospective | 50 / 296 | Frozen neural adapter | **1.3164** | **15.76% [13.78, 17.76]** | **3.0777** | **1.8713** | **0.3366** | −0.0368 | **0.800** | **1.032** |

Continuous CRPS skill is positive at every lead. In reused development it declines from 30.65% at W1 to 7.91% at W6 (30.65, 24.51, 14.53, 11.88, 9.01, 7.91%). In the no-retraining retrospective it is 29.02, 20.35, 13.42, 11.94, 10.98, and 10.50% from W1 to W6, with every paired interval above zero. Yearwise retrospective point skills are 15.72% in 2022, 14.23% in 2023, and 17.48% in 2024. Pooled retrospective RMSE and MAE improve by 10.02% and 8.07%, and ACC by 0.0628. Signed-bias change remains unresolved at −0.020 mm day\(^{-1}\) (−0.137 to 0.093). Coverage improves substantially but remains imperfect: nominal 90% coverage is 0.800 rather than 0.900.

### 5.2 The categorical ranking changes with lead

On the same 208 reused-development starts and common support, the neural adapter lowers pooled quintile RPS from 0.248478 to 0.206283, a 16.98% reduction (13.40–20.83%). Persistence++ and combined PBC are better pooled at 0.196170 and 0.197027. The ordering reverses at W1: neural scores 0.151680 and beats Persistence++ by 7.73% (4.39–12.38%) and combined by 8.32% (4.46–13.23%). W2 is unresolved; both persistence-based methods are better at W3–W6. This reused evidence suggests an early-lead neural specialist, not universal neural superiority.

**Figure 2 (full-width, two panels).** Frozen 2022–2024 quintile performance on 296 common starts. **A:** same-case RPS for raw FuXi, neural location + spread, Persistence++, and combined PBC. **B:** paired neural score reductions with exploratory 95% intervals; filled markers identify intervals above zero and open markers unresolved contrasts. Source: [receipt-bound operational categorical figure](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z/figures/operational_categorical_transfer.pdf). The development and later-era continuous lead curves remain available in Appendix D from the [continuous figure asset](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/presentation/deliverables/fuxi_allseason_probabilistic_paper_20260822T192948Z/figures/continuous_crps_skill_by_lead.pdf).

The frozen 2022–2024 comparison confirms transfer versus raw FuXi. Neural quintile RPS is 0.184063 versus 0.239489 raw, a 23.14% reduction (15.87–30.60%); semidecile RPS is 0.161821 versus 0.211836, a 23.61% reduction (16.09–31.41%); and the descriptive upper-q95 Brier score improves 12.24% (9.01–16.54%). At W1, neural beats Persistence++ by 10.68% (3.87–17.06%) for quintiles, 10.75% (4.12–16.34%) for semideciles, and 13.73% (8.75–19.03%) at q95.

Pooled multi-threshold dominance is not established. Persistence++ is numerically better than neural for quintiles (0.182154; neural difference −1.05%, −5.77 to 4.25%) and semideciles (0.160485; −0.83%, −5.31 to 4.05%), but both paired intervals cross zero. W2 is unresolved, and W3–W6 point estimates favor persistence while every neural-versus-Persistence++ and neural-versus-combined interval crosses zero. Neural is best for pooled q95, beating Persistence++ by 2.19% (0.40–3.95%) and combined by 2.41% (1.51–3.46%). This single upper-threshold diagnostic is not evidence of extreme-event, drought, or flood skill. With only three year clusters, all intervals are exploratory, condition on the mean score across three neural seeds, and do not adjust for the 252 reported comparisons.

### 5.3 More capacity and deterministic loss mixing do not help

Table 2 reports two predeclared falsification checks. A 157,570-parameter model improves validation CRPS by only 0.0446% over the 42,434-parameter base and fails the year/materiality guard; the 293,762-parameter model is worse. A parameter-matched summary-only control is also nearly tied, so member-level encoding is not established as essential. Adding an MSE term progressively worsens reused-development CRPS, and MSE-only training is worst.

**Table 2. Negative ablations.** Capacity is a validation-only screen. Objective sensitivity uses post-hoc reused-development values and is not an untouched test.

| Ablation | Variant | Parameters or \(\alpha_{\mathrm{MSE}}\) | CRPS | Frozen decision |
|---|---|---:|---:|---|
| Capacity (validation) | Small set | 19,618 | 1.334243 | Worse than base |
| Capacity (validation) | **Base set** | **42,434** | **1.334153** | **Retained** |
| Capacity (validation) | Medium set | 157,570 | 1.333559 | +0.0446% only; fails guard |
| Capacity (validation) | Large set | 293,762 | 1.334791 | Worse than base |
| Capacity (validation) | Summary matched | 43,058 | 1.333589 | +0.0423% only; 1/3 seeds improves |
| Objective (reused development) | **CRPS only** | **0.00** | **1.3874** | **Retained** |
| Objective (reused development) | CRPS + MSE | 0.10 | 1.3913 | Worse |
| Objective (reused development) | CRPS + MSE | 0.25 | 1.3981 | Worse |
| Objective (reused development) | CRPS + MSE | 0.50 | 1.4066 | Worse |
| Objective (reused development) | MSE only | 1.00 | 1.5611 | Worse |

The hybrid objective is \(L_\alpha=(1-\alpha)\operatorname{CRPS}+\alpha(C_0/E_0)\operatorname{MSE}\), with its scale fitted on training data. These results rule out two simple remedies for the observed long-lead gap; they do not rule out other architectures, objectives, or hybrid forecast designs.

## 6. Climate relevance, limitations, and broader impact

The intervention reuses an existing AI ensemble and a regional observation archive rather than training a new global model. The selected post-processor has 42,434 parameters. Its three immutable primary model loops took 83.9, 86.2, and 79.8 seconds (249.9 seconds total) on one NVIDIA A30 after cached preprocessing. This is model-loop timing, not end-to-end latency, energy use, or an emissions estimate. Better-calibrated weekly distributions could support risk-aware products, but we have not measured sectoral utility, cost–loss value, user outcomes, or local warning skill.

The 2020–2021 cohort was reused during development and contains only two year clusters; the 2022–2024 audit is post-selection, retrospective, and contains only three. Training and scoring are regional, at coarse 1.5° resolution, using IMD as the sole fitting and primary target. Weekly means do not test daily timing, drought duration, flash floods, extremes, or impacts. Signed bias is unresolved, residual undercoverage remains, and the summary-only model nearly ties the member encoder. Persistence/PBC is better pooled and at W3–W6 in reused development; later-era quintile/semidecile differences are unresolved despite similar point ordering. The later archive changes both era and member count, so transfer does not isolate either shift. Prospective robustness, latency, decision utility, and 2025 performance remain untested.

## 7. Conclusion

A frozen 42,434-parameter distributional adapter improves FuXi-S2S weekly rainfall accuracy and reliability over India and retains skill in a later 50-member retrospective without retraining. Categorical results identify it as a robust early-lead specialist; pooled later-era quintile/semidecile differences from Persistence++ remain unresolved. Negative ablations and this lead dependence favor compact neural calibration complemented by strong classical baselines.

<!-- Four-page main text ends. -->

---

## References outline

Use the venue's final bibliography style and verified metadata from the linked primary sources; do not infer missing author lists or titles from this draft.

1. Chen et al. (2024), **FuXi-S2S**, primary article: <https://doi.org/10.1038/s41467-024-50714-1>.
2. Horat and Lerch (2024), global neural S2S categorical post-processing, primary article: <https://doi.org/10.1175/MWR-D-23-0150.1>.
3. Scheuerer et al. (2020), regional neural S2S precipitation post-processing, primary article: <https://doi.org/10.1175/MWR-D-20-0096.1>.
4. Höhlein et al. (2024), permutation-invariant ensemble post-processing, primary article: <https://doi.org/10.1175/AIES-D-23-0070.1>.
5. Noh and Ahn (2026), spatially conditioned S2S rainfall correction, primary article: <https://doi.org/10.1038/s41612-026-01430-8>.
6. Guan et al. (2026), probabilistic bias correction, [preprint](https://arxiv.org/abs/2604.16238) and [official code](https://github.com/mouatadid/pbc).

## Appendix outline

- **A. Architecture and invariance:** layer table, exact 42,434-parameter count, identity initialization, permutation-equivariance checks, and 51/50-member compatibility.
- **B. Data and integrity contract:** initialization calendars, purge rules, 27 × 27 grid and 171-cell support, IMD coverage weighting, eligible 2022–2024 dates, daily-to-weekly validation, normalization hashes, and the explicit 2025 firewall.
- **C. Metrics and uncertainty:** area weights, finite-ensemble CRPS implementation, ACC and spread/error definitions, central interval construction, equality-aware score, RPSS reference, and two-stage block bootstrap algorithm.
- **D. Continuous diagnostics:** W1–W6 intervals, yearwise and seasonal results, seed sensitivity, reliability curves, rank histograms, and 50/80/90% coverage. Keep upper-tail diagnostics descriptive; make no event-skill claim.
- **E. Common-support categorical comparison:** all seven methods, threshold masks, strict-CDF tie handling, nominal-formula sensitivity, and all paired leadwise effects on reused 2020–2021 development data.
- **F. Ablations:** complete capacity guard, per-year/per-seed validation values, summary-only control, location-only control, and CRPS–MSE objective results.
- **G. Operational categorical extension:** full quintile, semidecile, and upper-q95 tables; all 252 unadjusted paired comparisons; seed-score averaging; three-year two-stage resampling; lag provenance; and manifest, semantic, bootstrap, and Slurm-gate reconstruction. Label every result post-hoc and retrospective.
- **H. Reproducibility and compute:** commands, source snapshots, environment, hardware, artifact inventory, timing scope, and receipt validation.

## Exact evidence links and hashes

Every fixed number in the main text must trace to one of these immutable sources. Paths are local to the evidence workspace.

| Evidence role | Immutable source | SHA-256 |
|---|---|---|
| Neural architecture, split, continuous metrics, coverage, timing | [neural manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z/manifest.json) | `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91` |
| Validation-only capacity selection | [capacity manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z/manifest.json) | `2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c` |
| Locked capacity winner on reused development | [capacity-development manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_capacity_development_evaluation/full_20260822T223100Z/manifest.json) | `883aa0b2955b658b9fd164ac839103a4e80a33da2cdbd5a22b6b8fbf4515ec60` |
| CRPS–MSE objective ablation | [loss manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_hybrid_loss_ablation/full_final_20260822T141844Z/manifest.json) | `a3e77e4cf4e6485a756d99b68fec102fece37ef096725e496fe4a27819828f5e` |
| PBC components and tie-aware score | [PBC manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json) | `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6` |
| Accepted common-support categorical comparison | [categorical manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z/manifest.json) | `44ca7130e626bfadc71d311d0002dbb39b67aebbf16d0dfb2802999b6be4bdbc` |
| Accepted 2022–2024 continuous audit | [operational manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/manifest.json) | `7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad` |
| Operational audit Slurm gate | [gate receipt](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/slurm_gate_receipt.json) | `65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd` |
| Operational audit semantic reconstruction | [semantic receipt](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/evaluation/postflight_semantic_audit.json) | `82a7d2950d1323e9ec70362d68a449e6bbcb2bad15eb29a7ba1594c8862ea4c8` |
| Receipt-gated paper tables and figures | [paper-bundle manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/presentation/deliverables/fuxi_allseason_probabilistic_paper_20260822T192948Z/manifest.json) | `cf9d17dd335a1f680d0a7fb2f5f7a45ac7b556bb5a621d0bf564157cdc4be0d9` |
| Primary-source related-work map | [related-work map](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/docs/ALLSEASON_RELATED_WORK_MAP_20260822.md) | `60fbff8a99d0c6a1e444f88b232af23ae331d72fd761819b6fd58db845e38139` |
| Accepted 2022–2024 categorical audit | [operational categorical manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/manifest.json) | `cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d` |
| Operational categorical semantic reconstruction | [semantic receipt](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/evaluation/postflight_semantic_audit.json) | `00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c` |
| Operational categorical Slurm gate | [gate receipt](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/slurm_gate_receipt.json) | `cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c` |
| Operational categorical paired bootstrap | [bootstrap table](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/metrics/paired_two_stage_bootstrap.csv) | `632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188` |
| Receipt-bound operational categorical paper supplement | [derived bundle manifest](/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z/manifest.json) | `bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a` |

## Pre-submission checklist

- [x] Integrate only the independently reconstructed, source-locked 2022–2024 categorical full result; verify its manifest, semantic, bootstrap, and Slurm-gate SHA-256 values.
- [ ] Verify every operational continuous value against the single accepted full manifest with SHA-256 `7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad`; never use a smoke, retry, log, or failed output.
- [ ] Keep “reused development” beside every 2020–2021 result and “post-hoc operational-era retrospective; no retraining or selection” beside every 2022–2024 result.
- [ ] Keep the 2025 forecast and target sealed; include no 2025 result or implication.
- [ ] Preserve score averaging across seeds 42/43/44; do not average parameters, corrections, members, or forecasts.
- [ ] Show signed bias without favorable emphasis and state that its change is unresolved.
- [ ] Retain residual undercoverage and the summary-only near-tie; distinguish the resolved pooled/W3–W6 PBC advantage in reused development from unresolved later-era differences.
- [ ] Call the PBC implementation a frozen-split India/IMD adaptation, never a reproduction of Guan et al.
- [ ] Make no global-training, district-scale, extremes, drought, flood, prospective-validation, deployability, utility, end-to-end latency, energy, or emissions claim.
- [ ] Assemble Figure 1 and place the verified two-panel Figure 2; confirm its caption says post-hoc 2022–2024 retrospective and the continuous appendix distinguishes reused development from retrospective transfer.
- [ ] Format references from the linked primary records and verify the final CCAI template, page limit, anonymity policy, and appendix rules.
- [ ] Run a final receipt/hash audit and archive the exact compiled manuscript source, figures, tables, bibliography, and build environment.
