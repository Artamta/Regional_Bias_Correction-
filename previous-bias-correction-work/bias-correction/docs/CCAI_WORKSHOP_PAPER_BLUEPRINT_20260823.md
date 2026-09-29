# CCAI workshop paper blueprint: compact probabilistic FuXi calibration

Date: 23 August 2026  
Status: four-page, reviewer-safe blueprint; not submission prose  
Evidence cutoff: only completed, immutable artifacts listed below  
Operational-era status: **accepted continuous and categorical full audits**;
Slurm jobs 110580 and 110593, 296 cases each  
Final-test status: the 2025 forecast and target remain sealed and unopened

## Working title

> **A 42k-Parameter Ensemble Adapter for Lead-Dependent Calibration of
> FuXi-S2S Rainfall over India**

Shorter backup title:

> **Lightweight Probabilistic Calibration of FuXi-S2S Rainfall over India**

The first title is preferable because it advertises both the compute-constrained
method and the scientifically important neural--classical lead crossover.

## One-sentence thesis

A 42,434-parameter, member-preserving adapter trained directly with CRPS
substantially improves FuXi's continuous weekly rainfall ensemble and shows
lead-dependent neural--persistence complementarity: without retraining on 296
2022--2024 50-member starts, it retains 15.76% continuous CRPS skill, reduces
raw quintile RPS by 23.14%, beats Persistence++ at W1, and is statistically
indistinguishable from it for pooled ordinary categorical RPS.

This is a paper about inexpensive regional probabilistic calibration and
lead-dependent method complementarity. It is not a claim of universal neural
dominance, universal bias removal, global training, or operational deployment.

## Intended reviewer takeaway

One small regional post-processor can repair much of the underdispersion and
continuous-error problem in a modern AI S2S ensemble. The fair categorical
comparisons reveal a useful boundary: learned ensemble correction is strongest
at W1, while observation-lag persistence is difficult to beat pooled and at
later leads. In the later-era cohort the ordinary pooled neural--persistence
difference is unresolved rather than a replicated classical win. The negative
capacity and hybrid-loss results show that the missing ingredient is not simply
a larger network or a deterministic penalty.

## Contributions to claim

1. **Compact member-preserving calibration.** A permutation-invariant residual
   adapter predicts spatial log-rainfall location and spread fields and
   transforms every FuXi member, retaining the empirical ensemble as the
   predictive distribution. The selected model has 42,434 parameters and exact
   raw-ensemble identity initialization.
2. **Proper-score training under regional constraints.** The adapter is fitted
   over a 27 x 27 India box using area-weighted finite-ensemble CRPS, all
   seasons, six weekly leads, and a purged 2002--2017/2018--2019
   training/selection split.
3. **Genuinely common-support neural--PBC comparisons.** Frozen neural seeds,
   Debias++, Persistence++, combined PBC, moment calibration, and raw FuXi are
   scored on the same 208 development initializations, thresholds, IMD truth,
   spatial support, and tie-aware score; the locked core methods are then
   rescored on all 296 eligible later-era starts. These comparisons expose W1
   neural strength and show that a pooled number alone hides lead dependence.
4. **Falsification-oriented evidence.** Capacity from 19,618 to 293,762
   parameters and CRPS--MSE hybrid objectives do not yield a robust improvement.
   In a source-locked, no-retraining 2022--2024 retrospective audit across the
   51-to-50-member shift, the frozen adapter retains 15.76% pooled CRPS skill
   (paired 95% interval 13.78--17.76%) and positive skill at every lead. In the
   matched categorical audit it reduces raw quintile RPS by 23.14%
   (exploratory 95% interval 15.87--30.60%), wins W1 against
   Persistence++/combined PBC, and is pooled statistically indistinguishable
   from both under ordinary categorical RPS.

Do not claim that the learned set encoder itself is essential: the summary-only
model is nearly tied. The defensible novelty is compact distributional
location/spread calibration plus the controlled comparison, not merely the use
of a set network.

## Four-page structure and budget

References and any permitted appendix are outside this budget.

| Part | Pages | Required content |
|---|---:|---|
| Title and abstract | 0.30 | Motivation, method, development result, categorical crossover, accepted later-era audit |
| 1. Motivation and contributions | 0.50 | India S2S uncertainty need; why calibration rather than a new forecast model; four contributions |
| 2. Related work and positioning | 0.25 | FuXi-S2S; neural S2S post-processing; set post-processing; recent PBC; precise distinction |
| 3. Data and method | 0.85 | Split, 51-member hindcast, India/IMD target, adapter equations, CRPS objective, Figure 1 |
| 4. Evaluation protocol | 0.40 | Continuous and categorical contracts, seed handling, paired bootstrap, later-era gate |
| 5. Results and ablations | 1.30 | Figure 2, Tables 1--2, lead crossover, no-retraining later-era result |
| 6. Climate relevance and limitations | 0.30 | Low-cost reuse of AI forecasts; decision boundary; temporal and spatial limits |
| 7. Conclusion | 0.10 | One restrained takeaway; no new number |
| **Total** | **4.00** | |

Suggested physical layout:

- Page 1: abstract, motivation/contributions, compact Figure 1.
- Page 2: related work, data/method equations, evaluation protocol.
- Page 3: Figure 2 and the continuous headline in Table 1.
- Page 4: Table 2, operational-era branch, climate relevance, limitations, and
  conclusion.

## Abstract skeleton (fill slots; do not expand yet)

Target length: 145--175 words.

1. **Climate/forecasting problem, one sentence:** calibrated subseasonal
   rainfall uncertainty over India matters for anticipatory planning, but a
   strong global AI ensemble can remain regionally underdispersed and
   miscalibrated.
2. **Gap, one sentence:** existing neural post-processors often predict
   categories or deterministic fields, while recent PBC is a strong classical
   probability baseline; the value of a tiny member-preserving adapter is
   unclear.
3. **Method, one sentence:** introduce a 42,434-parameter permutation-invariant
   residual adapter that learns log-location and spread fields and optimizes
   the finite-ensemble CRPS.
4. **Protocol, one sentence:** fit on 2002--2017, select on 2018--2019, report
   2020--2021 explicitly as reused development evidence, and compare all
   categorical methods on identical tie-aware thresholds/support.
5. **Fixed continuous result, one sentence:** on 208 reused-development
   initializations, CRPS falls from 1.6590 to 1.3874, a 16.37% reduction
   (paired 95% interval 14.28--18.16%), while 90% coverage rises from 0.427 to
   0.789 and spread/error moves from 0.562 to 0.967.
6. **Fixed categorical result, one sentence:** on reused development, neural
   beats both classical comparators at W1, is unresolved at W2, and is worse at
   W3--W6; in the 296-start later-era retrospective it reduces raw quintile RPS
   by 23.14% (exploratory 95% interval 15.87--30.60%), again wins W1, and is
   pooled statistically indistinguishable from Persistence++/combined PBC.
7. **Operational result, exactly one sentence:** among 296 protocol-eligible
   2022--2024 starts with 50 members, the frozen adapter achieved 15.76% pooled
   CRPS skill (paired 95% interval 13.78--17.76%), with positive skill from W1
   (29.02%) through W6 (10.50%); no retraining or selection was performed.
8. **Implication, one sentence:** lightweight neural and lag-based correction
   should be viewed as lead-dependent complements, while the successful
   retrospective transfer audit motivates prospective evaluation rather than
   establishing deployment readiness.

Never put an operational number into the abstract from a log, smoke run,
incomplete staging directory, or unaudited CSV. The values above come only
from the accepted full manifest and its bound audit receipts.

## Method notation and equations to present

Use initialization \(i\), member \(m=1,\ldots,M\), lead week
\(\ell=1,\ldots,6\), and grid cell \(g\). Hindcast evaluation has \(M=51\);
the operational-era audit has \(M=50\). Rainfall is in
mm day\(^{-1}\).

### Member representation

For nonnegative raw rainfall \(x_{im\ell g}\), work in stabilized space:

\[
u_{im\ell g}=\log(1+x_{im\ell g}),\qquad
\mu_{i\ell g}=\frac{1}{M}\sum_m u_{im\ell g},\qquad
d_{im\ell g}=u_{im\ell g}-\mu_{i\ell g}.
\]

The shared pointwise encoder and symmetric pooling are

\[
z_{i\ell g}=\frac{1}{M}\sum_m
\phi\!\left(u_{im\ell g},d_{im\ell g}\right),\qquad
r_{i\ell g}=\sqrt{\frac{1}{M}\sum_m d_{im\ell g}^{2}}.
\]

The compact 3-D residual backbone acts jointly over lead, latitude, and
longitude:

\[
(\delta_{i\ell g},a_{i\ell g})
=h_\theta\!\left[z_{i\ell g},\mu_{i\ell g},r_{i\ell g},c_{i\ell g}\right],
\qquad s_{i\ell g}=\exp\!\left(2\tanh a_{i\ell g}\right).
\]

The seven context channels \(c\) are training-only normalized IMD
climatology, day-of-year sine/cosine, latitude, longitude, lead, and the fixed
support mask. The two output heads start at zero.

### Memberwise residual transform

Present the implementation-equivalent residual form because it makes the
identity property clear:

\[
\widetilde u_{im\ell g}
=\max\!\left\{0,
u_{im\ell g}+\delta_{i\ell g}
+(s_{i\ell g}-1)d_{im\ell g}\right\},
\qquad
\widetilde x_{im\ell g}=\exp(\widetilde u_{im\ell g})-1.
\]

When \(\delta=0\) and \(s=1\), the corrected members equal the raw members;
permuting members only permutes outputs. State both as tested model properties.

### Training objective

For one grid point and verifying rainfall \(y\), use the empirical
finite-ensemble CRPS:

\[
\operatorname{CRPS}(\widetilde x,y)
=\frac{1}{M}\sum_m|\widetilde x_m-y|
-\frac{1}{2M^2}\sum_{m,n}|\widetilde x_m-\widetilde x_n|.
\]

Average over valid cells using latitude-area weights, then over leads and
initializations. This is the only retained training loss. Explain in one phrase
that memberwise MSE would reward collapse toward a conditional mean; do not add
a KL-divergence story because each case supplies one verifying realization,
not a target distribution.

### Shared categorical endpoint

For training-only quintile thresholds, use strict CDFs
\(\widehat F(q)=M^{-1}\sum_m\mathbf 1(\widetilde x_m<q)\) and
\(O(q)=\mathbf 1(y<q)\). Let \(Q^+_{i\ell g}\) contain each distinct positive
physical threshold once. Then

\[
R_{i\ell g}=\frac{1}{|Q^+_{i\ell g}|}
\sum_{q\in Q^+_{i\ell g}}
\left[\widehat F_{i\ell g}(q)-O_{i\ell g}(q)\right]^2.
\]

Rows with no positive threshold are excluded and the area-weight denominator
is recomputed on the identical threshold-only mask for every method and
reference. Define RPSS as \(1-\overline R_{\rm method}/\overline R_{\rm
train\text{-}empirical\ climatology}\). Keep the Guan-style four-slot score as
a secondary sensitivity and call the implemented method a frozen-split PBC
adaptation, not a reproduction.

### Uncertainty and seed aggregation

- The statistical unit is one initialization with all members and all six
  weeks attached.
- Resample years, then circular 13-initialization blocks within each sampled
  year; never bootstrap members or grid cells as independent samples.
- Neural seeds 42/43/44 are optimization replicates. Score each forecast
  separately, then average scores for each identical case/lead; never average
  model parameters, corrections, or forecasts.
- Say that 2020--2021 intervals are exploratory because they contain only two
  year clusters. The 2022--2024 audit has only three year clusters and is a
  retrospective transfer audit, not a prospective operational trial.

## Exact main visual and table plan

### Figure 1 — adapter schematic (single column, about 0.28 page)

Flow, left to right:

`unordered 51/50 FuXi members` -> `log1p members + deviations` -> `shared
member encoder and symmetric mean` -> `log-location, RMS spread, seven context
channels` -> `compact lead x latitude x longitude residual Conv3D` ->
`delta and positive spread` -> `memberwise corrected ensemble`.

Annotate only three facts: 42,434 parameters; exact identity at initialization;
all members retained at evaluation. Do not draw a second deterministic output.

### Figure 2 — accepted later-era categorical transfer (full width)

Use the already verified two-panel asset at
`presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z/figures/operational_categorical_transfer.pdf`.
Panel A shows same-case 2022--2024 quintile RPS for raw FuXi, neural,
Persistence++, and combined PBC. Panel B shows paired neural score reductions
and exploratory 95% intervals, with filled markers only when the full interval
is above zero. Its source bundle manifest SHA-256 is
`bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a`.

The figure makes the central boundary visible: neural beats raw at W1--W6 and
beats both classical methods at W1; W2--W6 neural--classical intervals are
unresolved even though W3--W6 point estimates favor PBC. In the caption state
“post-hoc 2022--2024 retrospective; 296 starts; no retraining or selection;
three year clusters.” Keep the semidecile and descriptive upper-q95 results in
text or the appendix. Move the continuous W1--W6 transfer curves to Appendix D;
Table 1 already carries the pooled continuous headline.

### Table 1 — pooled continuous performance and transfer (full width)

Columns:

`cohort | evidence label | members | initializations | method | CRPS | CRPS
skill (95% interval) | RMSE | ACC | signed bias | 90% coverage | spread/error`.

Fixed 2020--2021 rows:

| Method | CRPS | CRPS skill | RMSE | ACC | Bias | Coverage 90 | Spread/error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw FuXi | 1.6590 | 0 | 3.5452 | .2312 | -.1206 | .427 | .562 |
| Neural location+spread | **1.3874** | **16.37% [14.28, 18.16]** | **3.0994** | **.3521** | -.1578 | **.789** | **.967** |

Accepted post-selection 2022--2024 retrospective rows:

| Method | CRPS | CRPS skill | RMSE | ACC | Bias | Coverage 90 | Spread/error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw FuXi, 50 members | 1.5626 | 0 | 3.4203 | .2738 | -.0164 | .415 | .566 |
| Frozen neural adapter | **1.3164** | **15.76% [13.78, 17.76]** | **3.0777** | **.3366** | -.0368 | **.800** | **1.032** |

Do not bold signed bias in the development table: its change is -0.037
mm day\(^{-1}\) with interval -0.167 to +0.091, so there is no supported
bias-improvement claim. The operational-era bias change is likewise unsupported:
-0.020 mm day\(^{-1}\), interval -0.137 to +0.093.

### Table 2 — compact negative ablations (single column)

Use two clearly labelled blocks because the endpoints differ.

**A. Validation-only capacity screen:**

| Model | Parameters | Validation CRPS | Decision |
|---|---:|---:|---|
| Small set | 19,618 | 1.334243 | worse than base |
| **Base set** | **42,434** | 1.334153 | **retained** |
| Medium set | 157,570 | 1.333559 | +0.0446% only; fails year/materiality guard |
| Large set | 293,762 | 1.334791 | worse than base |
| Summary matched | 43,058 | 1.333589 | +0.0423% only; 1/3 seeds improves |

**B. Reused-development objective sensitivity:**

| Objective | CRPS |
|---|---:|
| **CRPS only** | **1.3874** |
| Hybrid (MSE weight \(\alpha=0.10\)) | 1.3913 |
| Hybrid (MSE weight \(\alpha=0.25\)) | 1.3981 |
| Hybrid (MSE weight \(\alpha=0.50\)) | 1.4066 |
| MSE only | 1.5611 |

Caption: selection was made from validation; development values are post-hoc
ablation evidence. The hybrid objective is
\(L_\alpha=(1-\alpha)\,CRPS+\alpha(C_0/E_0)\,MSE\), with the scale fitted on
training data. Do not describe the medium network as tied through a significance
test; it merely failed the frozen promotion rule and year guard.

## Results narrative order

1. Start with the continuous forecast: raw FuXi is underdispersed; the selected
   adapter improves CRPS, RMSE, ACC, and interval coverage in reused development
   years, while signed bias is not demonstrably improved.
2. Show the lead dependence: continuous CRPS skill remains positive from W1 to
   W6 but decreases with lead.
3. Show the reused-development categorical comparison: neural is the W1
   specialist, W2 is unresolved, and persistence/PBC is stronger from W3
   onward and pooled. Do not bury that development-cohort classical win.
4. Present both accepted operational-era results as frozen transfer evidence:
   continuous CRPS skill is 15.76% and positive in every lead/year; categorical
   neural skill versus raw is 23.14% for quintiles and 23.61% for semideciles,
   W1 neural strength persists, and pooled ordinary neural--PBC differences
   are unresolved. No retraining, selection, parameter averaging, or forecast
   averaging was performed.
5. Close with negative ablations: extra width and deterministic loss mixing do
   not solve the remaining long-lead problem.

## Accepted operational-era audit

The source-locked full run is Slurm job 110580. It retained every eligible
all-season initialization (296: 104 in 2022, 104 in 2023, and 88 in 2024), used
50 members, and performed no training, fine-tuning, blending, or selection.
The manifest, post-flight semantic audit, Slurm receipt, all 82 declared
artifacts, eight source snapshots, normalization, daily-to-weekly aggregation,
and 2022--2024 store whitelist were independently verified. The 2025 forecast
and target remained sealed and unopened.

This satisfies the predeclared **strong transfer** branch: pooled CRPS-skill is
15.76% [13.78, 17.76], every lead interval is above zero, and yearwise point
skills are 15.72% (2022), 14.23% (2023), and 17.48% (2024). Say “retained skill
in a no-retraining, later-era retrospective audit”; do not say “independent
test,” “prospectively validated,” or “operationally validated.”

## Accepted operational-era categorical audit

The source-locked categorical full run is Slurm job 110593. It rescored the
locked neural adapter and frozen projected classical baselines on the same 296
eligible 2022--2024 starts, training-derived thresholds, IMD truth, and support.
It performed no training, fine-tuning, selection, or fitted blending. Neural
headline rows are the arithmetic mean of scores from seeds 42/43/44 for each
identical case and lead; no forecasts, probabilities, corrections, or model
parameters were averaged. The 2025 stores remained sealed and unopened.

Pooled neural quintile RPS is .184063 versus .239489 raw, a 23.14% reduction
with exploratory 95% interval [15.87, 30.60]. Semidecile RPS is .161821 versus
.211836, a 23.61% reduction [16.09, 31.41]. At W1, the neural quintile score
beats Persistence++ by 10.68% [3.87, 17.06] and combined PBC by 9.42%
[3.80, 15.51]. Pooled ordinary neural scores are numerically slightly worse
than Persistence++ and combined PBC, but every corresponding interval crosses
zero. The descriptive pooled upper-q95 Brier score is lowest for neural
(.056665 versus .064569 raw), a 12.24% reduction [9.01, 16.54]; this is not an
event or extremes endpoint.

These intervals are exploratory because only three year clusters are
available. The bootstrap table has 252 paired comparisons with no multiplicity
adjustment, and neural contrasts condition on the per-case mean of per-seed
scores. The result supports later-era categorical transfer and lead-dependent
complementarity, not global training, prospective validation, deployment, 2025
performance, or drought/heavy-rain event skill.

The immutable provenance anchors are manifest SHA-256
`cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d`,
semantic-audit SHA-256
`00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c`,
Slurm-receipt SHA-256
`cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c`,
and paired-bootstrap CSV SHA-256
`632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188`.

## Claim language

### Supported now

- “On reused 2020--2021 development data, the 42,434-parameter adapter reduced
  continuous CRPS by 16.37% relative to raw FuXi (paired 95% interval
  14.28--18.16%).”
- “CRPS skill was positive at all six leads in this development evaluation,
  declining from 30.65% at W1 to 7.91% at W6.”
- “The 90% empirical interval coverage increased from 0.427 to 0.789 and the
  pooled spread/error ratio moved from 0.562 to 0.967.”
- “Under a common tie-aware categorical score, the neural adapter led at W1,
  W2 differences were unresolved, and Persistence++/combined PBC led at
  W3--W6.”
- “Neither increasing model width to 293,762 parameters nor adding an MSE term
  produced a robust improvement under the frozen selection/evaluation rules.”
- “Without retraining, the frozen adapter retained 15.76% pooled CRPS skill
  (paired 95% interval 13.78--17.76%) across the 2022--2024 50-member
  operational-era archive, with positive skill at all six leads.”
- “In that audit, RMSE improved 10.02%, MAE 8.07%, and ACC by 0.063; signed-bias
  change remained unresolved.”
- “Operational-era 50/80/90% coverage rose from .187/.338/.415 to
  .519/.720/.800, while spread/error moved from .566 to 1.032.”
- “In the post-hoc 296-start 2022--2024 categorical retrospective, the frozen
  neural adapter reduced raw FuXi quintile RPS by 23.14% (exploratory 95%
  interval 15.87--30.60%) and semidecile RPS by 23.61%
  (16.09--31.41%), without retraining or selection.”
- “The later-era neural adapter beat Persistence++ and combined PBC at W1;
  pooled ordinary neural--classical differences were unresolved.”
- “Its pooled descriptive upper-q95 Brier score was lowest and improved raw by
  12.24% [9.01, 16.54], but this threshold diagnostic is not an event or
  extremes result.”

### Forbidden

- “Independent test,” “prospective,” “operationally validated,” “state of the
  art,” or “production ready.”
- “Whole-world/global training”: the source FuXi archive is global, but fitting
  and IMD verification are regional.
- “Universal bias correction”: signed bias did not significantly improve in
  2020--2021.
- “The member encoder is essential”: the summary-only network is nearly tied.
- “Combined PBC beats Persistence++”: their pooled difference is unresolved and
  Persistence++ is numerically better.
- “Neural beats Persistence++/combined PBC pooled in the operational-era
  ordinary categorical comparison”: neural is numerically slightly worse and
  the paired intervals cross zero.
- “Improves extremes,” “drought prediction,” or “flood prediction”: current
  q95 diagnostics have paired uncertainty but remain descriptive thresholds;
  no event-focused decision study exists.
- Any 2025 result or suggestion that 2025 was inspected.

## Related-work positioning in four sentences

1. [FuXi-S2S](https://doi.org/10.1038/s41467-024-50714-1) supplies the global
   AI ensemble; this work post-processes its rainfall members regionally rather
   than replacing the forecast system.
2. Neural S2S precipitation post-processing has produced categorical or
   deterministic forecasts, including
   [Scheuerer et al.](https://doi.org/10.1175/MWR-D-20-0096.1),
   [Horat and Lerch](https://doi.org/10.1175/MWR-D-23-0150.1), and
   [Noh and Ahn](https://doi.org/10.1038/s41612-026-01430-8); this study retains
   a continuous memberwise predictive distribution through W6.
3. Permutation-invariant post-processing motivates the set representation
   ([Höhlein et al.](https://doi.org/10.1175/AIES-D-23-0070.1)), while the
   summary-only near-tie here prevents an exaggerated representation claim.
4. [Guan et al.](https://arxiv.org/abs/2604.16238) establishes a strong recent
   PBC benchmark; the present Debias++/Persistence++/combined implementation is
   a frozen-split India/IMD adaptation with equality-aware dry thresholds, not
   a reproduction of their rolling protocol.

## Climate relevance paragraph ingredients

- The intervention reuses an existing AI weather ensemble and a regional
  observation archive instead of training a new global forecast model.
- The selected post-processor contains 42,434 parameters. In the immutable
  neural manifest, the three primary model loops took 83.9, 86.2, and 79.8 s
  (249.9 s total) on one NVIDIA A30 after cached preprocessing. Label this as
  model-loop time, not end-to-end pipeline cost or an emissions estimate.
- Better-calibrated weekly rainfall distributions can support risk-aware
  forecast products, but no sectoral utility, cost--loss value, user study, or
  local warning skill has yet been demonstrated.
- The practical contribution is a low-cost calibration layer and a diagnostic
  of when neural versus persistence information is useful—not an autonomous
  decision system.

## Limitations to print, not hide

1. The 2020--2021 cohort has prior development exposure and only two year
   clusters; its intervals do not create independent confirmation.
2. The 2022--2024 audits are retrospective and post-selection, and three years
   still weakly represent interannual variability. Categorical intervals are
   exploratory across 252 unadjusted paired comparisons and condition on the
   arithmetic mean of per-seed scores, not one deployable forecast.
3. Training and scoring cover a coarse 27 x 27, 1.5-degree India box. This is
   not district-scale verification.
4. Weekly means through W6 do not evaluate daily timing, flash floods, drought
   duration, or event impacts.
5. IMD is the sole fitting/primary target; observation and product uncertainty
   are not characterized through an independent gridded or station product.
6. Signed bias is not significantly improved, and the method should be called
   calibration rather than universal bias correction.
7. Although substantially improved, nominal 90% coverage reaches 0.789 in
   reused development and 0.800 in the operational-era audit, so residual
   undercoverage remains.
8. In reused development, Persistence++/combined PBC is stronger pooled and at
   W3--W6. In the later-era cohort, those later-lead point estimates mostly
   retain the same direction but pooled and later-lead ordinary differences
   are unresolved; the neural method is not uniformly best.
9. The summary-only near-tie means member-level encoding contributes at most a
   modest incremental gain in the present setting.
10. The operational archive changes from 51 to 50 members and era; any transfer
   result confounds these real distribution shifts rather than isolating them.
11. The 2025 final control remains sealed and cannot be implied by earlier
    periods.

## Reviewer-risk mitigation checklist

| Likely concern | Evidence/action in the paper |
|---|---|
| “This is only a small CNN.” | Lead with exact member exchangeability/identity, direct CRPS, variable-member evaluation, and the common-support neural--PBC finding—not parameter count alone. |
| “The test set was reused.” | Put “reused development” in abstract, Table 1, Figure 2, and limitations; never use “test” unqualified. |
| “Baselines are incomparable.” | State identical cases, members, IMD truth, persisted thresholds, support, references, and paired resamples; cite the accepted comparison manifest. |
| “Why not make the model larger?” | Show the predeclared capacity table and promotion guard; medium gains only 0.0446% and fails year robustness. |
| “Why not add MSE or KL?” | Show the hybrid-loss degradation; explain that CRPS already scores distribution quality and one observation is not a target density for KL. |
| “Member preservation may not matter.” | Agree with the evidence: summary-only nearly ties; frame member encoding as an ablation, not the central causal claim. |
| “PBC wins overall.” | Separate cohorts. Reused development shows W1 neural strength and W3--W6/pooled classical strength; later-era W1 neural strength persists, but pooled ordinary neural--classical differences are unresolved. |
| “Can this be used operationally?” | The no-retraining later-era audit passed, but it remains retrospective and does not establish latency, prospective robustness, or decision utility. |
| “Only two or three years support inference.” | Use initialization blocks nested within year resampling, report intervals, explicitly call interannual uncertainty weak, and disclose 252 unadjusted later-era categorical comparisons plus conditioning on the mean seed score. |
| “Rainfall zeros break quantile scores.” | Define strict CDFs, collapse duplicate physical cuts, remove deterministic q=0 cuts, and keep the nominal four-slot formula secondary. |
| “The paper overclaims extremes or impacts.” | Remove event/extreme claims from the four-page paper unless a separately frozen event protocol with uncertainty is completed. |
| “Compute claims omit preprocessing.” | Report model-loop timing and hardware only; do not convert it to carbon or end-to-end cost without measurement. |

## Evidence ledger for every fixed number above

All paths are relative to the repository root.

| Evidence | Immutable source | SHA-256 |
|---|---|---|
| Canonical neural architecture, splits, continuous metrics, coverage, timing | `resultsv2/fuxi_allseason_ensemble_calibration/full_publication_20260822T115253Z/manifest.json` | `94b80712df3dcb55e3478b8cfc5262ba4d300420c76b5680424e9005d67eeb91` |
| Validation-only capacity selection | `resultsv2/fuxi_allseason_capacity_ablation/full_20260822T220000Z/manifest.json` | `2e014a50d72395d90c3b9ee59156a4de2ad1a953ad29fc58ae5aa9c8bdb7e24c` |
| Locked capacity winner on 2020--2021 development | `resultsv2/fuxi_allseason_capacity_development_evaluation/full_20260822T223100Z/manifest.json` | `883aa0b2955b658b9fd164ac839103a4e80a33da2cdbd5a22b6b8fbf4515ec60` |
| CRPS--MSE objective ablation | `resultsv2/fuxi_allseason_hybrid_loss_ablation/full_final_20260822T141844Z/manifest.json` | `a3e77e4cf4e6485a756d99b68fec102fece37ef096725e496fe4a27819828f5e` |
| PBC V2 components and tie-aware score | `resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z/manifest.json` | `c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6` |
| Accepted neural--PBC common-support comparison and crossover | `resultsv2/fuxi_allseason_categorical_comparison_v2/full_20260822T183010Z/manifest.json` | `44ca7130e626bfadc71d311d0002dbb39b67aebbf16d0dfb2802999b6be4bdbc` |
| Accepted 2022--2024 no-retraining operational-era audit | `resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/manifest.json` | `7485db3094b894ec8b5dc2e060ba3c541b533f7e359bf02a6a5a63dac96db2ad` |
| Operational-era Slurm gate receipt | `resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/slurm_gate_receipt.json` | `65c319854aa805a470f3f65bd5b21dd286f89573d73f21c1f5b64c8202152edd` |
| Operational-era semantic audit receipt | `resultsv2/fuxi_allseason_operational_era_audit/full_20260822T190829Z/evaluation/postflight_semantic_audit.json` | `82a7d2950d1323e9ec70362d68a449e6bbcb2bad15eb29a7ba1594c8862ea4c8` |
| Accepted 2022--2024 operational-era categorical retrospective | `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/manifest.json` | `cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d` |
| Operational-era categorical semantic audit | `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/evaluation/postflight_semantic_audit.json` | `00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c` |
| Operational-era categorical Slurm gate receipt | `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/slurm_gate_receipt.json` | `cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c` |
| Operational-era categorical paired bootstrap | `resultsv2/fuxi_allseason_operational_categorical_comparison/full_20260822T203000Z/metrics/paired_two_stage_bootstrap.csv` | `632a7b188198f8229f9a6ef00ba9cf5d32c0f7935c0aeae965f2c091030e8188` |
| Receipt-bound operational categorical paper supplement | `presentation/deliverables/fuxi_allseason_operational_categorical_paper_20260822T204756Z/manifest.json` | `bafd1ee4a3704e6ae52c657dd4f07af72837b618a8121944a434aaf6f75aaa9a` |
| Receipt-gated paper table/figure bundle | `presentation/deliverables/fuxi_allseason_probabilistic_paper_20260822T192948Z/manifest.json` | `cf9d17dd335a1f680d0a7fb2f5f7a45ac7b556bb5a621d0bf564157cdc4be0d9` |
| Primary-source related-work map | `docs/ALLSEASON_RELATED_WORK_MAP_20260822.md` | `60fbff8a99d0c6a1e444f88b232af23ae331d72fd761819b6fd58db845e38139` |

Operational continuous and categorical numbers must come only from their
respective accepted full manifests above; never mix them with smoke or failed
pre-score attempts.

## Final pre-submission kill list

- Confirm that every operational number traces to the correct accepted
  continuous or categorical full manifest; never copy a favorable value from a
  retry or smoke output.
- Make the evidence label visible beside every 2020--2021 and 2022--2024
  result.
- Use only the receipt-bound later-era categorical Figure 2 and ensure Table 1
  shows signed bias, including the non-improvement.
- Label later-era categorical intervals exploratory, disclose three year
  clusters and 252 unadjusted comparisons, and state that neural contrasts
  condition on the mean of per-seed scores.
- Do not call the PBC adaptation a reproduction.
- Do not claim global training, event skill, district guidance, prospective
  verification, or 2025 evidence.
- Keep four contributions, two figures, and two tables; move full seasonal,
  reliability, rank, seed, and sensitivity diagnostics to the appendix or
  artifact bundle.
- Report optimization seeds as sensitivity, not extra meteorological samples.
- Keep the title/abstract frozen to the accepted strong-transfer branch unless
  a separately predeclared analysis supplies new evidence.
