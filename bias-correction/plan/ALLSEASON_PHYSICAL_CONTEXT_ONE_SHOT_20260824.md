# Frozen one-shot all-season physical-context experiment

Date frozen: 2026-08-24

## Question

Can case-dependent FuXi atmospheric state make the accepted neural
location-and-spread calibrator improve categorical RPS consistently enough to
beat the existing categorical baselines without using PBC or observations as
model inputs?

The accepted equal-weight neural seed-CDF pool already has a lower pooled
2018--2019 validation RPS than Combined PBC (0.186499 versus 0.188733), but it
is slightly worse in 2019 and worse than Persistence++ overall (0.185356).
The target is therefore a stable neural-only improvement, not merely a lower
pooled point estimate than PBC.

## Frozen candidate

Keep the accepted 51-member precipitation input, exchangeable member encoder,
location-and-spread member transformation, 24-channel backbone, weighted CRPS
training objective, optimizer, split, and seeds unchanged. Append these ten
lead-aligned weekly FuXi ensemble-mean context fields:

1. T2M mean
2. TCWV mean
3. q850 mean
4. u850 mean
5. v850 mean
6. q850 times u850 moisture-flux mean
7. q850 times v850 moisture-flux mean
8. z500 mean
9. mean sea-level pressure mean
10. positive outgoing longwave radiation mean (`OLR = -TTR`)

The nine fields after T2M are exactly the previously selected JJAS
`physical_full_compact` bank. No field subset, feature weighting, topography,
lead gate, loss, capacity, or hyperparameter search is permitted. Topography
is excluded because it is fixed by grid cell and the accepted model already
receives latitude, longitude, train-only rainfall climatology, season, and a
spatial convolutional context; it cannot supply the missing case-specific
late-lead information.

Each field is normalized independently by lead using only 2002--2017 training
cases and positive target-area weights. Raw physical fields are aligned by
initialization, lead, latitude, and longitude. No target observation enters
the physical cache.

## Controls

- Accepted frozen neural seed-CDF pool.
- A same-width control trained with ten exact-zero extra channels, so the
  physical candidate and control have identical parameter counts.
- Raw FuXi categorical, Combined PBC, and Persistence++ are benchmark-only;
  they do not select feature subsets or training checkpoints.

## Selection and claim gates

The physical candidate is retained only if, versus the matched same-width
zero control on purged 2018--2019 validation, it:

- lowers pooled categorical RPS;
- lowers W2--W6 pooled categorical RPS;
- lowers categorical RPS in both 2018 and 2019;
- has at least two of three matched seeds lower in pooled categorical RPS; and
- worsens mean ensemble CRPS by no more than 0.5%.

After this decision is written, compare the selected neural CDF pool with the
immutable PBC and Persistence++ CDF arrays. A neural superiority statement
requires lower pooled RPS, lower RPS in both years, and a positive paired
moving-block-bootstrap 95% interval for relative RPS reduction.

## Evidence boundary

This is an exploratory reused-validation experiment. The physical feature
hypothesis was informed by the prior 2018--2019 JJAS screen, 2020--2021 has
already been used for development elsewhere, and the operational 2025 archive
does not contain all ten required fields. Even a successful result is not an
independent NeurIPS test and must be described as an ablation or supporting
result unless evaluated on a genuinely untouched compatible cohort.
