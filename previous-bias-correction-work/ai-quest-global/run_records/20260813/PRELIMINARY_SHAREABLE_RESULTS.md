# Preliminary precipitation-postprocessing results

**Updated:** 16 August 2026  
**Status:** exploratory research result; not an official AI Weather Quest score and not submission evidence.

## Shareable result

We developed a compact probabilistic postprocessor for FuXi subseasonal precipitation forecasts. On a small exploratory FuXi–IMERG benchmark, a calibrated three-seed spatial ensemble achieved positive skill relative to uniform climatological probabilities:

| Metric | 2019 calibration/selection | 2020 retrospective |
|---|---:|---:|
| RPSS versus uniform climatology | **+2.92%** | **+3.66%** |
| RPS reduction versus raw FuXi category probabilities | **22.21%** | **21.95%** |
| RPS reduction versus calibrated FuXi probabilities | **1.59%** | **1.93%** |

The distinction between these metrics matters. The approximately 22% number is a reduction relative to a weak, uncalibrated FuXi probability anchor. The more meaningful improvement over uniform climatology is approximately **3–4% RPSS**.

Temporal attention did not provide a robust improvement. Its calibrated validation advantage was only 0.015%, while the uncalibrated three-seed attention ensemble was 0.73% worse and used 30% more parameters. The simpler spatial ensemble was therefore retained.

### Suggested wording

> In a preliminary 16-case global FuXi–IMERG experiment, our calibrated probabilistic postprocessor achieved +2.9% RPSS on the calibration/model-selection slice and +3.7% RPSS in a retrospective 2020 analysis. It reduced RPS by about 1.9% beyond calibrated FuXi probabilities. A full ERA5-based evaluation is now running, so these figures should be interpreted as promising engineering evidence rather than official competition skill.

Do not describe these values as an official ERA5 score, a leaderboard score, or a 22% RPSS improvement.

## What is being “vibe-coded”?

This project is **not training a weather simulator from scratch**. FuXi already generates the physical 51-member, 42-day weather ensemble. The code being developed is the probability and evaluation layer around FuXi:

```text
FuXi 51-member weather forecast
              |
       weekly rainfall totals
              |
ERA5 climatological quintile boundaries
              |
raw five-category probabilities (p0)
              |
small neural/statistical correction model
              |
calibration toward uniform when confidence is weak
              |
five probabilities for D19–25 and D26–32
```

More concretely, the repository implements:

1. **Data engineering:** align 2,080 historical FuXi forecasts with ERA5, construct exactly 100 prior-only climatology samples, and prevent observation leakage.
2. **Probabilistic forecasting:** convert the 51 FuXi members into five rainfall-category probabilities with positive Jeffreys smoothing.
3. **Postprocessing models:** compare raw FuXi, calibration, Debias++, a spatial U-Net control, and a compact six-week Transformer/U-Net adapter.
4. **Scientific selection:** train on 2002–2017, select on 2018–2019, and open 2020–2021 only after the candidate is frozen.
5. **Submission safety:** validate grids, probability sums, provenance, permissions, and official package structure before any upload.

The planned six-week adapter has 151,573 parameters and predicts correction logits around FuXi probabilities. The currently reported IMERG result comes from the earlier 111,909-parameter target-window spatial model, not yet from the full ERA5-trained six-week adapter.

## ERA5 status

The real global six-hourly WeatherBench2 ERA5 source has now been found and validated. A full 2,080-case cache build is running as Slurm job `98122`. Cache construction is independent of GFS.

A two-case ERA5 infrastructure smoke produced aligned-mask proxy RPS values of `0.824961` for uniform and `0.946996` for raw FuXi probabilities, or **−14.79% RPSS**. This is only an infrastructure check, not a performance estimate, and no ERA5-trained neural score exists yet.

The first ad hoc smoke calculation reported −5.33% because the ascending-latitude land mask was applied to descending-latitude targets. Correcting the mask orientation gives −14.79%. Cache generation was unaffected.

## Evidence limits

- The exploratory benchmark contains only eight 2017–2018 training cases, four 2019 calibration/selection cases, and four already-opened 2020 retrospective cases.
- It uses IMERG Final V07B, 75-sample boundaries, and global support weighting rather than ERA5T, 100-sample boundaries, and the official land/aridity mask.
- Most of the improvement relative to raw FuXi comes from calibration toward uniform probabilities.
- The current local absolute RPS is an area-weighted proxy; RPSS is the safer comparison because the common scaling cancels. Package-exact scoring must be verified before official reporting.
- Written FuXi competition permission, official-mask provenance, and a completed live forecast remain hard submission gates.

## Reproducibility

- Exploratory comparison: `runs/imerg_real_smoke_20260816_fast_ablation_results_complete/comparison.csv`
- Exploratory hashes/configuration: `runs/imerg_real_smoke_20260816_fast_ablation_results_complete/manifest.json`
- ERA5 cache builder: `build_precip_cache.py`
- Probability model: `precip_model.py`
- Training: `train_precip.py`
- Candidate selection: `select_precip_candidate.py`
- Frozen methodology: `PRECIP_QUEST_20260813.md`
