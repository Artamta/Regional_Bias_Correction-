# `quest_plots`

Reusable, presentation-ready figures for the global AI Weather Quest precipitation
adapter. The default command consumes the completed FuXi–IMERG smoke artifacts and
builds architecture, workflow, training, prediction, score, and spatial-improvement
figures as both PNG and PDF. It also writes one complete Q1–Q5 baseline/model/ground-
truth comparison for each lead window.

The generated scorecard adds multicategory Brier and log scores. It uses RPS as the
ordered-category CRPS analogue and does not fabricate continuous CRPS from bin midpoints.

`AI_QUEST_GROUP_REPORT.pdf` is a one-page landscape summary designed for direct sharing.

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python -m quest_plots
```

Use `python -m quest_plots --help` to override the cache, comparison CSV, seed
histories/checkpoints, example case, formats, or output directory. The report builder
refuses missing sources and records SHA-256 hashes in its output manifest.

The default evidence is exploratory IMERG, not official ERA5 validation. That boundary
is printed directly on every result-bearing figure.

The generated `SHORT_REPORT.md` gives a one-page explanation of the objective, result,
spatial plot, evidence boundary, and next steps.
