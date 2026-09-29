# Regional Bias Correction Research Archive

Research code for subseasonal-to-seasonal (S2S) rainfall calibration. This
repository consolidates two earlier workstreams while keeping their scientific
contracts, tests, and documentation separate.

## Projects

| Project | Scope | Start here |
|---|---|---|
| [`bias-correction/`](bias-correction/) | Six-week FuXi-to-IMD rainfall post-processing over India, including deterministic and ensemble calibration experiments | [`bias-correction/README.md`](bias-correction/README.md) |
| [`ai-quest-global/`](ai-quest-global/) | Global five-category probability adapter and the frozen 2026-08-13 precipitation workflow | [`ai-quest-global/README.md`](ai-quest-global/README.md) |

The projects share a weather-calibration theme, but they use different targets,
grids, evidence boundaries, and operational contracts. Results from one project
must not be presented as evidence for the other.

## Repository status

This is a source-and-provenance archive of previous research work. It includes
code, tests, Slurm launchers, analysis plans, scientific notes, and lightweight
manifests. Raw forecasts, observations, caches, model checkpoints, virtual
environments, logs, and generated result packages are intentionally excluded.
See [`docs/MIGRATION_NOTES.md`](docs/MIGRATION_NOTES.md) for the migration
boundary.

Several workflows require data under site-specific storage paths, a Slurm GPU
cluster, and external packages. Read each project's README and scientific
contract before attempting a full run.

## Quick start

Clone the repository and choose one project:

```bash
git clone https://github.com/Artamta/Regional_Bias_Correction-.git
cd Regional_Bias_Correction-
```

For the India workflow:

```bash
cd bias-correction
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=src:evaluate:hpc_compat \
  python -m pytest -q -p no:cacheprovider \
  tests/test_project_layout.py \
  tests/test_ensemble_calibration_core.py \
  tests/test_fuxi_pbc_core.py \
  tests/test_india_s2s_benchmark_scoring.py \
  tests/test_india_s2s_benchmark_uncertainty.py \
  tests/test_india_s2s_block_bootstrap.py
```

The command above runs the source-only core checks. The complete historical
India suite also resolves a sibling `neural_adapter` package, benchmark-study
fixtures, and `s2s_verification` code that are not part of this snapshot. Their
original locations and roles are described in
[`bias-correction/docs/REPOSITORY_MAP.md`](bias-correction/docs/REPOSITORY_MAP.md).

For the global probability workflow:

```bash
cd ai-quest-global
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider
```

The global requirements include the competition-specific `AI-WQ-package`.
Submission code remains permission- and credential-gated; do not treat a local
smoke run as authorization to upload a forecast.

## Reproducibility and data policy

- Keep training, validation, retrospective, and sealed evaluation periods
  distinct as documented in each project.
- Never commit credentials, `.env` files, raw forecast/observation archives,
  Zarr or NetCDF datasets, checkpoints, or generated run directories.
- Treat completed run directories and manifests as immutable provenance units.
- Run synthetic or preflight checks before expensive Slurm jobs.
- Report negative and withdrawn experiments with the same care as selected
  results.

Contribution and review expectations are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
