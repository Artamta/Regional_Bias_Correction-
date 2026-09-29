# Migration notes

## Snapshot boundary

This repository was assembled on 30 September 2026 from two existing research
workspaces and records them as previous work:

- `bias-correction/`: the India FuXi-to-IMD post-processing workstream;
- `ai-quest-global/`: the global probability-adapter and precipitation-quest
  workstream.

The migration copied source code, tests, dependency declarations, Slurm
launchers, plans, scientific documentation, lightweight result indexes, and
small run records. The original scientific wording and evidence limitations
were retained.

## Deliberate exclusions

The following local-only or generated content was not migrated:

- raw forecasts and observations;
- Zarr, NetCDF, GRIB, and other bulk data products;
- feature caches and partial cache shards;
- virtual environments and Python/tool caches;
- model checkpoints and trained-model directories;
- Slurm logs and scheduler output;
- generated figures, handoff bundles, presentations, and result payloads;
- nested Git metadata and workspace-specific agent configuration.

These exclusions reduce the snapshot from tens of gigabytes to a source-sized
repository and prevent accidental publication of data or credentials. Some
documentation refers to excluded immutable artifacts by their original paths
and hashes; those references are provenance records, not claims that the
artifacts are stored in this Git repository.

## External dependencies

The complete India test and experiment environment expects a sibling
`neural_adapter` source tree, benchmark-study fixtures, `s2s_verification`
code, and site-specific forecast/observation storage. The global project may
require the competition-specific `AI-WQ-package`, official inputs, registered
credentials, and explicit upload permission. These dependencies are not
vendored here.

## Scientific separation

The two projects must remain separate in evaluation and reporting. They use
different targets, spatial domains, time periods, category definitions, and
evidence contracts. Migration into one repository does not make their metrics
directly comparable.
