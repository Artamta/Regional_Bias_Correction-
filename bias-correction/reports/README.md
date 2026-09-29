# Reports

## Current publication-focused core

`core3_20260822T050125Z/` is the compact three-figure paper package. It:

- shows the exact three-member raw-identity architecture and split/evidence
  timeline, including the sealed 2025 boundary;
- reports paired E2 gains by lead while exposing the weaker heavy/extreme-rain
  evidence;
- shows why the post-hoc raw-mean projection fails to transfer across the IMD
  grid and gauge-derived 1.5° cells;
- labels the distinct bias estimands explicitly: E2 uses
  `|pooled signed bias|`, while E3 uses `mean |case bias|`;
- identifies raw identity versus raw FuXi as a secondary E3 comparison, not the
  E3 primary estimand;
- binds the raw-identity training manifest
  `09317899c7d8c1d21952a23586499f195cf47f6b24ee3ca733580a38dd8d5463`,
  canonical E2 manifest
  `bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`,
  and canonical E3 manifest
  `5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`;
- requires the E2 checkpoint paths and SHA-256 values to reproduce the exact
  three-seed mapping in the validated training manifest, and requires the E3
  primary tuple to be selected adapter versus raw FuXi RMSE;
- parses the exact verified bytes of seven metric CSVs and opens no checkpoint,
  raw array/store, raw station file, or 2025 data; and
- contains 12 hash-verified artifacts: three PNG/PDF/CSV figure families,
  captions, README, and the exact standalone builder snapshot. Its verifier
  rejects both file and directory symlinks anywhere in the package tree.

Its package-manifest SHA-256 is
`f367d158a6b9e1b595045134ebbdfe3af88a72985b9b62f5409d3917613fa506`;
the packaged builder SHA-256 is
`afa20561c9a6cd3c86b6699dd47eb982b6028b177aa62ca6c05868afd1a02d0b`.

## Expanded sealed paper evidence

`paper_evidence_20260822T030741Z/` is the expanded five-figure E2/E3 reporting
package. It is retained because it:

- binds the canonical E2 manifest
  `bc9fa96182906f736dc542000ed62f1ddd70460448f07e679943efcffceeeeec`;
- binds the canonical E3 manifest
  `5404867e63f0fd6d3b09799c32727f64c36a62489b5e1c27310b8ca33463d249`;
- reads each of the five source CSVs once, verifies those exact bytes, and
  parses the same in-memory bytes without reopening a path;
- records that this reporting builder opened no raw data, prediction store,
  target array, raw station file, or 2025 data;
- separately discloses that upstream E3 scanned a mixed station container with
  45,910 unselected 2025+ rows while selecting, materializing, and scoring no
  2025 station value;
- enforces one strictly Boolean E3 primary-estimand row and cross-checks it
  against the structured E3 manifest;
- uses atomic no-clobber publication and rejects every undeclared file at any
  depth, including a nested `PACKAGE_MANIFEST.json`;
- contains 23 hash-verified outputs, including PDF and PNG figures, CSV and
  Markdown tables, captions, and the exact reporting-code snapshot; and
- passed an independent `--verify-only` check after publication.

Its package-manifest SHA-256 is
`b959dcc4d8d85a0acde24b0c27c0ed8ceccadb7bdebbb524aa4b0591e18d1afa`.
Use `paper_evidence_20260822T030741Z/PACKAGE_MANIFEST.json` as the authoritative
file inventory.

The evidence labels are intentionally conservative: E2 is a 2022–2024
retrospective development audit, and E3 is a 2024 external-observational-target
sensitivity. Neither is described as an untouched final test.
