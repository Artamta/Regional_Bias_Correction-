#!/usr/bin/env python
"""Package the verified lead-adaptive model for simple inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


TOOLS_ROOT = Path(__file__).resolve().parent
HERE = TOOLS_ROOT.parent
RESULT = (
    HERE
    / "results/fuxi_imerg_spatiotemporal/full_scratch_20260810T003927Z"
)
PARENT = HERE / "results/fuxi_imerg_jjas_5yr/full_20260809T233638Z"
SPATIAL_BUNDLE = (
    HERE
    / "trained_model/fuxi_imerg_2014_2018_v1/residual_unet_ensemble.pt"
)
OUTPUT = HERE / "trained_model/fuxi_imerg_spatiotemporal_2014_2018_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    (OUTPUT / "code/fuxi_adapter").mkdir(parents=True)
    source = torch.load(SPATIAL_BUNDLE, map_location="cpu", weights_only=True)
    if source["format_version"] != "fuxi-imerg-residual-unet-v1":
        raise ValueError("unexpected spatial bundle")
    spatial_by_seed = {int(item["seed"]): item for item in source["models"]}
    models = []
    for seed in (42, 43, 44):
        temporal_path = RESULT / f"models/seed_{seed}/checkpoints/best.pt"
        temporal = torch.load(temporal_path, map_location="cpu", weights_only=False)
        models.append(
            {
                "seed": seed,
                "spatial_state_dict": spatial_by_seed[seed]["model_state_dict"],
                "temporal_state_dict": temporal["model_state_dict"],
                "temporal_best_epoch": int(temporal["best_epoch"]),
                "temporal_best_validation_loss": float(
                    temporal["best_validation_loss"]
                ),
            }
        )
    bundle = {
        "format_version": "fuxi-imerg-lead-adaptive-v1",
        "latitude": source["latitude"],
        "longitude": source["longitude"],
        "support": source["support"],
        "daily_imerg_climatology": source["daily_imerg_climatology"],
        "normalization": source["normalization"],
        "models": models,
        "lead_sources": [
            "spatial",
            "spatial",
            "spatial",
            "spatial",
            "temporal",
            "temporal",
        ],
        "train_years": [2014, 2015, 2016, 2017, 2018],
        "validation_year": 2019,
        "exploratory_test_years": [2020, 2021],
        "result_manifest_sha256": sha256_file(RESULT / "manifest.json"),
        "spatial_parent_manifest_sha256": sha256_file(PARENT / "manifest.json"),
    }
    bundle_path = OUTPUT / "lead_adaptive_ensemble.pt"
    torch.save(bundle, bundle_path)

    for source_path in (
        TOOLS_ROOT / "trained_spatiotemporal_model.py",
        TOOLS_ROOT / "trained_model.py",
    ):
        shutil.copy2(source_path, OUTPUT / source_path.name)
    for source_path in (
        PARENT / "code/fuxi_adapter/__init__.py",
        PARENT / "code/fuxi_adapter/models.py",
    ):
        shutil.copy2(source_path, OUTPUT / "code/fuxi_adapter" / source_path.name)

    readme = """# Lead-adaptive FuXi–IMERG model

This package contains the three-seed spatial and spatiotemporal U-Nets.
2019 validation selects the spatial model for W1–W4 and temporal attention for
W5–W6. The 2020–2021 result is exploratory because those years were previously
examined.

```python
from trained_spatiotemporal_model import predict_fuxi_shards

weekly_tp = predict_fuxi_shards(["20200602.nc"])
```

Input shards must use the audited 51-member, 42-day, 27×27 FuXi TP contract.
The output is weekly mean rainfall in mm day-1.
"""
    (OUTPUT / "README.md").write_text(readme)
    files = {}
    for path in sorted(item for item in OUTPUT.rglob("*") if item.is_file()):
        files[str(path.relative_to(OUTPUT))] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "format_version": "fuxi-imerg-lead-adaptive-v1",
        "source_result": str(RESULT),
        "source_result_verified": True,
        "files": files,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
