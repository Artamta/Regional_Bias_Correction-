"""Contract tests for the presentation-only clear-story packager."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import package_clear_story as package_story  # noqa: E402


def _write(path: Path, payload: bytes = b"x" * 10_001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture_sources(root: Path) -> tuple[Path, Path, Path]:
    spatial = root / "spatial"
    acc = root / "acc"
    jjas = root / "jjas"
    for directory in (spatial, acc, jjas):
        directory.mkdir(parents=True)

    spatial_artifacts = {}
    for stem in package_story.SPATIAL_FILES:
        for suffix in (".png", ".pdf"):
            path = spatial / f"{stem}{suffix}"
            _write(path)
            spatial_artifacts[path.name] = package_story.sha256_file(path)
    (spatial / "spatial_atlas_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_scope": (
                    "2020-2021 exploratory/reused locked hindcasts; "
                    "not independent confirmation"
                ),
                "selected_configuration": "physical_full_compact",
                "visual_interpolation_used_for_metrics": False,
                "cases": 70,
                "lead_weeks": list(range(1, 7)),
                "artifacts": spatial_artifacts,
            }
        ),
        encoding="utf-8",
    )

    acc_artifacts = {}
    for stem in package_story.ACC_FILES:
        for suffix in (".png", ".pdf"):
            path = acc / f"{stem}{suffix}"
            _write(path)
            acc_artifacts[path.name] = package_story.sha256_file(path)
    (acc / "manifest.json").write_text(
        json.dumps(
            {
                "evaluation_scope": (
                    "2020-2021 exploratory/reused locked hindcasts; "
                    "not independent confirmation"
                ),
                "genuine_independent_confirmation": False,
                "source_arrays_opened": False,
                "cases": 70,
                "paired_points": 420,
                "figures": acc_artifacts,
            }
        ),
        encoding="utf-8",
    )

    jjas_artifacts = {}
    for stem in package_story.JJAS_FILES:
        for suffix in (".png", ".pdf"):
            path = jjas / f"{stem}{suffix}"
            _write(path)
            jjas_artifacts[path.name] = package_story.sha256_file(path)
    summary = jjas / "jjas_initialization_month_by_lead_summary.csv"
    _write(summary)
    jjas_artifacts[summary.name] = package_story.sha256_file(summary)
    (jjas / "diagnostic_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_scope": (
                    "2020-2021 exploratory/reused locked hindcasts; "
                    "not independent confirmation"
                ),
                "evaluation_role": "exploratory_reused_hindcast_evaluation",
                "selected_configuration": "physical_full_compact",
                "test_years": [2020, 2021],
                "genuine_independent_test": False,
                "parameter_updates": 0,
                "uncertainty": {
                    "p_values_computed": False,
                    "significance_claimed": False,
                },
                "artifacts": jjas_artifacts,
            }
        ),
        encoding="utf-8",
    )
    return spatial, acc, jjas


def test_package_orders_assets_and_preserves_honest_scope(tmp_path: Path) -> None:
    spatial, acc, jjas = _fixture_sources(tmp_path / "sources")
    output = package_story.package(spatial, acc, jjas, tmp_path / "package")
    assert len(list(output.glob("0*.png"))) == 6
    assert len(list(output.glob("0*.pdf"))) == 6
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["independent_confirmation_claimed"] is False
    assert manifest["statistical_significance_claimed"] is False
    assert manifest["metrics_recomputed_by_packager"] is False
    assert manifest["training_years_inclusive"] == [2002, 2017]
    story = (output / "STORY_AND_SLIDE_NOTES.md").read_text(encoding="utf-8")
    assert "not independent confirmation" in story
    assert "absolute mean bias" in story
    assert "329/420" in story


def test_package_refuses_existing_destination(tmp_path: Path) -> None:
    spatial, acc, jjas = _fixture_sources(tmp_path / "sources")
    output = tmp_path / "package"
    output.mkdir()
    with pytest.raises(FileExistsError):
        package_story.package(spatial, acc, jjas, output)


def test_package_refuses_interpolation_in_metrics(tmp_path: Path) -> None:
    spatial, acc, jjas = _fixture_sources(tmp_path / "sources")
    manifest_path = spatial / "spatial_atlas_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_interpolation_used_for_metrics"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="interpolation"):
        package_story.package(spatial, acc, jjas, tmp_path / "package")
