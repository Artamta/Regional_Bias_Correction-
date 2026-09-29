"""Tests for non-destructive, evidence-labeled presentation packaging."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import pytest


PRESENTATION = Path(__file__).resolve().parents[1] / "presentation"
if str(PRESENTATION) not in sys.path:
    sys.path.insert(0, str(PRESENTATION))

import package_presentation_assets as packaging  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_family(directory: Path, stem: str, suffixes: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(2.4, 1.35))
    axis.plot([0, 1], [0, 1], color="#198C7A")
    axis.set_title(stem)
    for suffix in suffixes:
        figure.savefig(directory / f"{stem}{suffix}")
    plt.close(figure)


def _artifact_mapping(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "figures").iterdir())
        if path.is_file()
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def packaging_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    method = tmp_path / "method"
    _render_family(
        method,
        packaging.ARCHITECTURE_STEM,
        (".png", ".pdf", ".svg"),
    )
    _render_family(
        method,
        packaging.TRAINING_STEM,
        (".png", ".pdf"),
    )

    validation = tmp_path / "validation"
    _render_family(
        validation / "figures",
        "03_india_weighted_metrics_by_lead",
        (".png", ".pdf"),
    )
    _write_json(
        validation / "postprocessing_manifest.json",
        {
            "status": "complete",
            "created_utc": "2026-08-12T00:00:00+00:00",
            "source_training_mode": "full",
            "validation_years": [2018, 2019],
            "evaluation_scope": "blocked validation only; no 2020+ prediction or metric",
            "selected_best_physical_candidate": "physical_full_compact",
            "artifacts": _artifact_mapping(validation),
        },
    )

    exploratory = tmp_path / "exploratory"
    _render_family(
        exploratory / "figures",
        "01_exploratory_skill_by_lead",
        (".png", ".pdf"),
    )
    _write_json(
        exploratory / "manifest.json",
        {
            "status": "complete",
            "test_years": [2020, 2021],
            "evaluation_role": "exploratory_reused_hindcast_evaluation",
            "evaluation_scope": (
                "2020-2021 exploratory/reused hindcast test; "
                "not independent confirmation"
            ),
            "selected_configuration": "physical_full_compact",
            "selection_locked_before_test": True,
            "selection_locked_before_target_access": True,
            "test_used_for_selection": False,
            "parameter_updates": 0,
            "reused_test_period": True,
            "genuine_independent_test": False,
            "artifacts": _artifact_mapping(exploratory),
        },
    )
    return method, validation, exploratory


def test_validation_only_package_has_honest_placeholders_and_preserves_sources(
    tmp_path: Path,
    packaging_sources: tuple[Path, Path, Path],
) -> None:
    method, validation, _ = packaging_sources
    source_hashes = {
        path: _sha256(path)
        for root in (method, validation)
        for path in root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "package"

    result = packaging.package_assets(
        validation,
        output,
        method_assets_directory=method,
    )

    assert result == output.resolve()
    assert (output / "CONTACT_SHEET.png").is_file()
    assert (output / "CONTACT_SHEET.pdf").is_file()
    assert (output / "SLIDE_INDEX.md").is_file()
    index = pd.read_csv(output / "SLIDE_INDEX.csv")
    assert len(index) == 5
    assert set(index.status) == {
        "available",
        "pending — no locked output supplied",
        "unavailable",
    }
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "not an independent test" in readme
    assert "No 2025 metric or result may be claimed" in readme
    assert "did **not** read target" in readme
    manifest = json.loads(
        (output / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["arrays_read"] is False
    assert manifest["metrics_computed"] is False
    assert manifest["exploratory_reused_test"]["included"] is False
    assert manifest["genuine_independent_2025_evaluation"]["available"] is False
    assert source_hashes == {path: _sha256(path) for path in source_hashes}


def test_locked_exploratory_assets_require_structured_reuse_contract(
    tmp_path: Path,
    packaging_sources: tuple[Path, Path, Path],
) -> None:
    method, validation, exploratory = packaging_sources
    _render_family(
        method,
        packaging.ARCHITECTURE_STEM_V2,
        (".png", ".pdf", ".svg"),
    )
    _render_family(
        method,
        packaging.TRAINING_STEM_V2,
        (".png", ".pdf"),
    )
    output = tmp_path / "package_with_test"

    packaging.package_assets(
        validation,
        output,
        exploratory_test_directory=exploratory,
        method_assets_directory=method,
    )

    index = pd.read_csv(output / "SLIDE_INDEX.csv")
    exploratory_rows = index.loc[index.section.eq("04_exploratory_reused_test")]
    assert len(exploratory_rows) == 1
    assert exploratory_rows.iloc[0].status == "available"
    assert "not independent confirmation" in exploratory_rows.iloc[0].claim_scope
    assert not index.status.str.contains("pending").any()
    assert index.iloc[-1].evidence_years == "2025"
    assert index.iloc[0].source_stem == packaging.ARCHITECTURE_STEM_V2
    assert index.iloc[1].source_stem == packaging.TRAINING_STEM_V2

    bad_manifest = json.loads(
        (exploratory / "manifest.json").read_text(encoding="utf-8")
    )
    bad_manifest.pop("genuine_independent_test")
    _write_json(exploratory / "manifest.json", bad_manifest)
    with pytest.raises(ValueError, match="genuine_independent_test"):
        packaging.package_assets(
            validation,
            tmp_path / "rejected",
            exploratory_test_directory=exploratory,
            method_assets_directory=method,
        )
    assert not (tmp_path / "rejected").exists()


def test_packager_refuses_to_write_inside_source_results(
    packaging_sources: tuple[Path, Path, Path],
) -> None:
    method, validation, _ = packaging_sources
    with pytest.raises(ValueError, match="inside"):
        packaging.package_assets(
            validation,
            validation / "presentation_package",
            method_assets_directory=method,
        )
