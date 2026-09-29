"""Synthetic contract tests for the sealed E2/E3 evidence packager."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import build_paper_evidence_package as reporting


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _artifact_hashes(root: Path, relatives: tuple[str, ...]) -> dict[str, str]:
    return {relative: _sha256(root / relative) for relative in relatives}


def _refresh_declared_artifact(root: Path, relative: str) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative] = _sha256(root / relative)
    _write_json(manifest_path, manifest)
    return _sha256(manifest_path)


def _effect(
    metrics: dict[str, dict[str, float]],
    candidate: str,
    reference: str,
    metric: str,
) -> float:
    columns = {
        "rmse": "rmse",
        "mae": "mae",
        "bias": "bias",
        "absolute_bias": "absolute_bias",
        "acc": "acc",
    }
    column = columns[metric]
    if metric in ("rmse", "mae", "absolute_bias"):
        return metrics[reference][column] - metrics[candidate][column]
    return metrics[candidate][column] - metrics[reference][column]


@pytest.fixture
def sealed_sources(tmp_path: Path) -> tuple[Path, Path, str, str]:
    e2 = tmp_path / "e2"
    e2_metrics = {
        "raw_fuxi": {"rmse": 6.0, "mae": 4.0, "bias": -0.20, "acc": 0.30},
        "log_bias": {"rmse": 5.8, "mae": 3.8, "bias": -0.50, "acc": 0.32},
        "legacy_anchored_adapter": {
            "rmse": 5.6,
            "mae": 3.6,
            "bias": -0.60,
            "acc": 0.34,
        },
        "raw_identity": {"rmse": 5.5, "mae": 3.5, "bias": -0.80, "acc": 0.36},
        "raw_identity_raw_mean_preserved": {
            "rmse": 5.45,
            "mae": 3.65,
            "bias": -0.18,
            "acc": 0.35,
        },
    }
    pooled_rows = []
    lead_rows = []
    for method, values in e2_metrics.items():
        pooled_rows.append(
            {
                "method": method,
                "method_label": reporting.METHOD_LABELS[method],
                "rmse_mm_day": values["rmse"],
                "mae_mm_day": values["mae"],
                "bias_mm_day": values["bias"],
                "acc": values["acc"],
                "case_lead_count": 600,
            }
        )
        for lead in range(1, 7):
            lead_rows.append(
                {
                    "lead_week": lead,
                    "method": method,
                    "method_label": reporting.METHOD_LABELS[method],
                    "rmse_mm_day": values["rmse"] + 0.05 * lead,
                    "mae_mm_day": values["mae"] + 0.03 * lead,
                    "bias_mm_day": values["bias"] - 0.01 * lead,
                    "acc": values["acc"] - 0.02 * lead,
                    "case_lead_count": 100,
                }
            )
    e2_effect_rows = []
    for candidate, baseline in reporting.E2_FOREST_COMPARISONS:
        for source_metric, normalized in (
            ("rmse_mm_day", "rmse"),
            ("mae_mm_day", "mae"),
            ("bias_mm_day", "absolute_bias"),
            ("acc", "acc"),
        ):
            left = e2_metrics[candidate]
            right = e2_metrics[baseline]
            if normalized == "acc":
                value = left["acc"] - right["acc"]
                prefix = "candidate minus baseline ACC"
            elif normalized == "absolute_bias":
                value = abs(right["bias"]) - abs(left["bias"])
                prefix = "absolute pooled bias baseline minus candidate"
            else:
                value = right[normalized] - left[normalized]
                prefix = f"baseline minus candidate {source_metric}"
            e2_effect_rows.append(
                {
                    "scope_type": "pooled",
                    "scope": "W1-W6",
                    "region": "all_india",
                    "candidate": candidate,
                    "baseline": baseline,
                    "source_metric": source_metric,
                    "effect": value,
                    "ci_lower_2p5": value - 0.1,
                    "ci_upper_97p5": value + 0.1,
                    "bootstrap_probability_improved": 0.9 if value > 0 else 0.1,
                    "n_starts": 100,
                    "n_leads_per_start": 6,
                    "definition": (
                        f"{prefix}; paired year-stratified circular moving-block "
                        "bootstrap with actual block length 13 initializations"
                    ),
                }
            )
    _write_csv(e2 / reporting.E2_REQUIRED_ARTIFACTS[0], pooled_rows)
    _write_csv(e2 / reporting.E2_REQUIRED_ARTIFACTS[1], lead_rows)
    _write_csv(e2 / reporting.E2_REQUIRED_ARTIFACTS[2], e2_effect_rows)
    e2_manifest = {
        "status": "complete",
        "canonical": True,
        "scientific_eligible": True,
        "smoke": False,
        "scientific_status": (
            "post-hoc matched 2022-2024 canonical E2 development audit; "
            "no untouched-final-test claim"
        ),
        "audit_years": [2022, 2023, 2024],
        "final_initialization_year_quarantined": 2025,
        "final_2025_store_opened": False,
        "methods": list(reporting.E2_METHODS),
        "bootstrap": {
            "draws": 10_000,
            "block_length_initializations": 13,
            "seed": 20260822,
            "all_six_leads_retained": True,
        },
        "artifacts": _artifact_hashes(e2, reporting.E2_REQUIRED_ARTIFACTS),
    }
    _write_json(e2 / "manifest.json", e2_manifest)
    e2_hash = _sha256(e2 / "manifest.json")

    e3 = tmp_path / "e3"
    e3_metrics = {
        "raw_fuxi": {
            "rmse": 8.0,
            "mae": 5.3,
            "bias": 2.4,
            "absolute_bias": 2.5,
            "acc": 0.40,
        },
        "log_bias": {
            "rmse": 7.6,
            "mae": 5.0,
            "bias": 1.9,
            "absolute_bias": 2.1,
            "acc": 0.43,
        },
        "selected_adapter": {
            "rmse": 7.7,
            "mae": 5.05,
            "bias": 1.85,
            "absolute_bias": 2.05,
            "acc": 0.425,
        },
        "raw_identity": {
            "rmse": 7.5,
            "mae": 4.9,
            "bias": 1.8,
            "absolute_bias": 2.0,
            "acc": 0.435,
        },
        "raw_identity_raw_mean_preserved": {
            "rmse": 7.9,
            "mae": 5.25,
            "bias": 2.45,
            "absolute_bias": 2.55,
            "acc": 0.405,
        },
    }
    e3_summary_rows = []
    for method, values in e3_metrics.items():
        common = {
            "method": method,
            "initializations": 30,
            "mean_common_grid_cells": 96.0,
            "mean_station_locations": 319.0,
            "rmse_mean": values["rmse"],
            "mae_mean": values["mae"],
            "bias_mean": values["bias"],
            "absolute_bias_mean": values["absolute_bias"],
            "acc_mean": values["acc"],
        }
        e3_summary_rows.append(
            {**common, "scope_type": "pooled", "scope": "W1-W6", "case_leads": 180}
        )
        for lead in range(1, 7):
            e3_summary_rows.append(
                {**common, "scope_type": "lead", "scope": f"W{lead}", "case_leads": 30}
            )
    e3_effect_rows = []
    for candidate, reference in reporting.E3_COMPARISONS:
        comparison = f"{candidate}_vs_{reference}"
        for metric in ("rmse", "mae", "acc", "bias", "absolute_bias"):
            value = _effect(e3_metrics, candidate, reference, metric)
            direction = (
                "reference_minus_candidate"
                if metric in ("rmse", "mae", "absolute_bias")
                else "candidate_minus_reference"
            )
            for block in (4, 8, 13):
                e3_effect_rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "effect_definition": direction,
                        "positive_favors_candidate": metric != "bias",
                        "block_length_initializations": block,
                        "analysis_role": (
                            "primary_uncertainty"
                            if block == 13
                            else "predeclared_sensitivity"
                        ),
                        "bootstrap_draws": 2000,
                        "initializations": 30,
                        "case_leads": 180,
                        "point_effect": value,
                        "bootstrap_mean": value,
                        "ci_lower_2p5": value - 0.1,
                        "ci_upper_97p5": value + 0.1,
                        "probability_effect_gt_zero": 0.9 if value > 0 else 0.1,
                        "primary_estimand": (
                            comparison == "selected_adapter_vs_raw_fuxi"
                            and metric == "rmse"
                            and block == 13
                        ),
                    }
                )
    _write_csv(e3 / reporting.E3_REQUIRED_ARTIFACTS[0], e3_summary_rows)
    _write_csv(e3 / reporting.E3_REQUIRED_ARTIFACTS[1], e3_effect_rows)
    e3_manifest = {
        "status": "complete_frozen_external_target_sensitivity",
        "canonical_artifact": True,
        "training_performed": False,
        "selection_calibration_or_blending_performed": False,
        "2025_metric_computed": False,
        "2025_prediction_opened": False,
        "2025_station_value_selected": False,
        "methods": list(reporting.E3_METHODS),
        "scientific_status": (
            "frozen independent-observational-target sensitivity; "
            "not untouched temporal final"
        ),
        "dates": {
            "initialization_years": [2024],
            "lead_weeks": [1, 2, 3, 4, 5, 6],
            "initialization_count": 30,
            "case_leads": 180,
            "verification_date_min": "2024-06-03",
            "verification_date_max": "2024-11-10",
        },
        "bootstrap": {
            "draws": 2000,
            "primary_block_length": 13,
            "sensitivity_block_lengths": [4, 8],
            "seed": 20260822,
        },
        "primary_estimand": {
            "comparison": "selected_adapter_vs_raw_fuxi",
            "metric": "rmse",
            "definition": "reference_minus_candidate",
            "point_effect": _effect(e3_metrics, "selected_adapter", "raw_fuxi", "rmse"),
            "ci_lower_2p5": _effect(e3_metrics, "selected_adapter", "raw_fuxi", "rmse")
            - 0.1,
            "ci_upper_97p5": _effect(e3_metrics, "selected_adapter", "raw_fuxi", "rmse")
            + 0.1,
            "probability_effect_gt_zero": 0.9,
            "circular_block_length_initializations": 13,
            "bootstrap_draws": 2000,
            "all_six_leads_attached": True,
        },
        "station_truth_boundary": {
            "container_date_min": "2023-12-31",
            "container_date_max": "2025-02-10",
            "container_rows_scanned": 543_518,
            "unselected_2025_plus_rows": 45_910,
            "rainfall_converted_only_after_exact_2024_date_and_station_filter": True,
            "selected_snapshot": "inputs/station_truth_selected_2024.csv.gz",
        },
        "extended_prediction_manifest_sha256": e2_hash,
        "artifacts": _artifact_hashes(e3, reporting.E3_REQUIRED_ARTIFACTS),
    }
    _write_json(e3 / "manifest.json", e3_manifest)
    return e2, e3, e2_hash, _sha256(e3 / "manifest.json")


def test_builds_complete_hash_verified_package_from_sealed_metric_artifacts(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
) -> None:
    e2, e3, e2_hash, e3_hash = sealed_sources
    source_hashes = {
        path: _sha256(path)
        for root in (e2, e3)
        for path in root.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "paper_evidence"

    result = reporting.build_package(
        e2,
        e3,
        output,
        e2_manifest_sha256=e2_hash,
        e3_manifest_sha256=e3_hash,
    )

    assert result == output.resolve()
    for index, stem in enumerate(
        (
            "imd_pooled_tradeoff",
            "imd_by_lead",
            "imd_paired_effects",
            "station_pooled",
            "station_block13_effects",
        ),
        start=1,
    ):
        family = f"figure_{index:02d}_{stem}"
        png = output / f"{family}.png"
        pdf = output / f"{family}.pdf"
        with Image.open(png) as image:
            image.verify()
        assert pdf.read_bytes().startswith(b"%PDF")
    for index, stem in enumerate(
        (
            "imd_pooled",
            "imd_by_lead",
            "imd_paired_effects",
            "station_pooled",
            "station_block13_effects",
        ),
        start=1,
    ):
        assert (output / f"table_{index:02d}_{stem}.csv").is_file()
        assert (output / f"table_{index:02d}_{stem}.md").is_file()

    manifest = reporting.verify_package(output)
    assert reporting.main(["--verify-only", "--output", str(output)]) == 0
    assert len(manifest["artifacts"]) == 23
    assert manifest["builder"]["source"] == "code/build_paper_evidence_package.py"
    assert (
        manifest["builder"]["sha256"]
        == manifest["artifacts"]["code/build_paper_evidence_package.py"]
    )
    assert manifest["access_boundary"]["metric_artifact_only"] is True
    assert (
        manifest["access_boundary"]["scope"] == "this reporting-builder invocation only"
    )
    assert manifest["access_boundary"]["builder_opened"]["2025_data"] is False
    upstream = manifest["access_boundary"]["upstream_e3_disclosure"]
    assert upstream["unselected_2025_plus_rows"] == 45_910
    assert upstream["2025_station_values_selected"] == 0
    assert upstream["2025_station_values_materialized"] == 0
    assert upstream["2025_station_values_scored"] == 0
    assert manifest["publication"]["no_clobber"] is True
    assert manifest["sources"]["e2"]["manifest_sha256"] == e2_hash
    assert manifest["sources"]["e3"]["manifest_sha256"] == e3_hash
    assert source_hashes == {path: _sha256(path) for path in source_hashes}
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "retrospective development audit" in readme
    assert "external-target sensitivity" in readme
    assert "not an untouched final test" in readme
    assert "45,910" in readme
    captions = (output / "FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
    assert "baseline minus candidate for RMSE, MAE, and absolute" in captions
    assert "candidate minus baseline for ACC" in captions

    nested = output / "nested" / "PACKAGE_MANIFEST.json"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*nested/PACKAGE_MANIFEST.json"):
        reporting.verify_package(output)


def test_tampered_metric_is_rejected_before_any_dataframe_load(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e2, e3, e2_hash, e3_hash = sealed_sources
    pooled = e2 / reporting.E2_REQUIRED_ARTIFACTS[0]
    pooled.write_bytes(pooled.read_bytes() + b"\n")
    loads: list[object] = []

    def forbidden_read(*args: object, **kwargs: object) -> pd.DataFrame:
        loads.append((args, kwargs))
        raise AssertionError("read_csv must not run before all checksums pass")

    monkeypatch.setattr(reporting.pd, "read_csv", forbidden_read)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        reporting.build_package(
            e2,
            e3,
            output,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )
    assert loads == []
    assert not output.exists()


def test_2025_access_flag_is_rejected_even_with_a_fresh_manifest_pin(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
) -> None:
    e2, e3, e2_hash, _ = sealed_sources
    manifest_path = e3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["2025_prediction_opened"] = True
    _write_json(manifest_path, manifest)
    output = tmp_path / "rejected_2025"

    with pytest.raises(ValueError, match="2025_prediction_opened"):
        reporting.build_package(
            e2,
            e3,
            output,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=_sha256(manifest_path),
        )
    assert not output.exists()


def test_checksum_bound_bytes_survive_a_deterministic_path_mutation_race(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e2, e3, e2_hash, e3_hash = sealed_sources
    pooled_path = e2 / reporting.E2_REQUIRED_ARTIFACTS[0]
    original_read_csv = reporting.pd.read_csv
    calls = 0

    def racing_read_csv(
        source: object, *args: object, **kwargs: object
    ) -> pd.DataFrame:
        nonlocal calls
        assert isinstance(source, io.BytesIO)
        if calls == 0:
            changed = original_read_csv(io.BytesIO(pooled_path.read_bytes()))
            changed.loc[changed["method"].eq("raw_fuxi"), "rmse_mm_day"] = 999.0
            changed.to_csv(pooled_path, index=False)
        calls += 1
        return original_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(reporting.pd, "read_csv", racing_read_csv)
    output = reporting.build_package(
        e2,
        e3,
        tmp_path / "race_safe",
        e2_manifest_sha256=e2_hash,
        e3_manifest_sha256=e3_hash,
    )

    assert calls == 5
    table = original_read_csv(output / "table_01_imd_pooled.csv")
    raw_rmse = table.loc[table["method"].eq("Raw FuXi"), "rmse_mm_day"].item()
    assert raw_rmse == pytest.approx(6.0)
    assert "999" in pooled_path.read_text(encoding="utf-8")
    reporting.verify_package(output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("second_true", "exactly one primary_estimand=True"),
        ("non_boolean", "strict booleans"),
        ("manifest_mismatch", "point_effect differs"),
    ),
)
def test_e3_primary_estimand_contract_rejects_ambiguous_or_unbound_flags(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
    mutation: str,
    message: str,
) -> None:
    e2, e3, e2_hash, e3_hash = sealed_sources
    effects_relative = reporting.E3_REQUIRED_ARTIFACTS[1]
    effects_path = e3 / effects_relative
    if mutation in {"second_true", "non_boolean"}:
        effects = pd.read_csv(effects_path)
        if mutation == "second_true":
            effects.loc[1, "primary_estimand"] = True
        else:
            effects["primary_estimand"] = effects["primary_estimand"].map(
                {True: "yes", False: "no"}
            )
        effects.to_csv(effects_path, index=False)
        e3_hash = _refresh_declared_artifact(e3, effects_relative)
    else:
        manifest_path = e3 / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["primary_estimand"]["point_effect"] += 1.0
        _write_json(manifest_path, manifest)
        e3_hash = _sha256(manifest_path)

    with pytest.raises(ValueError, match=message):
        reporting.build_package(
            e2,
            e3,
            tmp_path / f"rejected_{mutation}",
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_raced_empty_destination_is_not_replaced(
    tmp_path: Path,
    sealed_sources: tuple[Path, Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e2, e3, e2_hash, e3_hash = sealed_sources
    output = tmp_path / "raced_destination"
    original_publish = reporting._publish_directory_noreplace

    def race(staging: Path, destination: Path) -> None:
        destination.mkdir()
        original_publish(staging, destination)

    monkeypatch.setattr(reporting, "_publish_directory_noreplace", race)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        reporting.build_package(
            e2,
            e3,
            output,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".raced_destination.staging-*")) == []
