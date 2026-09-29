"""Synthetic contracts for the standalone core-three paper packager."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import build_core3_paper_package as core3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _artifact_hashes(root: Path, relatives: tuple[str, ...]) -> dict[str, str]:
    return {relative: _sha256(root / relative) for relative in relatives}


def _refresh_artifact(root: Path, relative: str) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative] = _sha256(root / relative)
    _write_json(manifest_path, manifest)
    return _sha256(manifest_path)


@pytest.fixture
def frozen_sources(tmp_path: Path) -> tuple[Path, Path, Path, str, str, str]:
    training = tmp_path / "training"
    checkpoint_hashes = {
        42: "a" * 64,
        43: "b" * 64,
        44: "c" * 64,
    }
    runs = []
    training_artifacts = {}
    for offset, seed in enumerate(core3.SEEDS):
        checkpoint = f"models/normal_climo_model/seed_{seed}/checkpoints/best.pt"
        training_artifacts[checkpoint] = checkpoint_hashes[seed]
        runs.append(
            {
                "seed": seed,
                "best_epoch": 14 + offset * 3,
                "best_validation_loss": 0.475 - offset * 0.002,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hashes[seed],
            }
        )
    training_manifest = {
        "status": "complete",
        "smoke": False,
        "observation_source": "IMD",
        "selected_model": core3.SELECTED_MODEL,
        "selected_alpha": 1.0,
        "training_anchor": "raw_fuxi",
        "uses_fitted_log_bias_in_neural_training": False,
        "log_bias_role": "reporting_only",
        "active_leads": [1, 2, 3, 4, 5, 6],
        "quarantined_final_initialization_years": [2025],
        "split_years": {
            "train": list(range(2002, 2018)),
            "validation": [2018, 2019],
            "test": [2020, 2021],
        },
        "split_counts": {"train": 560, "validation": 70, "test": 70},
        "features": list(core3.CONSUMED_CHANNELS),
        "code_sha256": {"models.py": core3.FROZEN_MODEL_CODE_SHA256},
        "training": {
            core3.SELECTED_MODEL: {
                "architecture": core3.SELECTED_ARCHITECTURE,
                "parameter_count": core3.PARAMETER_COUNT,
                "seeds": list(core3.SEEDS),
                "train_case_count": 560,
                "validation_case_count": 70,
                "inactive_lead_count": 0,
                "dropout": 0.3,
                "loss_coefficients": {"smooth_l1": 0.75, "acc": 0.2, "bias": 0.05},
                "runs": runs,
            }
        },
        "artifacts": training_artifacts,
    }
    _write_json(training / "manifest.json", training_manifest)
    training_hash = _sha256(training / "manifest.json")

    e2 = tmp_path / "e2"
    pooled_scores = {
        "raw_fuxi": (6.0, 4.0, -0.2, 0.30),
        "log_bias": (5.8, 3.8, -0.5, 0.32),
        "legacy_anchored_adapter": (5.6, 3.6, -0.6, 0.34),
        "raw_identity": (5.5, 3.5, -0.8, 0.36),
        "raw_identity_raw_mean_preserved": (5.49, 3.6, -0.2, 0.355),
    }
    pooled_rows = [
        {
            "method": method,
            "rmse_mm_day": values[0],
            "mae_mm_day": values[1],
            "bias_mm_day": values[2],
            "acc": values[3],
            "case_lead_count": 600,
        }
        for method, values in pooled_scores.items()
    ]
    by_lead_rows = []
    lead_effect_rows = []
    for method, values in pooled_scores.items():
        for lead in range(1, 7):
            rmse = values[0] + 0.04 * lead
            acc = values[3] - 0.01 * lead
            if method == "raw_identity":
                rmse = pooled_scores["raw_fuxi"][0] + 0.04 * lead - (0.75 - 0.08 * lead)
                acc = pooled_scores["raw_fuxi"][3] - 0.01 * lead + 0.08
            by_lead_rows.append(
                {
                    "lead_week": lead,
                    "method": method,
                    "rmse_mm_day": rmse,
                    "mae_mm_day": values[1] + 0.02 * lead,
                    "bias_mm_day": values[2],
                    "acc": acc,
                    "case_lead_count": 100,
                }
            )
    lead_scores = pd.DataFrame(by_lead_rows).set_index(["method", "lead_week"])
    for lead in range(1, 7):
        for metric in ("rmse_mm_day", "acc"):
            if metric == "rmse_mm_day":
                effect = (
                    lead_scores.loc[("raw_fuxi", lead), metric]
                    - lead_scores.loc[("raw_identity", lead), metric]
                )
            else:
                effect = (
                    lead_scores.loc[("raw_identity", lead), metric]
                    - lead_scores.loc[("raw_fuxi", lead), metric]
                )
            lead_effect_rows.append(
                {
                    "scope_type": "lead",
                    "scope": f"W{lead}",
                    "region": "all_india",
                    "candidate": "raw_identity",
                    "baseline": "raw_fuxi",
                    "source_metric": metric,
                    "effect": effect,
                    "ci_lower_2p5": effect - 0.05,
                    "ci_upper_97p5": effect + 0.05,
                    "bootstrap_probability_improved": 1.0,
                    "n_starts": 100,
                    "n_leads_per_start": 1,
                    "definition": "paired year-stratified circular moving-block bootstrap with actual block length 13 initializations",
                }
            )
    projection_effects = {
        "rmse_mm_day": (0.01, -0.02, 0.04),
        "mae_mm_day": (-0.10, -0.13, -0.07),
        "bias_mm_day": (0.60, 0.52, 0.68),
        "acc": (-0.005, -0.010, 0.001),
    }
    for metric, (effect, lower, upper) in projection_effects.items():
        definition = (
            "absolute pooled bias baseline minus candidate; positive favors candidate; paired year-stratified circular moving-block bootstrap with actual block length 13 initializations"
            if metric == "bias_mm_day"
            else "paired year-stratified circular moving-block bootstrap with actual block length 13 initializations"
        )
        lead_effect_rows.append(
            {
                "scope_type": "pooled",
                "scope": "W1-W6",
                "region": "all_india",
                "candidate": "raw_identity_raw_mean_preserved",
                "baseline": "raw_identity",
                "source_metric": metric,
                "effect": effect,
                "ci_lower_2p5": lower,
                "ci_upper_97p5": upper,
                "bootstrap_probability_improved": float(effect > 0),
                "n_starts": 100,
                "n_leads_per_start": 6,
                "definition": definition,
            }
        )

    intensity_effect_values = {
        "dry_lt1": {
            "rmse_mm_day": (0.55, 0.42, 0.66),
            "mae_mm_day": (0.31, 0.24, 0.38),
        },
        "light_1_5": {
            "rmse_mm_day": (0.98, 0.85, 1.10),
            "mae_mm_day": (0.64, 0.55, 0.72),
        },
        "moderate_5_10": {
            "rmse_mm_day": (0.80, 0.58, 1.01),
            "mae_mm_day": (0.41, 0.26, 0.54),
        },
        "heavy_10_20": {
            "rmse_mm_day": (0.21, 0.04, 0.37),
            "mae_mm_day": (0.08, -0.07, 0.21),
        },
        "extreme_ge20": {
            "rmse_mm_day": (0.23, -0.11, 0.45),
            "mae_mm_day": (-0.05, -0.35, 0.15),
        },
    }
    raw_errors = {
        "dry_lt1": (2.70, 1.56),
        "light_1_5": (4.35, 3.05),
        "moderate_5_10": (4.59, 3.35),
        "heavy_10_20": (6.76, 5.47),
        "extreme_ge20": (17.59, 14.64),
    }
    intensity_rows = []
    intensity_effect_rows = []
    for stratum_index, stratum in enumerate(core3.INTENSITY_ORDER):
        for method_index, method in enumerate(core3.E2_METHODS):
            rmse, mae = raw_errors[stratum]
            if method == "raw_identity":
                rmse -= intensity_effect_values[stratum]["rmse_mm_day"][0]
                mae -= intensity_effect_values[stratum]["mae_mm_day"][0]
            elif method != "raw_fuxi":
                rmse -= 0.1 * method_index
                mae -= 0.05 * method_index
            intensity_rows.append(
                {
                    "method": method,
                    "stratum": stratum,
                    "stratum_label": core3.INTENSITY_LABELS[stratum],
                    "cell_case_lead_count": 30_000 - 4_000 * stratum_index,
                    "rmse_mm_day": rmse,
                    "mae_mm_day": mae,
                    "bias_mm_day": 0.1,
                    "truth_mean_mm_day": 0.5 + 5.0 * stratum_index,
                    "prediction_mean_mm_day": 0.6 + 5.0 * stratum_index,
                }
            )
        for metric, (effect, lower, upper) in intensity_effect_values[stratum].items():
            intensity_effect_rows.append(
                {
                    "stratum": stratum,
                    "candidate": "raw_identity",
                    "baseline": "raw_fuxi",
                    "source_metric": metric,
                    "effect": effect,
                    "ci_lower_2p5": lower,
                    "ci_upper_97p5": upper,
                    "bootstrap_probability_improved": float(effect > 0),
                    "n_starts": 100,
                    "n_leads_per_start": 6,
                    "definition": "paired year-stratified circular moving-block bootstrap with actual block length 13 initializations; all six leads retained",
                }
            )
    e2_payloads = {
        core3.E2_REQUIRED_ARTIFACTS[0]: pooled_rows,
        core3.E2_REQUIRED_ARTIFACTS[1]: by_lead_rows,
        core3.E2_REQUIRED_ARTIFACTS[2]: lead_effect_rows,
        core3.E2_REQUIRED_ARTIFACTS[3]: intensity_rows,
        core3.E2_REQUIRED_ARTIFACTS[4]: intensity_effect_rows,
    }
    for relative, rows in e2_payloads.items():
        _write_csv(e2 / relative, rows)
    e2_manifest = {
        "status": "complete",
        "canonical": True,
        "scientific_eligible": True,
        "smoke": False,
        "audit_years": [2022, 2023, 2024],
        "final_initialization_year_quarantined": 2025,
        "final_2025_store_opened": False,
        "methods": list(core3.E2_METHODS),
        "audit_counts": {"2022": 35, "2023": 35, "2024": 30},
        "raw_identity_selection": {
            "model": core3.SELECTED_MODEL,
            "alpha": 1.0,
            "retrained_for_audit": False,
            "retuned_on_audit": False,
            "training_anchor": "raw_fuxi",
            "uses_fitted_log_bias_in_neural_training": False,
        },
        "input_provenance": {
            "raw_identity_manifest_sha256": training_hash,
            "raw_identity_checkpoints": [
                {
                    "path": (
                        f"/frozen/run/models/{core3.SELECTED_MODEL}/seed_{seed}/"
                        "checkpoints/best.pt"
                    ),
                    "sha256": checkpoint_hashes[seed],
                }
                for seed in core3.SEEDS
            ],
        },
        "bootstrap": {
            "draws": 10_000,
            "block_length_initializations": 13,
            "seed": 20260822,
            "all_six_leads_retained": True,
        },
        "projection": {"post_hoc": True, "operational_claim": False},
        "artifacts": _artifact_hashes(e2, core3.E2_REQUIRED_ARTIFACTS),
    }
    _write_json(e2 / "manifest.json", e2_manifest)
    e2_hash = _sha256(e2 / "manifest.json")

    e3 = tmp_path / "e3"
    e3_scores = {
        "raw_fuxi": (8.0, 5.3, 2.4, 2.5, 0.40),
        "log_bias": (7.6, 5.0, 1.9, 2.1, 0.43),
        "selected_adapter": (7.6, 5.0, 1.85, 2.05, 0.425),
        "raw_identity": (7.5, 4.9, 1.8, 2.0, 0.43),
        "raw_identity_raw_mean_preserved": (7.9, 5.25, 2.45, 2.55, 0.405),
    }
    summary_rows = [
        {
            "method": method,
            "scope_type": "pooled",
            "scope": "W1-W6",
            "initializations": 30,
            "case_leads": 180,
            "rmse_mean": values[0],
            "mae_mean": values[1],
            "bias_mean": values[2],
            "absolute_bias_mean": values[3],
            "acc_mean": values[4],
        }
        for method, values in e3_scores.items()
    ]
    effects_rows = []

    def add_e3_effects(
        candidate: str, reference: str, primary_rmse: bool = False
    ) -> None:
        candidate_scores = e3_scores[candidate]
        reference_scores = e3_scores[reference]
        columns = {"rmse": 0, "mae": 1, "absolute_bias": 3, "acc": 4}
        for metric, index in columns.items():
            effect = (
                candidate_scores[index] - reference_scores[index]
                if metric == "acc"
                else reference_scores[index] - candidate_scores[index]
            )
            effects_rows.append(
                {
                    "comparison": f"{candidate}_vs_{reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "effect_definition": (
                        "candidate_minus_reference"
                        if metric == "acc"
                        else "reference_minus_candidate"
                    ),
                    "block_length_initializations": 13,
                    "analysis_role": "primary_uncertainty",
                    "bootstrap_draws": 2_000,
                    "initializations": 30,
                    "case_leads": 180,
                    "point_effect": effect,
                    "bootstrap_mean": effect,
                    "ci_lower_2p5": effect - 0.05,
                    "ci_upper_97p5": effect + 0.05,
                    "probability_effect_gt_zero": float(effect > 0),
                    "primary_estimand": bool(primary_rmse and metric == "rmse"),
                }
            )

    add_e3_effects("selected_adapter", "raw_fuxi", primary_rmse=True)
    add_e3_effects("raw_identity", "raw_fuxi")
    add_e3_effects("raw_identity_raw_mean_preserved", "raw_identity")
    _write_csv(e3 / core3.E3_REQUIRED_ARTIFACTS[0], summary_rows)
    _write_csv(e3 / core3.E3_REQUIRED_ARTIFACTS[1], effects_rows)
    primary = next(row for row in effects_rows if row["primary_estimand"])
    e3_manifest = {
        "status": "complete_frozen_external_target_sensitivity",
        "canonical_artifact": True,
        "training_performed": False,
        "selection_calibration_or_blending_performed": False,
        "2025_metric_computed": False,
        "2025_prediction_opened": False,
        "2025_station_value_selected": False,
        "methods": list(core3.E3_METHODS),
        "extended_prediction_manifest_sha256": e2_hash,
        "dates": {
            "initialization_years": [2024],
            "lead_weeks": [1, 2, 3, 4, 5, 6],
            "initialization_count": 30,
            "case_leads": 180,
        },
        "bootstrap": {"draws": 2_000, "primary_block_length": 13, "seed": 20260822},
        "primary_estimand": {
            "comparison": primary["comparison"],
            "metric": primary["metric"],
            "point_effect": primary["point_effect"],
            "ci_lower_2p5": primary["ci_lower_2p5"],
            "ci_upper_97p5": primary["ci_upper_97p5"],
            "circular_block_length_initializations": 13,
            "bootstrap_draws": 2_000,
        },
        "station_truth_boundary": {
            "container_rows_scanned": 543_518,
            "container_date_max": "2025-02-10",
            "unselected_2025_plus_rows": 45_910,
            "rainfall_converted_only_after_exact_2024_date_and_station_filter": True,
        },
        "artifacts": _artifact_hashes(e3, core3.E3_REQUIRED_ARTIFACTS),
    }
    _write_json(e3 / "manifest.json", e3_manifest)
    e3_hash = _sha256(e3 / "manifest.json")
    return training, e2, e3, training_hash, e2_hash, e3_hash


def _build(sources: tuple[Path, Path, Path, str, str, str], output: Path) -> Path:
    training, e2, e3, training_hash, e2_hash, e3_hash = sources
    return core3.build_package(
        training,
        e2,
        e3,
        output,
        training_manifest_sha256=training_hash,
        e2_manifest_sha256=e2_hash,
        e3_manifest_sha256=e3_hash,
    )


def test_builds_three_figure_hash_verified_package(
    frozen_sources: tuple[Path, Path, Path, str, str, str], tmp_path: Path
) -> None:
    output = _build(frozen_sources, tmp_path / "core3")
    manifest = core3.verify_package(output)
    assert len(manifest["artifacts"]) == 12
    assert manifest["architecture_contract"]["ensemble_members"] == 3
    assert manifest["architecture_contract"]["parameter_count_per_member"] == 144_689
    assert manifest["bias_estimands"]["distinct_and_not_interchangeable"] is True
    assert manifest["e3_comparison_roles"]["raw_identity_vs_raw_fuxi"].startswith(
        "secondary"
    )
    assert manifest["access_boundary"]["builder_opened"]["2025_data"] is False
    assert len(manifest["access_boundary"]["loaded_source_files"]) == 10
    for stem in (
        "figure_01_architecture_evidence_timeline",
        "figure_02_lead_intensity_effects",
        "figure_03_cross_target_projection_failure",
    ):
        with Image.open(output / f"{stem}.png") as image:
            assert image.width >= 2500
            assert image.height >= 800
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        assert len(pd.read_csv(output / f"{stem}.csv")) > 0


def test_exact_bytes_are_parsed_after_verification(
    frozen_sources: tuple[Path, Path, Path, str, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, e2, _, _, _, _ = frozen_sources
    source = e2 / core3.E2_REQUIRED_ARTIFACTS[0]
    original_hash = _sha256(source)
    original_read_csv = core3.pd.read_csv
    mutated = False

    def mutate_after_payload_capture(value: object, *args: object, **kwargs: object):
        nonlocal mutated
        assert isinstance(value, io.BytesIO)
        if not mutated:
            source.write_text("corrupted after byte capture\n", encoding="utf-8")
            mutated = True
        return original_read_csv(value, *args, **kwargs)

    monkeypatch.setattr(core3.pd, "read_csv", mutate_after_payload_capture)
    output = _build(frozen_sources, tmp_path / "core3")
    manifest = core3.verify_package(output)
    assert mutated
    assert (
        manifest["sources"]["e2"]["verified_artifacts"][core3.E2_REQUIRED_ARTIFACTS[0]]
        == original_hash
    )


def test_rejects_wrong_manifest_pin(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, _, e2_hash, e3_hash = frozen_sources
    with pytest.raises(ValueError, match="manifest SHA-256 differs"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256="0" * 64,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_mutated_source_csv(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, training_hash, e2_hash, e3_hash = frozen_sources
    path = e2 / core3.E2_REQUIRED_ARTIFACTS[3]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_architecture_drift_even_with_refreshed_pin(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, _, e2_hash, e3_hash = frozen_sources
    manifest_path = training / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"][core3.SELECTED_MODEL]["parameter_count"] = 144_688
    _write_json(manifest_path, manifest)
    training_hash = _sha256(manifest_path)
    e2_manifest = json.loads((e2 / "manifest.json").read_text(encoding="utf-8"))
    e2_manifest["input_provenance"]["raw_identity_manifest_sha256"] = training_hash
    _write_json(e2 / "manifest.json", e2_manifest)
    e2_hash = _sha256(e2 / "manifest.json")
    e3_manifest = json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))
    e3_manifest["extended_prediction_manifest_sha256"] = e2_hash
    _write_json(e3 / "manifest.json", e3_manifest)
    e3_hash = _sha256(e3 / "manifest.json")
    with pytest.raises(ValueError, match="parameter count"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_secondary_e3_comparison_marked_primary(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, training_hash, e2_hash, _ = frozen_sources
    effects_path = e3 / core3.E3_REQUIRED_ARTIFACTS[1]
    effects = pd.read_csv(effects_path)
    index = effects.index[
        effects["candidate"].eq("raw_identity") & effects["metric"].eq("rmse")
    ][0]
    effects.loc[index, "primary_estimand"] = True
    effects.to_csv(effects_path, index=False, lineterminator="\n")
    e3_hash = _refresh_artifact(e3, core3.E3_REQUIRED_ARTIFACTS[1])
    with pytest.raises(ValueError, match="exactly one primary"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_refreshed_hash_for_inconsistent_e3_secondary_effect(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, training_hash, e2_hash, _ = frozen_sources
    effects_path = e3 / core3.E3_REQUIRED_ARTIFACTS[1]
    effects = pd.read_csv(effects_path)
    selected = (
        effects["candidate"].eq("raw_identity")
        & effects["reference"].eq("raw_fuxi")
        & effects["metric"].eq("rmse")
    )
    assert selected.sum() == 1
    effects.loc[selected, "point_effect"] = 999.0
    effects.loc[selected, "bootstrap_mean"] = 999.0
    effects.loc[selected, "ci_lower_2p5"] = 998.0
    effects.loc[selected, "ci_upper_97p5"] = 1000.0
    effects.to_csv(effects_path, index=False, lineterminator="\n")
    e3_hash = _refresh_artifact(e3, core3.E3_REQUIRED_ARTIFACTS[1])

    with pytest.raises(ValueError, match="E3 secondary effect does not reproduce"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_refreshed_hash_for_wrong_e3_primary_tuple(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, training_hash, e2_hash, _ = frozen_sources
    effects_path = e3 / core3.E3_REQUIRED_ARTIFACTS[1]
    effects = pd.read_csv(effects_path)
    selected = effects["primary_estimand"].eq(True)
    assert selected.sum() == 1
    # The synthetic log-bias and selected-adapter RMSE scores are equal.  This
    # drift therefore remains numerically self-consistent and must be rejected
    # by the semantic primary-estimand contract itself.
    effects.loc[selected, "candidate"] = "log_bias"
    effects.to_csv(effects_path, index=False, lineterminator="\n")
    e3_hash = _refresh_artifact(e3, core3.E3_REQUIRED_ARTIFACTS[1])

    with pytest.raises(ValueError, match="normalized primary estimand tuple"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_rejects_repinned_e2_checkpoint_seed_hash_mismatch(
    frozen_sources: tuple[Path, Path, Path, str, str, str]
) -> None:
    training, e2, e3, training_hash, _, _ = frozen_sources
    e2_manifest_path = e2 / "manifest.json"
    e2_manifest = json.loads(e2_manifest_path.read_text(encoding="utf-8"))
    e2_manifest["input_provenance"]["raw_identity_checkpoints"][0]["sha256"] = "d" * 64
    _write_json(e2_manifest_path, e2_manifest)
    e2_hash = _sha256(e2_manifest_path)

    e3_manifest_path = e3 / "manifest.json"
    e3_manifest = json.loads(e3_manifest_path.read_text(encoding="utf-8"))
    e3_manifest["extended_prediction_manifest_sha256"] = e2_hash
    _write_json(e3_manifest_path, e3_manifest)
    e3_hash = _sha256(e3_manifest_path)

    with pytest.raises(ValueError, match="checkpoint seed/SHA-256 mapping"):
        core3.load_verified_inputs(
            training,
            e2,
            e3,
            training_manifest_sha256=training_hash,
            e2_manifest_sha256=e2_hash,
            e3_manifest_sha256=e3_hash,
        )


def test_no_clobber_and_verifier_rejects_tampering(
    frozen_sources: tuple[Path, Path, Path, str, str, str], tmp_path: Path
) -> None:
    output = _build(frozen_sources, tmp_path / "core3")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(frozen_sources, output)
    png = output / "figure_01_architecture_evidence_timeline.png"
    png.write_bytes(png.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        core3.verify_package(output)


def test_verifier_rejects_extra_and_nested_manifest(
    frozen_sources: tuple[Path, Path, Path, str, str, str], tmp_path: Path
) -> None:
    output = _build(frozen_sources, tmp_path / "core3")
    nested = output / "nested"
    nested.mkdir()
    (nested / "PACKAGE_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nested PACKAGE_MANIFEST"):
        core3.verify_package(output)


def test_verifier_rejects_symlink_files_and_directories(
    frozen_sources: tuple[Path, Path, Path, str, str, str], tmp_path: Path
) -> None:
    output = _build(frozen_sources, tmp_path / "core3")
    readme = output / "README.md"
    original = readme.read_bytes()
    external_file = tmp_path / "external_readme.md"
    external_file.write_bytes(original)
    readme.unlink()
    readme.symlink_to(external_file)
    with pytest.raises(ValueError, match="package tree contains symlink"):
        core3.verify_package(output)

    readme.unlink()
    readme.write_bytes(original)
    external_directory = tmp_path / "external_directory"
    external_directory.mkdir()
    (external_directory / "payload.txt").write_text("outside\n", encoding="utf-8")
    (output / "linked_directory").symlink_to(
        external_directory, target_is_directory=True
    )
    with pytest.raises(ValueError, match="package tree contains symlink"):
        core3.verify_package(output)


def test_atomic_publish_refuses_destination_created_after_precheck(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    (destination / "owner.txt").write_text("owner\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="created during publication"):
        core3._publish_directory_noreplace(staging, destination)

    assert (staging / "candidate.txt").read_text(encoding="utf-8") == "candidate\n"
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "owner\n"
