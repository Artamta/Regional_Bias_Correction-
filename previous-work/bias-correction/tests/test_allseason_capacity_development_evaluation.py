"""Contracts for the frozen post-selection capacity evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_capacity_ablation as capacity
import fuxi_allseason_capacity_development_evaluation as evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _capacity_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected: str = "medium_158k",
    mode: str = "full",
) -> Path:
    root = tmp_path / "capacity-full"
    root.mkdir()
    cache = tmp_path / "cache.npy"
    cache.write_bytes(b"verified-test-cache")
    cache_hash = _sha256(cache)
    monkeypatch.setattr(capacity, "EXPECTED_CACHE_SHA256", cache_hash)

    relevant_sources = {
        "code/src/fuxi_allseason_capacity_ablation.py": Path(capacity.__file__),
        "code/src/fuxi_allseason_ensemble_calibration.py": Path(
            evaluation.base.__file__
        ),
        "code/src/fuxi_ensemble_calibration_core.py": evaluation.PROJECT_ROOT
        / "src/fuxi_ensemble_calibration_core.py",
        "code/src/fuxi_allseason_member_cache.py": evaluation.PROJECT_ROOT
        / "src/fuxi_allseason_member_cache.py",
    }
    source_hashes = {}
    artifact_hashes = {}
    for relative, source in relevant_sources.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = _sha256(destination)
        source_hashes[relative] = digest
        artifact_hashes[relative] = digest

    validation_rows = []
    for arm in capacity.EXPERIMENT_ARMS:
        if arm.name == capacity.BASE_CANDIDATE:
            value = 1.0
        elif arm.name == capacity.SUMMARY_CONTROL.name:
            value = 0.95
        elif arm.name == selected and selected != capacity.BASE_CANDIDATE:
            value = 0.98
        else:
            value = 1.01
        for seed in evaluation.SEEDS:
            for year in evaluation.base.VALIDATION_YEARS:
                for lead in range(1, 7):
                    validation_rows.append(
                        {
                            "split": "validation",
                            "candidate": arm.name,
                            "seed": seed,
                            "init": f"{year}-01-03",
                            "year": year,
                            "lead_week": lead,
                            "crps": value,
                        }
                    )
    validation_metrics = pd.DataFrame(validation_rows)
    selection = capacity.select_capacity(
        validation_metrics, expected_seeds=evaluation.SEEDS
    )
    assert selection["selected_candidate"] == selected
    selection["written_utc"] = "2026-08-22T00:00:00+00:00"
    selection["scientific_selection"] = True
    validation_path = root / "metrics/validation_case_metrics.csv"
    validation_path.parent.mkdir(parents=True)
    validation_metrics.to_csv(validation_path, index=False)
    artifact_hashes["metrics/validation_case_metrics.csv"] = _sha256(validation_path)
    _write_json(root / "selection.json", selection)
    artifact_hashes["selection.json"] = _sha256(root / "selection.json")

    runs = []
    for arm in capacity.EXPERIMENT_ARMS:
        for seed in evaluation.SEEDS:
            relative = f"models/{arm.name}/seed_{seed}/checkpoints/best.pt"
            checkpoint = root / relative
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm.name}:{seed}\n".encode())
            digest = _sha256(checkpoint)
            artifact_hashes[relative] = digest
            runs.append(
                {
                    "candidate": arm.name,
                    "seed": seed,
                    "parameter_count": arm.expected_parameter_count,
                    "checkpoint": relative,
                    "checkpoint_sha256": digest,
                }
            )
    manifest = {
        "experiment": capacity.EXPERIMENT,
        "status": "complete",
        "mode": mode,
        "smoke": mode == "smoke",
        "candidates": [asdict(candidate) for candidate in capacity.CANDIDATES],
        "controls": [asdict(capacity.SUMMARY_CONTROL)],
        "seeds": list(evaluation.SEEDS),
        "split_counts_selected": {"train": 1652, "validation": 196},
        "contract": {
            "test_metrics_consulted": False,
            "sealed_2025_target_opened": False,
            "train_years": list(evaluation.base.TRAIN_YEARS),
            "validation_years": list(evaluation.base.VALIDATION_YEARS),
        },
        "training": {
            "member_subsample": 16,
            "full_members_for_validation": 51,
            "objective": "area-weighted empirical finite-ensemble CRPS",
            "runs": runs,
        },
        "selection": selection,
        "cache": {
            "data_file": str(cache),
            "data_sha256": cache_hash,
            "source_fingerprint": "unit-test-source",
        },
        "source_snapshot_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
    }
    _write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def _minimal_case_rows(method: str, seed: str | int, value: float) -> pd.DataFrame:
    rows = []
    for lead in range(1, 7):
        rows.append(
            {
                "split": "test_development",
                "method": method,
                "method_label": method,
                "seed": seed,
                "init": "2020-01-03",
                "year": 2020,
                "season": "DJF",
                "lead_week": lead,
                "member_count": 51,
                "support_cells": 171,
                "crps": value,
                "rmse": 2.0 * value,
                "mae": 1.5 * value,
                "acc": 0.2,
                "bias": -0.1,
                "absolute_bias": 0.1,
                "ensemble_spread": 1.0,
                "ensemble_variance": 1.0,
                "mean_squared_error": 4.0,
                "spread_skill_ratio": 0.5,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_rows(
    methods: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    initializations = np.asarray(
        ["2020-01-03", "2020-01-10", "2021-01-03", "2021-01-10"],
        dtype="datetime64[D]",
    )
    raw_rows = []
    seed_rows = []
    for initialization in initializations:
        init = np.datetime_as_string(initialization, unit="D")
        for lead in range(1, 7):
            raw_rows.append(
                {
                    "method": "raw_fuxi",
                    "init": init,
                    "lead_week": lead,
                    "crps": 2.0,
                    "rmse": 3.0,
                    "mae": 2.5,
                    "acc": 0.1,
                    "bias": -0.2,
                }
            )
            for method in methods:
                for seed in evaluation.SEEDS:
                    seed_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "init": init,
                            "lead_week": lead,
                            "crps": 1.8,
                            "rmse": 2.7,
                            "mae": 2.2,
                            "acc": 0.2,
                            "bias": -0.1,
                        }
                    )
    raw = pd.DataFrame(raw_rows)
    seeds = pd.DataFrame(seed_rows)
    headline = pd.concat(
        [
            raw,
            *[
                seeds.loc[seeds.method.eq(method)]
                .groupby(["method", "init", "lead_week"], as_index=False)[
                    ["crps", "rmse", "mae", "acc", "bias"]
                ]
                .mean()
                for method in methods
            ],
        ],
        ignore_index=True,
    )
    return headline, seeds, initializations


def test_full_capacity_receipt_is_hash_gated_and_returns_locked_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _capacity_receipt(tmp_path, monkeypatch, selected="medium_158k")

    receipt = evaluation.validate_capacity_manifest(path)

    assert receipt.selected_candidate == "medium_158k"
    assert receipt.selected_distinct_from_base is True
    assert set(receipt.checkpoint_records) == {
        (arm.name, seed)
        for arm in capacity.EXPERIMENT_ARMS
        for seed in evaluation.SEEDS
    }
    assert receipt.manifest_sha256 == _sha256(path)


def test_receipt_rejects_smoke_and_artifact_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _capacity_receipt(tmp_path, monkeypatch, mode="smoke")
    with pytest.raises(evaluation.DevelopmentEvaluationError, match="full capacity"):
        evaluation.validate_capacity_manifest(smoke)

    other = tmp_path / "other"
    other.mkdir()
    full = _capacity_receipt(other, monkeypatch)
    checkpoint = full.parent / "models/base_42k/seed_42/checkpoints/best.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(evaluation.DevelopmentEvaluationError, match="hash mismatch"):
        evaluation.validate_capacity_manifest(full)


def test_receipt_rejects_current_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _capacity_receipt(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    relative = "code/src/fuxi_allseason_capacity_ablation.py"
    frozen = path.parent / relative
    frozen.write_text("# internally consistent but not current\n")
    digest = _sha256(frozen)
    manifest["source_snapshot_sha256"][relative] = digest
    manifest["artifact_sha256"][relative] = digest
    _write_json(path, manifest)

    with pytest.raises(
        evaluation.DevelopmentEvaluationError, match="current source differs"
    ):
        evaluation.validate_capacity_manifest(path)


def test_receipt_recomputes_locked_winner_from_hashed_validation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _capacity_receipt(tmp_path, monkeypatch, selected="medium_158k")
    manifest = json.loads(path.read_text())
    selection = dict(manifest["selection"])
    selection["selected_candidate"] = "large_294k"
    selection["selected_parameter_count"] = 293_762
    _write_json(path.parent / "selection.json", selection)
    manifest["selection"] = selection
    manifest["artifact_sha256"]["selection.json"] = _sha256(
        path.parent / "selection.json"
    )
    _write_json(path, manifest)

    with pytest.raises(evaluation.DevelopmentEvaluationError, match="not reproducible"):
        evaluation.validate_capacity_manifest(path)


def test_selected_base_is_evaluated_once_without_fake_alias() -> None:
    assert evaluation.evaluation_candidate_names("base_42k") == ("base_42k",)
    with pytest.raises(ValueError, match="unknown selected"):
        evaluation.evaluation_candidate_names("not_a_candidate")

    weekwise = pd.DataFrame(
        {
            "method": ["base_42k"] * 6,
            "lead_week": list(range(1, 7)),
            "crps": np.linspace(1.0, 2.0, 6),
            "rmse": np.linspace(2.0, 3.0, 6),
            "acc": np.linspace(0.5, 0.1, 6),
            "bias": np.linspace(-0.2, 0.2, 6),
        }
    )
    comparison = evaluation.selected_vs_base_weekwise(weekwise, "base_42k")
    assert not comparison.distinct_comparison.any()
    assert set(comparison.comparison_status) == {
        "selected_is_base_no_distinct_improvement"
    }
    assert comparison.filter(like="selected_vs_base").isna().all().all()


def test_distinct_selected_weekwise_effects_are_directionally_defined() -> None:
    base_rows = pd.DataFrame(
        {
            "method": "base_42k",
            "lead_week": range(1, 7),
            "crps": 2.0,
            "rmse": 4.0,
            "acc": 0.2,
            "bias": -0.3,
        }
    )
    selected_rows = pd.DataFrame(
        {
            "method": "medium_158k",
            "lead_week": range(1, 7),
            "crps": 1.8,
            "rmse": 3.6,
            "acc": 0.25,
            "bias": -0.1,
        }
    )
    comparison = evaluation.selected_vs_base_weekwise(
        pd.concat((base_rows, selected_rows)), "medium_158k"
    )
    np.testing.assert_allclose(comparison.crps_skill_pct_selected_vs_base, 10.0)
    np.testing.assert_allclose(comparison.rmse_skill_pct_selected_vs_base, 10.0)
    np.testing.assert_allclose(comparison.acc_delta_selected_vs_base, 0.05)
    np.testing.assert_allclose(comparison.signed_bias_delta_selected_vs_base, 0.2)


def test_headline_averages_scores_only_and_keeps_every_seed_row() -> None:
    raw = _minimal_case_rows("raw_fuxi", "not_applicable", 2.0)
    seed_frames = [
        _minimal_case_rows("base_42k", seed, value)
        for seed, value in zip(evaluation.SEEDS, (1.0, 2.0, 3.0), strict=True)
    ]
    seeds = pd.concat(seed_frames, ignore_index=True)

    headline = evaluation.build_headline_case_metrics(raw, seeds, ("base_42k",))

    assert len(seeds) == 18
    assert len(headline) == 12
    neural = headline.loc[headline.method.eq("base_42k")]
    np.testing.assert_allclose(neural.crps, 2.0)
    assert set(neural.seed) == {"mean_of_seed_metrics_42_43_44"}


def test_bootstraps_retain_seed_identity_and_omit_same_model_comparison() -> None:
    methods = ("base_42k",)
    headline, seeds, initializations = _bootstrap_rows(methods)
    raw = headline.loc[headline.method.eq("raw_fuxi")]

    pooled, seed_raw, matched = evaluation.build_bootstrap_tables(
        headline,
        raw,
        seeds,
        initializations,
        methods,
        "base_42k",
        n_resamples=20,
    )

    assert set(pooled.comparison_scope) == {"mean_seed_scores_vs_raw"}
    assert set(pooled.metric) == set(evaluation.CORE_METRICS)
    assert set(seed_raw.optimization_seed) == set(evaluation.SEEDS)
    assert matched.empty
    assert list(matched.columns) == list(evaluation.BOOTSTRAP_COLUMNS)


def test_distinct_bootstrap_has_matched_seed_selected_vs_base_rows() -> None:
    methods = ("base_42k", "medium_158k")
    headline, seeds, initializations = _bootstrap_rows(methods)
    # Make selected distinguishable from base.
    chosen = seeds.method.eq("medium_158k")
    seeds.loc[chosen, ["crps", "rmse"]] *= 0.9
    seeds.loc[chosen, "acc"] += 0.05
    headline = pd.concat(
        [
            headline.loc[headline.method.ne("medium_158k")],
            seeds.loc[chosen]
            .groupby(["method", "init", "lead_week"], as_index=False)[
                ["crps", "rmse", "mae", "acc", "bias"]
            ]
            .mean(),
        ],
        ignore_index=True,
    )
    raw = headline.loc[headline.method.eq("raw_fuxi")]

    pooled, _, matched = evaluation.build_bootstrap_tables(
        headline,
        raw,
        seeds,
        initializations,
        methods,
        "medium_158k",
        n_resamples=20,
    )

    assert "mean_seed_scores_selected_vs_base" in set(pooled.comparison_scope)
    assert set(matched.optimization_seed) == set(evaluation.SEEDS)
    assert set(matched.baseline) == {"base_42k"}
    assert set(matched.method) == {"medium_158k"}


def test_validate_args_freezes_smoke_and_full_settings(tmp_path: Path) -> None:
    parser = evaluation.build_parser()
    manifest = tmp_path / "manifest.json"
    full = parser.parse_args(["--capacity-manifest", str(manifest)])
    evaluation.validate_args(full)
    assert full.bootstrap_samples == 2000

    smoke = parser.parse_args(["--capacity-manifest", str(manifest), "--smoke"])
    evaluation.validate_args(smoke)
    assert smoke.bootstrap_samples == 100

    wrong = parser.parse_args(
        ["--capacity-manifest", str(manifest), "--bootstrap-samples", "50"]
    )
    with pytest.raises(ValueError, match="canonical evaluation settings differ"):
        evaluation.validate_args(wrong)
    cpu = parser.parse_args(["--capacity-manifest", str(manifest), "--device", "cpu"])
    with pytest.raises(ValueError, match="requires CUDA"):
        evaluation.validate_args(cpu)


def test_main_publishes_atomically_after_receipt_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity_manifest = tmp_path / "capacity" / "manifest.json"
    capacity_manifest.parent.mkdir()
    capacity_manifest.write_text("{}")
    output = tmp_path / "development-smoke"
    receipt = evaluation.CapacityReceipt(
        manifest_path=capacity_manifest,
        root=capacity_manifest.parent,
        manifest={"experiment": capacity.EXPERIMENT},
        manifest_sha256="a" * 64,
        selected_candidate="base_42k",
        selected_distinct_from_base=False,
        checkpoint_records={},
        cache_path=tmp_path / "cache.npy",
    )
    monkeypatch.setattr(evaluation, "validate_capacity_manifest", lambda path: receipt)

    def fake_run(args, staging: Path, validated):
        assert validated is receipt
        evaluation.write_json(
            staging / "manifest.json",
            {"experiment": evaluation.EXPERIMENT, "status": "complete"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(evaluation, "run_experiment", fake_run)
    assert (
        evaluation.main(
            [
                "--capacity-manifest",
                str(capacity_manifest),
                "--output",
                str(output),
                "--smoke",
            ]
        )
        == 0
    )
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".*.incomplete-*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluation.main(
            [
                "--capacity-manifest",
                str(capacity_manifest),
                "--output",
                str(output),
                "--smoke",
            ]
        )


def test_slurm_launcher_requires_and_writes_postrun_gate_receipt() -> None:
    launcher = (
        evaluation.PROJECT_ROOT
        / "slurm/evaluate_allseason_capacity_development.sbatch"
    ).read_text(encoding="utf-8")

    assert "SMOKE_GATE_RECEIPT=" in launcher
    assert 'receipt.get("gate_status") == "passed"' in launcher
    assert 'receipt.get("manifest_sha256") == sha256_file(smoke_path)' in launcher
    assert '"post_run_audit_version": "capacity_dev_postrun_v1"' in launcher
    assert 'destination = root / "slurm_gate_receipt.json"' in launcher
    assert "os.replace(temporary, destination)" in launcher
