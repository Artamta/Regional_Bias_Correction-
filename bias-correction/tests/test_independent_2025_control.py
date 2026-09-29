from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest
import torch

import evaluate_independent_2025_control as evaluation
import freeze_independent_2025_control_selection as freezer


def _normalization(*, append_physical: bool = True) -> dict:
    names = list(evaluation._expected_input_channels())
    result = {
        "input_channels": names,
        "spatial_context": {
            "enabled": True,
            "full_domain_channels": [
                "log_fuxi_mean",
                "log_fuxi_spread",
                "fuxi_t2m_weekly",
            ],
            "normalization_fit": "training cases and positive target weights only",
            "target_support_cells": 171,
        },
        "climatology_attention": {
            "source": "IMD training years 2002-2017 only",
            "offsets_days": list(evaluation.benchmark.imd_experiment.OFFSETS_DAYS),
        },
    }
    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
    ):
        result[name] = {
            "mean_by_lead": [0.1] * 6,
            "std_by_lead": [1.1] * 6,
        }
    if append_physical:
        result["input_channels"].extend(evaluation.sweep.PHYSICAL_PREDICTOR_NAMES)
        result["spatial_context"]["full_domain_channels"].extend(
            evaluation.sweep.PHYSICAL_PREDICTOR_NAMES
        )
        for name in evaluation.sweep.PHYSICAL_PREDICTOR_NAMES:
            result[name] = {
                "mean_by_lead": [0.0] * 6,
                "std_by_lead": [1.0] * 6,
            }
        result["fuxi_physical_predictors"] = {
            "normalization_fit": "training cases and positive target weights only"
        }
    return result


def _frozen_fixture(
    tmp_path: Path,
    *,
    configuration: str = "physical_control",
    explicit_contract: bool = False,
) -> tuple[Path, Path]:
    run = tmp_path / "validation_run"
    run.mkdir()
    candidate = evaluation.sweep.CANDIDATE_BY_NAME[configuration]
    manifest = {
        "status": "complete",
        "smoke": False,
        "test_predictions_created": False,
        "train_years": list(range(2002, 2018)),
        "validation_years": [2018, 2019],
        "quarantined_years": [2020, 2021, 2022, 2023, 2024, 2025],
        "seeds": [42, 43, 44],
        "candidates": [asdict(candidate)],
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    normalization = _normalization()
    (run / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )
    (run / "models").mkdir()
    np.savez_compressed(
        run / "models" / "log_bias_anchor.npz",
        lead_month_residual=np.zeros((6, 12, 27, 27), dtype=np.float32),
        shrinkage=np.float32(10.0),
        target_scale=np.ones(6, dtype=np.float32),
    )
    checkpoint_entries = []
    for seed in (42, 43, 44):
        seed_root = run / "models" / configuration / f"seed_{seed}"
        checkpoint = seed_root / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        if candidate.model_kind == "temporal":
            model = evaluation.sweep.build_model(
                candidate,
                29 + len(evaluation.sweep.PHYSICAL_PREDICTOR_NAMES),
                np.ones(6, dtype=np.float32),
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": 7,
                    "best_validation_loss": 0.51,
                    "seed": seed,
                    "target_scale": np.ones(6, dtype=np.float32),
                    "lead_weights": np.full(6, 1.0 / 6.0),
                    "loss_coefficients": dict(evaluation.sweep.LOSS_COEFFICIENTS),
                },
                checkpoint,
            )
        else:
            # The physical-candidate refusal occurs before checkpoint loading.
            checkpoint.write_bytes(f"synthetic incompatible checkpoint {seed}".encode())
        checkpoint_hash = evaluation.sha256_file(checkpoint)
        record = {
            "status": "complete",
            "seed": seed,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint": str(checkpoint.relative_to(run)),
            "candidate": asdict(candidate),
            "best_epoch_zero_based": 7,
            "best_validation_loss": 0.51,
        }
        (seed_root / "run_record.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint_entries.append(
            {
                "seed": seed,
                "path": str(checkpoint.relative_to(run)),
                "sha256": checkpoint_hash,
            }
        )
    (run / "metrics").mkdir()
    pd.DataFrame(
        [
            {
                "configuration": configuration,
                "rank": 1,
                "mean_best_validation_loss": 0.51,
                "qualifies": True,
            }
        ]
    ).to_csv(run / "metrics" / "ranked_configurations.csv", index=False)
    selection = {
        "status": "frozen",
        "frozen_before_2025_access": True,
        "selection_data_end_year": 2019,
        "selected_configuration": configuration,
        "validation_run": str(run.resolve()),
        "validation_run_manifest_sha256": evaluation.sha256_file(
            run / "manifest.json"
        ),
        "normalization_sha256": evaluation.sha256_file(run / "normalization.json"),
        "log_bias_anchor_sha256": evaluation.sha256_file(
            run / "models" / "log_bias_anchor.npz"
        ),
        "checkpoints": checkpoint_entries,
    }
    control_normalization = evaluation.project_control_normalization(normalization)
    data_contract = evaluation.build_control_data_contract(
        candidate,
        control_normalization,
        manifest,
        normalization_sha256=evaluation.sha256_file(run / "normalization.json"),
        anchor_sha256=evaluation.sha256_file(
            run / "models" / "log_bias_anchor.npz"
        ),
    )
    selection["code_sha256"] = evaluation.live_code_hashes()
    selection["data_contract"] = data_contract
    selection["data_contract_sha256"] = evaluation.canonical_json_sha256(
        data_contract
    )
    if explicit_contract:
        selection["independent_input_contract"] = (
            evaluation.EXPECTED_CONTROL_CONTRACT
        )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    return run, selection_path


def test_control_preflight_projects_unused_physical_stats_without_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, selection = _frozen_fixture(tmp_path)

    def forbidden_open(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("preflight opened a data store")

    monkeypatch.setattr(evaluation.xr, "open_zarr", forbidden_open)
    frozen = evaluation.validate_frozen_control(run, selection)
    assert frozen.candidate.name == "physical_control"
    assert len(frozen.control_normalization["input_channels"]) == 29
    assert not set(evaluation.sweep.PHYSICAL_PREDICTOR_NAMES) & set(
        frozen.control_normalization["input_channels"]
    )
    assert frozen.checkpoint_seeds == (42, 43, 44)


def test_cli_preflight_is_metadata_only_and_creates_no_access_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run, selection = _frozen_fixture(tmp_path)
    output = tmp_path / "must_not_exist"

    def forbidden_data_access(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("preflight touched a 2025 predictor or IMD target")

    monkeypatch.setattr(evaluation.xr, "open_zarr", forbidden_data_access)
    monkeypatch.setattr(
        evaluation.diagnostic, "annual_observation", forbidden_data_access
    )
    monkeypatch.setattr(evaluation, "load_test_data", forbidden_data_access)
    monkeypatch.setattr(
        evaluation.sys,
        "argv",
        [
            "evaluate_independent_2025_control.py",
            "--validation-run",
            str(run),
            "--selection-manifest",
            str(selection),
            "--output",
            str(output),
            "--preflight-only",
        ],
    )
    evaluation.main()
    message = capsys.readouterr().out
    assert "neither 2025 predictors nor IMD targets" in message
    assert not output.exists()
    assert not selection.with_name(
        selection.name + ".independent_2025_access.json"
    ).exists()


def test_selected_full_physical_winner_is_refused_before_2025_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, selection = _frozen_fixture(
        tmp_path, configuration="physical_full_compact", explicit_contract=True
    )

    def forbidden_open(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("physical refusal opened a 2025 store")

    monkeypatch.setattr(evaluation.xr, "open_zarr", forbidden_open)
    monkeypatch.setattr(evaluation, "load_test_data", forbidden_open)
    monkeypatch.setattr(evaluation.diagnostic, "annual_observation", forbidden_open)
    with pytest.raises(ValueError, match="absent.*2025|absent from"):
        evaluation.validate_frozen_control(run, selection)


def test_checkpoint_hash_tampering_is_rejected(tmp_path: Path) -> None:
    run, selection = _frozen_fixture(tmp_path)
    checkpoint = run / "models" / "physical_control" / "seed_43" / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        evaluation.validate_frozen_control(run, selection)


def test_live_code_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    run, selection = _frozen_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["code_sha256"]["evaluation"] = "0" * 64
    selection.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live evaluation code differs"):
        evaluation.validate_frozen_control(run, selection)


def test_live_data_contract_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    run, selection = _frozen_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["data_contract"]["expected_test_cases"] = 34
    payload["data_contract_sha256"] = evaluation.canonical_json_sha256(
        payload["data_contract"]
    )
    selection.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live predictor/target/data contract differs"):
        evaluation.validate_frozen_control(run, selection)


def test_live_frozen_audit_reports_matching_code_checkpoints_and_contract(
    tmp_path: Path,
) -> None:
    run, selection = _frozen_fixture(tmp_path)
    frozen = evaluation.validate_frozen_control(run, selection)
    report = evaluation.audit_live_against_frozen(
        frozen,
        expected_selection_sha256=evaluation.sha256_file(selection),
    )
    assert report["all_live_vs_frozen_hashes_match"] is True
    assert report["code"]["match"] is True
    assert report["checkpoints"]["match"] is True
    assert report["data_contract"]["match"] is True
    assert (
        report["data_contract"]["frozen_sha256"]
        == report["data_contract"]["live_sha256"]
    )


def test_access_ledger_is_atomic_and_single_use(tmp_path: Path) -> None:
    run, selection = _frozen_fixture(tmp_path)
    frozen = evaluation.validate_frozen_control(run, selection)
    ledger = tmp_path / "access.json"
    evaluation.create_access_ledger(ledger, frozen, tmp_path / "result")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["test_year"] == 2025
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluation.create_access_ledger(ledger, frozen, tmp_path / "other")


def test_freezer_emits_manifest_that_passes_strict_preflight(tmp_path: Path) -> None:
    run, _ = _frozen_fixture(tmp_path)
    output = tmp_path / "frozen_by_utility.json"
    payload = freezer.freeze_selection(
        run,
        output,
        configuration="physical_control",
        attest_no_2025_access=True,
    )
    assert payload["frozen_before_2025_access"] is True
    assert len(payload["checkpoints"]) == 3
    assert evaluation.validate_frozen_control(run, output).candidate.name == "physical_control"
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))


def test_cuda_requirement_fails_before_staging_or_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, selection = _frozen_fixture(tmp_path)
    frozen = evaluation.validate_frozen_control(run, selection)
    output = tmp_path / "cuda_required_result"
    ledger = tmp_path / "cuda_access.json"
    evaluation.create_access_ledger(ledger, frozen, output)

    def forbidden_data_access(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("CUDA rejection occurred after 2025 data access")

    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(evaluation, "load_test_data", forbidden_data_access)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        evaluation.run_evaluation(
            frozen,
            output,
            boundary_path=evaluation.validation_plots.DEFAULT_INDIA_BOUNDARY,
            access_ledger=ledger,
            require_cuda=True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))


def test_failed_evaluation_never_publishes_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, selection = _frozen_fixture(tmp_path)
    frozen = evaluation.validate_frozen_control(run, selection)
    output = tmp_path / "final_result"
    ledger = tmp_path / "failed_access.json"
    evaluation.create_access_ledger(ledger, frozen, output)

    def synthetic_failure() -> None:
        raise RuntimeError("synthetic failure before any 2025 array")

    monkeypatch.setattr(evaluation, "load_test_data", synthetic_failure)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        evaluation.run_evaluation(
            frozen,
            output,
            boundary_path=evaluation.validation_plots.DEFAULT_INDIA_BOUNDARY,
            access_ledger=ledger,
        )
    assert not output.exists()
    staging = list(tmp_path.glob(f".{output.name}.partial-*"))
    assert len(staging) == 1
    manifest = json.loads((staging[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "synthetic failure" in manifest["failure"]


def test_successful_evaluation_atomically_publishes_only_complete_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, selection = _frozen_fixture(tmp_path)
    frozen = evaluation.validate_frozen_control(run, selection)
    output = tmp_path / "complete_result"
    ledger = tmp_path / "success_access.json"
    evaluation.create_access_ledger(ledger, frozen, output)

    cases, leads, height, width = 35, 6, 2, 2
    raw = np.full((cases, leads, height, width), 3.0, dtype=np.float32)
    corrected = np.full_like(raw, 2.0)
    truth = np.full_like(raw, 1.8)
    forecast = SimpleNamespace(
        initializations=np.arange(
            np.datetime64("2025-06-01"),
            np.datetime64("2025-06-01") + np.timedelta64(cases, "D"),
        ),
        latitude=np.linspace(8.0, 36.0, height),
        longitude=np.linspace(68.0, 96.0, width),
        ensemble_mean=raw,
    )
    synthetic = evaluation.TestData(
        forecast=forecast,
        t2m_weekly=np.full_like(raw, 300.0),
        truth=truth,
        climatology=np.full_like(raw, 1.0),
        weekly_coverage=np.ones_like(raw),
        training_climatology_daily=np.zeros((365, height, width), dtype=np.float32),
        support=np.ones((height, width), dtype=bool),
        area_weight_km2=np.ones((height, width), dtype=np.float64),
        source_stores=("synthetic-predictor",),
        training_imd_stores=("synthetic-training-climatology",),
        verification_imd_store="synthetic-verification",
        source_metadata_hashes={},
    )

    monkeypatch.setattr(evaluation, "load_test_data", lambda: synthetic)
    monkeypatch.setattr(evaluation, "build_frozen_features", lambda data, lock: None)
    monkeypatch.setattr(
        evaluation,
        "infer_frozen_ensemble",
        lambda features, data, lock: (
            corrected,
            np.zeros_like(corrected),
            np.zeros_like(corrected),
        ),
    )

    def synthetic_metrics(prediction, *args, **kwargs):
        is_raw = prediction is raw
        return {
            "rmse_mm_day": np.full((cases, leads), 2.0 if is_raw else 1.0),
            "acc": np.full((cases, leads), 0.2 if is_raw else 0.5),
            "bias_mm_day": np.full((cases, leads), 1.0 if is_raw else 0.2),
        }

    monkeypatch.setattr(evaluation, "weighted_case_metrics", synthetic_metrics)
    spatial_shape = (leads, height, width)
    monkeypatch.setattr(
        evaluation,
        "cellwise_rmse_skill_interval",
        lambda *args, **kwargs: (
            np.full(spatial_shape, 50.0),
            np.full(spatial_shape, 0.1),
            np.full(spatial_shape, 0.5),
            np.ones(spatial_shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(
        evaluation.validation_plots,
        "load_india_boundary",
        lambda path=None: ([], {"source": "synthetic checked boundary"}),
    )
    monkeypatch.setattr(evaluation, "plot_metric_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "plot_spatial_lead", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation, "plot_skill_contact_sheet", lambda *args, **kwargs: None
    )

    evaluation.run_evaluation(
        frozen,
        output,
        boundary_path=evaluation.validation_plots.DEFAULT_INDIA_BOUNDARY,
        access_ledger=ledger,
    )
    assert output.is_dir()
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["atomic_output"]["requested_final_directory"] == str(output)
    assert manifest["live_vs_frozen_provenance_lock"][
        "all_live_vs_frozen_hashes_match"
    ] is True


def test_circular_blocks_are_deterministic_and_wrap() -> None:
    first = evaluation.circular_moving_block_indices(
        35, draws=23, block_length=13, seed=812
    )
    second = evaluation.circular_moving_block_indices(
        35, draws=23, block_length=13, seed=812
    )
    assert first.shape == (23, 35)
    assert first.dtype == np.int16
    assert np.array_equal(first, second)
    for row in first:
        assert np.all((row[1:13] - row[:12]) % 35 == 1)
        assert np.all((row[14:26] - row[13:25]) % 35 == 1)


def test_paired_metric_bootstrap_reports_supported_improvement() -> None:
    cases, leads = 35, 6
    raw = {
        "rmse_mm_day": np.full((cases, leads), 2.0),
        "acc": np.full((cases, leads), 0.25),
        "bias_mm_day": np.full((cases, leads), 1.0),
    }
    corrected = {
        "rmse_mm_day": np.full((cases, leads), 1.0),
        "acc": np.full((cases, leads), 0.55),
        "bias_mm_day": np.full((cases, leads), 0.2),
    }
    indices = evaluation.circular_moving_block_indices(
        cases, draws=500, block_length=13, seed=3
    )
    result = evaluation.bootstrap_metric_summary(raw, corrected, indices)
    lead = result.loc[result.lead_week.ne("all")]
    assert len(result) == 21
    assert lead.bootstrap_supported_improvement_95.all()
    assert not any(
        token in column.lower()
        for column in result.columns
        for token in ("p_value", "p-value", "paired_p", "q_", "fdr", "signific")
    )
    rmse = lead.loc[lead.metric.eq("rmse_mm_day")]
    assert np.allclose(rmse.improvement_pct, 50.0)
    assert (rmse.improvement_ci_low > 0.0).all()


def test_weighted_metrics_and_cellwise_percentile_support_on_synthetic_fields() -> None:
    generator = np.random.default_rng(9)
    cases, leads, height, width = 20, 6, 5, 4
    truth = generator.uniform(1.0, 8.0, size=(cases, leads, height, width))
    climatology = np.full_like(truth, 2.0)
    raw = truth + 2.0
    corrected = truth + 0.5
    coverage = np.ones_like(truth)
    coverage[0, :, 0, 0] = 0.0
    area = np.ones((height, width))
    metrics = evaluation.weighted_case_metrics(
        corrected, truth, climatology, coverage, area
    )
    assert np.allclose(metrics["rmse_mm_day"], 0.5)
    assert np.allclose(metrics["bias_mm_day"], 0.5)
    indices = evaluation.circular_moving_block_indices(
        cases, draws=300, block_length=7, seed=4
    )
    support = np.ones((height, width), dtype=bool)
    skill, interval_low, interval_high, supported = (
        evaluation.cellwise_rmse_skill_interval(
            raw, corrected, truth, coverage, support, indices
        )
    )
    assert np.allclose(skill[:, support], 75.0)
    assert np.nanmin(interval_low) > 0.0
    assert np.nanmin(interval_high) > 0.0
    assert supported.all()


def test_metric_curve_figure_has_block_percentile_support_path(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "figures").mkdir(parents=True)
    shape = (35, 6)
    raw = {
        "rmse_mm_day": np.full(shape, 2.0),
        "acc": np.full(shape, 0.25),
        "bias_mm_day": np.full(shape, 0.8),
    }
    corrected = {
        "rmse_mm_day": np.full(shape, 1.4),
        "acc": np.full(shape, 0.45),
        "bias_mm_day": np.full(shape, 0.2),
    }
    indices = evaluation.circular_moving_block_indices(
        35, draws=300, block_length=13, seed=7
    )
    summary = evaluation.bootstrap_metric_summary(raw, corrected, indices)
    evaluation.plot_metric_curves(output, summary)
    assert (output / "figures" / "metrics_independent_2025_by_lead.png").stat().st_size > 1000
    assert (output / "figures" / "metrics_independent_2025_by_lead.pdf").stat().st_size > 1000


def test_spatial_2x3_figure_renders_from_synthetic_fields(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "figures").mkdir(parents=True)
    cases = 5
    latitude = np.linspace(0.0, 39.0, 27)
    longitude = np.linspace(60.0, 99.0, 27)
    yy, xx = np.meshgrid(latitude, longitude, indexing="ij")
    base_field = 2.0 + np.exp(-((yy - 22.0) ** 2 + (xx - 80.0) ** 2) / 90.0)
    truth = np.broadcast_to(base_field, (cases, 6, 27, 27)).copy()
    raw = truth + 0.8
    corrected = truth + 0.2
    coverage = np.ones_like(truth)
    support = np.ones((27, 27), dtype=bool)
    skill = np.full((6, 27, 27), 75.0)
    supported = np.zeros((6, 27, 27), dtype=bool)
    supported[:, 10:13, 10:13] = True
    segments, _ = evaluation.validation_plots.load_india_boundary()
    summary = pd.DataFrame(
        [
            {
                "metric": "rmse_mm_day",
                "lead_week": 1,
                "improvement_pct": 75.0,
                "improvement_pct_ci_low": 60.0,
                "improvement_pct_ci_high": 82.0,
            }
        ]
    )
    evaluation.plot_spatial_lead(
        output,
        lead=1,
        truth=truth,
        raw=raw,
        corrected=corrected,
        coverage=coverage,
        skill=skill,
        bootstrap_supported=supported,
        support=support,
        latitude=latitude,
        longitude=longitude,
        boundary_segments=segments,
        lead_summary=summary,
    )
    assert (output / "figures" / "spatial_independent_2025_W1.png").stat().st_size > 1000
    assert (output / "figures" / "spatial_independent_2025_W1.pdf").stat().st_size > 1000
    evaluation.plot_skill_contact_sheet(
        output,
        skill,
        supported,
        support,
        latitude,
        longitude,
        segments,
    )
    assert (
        output / "figures" / "spatial_rmse_skill_contact_sheet_2025.png"
    ).stat().st_size > 1000


def test_slurm_wrapper_enforces_cuda_and_known_node_exclusions() -> None:
    script = (
        Path(evaluation.__file__).resolve().parents[1]
        / "slurm"
        / "evaluate_independent_2025_control.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=GPU-AI_prio" in script
    assert "#SBATCH --exclude=cn2,cn3,cn15,cn17" in script
    assert "nvidia-smi" in script
    assert "--require-cuda" in script
