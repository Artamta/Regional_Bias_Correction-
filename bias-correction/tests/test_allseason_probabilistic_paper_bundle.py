"""Synthetic contracts for the all-season probabilistic paper bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import build_allseason_probabilistic_paper_bundle as bundle


def test_operational_receipt_contract_is_pinned_to_v3() -> None:
    assert bundle.RECEIPT_AUDIT_VERSION["operational"] == (
        "operational_era_postrun_v3"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _manifest(
    root: Path,
    role: str,
    artifacts: tuple[str, ...],
    **updates: object,
) -> tuple[dict, str]:
    payload: dict = {
        "experiment": bundle.EXPERIMENT_BY_ROLE[role],
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "scientific_status": f"synthetic scientific status for {role}",
        "contract": {
            "sealed_2025_target_opened": False,
            "sealed_unopened_years": [2025],
            "train_years": [2002, 2003],
        },
        "artifact_sha256": {relative: _sha(root / relative) for relative in artifacts},
    }
    payload.update(updates)
    _json(root / "manifest.json", payload)
    return payload, _sha(root / "manifest.json")


def _gate_receipt(
    root: Path,
    role: str,
    manifest_sha: str,
    **updates: object,
) -> None:
    payload: dict = {
        "experiment": bundle.EXPERIMENT_BY_ROLE[role],
        "gate_status": "passed",
        "post_run_audit_version": bundle.RECEIPT_AUDIT_VERSION[role],
        "manifest_sha256": manifest_sha,
        "job_id": "12345",
        "node": "cn20",
        "partition": "gpu_prio",
        "completed_utc": "2026-08-22T12:00:00+00:00",
    }
    if role != "categorical":
        payload["mode"] = "full"
    if role in {"pbc", "categorical"}:
        payload["scoring_contract_version"] = bundle.PRIMARY_CATEGORICAL_CONTRACT
    payload.update(updates)
    _json(root / "slurm_gate_receipt.json", payload)


def _continuous_rows(*, operational: bool) -> list[dict]:
    rows = []
    for lead in range(0, 7):
        scope = "W1-W6" if lead == 0 else f"W{lead}"
        effect = (12.0 if operational else 16.0) - 0.8 * lead
        if operational:
            rows.append(
                {
                    "comparison_scope": "paired_seed_score_average_vs_raw",
                    "optimization_seed": "mean_of_seed_scores",
                    "method": "base_42k",
                    "baseline": "raw_fuxi",
                    "lead_scope": scope,
                    "metric": "crps",
                    "effect": effect,
                    "ci_lower_2p5": effect - 2.0,
                    "ci_upper_97p5": effect + 2.0,
                    "bootstrap_probability_effect_positive": 0.99,
                    "paired_initializations": 12,
                    "source_year_clusters": 3,
                    "n_resamples": 100,
                    "block_length_initializations": 13,
                    "bootstrap": "year resampling plus circular blocks",
                }
            )
        else:
            rows.append(
                {
                    "comparison_scope": "mean_seed_scores_vs_raw",
                    "optimization_seed": "mean_of_seed_scores",
                    "method": "base_42k",
                    "baseline": "raw_fuxi",
                    "lead_scope": scope,
                    "metric": "crps",
                    "effect": effect,
                    "ci_lower": effect - 2.0,
                    "ci_upper": effect + 2.0,
                    "paired_initializations": 8,
                    "n_resamples": 100,
                    "block_length_initializations": 13,
                    "seed": 42,
                    "bootstrap": "year resampling plus circular blocks",
                }
            )
    return rows


@pytest.fixture
def paper_sources(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str], Path]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}

    neural = tmp_path / "neural"
    neural.mkdir()
    (neural / "README.md").write_text("neural\n", encoding="utf-8")
    _, hashes["neural"] = _manifest(
        neural,
        "neural",
        ("README.md",),
        slurm={"job_id": "100", "node": "cn11", "partition": "gpu_prio"},
    )
    paths["neural"] = neural

    capacity = tmp_path / "capacity"
    candidates = [
        ("small_20k", 20_000, -0.1, True),
        ("base_42k", 42_434, 0.0, True),
        ("medium_158k", 158_000, 0.04, True),
        ("large_294k", 294_000, -0.05, True),
        ("summary_matched_43k", 43_058, 0.03, False),
    ]
    capacity_rows = [
        {
            "candidate": name,
            "role": "parameter_matched_summary_control" if not encoder else "width_candidate",
            "mode": "summary_only" if not encoder else "location_spread",
            "member_encoder_used": encoder,
            "parameter_count": parameters,
            "mean_validation_crps": 1.33 - skill / 100.0,
            "crps_skill_pct_vs_base": skill,
            "all_years_noninferior_guard": name in {"base_42k", "summary_matched_43k"},
            "matched_seed_improvement_passes": 1,
            "eligible_promotion": False,
        }
        for name, parameters, skill, encoder in candidates
    ]
    _csv(capacity / "metrics/validation_summary.csv", capacity_rows)
    capacity_manifest, hashes["capacity"] = _manifest(
        capacity,
        "capacity",
        ("metrics/validation_summary.csv",),
        candidates=[{"name": row[0]} for row in candidates[:4]],
        controls=[{"name": candidates[4][0]}],
        selection={
            "selected_candidate": "base_42k",
            "reason": "no candidate passed all frozen validation guards",
        },
    )
    assert capacity_manifest["selection"]["selected_candidate"] == "base_42k"
    paths["capacity"] = capacity

    development = tmp_path / "capacity-development"
    _csv(
        development / "metrics/paired_block_bootstrap.csv",
        _continuous_rows(operational=False),
    )
    _, hashes["capacity_development"] = _manifest(
        development,
        "capacity_development",
        ("metrics/paired_block_bootstrap.csv",),
        capacity_receipt={
            "sha256": hashes["capacity"],
            "artifact_inventory_verified": True,
        },
    )
    _gate_receipt(
        development,
        "capacity_development",
        hashes["capacity_development"],
    )
    paths["capacity_development"] = development

    hybrid = tmp_path / "hybrid"
    profiles = ("crps_only", "hybrid_010", "hybrid_025", "hybrid_050", "mse_only")
    alphas = (0.0, 0.1, 0.25, 0.5, 1.0)
    validation_rows = []
    pooled_rows = []
    for index, (profile, alpha) in enumerate(zip(profiles, alphas, strict=True)):
        ratio = 1.0 + 0.003 * index
        validation_rows.append(
            {
                "profile": profile,
                "alpha_mse": alpha,
                "mean_validation_crps": 1.33 * ratio,
                "mean_validation_rmse": 3.0 + 0.02 * index,
                "mean_validation_coverage_error": 0.06 + 0.005 * index,
                "crps_ratio_vs_control": ratio,
                "eligible_joint_candidate": True,
            }
        )
        pooled_rows.append(
            {
                "method": profile,
                "crps": 1.38 + 0.02 * index,
                "rmse": 3.1 + 0.03 * index,
                "crps_skill_pct_vs_raw": 16.0 - index,
                "rmse_skill_pct_vs_raw": 12.0 - index,
                "case_count": 8,
            }
        )
    _csv(hybrid / "metrics/validation_profile_summary.csv", validation_rows)
    _csv(hybrid / "metrics/pooled_metrics.csv", pooled_rows)
    _, hashes["hybrid_loss"] = _manifest(
        hybrid,
        "hybrid_loss",
        ("metrics/validation_profile_summary.csv", "metrics/pooled_metrics.csv"),
        profiles=list(profiles),
        selection={"selected_profile": "crps_only"},
        slurm={"job_id": "101", "node": "cn18", "partition": "gpu_prio"},
    )
    paths["hybrid_loss"] = hybrid

    pbc = tmp_path / "pbc"
    pbc_rows = []
    for method, score in (
        ("debias_plus_plus", 0.21),
        ("persistence_plus_plus", 0.195),
    ):
        for lead in range(0, 7):
            reduction = 1.0 - score / 0.25
            pbc_rows.append(
                {
                    "score_contract": bundle.PRIMARY_CATEGORICAL_CONTRACT,
                    "lead_scope": "W1-W6" if lead == 0 else f"W{lead}",
                    "lead_week": lead,
                    "method": method,
                    "baseline": "raw_fuxi_categorical",
                    "rps_reduction_fraction": reduction,
                    "ci_lower_95": reduction - 0.02,
                    "ci_upper_95": reduction + 0.02,
                    "bootstrap_probability_improvement": 1.0,
                    "bootstrap_samples": 100,
                    "block_length_initializations": 13,
                    "bootstrap_scheme": "two_stage_year_resample_then_within_year_circular_blocks",
                    "test_year_count": 2,
                }
            )
    _csv(pbc / "metrics/paired_block_bootstrap.csv", pbc_rows)
    _, hashes["pbc"] = _manifest(
        pbc,
        "pbc",
        ("metrics/paired_block_bootstrap.csv",),
        scoring_contract_version=bundle.PRIMARY_CATEGORICAL_CONTRACT,
    )
    _gate_receipt(pbc, "pbc", hashes["pbc"])
    paths["pbc"] = pbc

    categorical = tmp_path / "categorical"
    methods = (
        "raw_fuxi_categorical",
        "moment_calibration",
        "summary_only",
        "location_spread",
        "debias_plus_plus",
        "persistence_plus_plus",
        "pbc_combined",
    )
    location_by_lead = (0.18, 0.198, 0.21, 0.215, 0.22, 0.225)
    scores_by_method = {
        "raw_fuxi_categorical": 0.25,
        "moment_calibration": 0.22,
        "summary_only": 0.205,
        "location_spread": sum(location_by_lead) / 6.0,
        "debias_plus_plus": 0.21,
        "persistence_plus_plus": 0.195,
        "pbc_combined": 0.20,
    }
    lead_scores = {
        method: (
            location_by_lead
            if method == "location_spread"
            else (score,) * 6
        )
        for method, score in scores_by_method.items()
    }
    categorical_rows = []
    weekwise_rows = []
    for method in methods:
        score = scores_by_method[method]
        reference = 0.225
        paper_score = score * 3.0
        paper_reference = 0.75
        categorical_rows.append(
            {
                "method": method,
                "method_label": method.replace("_", " ").title(),
                "score_contract": bundle.PRIMARY_CATEGORICAL_CONTRACT,
                "nominal_reference_contract": "synthetic",
                "rps": score,
                "paper_q80_retained_scoring_weight": 100.0,
                "nominal_climatology_rps": 0.23,
                "training_empirical_climatology_rps": reference,
                "n_initializations": 8,
                "rpss_vs_nominal_climatology": 1.0 - score / 0.23,
                "rpss_vs_training_empirical_climatology": 1.0 - score / reference,
                "paper_nominal_cut_q80_positive_rps": paper_score,
                "paper_fixed_nominal_climatology_rps": paper_reference,
                "paper_nominal_cut_rpss_vs_fixed_nominal_climatology": 1.0
                - paper_score / paper_reference,
            }
        )
        for lead, lead_score in enumerate(lead_scores[method], start=1):
            weekwise_rows.append(
                {
                    "lead_week": lead,
                    "method": method,
                    "method_label": method.replace("_", " ").title(),
                    "score_contract": bundle.PRIMARY_CATEGORICAL_CONTRACT,
                    "nominal_reference_contract": "synthetic",
                    "rps": lead_score,
                    "paper_q80_retained_scoring_weight": 100.0,
                    "nominal_climatology_rps": 0.23,
                    "training_empirical_climatology_rps": reference,
                    "n_initializations": 8,
                    "rpss_vs_nominal_climatology": 1.0 - lead_score / 0.23,
                    "rpss_vs_training_empirical_climatology": 1.0
                    - lead_score / reference,
                    "paper_nominal_cut_q80_positive_rps": lead_score * 3.0,
                    "paper_fixed_nominal_climatology_rps": paper_reference,
                    "paper_nominal_cut_rpss_vs_fixed_nominal_climatology": 1.0
                    - lead_score * 3.0 / paper_reference,
                }
            )
    comparison_rows = []
    for lead in range(0, 7):
        for method in (
            "moment_calibration",
            "summary_only",
            "location_spread",
            "pbc_combined",
        ):
            score = (
                scores_by_method[method]
                if lead == 0
                else lead_scores[method][lead - 1]
            )
            raw_score = 0.25
            reduction = 1.0 - score / raw_score
            comparison_rows.append(
                {
                    "score_contract": bundle.PRIMARY_CATEGORICAL_CONTRACT,
                    "metric": bundle.PRIMARY_CATEGORICAL_METRIC,
                    "lead_scope": "W1-W6" if lead == 0 else f"W{lead}",
                    "lead_week": lead,
                    "method": method,
                    "baseline": "raw_fuxi_categorical",
                    "method_score": score,
                    "baseline_score": raw_score,
                    "rps_reduction_fraction": reduction,
                    "ci_lower_95": reduction - 0.02,
                    "ci_upper_95": reduction + 0.02,
                    "bootstrap_probability_improvement": 1.0,
                    "bootstrap_samples": 100,
                    "block_length_initializations": 13,
                    "bootstrap_scheme": "two_stage_year_resample_then_within_year_circular_blocks",
                    "test_year_count": 2,
                }
            )
        location = (
            scores_by_method["location_spread"]
            if lead == 0
            else lead_scores["location_spread"][lead - 1]
        )
        combined = 0.20
        reduction = 1.0 - location / combined
        if lead == 2:
            lower, upper, probability = -0.02, 0.04, 0.65
        else:
            lower, upper = reduction - 0.02, reduction + 0.02
            probability = 1.0 if lower > 0.0 else 0.0
        comparison_rows.append(
            {
                "score_contract": bundle.PRIMARY_CATEGORICAL_CONTRACT,
                "metric": bundle.PRIMARY_CATEGORICAL_METRIC,
                "lead_scope": "W1-W6" if lead == 0 else f"W{lead}",
                "lead_week": lead,
                "method": "location_spread",
                "baseline": "pbc_combined",
                "method_score": location,
                "baseline_score": combined,
                "rps_reduction_fraction": reduction,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "bootstrap_probability_improvement": probability,
                "bootstrap_samples": 100,
                "block_length_initializations": 13,
                "bootstrap_scheme": "two_stage_year_resample_then_within_year_circular_blocks",
                "test_year_count": 2,
            }
        )
    _csv(categorical / "metrics/pooled_metrics.csv", categorical_rows)
    _csv(categorical / "metrics/weekwise_metrics.csv", weekwise_rows)
    _csv(categorical / "metrics/paired_block_bootstrap.csv", comparison_rows)
    _, hashes["categorical"] = _manifest(
        categorical,
        "categorical",
        (
            "metrics/pooled_metrics.csv",
            "metrics/weekwise_metrics.csv",
            "metrics/paired_block_bootstrap.csv",
        ),
        scoring_contract_version=bundle.PRIMARY_CATEGORICAL_CONTRACT,
        methods=list(methods),
        input_receipts={
            "neural_manifest": {"sha256": hashes["neural"]},
            "pbc_manifest": {"sha256": hashes["pbc"]},
        },
        evaluation={
            "bootstrap_inference_status": (
                "exploratory paired uncertainty only; two year clusters do not create an independent test"
            )
        },
    )
    _gate_receipt(
        categorical,
        "categorical",
        hashes["categorical"],
        neural_manifest_sha256=hashes["neural"],
        pbc_manifest_sha256=hashes["pbc"],
    )
    paths["categorical"] = categorical

    operational = tmp_path / "operational"
    _csv(
        operational / "metrics/paired_two_stage_bootstrap.csv",
        _continuous_rows(operational=True),
    )
    _json(
        operational / "evaluation/postflight_semantic_audit.json",
        {"status": "passed"},
    )
    _, hashes["operational"] = _manifest(
        operational,
        "operational",
        (
            "metrics/paired_two_stage_bootstrap.csv",
            "evaluation/postflight_semantic_audit.json",
        ),
        canonical=True,
        scientific_eligible=True,
        capacity_receipt={
            "sha256": hashes["capacity"],
            "artifact_inventory_verified": True,
        },
    )
    _gate_receipt(
        operational,
        "operational",
        hashes["operational"],
        postflight_semantic_audit_sha256=_sha(
            operational / "evaluation/postflight_semantic_audit.json"
        ),
    )
    paths["operational"] = operational

    return paths, hashes, tmp_path / "deliverables"


def test_builds_atomic_complete_hashed_bundle(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    output = deliverables / "paper-bundle"
    completed = bundle.build_bundle(
        paths, hashes, output, allowed_output_root=deliverables
    )
    assert completed == output.resolve()
    manifest = json.loads((completed / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["smoke"] is False
    assert manifest["contract"]["sealed_2025_target_opened"] is False
    actual = {
        str(path.relative_to(completed)): _sha(path)
        for path in completed.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert manifest["artifact_sha256"] == actual
    assert len(actual) == 15
    assert len(pd.read_csv(completed / "tables/continuous_crps_skill_by_lead.csv")) == 14
    assert len(pd.read_csv(completed / "tables/categorical_rps_comparison.csv")) == 7
    assert len(pd.read_csv(completed / "tables/categorical_rps_by_lead.csv")) == 42
    pairwise = pd.read_csv(
        completed / "tables/categorical_location_vs_combined_by_lead.csv"
    )
    assert len(pairwise) == 7
    lead_directions = pairwise.loc[pairwise.lead_week.between(1, 6)].set_index(
        "lead_week"
    ).interval_interpretation.to_dict()
    assert lead_directions == {
        1: "location_spread_lower_rps",
        2: "interval_crosses_zero",
        3: "pbc_combined_lower_rps",
        4: "pbc_combined_lower_rps",
        5: "pbc_combined_lower_rps",
        6: "pbc_combined_lower_rps",
    }
    assert len(pd.read_csv(completed / "tables/capacity_ablation.csv")) == 5
    assert len(pd.read_csv(completed / "tables/loss_ablation.csv")) == 5
    for stem in (
        "continuous_crps_skill_by_lead",
        "categorical_rps_comparison",
        "capacity_and_loss_ablation",
    ):
        assert (completed / f"figures/{stem}.pdf").read_bytes().startswith(b"%PDF")
        with Image.open(completed / f"figures/{stem}.png") as image:
            assert image.width >= 1500
            assert image.height >= 800
    claims = json.loads((completed / "claim_boundaries.json").read_text())
    assert {row["id"] for row in claims["mandatory_boundaries"]} >= {
        "development_reuse",
        "later_era_retrospective",
        "sealed_final_year",
        "bias",
    }
    pair_claim = claims["numeric_claims"][
        "location_spread_vs_combined_pbc_by_lead"
    ]
    assert [row["interval_interpretation"] for row in pair_claim["lead_results"]] == [
        "location_spread_lower_rps",
        "interval_crosses_zero",
        "pbc_combined_lower_rps",
        "pbc_combined_lower_rps",
        "pbc_combined_lower_rps",
        "pbc_combined_lower_rps",
    ]
    assert "two year" in pair_claim["inference_status"]
    receipts = json.loads((completed / "source_receipts.json").read_text())
    assert receipts["cross_source_bindings_verified"] == {
        "capacity_to_postselection_development": True,
        "capacity_to_operational_era": True,
        "neural_and_pbc_to_shared_categorical": True,
    }


def test_rejects_tampered_declared_artifact_before_output(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    with (paths["capacity"] / "metrics/validation_summary.csv").open("a") as stream:
        stream.write("tampered\n")
    output = deliverables / "tampered"
    with pytest.raises(bundle.BundleContractError, match="artifact checksum differs"):
        bundle.build_bundle(paths, hashes, output, allowed_output_root=deliverables)
    assert not output.exists()


def test_rejects_smoke_even_when_new_manifest_hash_is_supplied(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    manifest_path = paths["categorical"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["smoke"] = True
    _json(manifest_path, manifest)
    hashes["categorical"] = _sha(manifest_path)
    receipt_path = paths["categorical"] / "slurm_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest_sha256"] = hashes["categorical"]
    _json(receipt_path, receipt)
    with pytest.raises(bundle.BundleContractError, match="smoke evidence"):
        bundle.build_bundle(
            paths,
            hashes,
            deliverables / "smoke",
            allowed_output_root=deliverables,
        )


def test_rejects_stale_slurm_receipt(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    receipt_path = paths["operational"] / "slurm_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest_sha256"] = "0" * 64
    _json(receipt_path, receipt)
    with pytest.raises(bundle.BundleContractError, match="Slurm receipt field"):
        bundle.build_bundle(
            paths,
            hashes,
            deliverables / "stale-receipt",
            allowed_output_root=deliverables,
        )


def test_rejects_superseded_operational_v2_receipt(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    receipt_path = paths["operational"] / "slurm_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["post_run_audit_version"] = "operational_era_postrun_v2"
    _json(receipt_path, receipt)
    output = deliverables / "superseded-operational-receipt"
    with pytest.raises(bundle.BundleContractError, match="post_run_audit_version"):
        bundle.build_bundle(
            paths,
            hashes,
            output,
            allowed_output_root=deliverables,
        )
    assert not output.exists()


def test_rejects_sealed_year_path_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "run_2025" / "manifest.json"
    with pytest.raises(bundle.BundleContractError, match="sealed year"):
        bundle.verify_source(
            "neural", forbidden, expected_manifest_sha256="0" * 64
        )


def test_store_path_gate_allows_archive_range_label() -> None:
    manifest = {
        "contract": {
            "sealed_2025_target_opened": False,
            "final_2025_store_opened": False,
            "forecast_year_store_whitelist": [2022, 2023, 2024],
            "sealed_unopened_years": [2025],
        },
        "forecast_store_paths": [
            "/archive/model-run__2020_2025_ens50/tp/common_1p5/2024.zarr"
        ],
        "opened_store_paths_exact": [
            "/archive/model-run__2020_2025_ens50/tp/common_1p5/2024.zarr"
        ],
    }
    bundle._verify_sealed_year_contract("operational", manifest)


def test_store_path_gate_rejects_actual_sealed_year_leaf() -> None:
    manifest = {
        "contract": {
            "sealed_2025_target_opened": False,
            "final_2025_store_opened": False,
            "forecast_year_store_whitelist": [2022, 2023, 2024],
            "sealed_unopened_years": [2025],
        },
        "forecast_store_paths": [
            "/archive/model-run__2020_2025_ens50/tp/common_1p5/2025.zarr"
        ],
    }
    with pytest.raises(bundle.BundleContractError, match="store path references 2025"):
        bundle._verify_sealed_year_contract("operational", manifest)


def test_refuses_existing_or_outside_output(
    paper_sources: tuple[dict[str, Path], dict[str, str], Path]
) -> None:
    paths, hashes, deliverables = paper_sources
    existing = deliverables / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(bundle.BundleContractError, match="overwrite"):
        bundle.build_bundle(paths, hashes, existing, allowed_output_root=deliverables)
    with pytest.raises(bundle.BundleContractError, match="must be under"):
        bundle.build_bundle(
            paths,
            hashes,
            deliverables.parent / "outside",
            allowed_output_root=deliverables,
        )
