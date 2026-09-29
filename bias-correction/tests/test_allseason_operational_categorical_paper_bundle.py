"""Contracts for the operational-era categorical paper supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import build_allseason_operational_categorical_paper_bundle as bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _source_tables(root: Path) -> None:
    methods = bundle.EXPECTED_METHODS
    lead_scores = {
        "raw_fuxi_categorical": (0.25, 0.25, 0.25, 0.25, 0.25, 0.25),
        "base_42k": (0.13, 0.18, 0.20, 0.21, 0.22, 0.23),
        "debias_plus_plus": (0.16, 0.19, 0.205, 0.215, 0.225, 0.235),
        "persistence_plus_plus": (0.15, 0.185, 0.195, 0.20, 0.205, 0.21),
        "pbc_combined": (0.148, 0.184, 0.194, 0.199, 0.204, 0.209),
    }
    pooled_rows: list[dict] = []
    weekwise_rows: list[dict] = []
    for family, scale in (("quintile", 1.0), ("semidecile", 0.9)):
        for method in methods:
            scores = tuple(scale * value for value in lead_scores[method])
            pooled = sum(scores) / 6.0
            pooled_rows.append(
                {
                    "family": family,
                    "method": method,
                    "method_label": bundle.METHOD_LABELS[method],
                    "score_contract": bundle.SCORING_CONTRACT,
                    "rps": pooled,
                    "rpss_vs_training_empirical_climatology": 1.0 - pooled / 0.24,
                    "n_initializations": 6,
                }
            )
            for lead_week, score in enumerate(scores, start=1):
                weekwise_rows.append(
                    {
                        "family": family,
                        "lead_week": lead_week,
                        "method": method,
                        "method_label": bundle.METHOD_LABELS[method],
                        "score_contract": bundle.SCORING_CONTRACT,
                        "rps": score,
                        "rpss_vs_training_empirical_climatology": 1.0 - score / 0.24,
                        "n_initializations": 6,
                    }
                )
    _csv(root / "metrics/pooled_rps.csv", pooled_rows)
    _csv(root / "metrics/weekwise_rps.csv", weekwise_rows)

    pooled_lookup = {(row["family"], row["method"]): row["rps"] for row in pooled_rows}
    lead_lookup = {
        (row["family"], row["lead_week"], row["method"]): row["rps"]
        for row in weekwise_rows
    }
    bootstrap_rows: list[dict] = []
    for family, metric in bundle.FAMILY_METRICS.items():
        for lead_week in range(0, 7):
            neural = (
                pooled_lookup[(family, "base_42k")]
                if lead_week == 0
                else lead_lookup[(family, lead_week, "base_42k")]
            )
            for baseline in (
                "raw_fuxi_categorical",
                "persistence_plus_plus",
                "pbc_combined",
            ):
                baseline_score = (
                    pooled_lookup[(family, baseline)]
                    if lead_week == 0
                    else lead_lookup[(family, lead_week, baseline)]
                )
                effect = 1.0 - neural / baseline_score
                if baseline == "raw_fuxi_categorical":
                    half_width = 0.02
                elif lead_week == 1:
                    half_width = 0.025
                else:
                    half_width = 0.15
                bootstrap_rows.append(
                    {
                        "score_contract": bundle.SCORING_CONTRACT,
                        "metric": metric,
                        "metric_contract": bundle.SCORING_CONTRACT,
                        "lead_scope": "W1-W6" if lead_week == 0 else f"W{lead_week}",
                        "lead_week": lead_week,
                        "method": "base_42k",
                        "baseline": baseline,
                        "method_score": neural,
                        "baseline_score": baseline_score,
                        "score_reduction_fraction": effect,
                        "ci_lower_95": effect - half_width,
                        "ci_upper_95": effect + half_width,
                        "bootstrap_probability_improvement": 0.99 if effect > half_width else 0.5,
                        "threshold_retained_scoring_weight": float("nan"),
                        "bootstrap_samples": 100,
                        "block_length_initializations": 13,
                        "bootstrap_seed": 20260823,
                        "bootstrap_scheme": "two_stage_year_resample_then_within_year_circular_13_init_blocks",
                        "source_year_clusters": 3,
                        "small_year_cluster_limitation": True,
                    }
                )
    _csv(root / "metrics/paired_two_stage_bootstrap.csv", bootstrap_rows)


def _bind_manifest_and_gates(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = root / "manifest.json"
    manifest_sha = _sha(manifest_path)
    semantic_path = root / "evaluation/postflight_semantic_audit.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["manifest_sha256"] = manifest_sha
    _json(semantic_path, semantic)
    semantic_sha = _sha(semantic_path)
    gate_path = root / "slurm_gate_receipt.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["manifest_sha256"] = manifest_sha
    gate["postflight_semantic_audit_sha256"] = semantic_sha
    _json(gate_path, gate)
    monkeypatch.setattr(bundle, "CANONICAL_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(bundle, "CANONICAL_SEMANTIC_SHA256", semantic_sha)
    monkeypatch.setattr(bundle, "CANONICAL_GATE_SHA256", _sha(gate_path))


@pytest.fixture
def synthetic_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "full_synthetic"
    root.mkdir()
    _source_tables(root)
    (root / "README.md").write_text("synthetic accepted full source\n", encoding="utf-8")
    declared = {
        "README.md",
        "metrics/pooled_rps.csv",
        "metrics/weekwise_rps.csv",
        "metrics/paired_two_stage_bootstrap.csv",
    }
    snapshot = {"code/src/driver.py": "a" * 64}
    initializations = [
        "2022-01-03",
        "2022-01-06",
        "2023-01-02",
        "2023-01-05",
        "2024-01-01",
        "2024-01-04",
    ]
    manifest = {
        "experiment": bundle.SOURCE_EXPERIMENT,
        "audit_contract_version": bundle.SOURCE_CONTRACT,
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "canonical": True,
        "scientific_eligible": True,
        "scientific_status": "synthetic post-hoc 2022-2024 retrospective",
        "evidence_label": "synthetic post-hoc 2022-2024 retrospective; no retraining or selection",
        "scoring_contract_version": bundle.SCORING_CONTRACT,
        "methods": list(bundle.EXPECTED_METHODS),
        "seeds": [42, 43, 44],
        "seed_handling": {
            "headline_aggregation": "arithmetic mean of per-seed scores on identical case/lead rows",
            "probabilities_averaged_across_seeds": False,
        },
        "contract": {
            "sealed_2025_target_opened": False,
            "final_2025_store_opened": False,
            "forecast_year_store_whitelist": [2022, 2023, 2024],
            "verification_year_store_whitelist": [2022, 2023, 2024],
            "maximum_verification_date": "2024-12-29",
        },
        "cases": {
            "evaluated_counts": {"2022": 2, "2023": 2, "2024": 2},
            "evaluated_initializations": initializations,
        },
        "opened_input_paths_exact": [f"/data/{year}.zarr" for year in (2022, 2023, 2024)],
        "opened_store_paths_exact": [f"/data/{year}.zarr" for year in (2022, 2023, 2024)],
        "source_snapshot_sha256": snapshot,
        "artifact_sha256": {relative: _sha(root / relative) for relative in sorted(declared)},
    }
    _json(root / "manifest.json", manifest)
    _json(
        root / "evaluation/postflight_semantic_audit.json",
        {
            "experiment": bundle.SOURCE_EXPERIMENT,
            "post_run_audit_version": "operational_categorical_postrun_v1",
            "status": "passed",
            "mode": "full",
            "manifest_sha256": "pending",
            "case_count": 6,
            "bootstrap_rows_recomputed": 42,
            "bootstrap_draws_recomputed": 100,
            "aggregate_tables_recomputed": True,
            "headline_neural_score_average_recomputed": True,
            "smoke_full_exact_source_snapshot_identity": True,
            "lag_receipts_rechecked": True,
            "no_2025_year_store_component": True,
            "source_snapshot_sha256": snapshot,
        },
    )
    _json(
        root / "slurm_gate_receipt.json",
        {
            "experiment": bundle.SOURCE_EXPERIMENT,
            "audit_contract_version": bundle.SOURCE_CONTRACT,
            "post_run_audit_version": "operational_categorical_postrun_v1",
            "gate_status": "passed",
            "mode": "full",
            "manifest_sha256": "pending",
            "postflight_semantic_audit_sha256": "pending",
            "job_id": "12345",
            "node": "cn1",
            "partition": "gpu_prio",
            "completed_utc": "2026-08-22T20:29:33+00:00",
            "source_snapshot_sha256": snapshot,
        },
    )
    monkeypatch.setattr(bundle, "CANONICAL_SOURCE_DIR", root)
    monkeypatch.setattr(bundle, "EXPECTED_DECLARED_ARTIFACTS", frozenset(declared))
    monkeypatch.setattr(bundle, "EXPECTED_CASES", 6)
    monkeypatch.setattr(bundle, "EXPECTED_YEAR_COUNTS", {"2022": 2, "2023": 2, "2024": 2})
    monkeypatch.setattr(bundle, "EXPECTED_BOOTSTRAP_ROWS", 42)
    monkeypatch.setattr(bundle, "EXPECTED_BOOTSTRAP_DRAWS", 100)
    _bind_manifest_and_gates(root, monkeypatch)
    return root


def test_constants_pin_only_the_accepted_full_result() -> None:
    assert bundle.CANONICAL_SOURCE_DIR.name == "full_20260822T203000Z"
    assert bundle.CANONICAL_MANIFEST_SHA256 == "cb2657aa19313b5efccb35250d555df1302e1f6ad5d16c094e343d8a3f383c1d"
    assert bundle.CANONICAL_SEMANTIC_SHA256 == "00d2db2765c3eeb969126671617ed77b794df9169a0768011e4ebf6564567c6c"
    assert bundle.CANONICAL_GATE_SHA256 == "cec53cbfb094dfa82010ec6c7521a596b848bf1cb21cf3189cc384c43791289c"
    assert "smoke" not in str(bundle.CANONICAL_SOURCE_DIR)


def test_build_bundle_writes_receipted_tables_and_publication_figure(
    synthetic_source: Path, tmp_path: Path
) -> None:
    allowed = tmp_path / "deliverables"
    output = allowed / "fuxi_allseason_operational_categorical_paper_test"
    completed = bundle.build_bundle(
        synthetic_source,
        bundle.CANONICAL_MANIFEST_SHA256,
        output,
        allowed_output_root=allowed,
    )
    manifest = json.loads((completed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["source"]["manifest_sha256"] == bundle.CANONICAL_MANIFEST_SHA256
    physical = {
        str(path.relative_to(completed))
        for path in completed.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert physical == set(manifest["artifact_sha256"])
    for relative, digest in manifest["artifact_sha256"].items():
        assert _sha(completed / relative) == digest
    quintile = pd.read_csv(completed / "tables/quintile_neural_contrasts.csv")
    assert len(quintile) == 21
    assert set(quintile.loc[quintile.lead_week.between(2, 6) & quintile.baseline.ne("raw_fuxi_categorical"), "inference"]) == {"unresolved_95pct_interval"}
    with Image.open(completed / "figures/operational_categorical_transfer.png") as image:
        assert image.width >= 1800
        assert image.height >= 1300
        assert min(image.info.get("dpi", (0, 0))) >= 299
    assert (completed / "figures/operational_categorical_transfer.pdf").read_bytes().startswith(b"%PDF")


def test_source_artifact_tamper_fails_before_output(
    synthetic_source: Path, tmp_path: Path
) -> None:
    with (synthetic_source / "metrics/weekwise_rps.csv").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    allowed = tmp_path / "deliverables"
    output = allowed / "fuxi_allseason_operational_categorical_paper_tampered"
    with pytest.raises(bundle.PaperBundleError, match="artifact hash differs"):
        bundle.build_bundle(
            synthetic_source,
            bundle.CANONICAL_MANIFEST_SHA256,
            output,
            allowed_output_root=allowed,
        )
    assert not output.exists()


def test_smoke_manifest_is_rejected_even_when_rebound(
    synthetic_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = synthetic_source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smoke"] = True
    _json(manifest_path, manifest)
    _bind_manifest_and_gates(synthetic_source, monkeypatch)
    with pytest.raises(bundle.PaperBundleError, match="source field 'smoke' differs"):
        bundle.verify_source(
            synthetic_source,
            expected_manifest_sha256=bundle.CANONICAL_MANIFEST_SHA256,
        )


def test_sealed_2025_store_leaf_is_rejected_when_rebound(
    synthetic_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = synthetic_source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["opened_store_paths_exact"].append("/archive/model_2020_2025/2025.zarr")
    _json(manifest_path, manifest)
    _bind_manifest_and_gates(synthetic_source, monkeypatch)
    with pytest.raises(bundle.PaperBundleError, match="opened 2025 store"):
        bundle.verify_source(
            synthetic_source,
            expected_manifest_sha256=bundle.CANONICAL_MANIFEST_SHA256,
        )


def test_cli_requires_explicit_manifest_hash() -> None:
    with pytest.raises(SystemExit):
        bundle.build_parser().parse_args([])
