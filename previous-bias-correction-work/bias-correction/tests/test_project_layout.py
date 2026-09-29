"""Regression checks for the compact workspace layout."""

from __future__ import annotations

import re
from pathlib import Path

import project_paths


ROOT = Path(__file__).resolve().parents[1]


def test_root_is_code_free_and_plan_contains_documents_only() -> None:
    assert list(ROOT.glob("*.py")) == []
    plan_entries = sorted((ROOT / "plan").iterdir())
    assert plan_entries
    assert all(path.is_file() and path.suffix == ".md" for path in plan_entries)


def test_canonical_project_roots_resolve_to_the_workspace() -> None:
    assert project_paths.PROJECT_ROOT == ROOT
    assert project_paths.SOURCE_ROOT == ROOT / "src"
    assert project_paths.EVALUATE_ROOT == ROOT / "evaluate"
    assert project_paths.RESULTS_ROOT == ROOT / "results"
    assert project_paths.CACHE_ROOT == ROOT / "cache"
    assert project_paths.NEURAL_ADAPTER_SRC == ROOT.parent / "neural_adapter" / "src"


def test_every_slurm_python_target_exists_in_src_or_evaluate() -> None:
    launchers = sorted((ROOT / "slurm").glob("*.sbatch"))
    assert launchers
    required_launchers = {
        "run_global_pretrain_india_smoke.sbatch",
        "run_global_pretrain_india_full.sbatch",
    }
    assert required_launchers <= {path.name for path in launchers}
    target_pattern = re.compile(r"(?:src|evaluate)/[a-z0-9_]+\.py")
    for launcher in launchers:
        targets = set(target_pattern.findall(launcher.read_text(encoding="utf-8")))
        assert targets, f"no compact-layout Python target in {launcher.name}"
        for target in targets:
            assert (ROOT / target).is_file(), f"missing {target} from {launcher.name}"
