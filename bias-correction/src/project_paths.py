"""Canonical filesystem roots for the bias-correction workspace.

Code must distinguish its own source directory from the project root.  This
module keeps result, cache, documentation, and sibling-package paths stable
after the source tree moved out of the repository root.
"""

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
CLEAN_ROOT = PROJECT_ROOT.parent

EVALUATE_ROOT = PROJECT_ROOT / "evaluate"
TOOLS_ROOT = PROJECT_ROOT / "tools"
RESULTS_ROOT = PROJECT_ROOT / "results"
CACHE_ROOT = PROJECT_ROOT / "cache"
DOCS_ROOT = PROJECT_ROOT / "docs"
SLURM_ROOT = PROJECT_ROOT / "slurm"
PRESENTATION_ROOT = PROJECT_ROOT / "presentation"

NEURAL_ADAPTER_SRC = CLEAN_ROOT / "neural_adapter" / "src"
