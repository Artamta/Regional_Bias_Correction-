from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluate"
    / "plot_jjas_average_acc_curve.py"
)
SPEC = importlib.util.spec_from_file_location("plot_jjas_average_acc_curve", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
jjas_acc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jjas_acc)


def test_contract_uses_correct_label_and_no_imd_method_curve() -> None:
    assert jjas_acc.METHOD_LABELS["corrected"] == "Corrected Forecast"
    assert set(jjas_acc.METHOD_ORDER) == {"raw_fuxi", "log_bias", "corrected"}
    assert "imd" not in jjas_acc.METHOD_ORDER
    assert "No IMD self-curve" in jjas_acc.REFERENCE_NOTE


def test_real_locked_csvs_generate_fresh_png_pdf_and_provenance(
    tmp_path: Path,
) -> None:
    source_paths = jjas_acc.locked_acc._metric_paths(jjas_acc.DEFAULT_RESULT_DIR)
    before = {
        name: jjas_acc.sha256_file(path) for name, path in source_paths.items()
    }
    data = jjas_acc.locked_acc.load_acc_figure_data(jjas_acc.DEFAULT_RESULT_DIR)
    output = jjas_acc.generate_figure(data, tmp_path / "new_figure")

    png = output / f"{jjas_acc.FIGURE_STEM}.png"
    pdf = output / f"{jjas_acc.FIGURE_STEM}.pdf"
    assert png.is_file() and png.stat().st_size > 50_000
    assert pdf.is_file() and pdf.stat().st_size > 10_000
    assert before == {
        name: jjas_acc.sha256_file(path) for name, path in source_paths.items()
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_arrays_opened"] is False
    assert manifest["metrics_recomputed"] is False
    assert manifest["model_refit"] is False
    assert manifest["imd_role"].startswith("verification reference")
    assert manifest["method_labels"]["corrected"] == "Corrected Forecast"
    assert "p-value" in manifest["unused_inference_fields"]
    assert set(manifest["figures"]) == {png.name, pdf.name}


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "belongs_to_user"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    data = jjas_acc.locked_acc.load_acc_figure_data(jjas_acc.DEFAULT_RESULT_DIR)
    with pytest.raises(FileExistsError, match="fresh JJAS ACC"):
        jjas_acc.generate_figure(data, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
