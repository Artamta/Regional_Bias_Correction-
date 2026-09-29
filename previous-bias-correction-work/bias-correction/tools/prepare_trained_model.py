"""Create the reusable trained-model bundle and training-loss figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

import fuxi_imerg_experiment as experiment


TOOLS_ROOT = Path(__file__).resolve().parent
HERE = TOOLS_ROOT.parent
DEFAULT_RESULTS = HERE / "results" / "fuxi_imerg_jjas_5yr"
DEFAULT_OUTPUT = HERE / "trained_model" / "fuxi_imerg_2014_2018_v1"
COLORS = {42: "#0072B2", 43: "#E69F00", 44: "#009E73"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def daily_climatology():
    dates = []
    values = []
    support = latitude = longitude = None
    for year in experiment.TRAIN_YEARS:
        path = experiment.IMERG_DAILY / f"{year}.zarr"
        with xr.open_zarr(path, consolidated=True) as dataset:
            dates.append(np.asarray(dataset.time.values, dtype="datetime64[D]"))
            values.append(
                np.asarray(dataset.observation.load().values, dtype=np.float32)
            )
            fraction = experiment.collapse_fraction(dataset, (27, 27))
            if support is None:
                support = fraction > 0.0
                latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
                longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
            elif not np.array_equal(support, fraction > 0.0):
                raise ValueError("IMERG support changes between training years")
    dates = np.concatenate(dates)
    values = np.concatenate(values)
    climatology = experiment.build_training_climatology(dates, values, support)
    return climatology, latitude, longitude, support


def plot_losses(run: Path, output: Path) -> pd.DataFrame:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    summary = []
    for checkpoint in sorted(run.glob("models/seed_*/checkpoints/best.pt")):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seed = int(saved["seed"])
        history_path = checkpoint.parents[1] / "logs" / "training_history.csv"
        history = pd.read_csv(history_path)
        best_epoch = int(saved["best_epoch"])
        color = COLORS[seed]
        axes[0].plot(
            history.epoch,
            history.train_loss,
            color=color,
            linewidth=1.8,
            label=f"Seed {seed}",
        )
        axes[1].plot(
            history.epoch,
            history.validation_loss,
            color=color,
            linewidth=1.8,
            label=f"Seed {seed}",
        )
        axes[1].scatter(
            best_epoch,
            saved["best_validation_loss"],
            color=color,
            marker="*",
            s=90,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        summary.append(
            {
                "seed": seed,
                "epochs_run": len(history),
                "best_epoch_zero_based": best_epoch,
                "best_validation_loss": float(saved["best_validation_loss"]),
                "final_train_loss": float(history.train_loss.iloc[-1]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )

    for label, axis in zip(("(a) Training", "(b) Validation"), axes):
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Area-weighted Smooth-L1 loss")
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[1].text(
        0.98,
        0.04,
        "Stars mark selected checkpoints",
        transform=axes[1].transAxes,
        ha="right",
        color="0.35",
        fontsize=8.5,
    )
    figure.suptitle(
        "Residual U-Net optimization\nTrain: 2014–2018; validation: 2019",
        fontsize=13,
        fontweight="semibold",
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(summary).sort_values("seed")


def make_bundle(run: Path, output: Path, training: pd.DataFrame) -> None:
    normalization = json.loads((run / "normalization.json").read_text())
    climatology, latitude, longitude, support = daily_climatology()
    models = []
    for row in training.itertuples(index=False):
        saved = torch.load(row.checkpoint, map_location="cpu", weights_only=True)
        models.append(
            {
                "seed": int(saved["seed"]),
                "best_epoch": int(saved["best_epoch"]),
                "best_validation_loss": float(saved["best_validation_loss"]),
                "checkpoint_sha256": row.checkpoint_sha256,
                "model_state_dict": saved["model_state_dict"],
            }
        )
    bundle = {
        "format_version": "fuxi-imerg-residual-unet-v1",
        "source_run": str(run.resolve()),
        "architecture": {
            "name": "ResidualUNet",
            "input_channels": 9,
            "base_channels": 16,
            "dropout": 0.1,
            "parameter_count": 110545,
        },
        "training_years": [2014, 2015, 2016, 2017, 2018],
        "validation_year": 2019,
        "target": "IMERG Final V07B weekly rainfall",
        "normalization": normalization,
        "latitude": torch.from_numpy(latitude.copy()),
        "longitude": torch.from_numpy(longitude.copy()),
        "support": torch.from_numpy(support.copy()),
        "daily_imerg_climatology": torch.from_numpy(climatology.copy()),
        "models": models,
    }
    torch.save(bundle, output / "residual_unet_ensemble.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = args.run
    if run is None:
        runs = sorted(DEFAULT_RESULTS.glob("full_*"))
        if not runs:
            raise FileNotFoundError("no full experiment result found")
        run = runs[-1]
    run = run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    training = plot_losses(run, output / "training_loss")
    training.to_csv(output / "training_summary.csv", index=False)
    make_bundle(run, output, training)
    shutil.copy2(TOOLS_ROOT / "trained_model.py", output / "trained_model.py")
    shutil.copytree(run / "code", output / "code", dirs_exist_ok=True)

    bundle = output / "residual_unet_ensemble.pt"
    summary = {
        "source_run": str(run),
        "bundle": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "seeds": training.seed.tolist(),
        "loss_figure_png": str(output / "training_loss.png"),
        "loss_figure_pdf": str(output / "training_loss.pdf"),
    }
    summary["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
