"""Train the small global probabilistic TP adapter.

The cache contains one sample per FuXi initialization.  Nothing in this file
opens the 13 TB source archive; preprocessing is intentionally separate.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import EXPERIMENT, PATHS, serializable_config
from model import TPProbUNet


LICENSE_WARNING = (
    "FuXi forecasts and FuXi-derived weights are research-only here. "
    "Do not submit them to a competition without written FuXi permission."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    return device


def _read_years(init_values: np.ndarray) -> np.ndarray:
    values = np.asarray(init_values)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64) // 10_000
    return np.asarray([int(str(value)[:4]) for value in values], dtype=np.int16)


SPLIT_NAMES = ("train", "validation", "test")


def _read_npz_provenance(source: Any) -> dict[str, Any] | None:
    if "provenance_json" not in source:
        return None
    try:
        provenance = json.loads(str(source["provenance_json"].item()))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("NPZ provenance_json is not valid scalar JSON") from error
    if not isinstance(provenance, dict):
        raise ValueError("NPZ provenance_json must decode to an object")
    return provenance


def _split_years(
    init_values: np.ndarray,
    split_values: np.ndarray,
) -> dict[str, tuple[int, ...]]:
    """Return years represented by each explicit split, rejecting leakage."""

    years = _read_years(init_values)
    labels = np.asarray(split_values).astype(str)
    if labels.shape != years.shape:
        raise ValueError("NPZ split must contain one label per initialization")
    unexpected = sorted(set(labels).difference(SPLIT_NAMES))
    if unexpected:
        raise ValueError(f"NPZ split contains unexpected labels {unexpected}")
    for year in np.unique(years):
        year_labels = set(labels[years == year])
        if len(year_labels) != 1:
            raise ValueError(
                f"calendar year {int(year)} is assigned to multiple NPZ splits"
            )
    return {
        name: tuple(int(value) for value in np.unique(years[labels == name]))
        for name in SPLIT_NAMES
    }


def _validate_declared_split_years(
    actual: dict[str, tuple[int, ...]],
    provenance: dict[str, Any] | None,
) -> None:
    if provenance is None or "split_years" not in provenance:
        return
    declared = provenance["split_years"]
    if not isinstance(declared, dict):
        raise ValueError("NPZ provenance split_years must be an object")
    for name in SPLIT_NAMES:
        if name not in declared:
            raise ValueError(f"NPZ provenance split_years is missing {name!r}")
        try:
            declared_years = tuple(sorted({int(value) for value in declared[name]}))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"NPZ provenance split_years[{name!r}] must contain years"
            ) from error
        if declared_years != actual[name]:
            raise ValueError(
                "NPZ split labels disagree with provenance split_years for "
                f"{name}: labels={list(actual[name])}, declared={list(declared_years)}"
            )


class PreparedCases(Dataset):
    """Read selected cases from either the compact Zarr cache or an NPZ file."""

    def __init__(
        self,
        path: Path,
        years: tuple[int, ...] | None = None,
        *,
        split: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._npz: Any | None = None
        self._zarr: Any | None = None
        self.split_name: str | None = None
        self.split_years: dict[str, tuple[int, ...]] | None = None
        if self.path.suffix == ".npz":
            self._npz = np.load(self.path, allow_pickle=False)
            init_values = self._npz["init_dates"]
        else:
            try:
                import zarr
            except ImportError as error:
                raise RuntimeError("zarr is required to read the prepared cache") from error
            self._zarr = zarr.open_group(str(self.path), mode="r")
            status = self._zarr.attrs.get("status", "")
            if status and status != "complete":
                raise RuntimeError(f"cache status is {status!r}, not 'complete'")
            init_key = "init_yyyymmdd" if "init_yyyymmdd" in self._zarr else "init_dates"
            init_values = self._zarr[init_key][:]

        available_years = _read_years(init_values)
        self.available_years = tuple(
            int(value) for value in np.unique(available_years)
        )
        if self._npz is not None and "split" in self._npz:
            labels = np.asarray(self._npz["split"]).astype(str)
            self.split_years = _split_years(init_values, labels)
            _validate_declared_split_years(
                self.split_years, _read_npz_provenance(self._npz)
            )
            if split is None:
                if years is None:
                    raise ValueError(
                        "an explicit split or its exact years are required for this NPZ"
                    )
                requested = tuple(sorted({int(value) for value in years}))
                matches = [
                    name
                    for name, split_years in self.split_years.items()
                    if split_years == requested
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"requested years {requested} do not exactly match one NPZ split"
                    )
                split = matches[0]
            if split not in SPLIT_NAMES:
                raise ValueError(f"unknown prepared-cache split {split!r}")
            actual_years = self.split_years[split]
            if years is not None:
                requested = tuple(sorted({int(value) for value in years}))
                if requested != actual_years:
                    raise ValueError(
                        f"requested years {requested} disagree with NPZ split "
                        f"{split!r} years {actual_years}"
                    )
            self.split_name = split
            self.indices = np.flatnonzero(labels == split)
        else:
            if split is not None:
                raise ValueError("prepared cache has no explicit split labels")
            if years is None:
                raise ValueError("years are required for a legacy prepared cache")
            self.indices = np.flatnonzero(
                np.isin(available_years, np.asarray(years))
            )
        if not len(self.indices):
            selector = f"split {split!r}" if split is not None else f"years {years}"
            raise ValueError(f"no prepared cases found for {selector}")
        self.selected_years = tuple(
            int(value) for value in np.unique(available_years[self.indices])
        )

        source = self._npz if self._npz is not None else self._zarr
        assert source is not None
        required = ("features", "p0", "target")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"prepared cache is missing {missing}")
        if source["features"].shape[1:3] != (2, 18):
            raise ValueError("features must have shape [case, 2, 18, lat, lon]")
        if source["p0"].shape[1:3] != (2, 5):
            raise ValueError("p0 must have shape [case, 2, 5, lat, lon]")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        source = self._npz if self._npz is not None else self._zarr
        assert source is not None
        features = np.asarray(source["features"][index], dtype=np.float32)
        anchor = np.asarray(source["p0"][index], dtype=np.float32)
        target = np.asarray(source["target"][index], dtype=np.int64)
        if not np.isfinite(features).all() or not np.isfinite(anchor).all():
            raise ValueError(f"case {index} contains non-finite model inputs")
        return (
            torch.from_numpy(features),
            torch.from_numpy(anchor),
            torch.from_numpy(target),
        )

    def metadata(self) -> dict[str, Any]:
        if self._npz is not None:
            metadata: dict[str, Any] = {
                "feature_names": (
                    [str(value) for value in self._npz["feature_names"].tolist()]
                    if "feature_names" in self._npz
                    else list(EXPERIMENT.feature_names)
                ),
                "tp_quantile_mean": (
                    np.asarray(self._npz["tp_quantile_mean"], dtype=np.float32).tolist()
                    if "tp_quantile_mean" in self._npz
                    else []
                ),
                "tp_quantile_std": (
                    np.asarray(self._npz["tp_quantile_std"], dtype=np.float32).tolist()
                    if "tp_quantile_std" in self._npz
                    else []
                ),
                "source_store": "",
                "selected_years": list(self.selected_years),
            }
            if self.split_name is not None:
                metadata["selected_split"] = self.split_name
            if self.split_years is not None:
                metadata["split_years"] = {
                    name: list(values) for name, values in self.split_years.items()
                }
            provenance = _read_npz_provenance(self._npz)
            if provenance is not None:
                metadata["provenance"] = provenance
                metadata["source_store"] = str(provenance.get("fuxi_source", ""))
            if "cache_contract_sha256" in self._npz:
                metadata["cache_contract_sha256"] = str(
                    self._npz["cache_contract_sha256"].item()
                )
            return metadata
        assert self._zarr is not None
        attrs = dict(self._zarr.attrs)
        metadata = {
            "feature_names": list(attrs.get("feature_names", EXPERIMENT.feature_names)),
            "tp_quantile_mean": attrs.get("tp_quantile_mean", []),
            "tp_quantile_std": attrs.get("tp_quantile_std", []),
            "source_store": attrs.get("source_store", ""),
            "selected_years": list(self.selected_years),
        }
        if "cache_contract_sha256" in attrs:
            metadata["cache_contract_sha256"] = str(attrs["cache_contract_sha256"])
        return metadata


class SyntheticCases(Dataset):
    """Small deterministic data used only to test the complete training path."""

    def __init__(self, cases: int, seed: int, height: int = 17, width: int = 24) -> None:
        rng = np.random.default_rng(seed)
        anchor = rng.dirichlet(np.ones(5), size=(cases, 2, height, width))
        self.p0 = torch.from_numpy(anchor.transpose(0, 1, 4, 2, 3).astype(np.float32))
        features = rng.normal(size=(cases, 2, 18, height, width)).astype(np.float32)
        features[:, :, :5] = np.log(np.maximum(self.p0.numpy(), 1.0e-8))
        self.features = torch.from_numpy(features)
        signal = features[:, :, 5] + 0.35 * features[:, :, 7]
        bins = np.quantile(signal, (0.2, 0.4, 0.6, 0.8))
        self.target = torch.from_numpy(np.digitize(signal, bins).astype(np.int64))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[item], self.p0[item], self.target[item]

    def metadata(self) -> dict[str, Any]:
        return {"feature_names": list(EXPERIMENT.feature_names)}


def training_year_partitions(
    train_data: PreparedCases,
    validation_data: PreparedCases,
) -> dict[str, list[int]]:
    """Describe the actual cache partitions used by a training run."""

    if train_data.path != validation_data.path:
        raise ValueError("training and validation datasets must use the same cache")
    if train_data.split_years is not None or validation_data.split_years is not None:
        if train_data.split_years != validation_data.split_years:
            raise ValueError("training and validation caches disagree on explicit splits")
        if train_data.split_name != "train" or validation_data.split_name != "validation":
            raise ValueError("explicit cache splits were not selected as train/validation")
        assert train_data.split_years is not None
        return {
            name: list(train_data.split_years[name]) for name in SPLIT_NAMES
        }

    train_years = set(train_data.selected_years)
    validation_years = set(validation_data.selected_years)
    if train_years.intersection(validation_years):
        raise ValueError("training and validation years overlap")
    if train_data.available_years != validation_data.available_years:
        raise ValueError("training and validation caches expose different years")
    test_years = set(train_data.available_years).difference(
        train_years, validation_years
    )
    return {
        "train": sorted(train_years),
        "validation": sorted(validation_years),
        "test": sorted(test_years),
    }


def spatial_weights(latitude: np.ndarray, spatial_fraction: np.ndarray) -> torch.Tensor:
    latitude = np.asarray(latitude, dtype=np.float32)
    spatial_fraction = np.asarray(spatial_fraction, dtype=np.float32)
    if spatial_fraction.ndim != 2 or spatial_fraction.shape[0] != latitude.size:
        raise ValueError("spatial fraction latitude dimension does not match latitude")
    # Taking cos(deg2rad(90)) in float32 produces a small negative number.
    # Compute in float64 and clip round-off so polar weights remain nonnegative.
    cosine_latitude = np.clip(
        np.cos(np.deg2rad(latitude.astype(np.float64))), 0.0, None
    )
    weights = cosine_latitude[:, None] * (spatial_fraction >= 0.5)
    if not np.any(weights > 0):
        raise ValueError("spatial support contains no scoring cells")
    return torch.from_numpy(weights.astype(np.float32))


def prepared_spatial_fraction(source: Any) -> np.ndarray:
    """Read generic exploratory support, falling back to the legacy land key."""

    if "spatial_support_fraction" in source:
        key = "spatial_support_fraction"
    elif "land_fraction" in source:
        key = "land_fraction"
    else:
        raise KeyError(
            "prepared cache needs spatial_support_fraction or land_fraction"
        )
    return np.asarray(source[key][:], dtype=np.float32)


def rps_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Official-category RPS averaged over valid cases, leads and land cells."""

    numerator, denominator = rps_totals(probabilities, target, weights)
    return numerator / denominator


def rps_totals(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted RPS sum and weight for exact aggregation across batches."""

    if probabilities.ndim != 5 or probabilities.shape[2] != 5:
        raise ValueError("probabilities must be [batch, period, 5, lat, lon]")
    if target.shape != probabilities.shape[:2] + probabilities.shape[-2:]:
        raise ValueError("target does not match probability cases, periods and grid")
    if weights.shape != probabilities.shape[-2:]:
        raise ValueError("spatial weights do not match the probability grid")

    valid = (target >= 0) & (target <= 4)
    categories = torch.arange(5, device=target.device).view(1, 1, 5, 1, 1)
    observed_cdf = (target.unsqueeze(2) <= categories).to(probabilities.dtype)
    forecast_cdf = probabilities.cumsum(dim=2)
    cell_rps = ((forecast_cdf - observed_cdf) ** 2).sum(dim=2)
    combined = weights.view(1, 1, *weights.shape) * valid
    denominator = combined.sum()
    if denominator <= 0:
        raise ValueError("batch contains no valid scoring cells")
    return (cell_rps * combined).sum(), denominator


def weighted_modal_quintile_accuracy(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return diagnostic modal-quintile accuracy, not meteorological ACC."""

    numerator, denominator = weighted_modal_quintile_accuracy_totals(
        probabilities, target, weights
    )
    return numerator / denominator


def weighted_modal_quintile_accuracy_totals(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact weighted totals for diagnostic category accuracy.

    This measures whether the most probable quintile equals the observed
    quintile. It is a simple diagnostic and is not the anomaly correlation
    coefficient (ACC); RPS remains the probabilistic selection metric.
    """

    if probabilities.ndim != 5 or probabilities.shape[2] != 5:
        raise ValueError("probabilities must be [batch, period, 5, lat, lon]")
    if target.shape != probabilities.shape[:2] + probabilities.shape[-2:]:
        raise ValueError("target does not match probability cases, periods and grid")
    if weights.shape != probabilities.shape[-2:]:
        raise ValueError("spatial weights do not match the probability grid")

    valid = (target >= 0) & (target <= 4)
    combined = weights.view(1, 1, *weights.shape) * valid
    denominator = combined.sum()
    if denominator <= 0:
        raise ValueError("batch contains no valid scoring cells")
    predicted = probabilities.argmax(dim=2)
    correct = (predicted == target).to(probabilities.dtype)
    return (correct * combined).sum(), denominator


def optimizer_for(model: nn.Module, learning_rate: float, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith("bias"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def run_epoch(
    model: TPProbUNet,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    correction_penalty: float,
    gradient_clip: float,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rps_numerator = 0.0
    total_rps_denominator = 0.0
    total_accuracy_numerator = 0.0
    total_accuracy_denominator = 0.0
    samples = 0
    for features, anchor, target in loader:
        features = features.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            corrections = model.forward_corrections(features)
            probabilities = torch.softmax(
                torch.log(anchor.clamp_min(1.0e-8)) + corrections, dim=2
            )
            score_numerator, score_denominator = rps_totals(
                probabilities, target, weights
            )
            accuracy_numerator, accuracy_denominator = (
                weighted_modal_quintile_accuracy_totals(
                    probabilities, target, weights
                )
            )
            score = score_numerator / score_denominator
            loss = score + correction_penalty * corrections.square().mean()

        if optimizer is not None:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        batch = features.shape[0]
        total_loss += float(loss.detach().cpu()) * batch
        total_rps_numerator += float(score_numerator.detach().cpu())
        total_rps_denominator += float(score_denominator.detach().cpu())
        total_accuracy_numerator += float(accuracy_numerator.detach().cpu())
        total_accuracy_denominator += float(accuracy_denominator.detach().cpu())
        samples += batch
    return (
        total_loss / samples,
        total_rps_numerator / total_rps_denominator,
        total_accuracy_numerator / total_accuracy_denominator,
    )


def save_curve(
    history: list[dict[str, float]],
    destination: Path,
    *,
    note: str | None = None,
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(epochs, [row["train_rps"] for row in history], label="Train")
    axes[0].plot(
        epochs, [row["validation_rps"] for row in history], label="Validation"
    )
    axes[0].set(xlabel="Epoch", ylabel="Area-weighted RPS", title="Probabilistic score")

    train_accuracy_key = (
        "train_weighted_modal_quintile_accuracy_diagnostic_not_acc"
    )
    validation_accuracy_key = (
        "validation_weighted_modal_quintile_accuracy_diagnostic_not_acc"
    )
    axes[1].plot(
        epochs, [row[train_accuracy_key] for row in history], label="Train"
    )
    axes[1].plot(
        epochs,
        [row[validation_accuracy_key] for row in history],
        label="Validation",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Area-weighted fraction correct",
        title="Modal-quintile accuracy\n(diagnostic; not meteorological ACC)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    if note:
        fig.suptitle(note, fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=PATHS.cache_store)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=EXPERIMENT.batch_size)
    parser.add_argument("--max-epochs", type=int, default=EXPERIMENT.max_epochs)
    parser.add_argument("--patience", type=int, default=EXPERIMENT.patience)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=EXPERIMENT.seed)
    parser.add_argument(
        "--period-attention",
        action="store_true",
        help="mix the two forecast periods at the U-Net bottleneck",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="attention heads used with --period-attention",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.10,
        help="Transformer dropout used with --period-attention",
    )
    parser.add_argument("--smoke", action="store_true", help="run a tiny synthetic training job")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or PATHS.runs_root / f"tp_prob_unet_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()

    if args.smoke:
        train_data: Dataset = SyntheticCases(8, args.seed)
        validation_data: Dataset = SyntheticCases(4, args.seed + 1)
        latitude = np.linspace(90.0, -90.0, 17)
        spatial_fraction = np.ones((17, 24), dtype=np.float32)
        normalization = train_data.metadata()  # type: ignore[attr-defined]
        year_partitions = {name: [] for name in SPLIT_NAMES}
    else:
        explicit_splits = False
        if args.cache.suffix == ".npz":
            with np.load(args.cache, allow_pickle=False) as archive:
                explicit_splits = "split" in archive
        if explicit_splits:
            train_data = PreparedCases(args.cache, split="train")
            validation_data = PreparedCases(args.cache, split="validation")
        else:
            train_data = PreparedCases(args.cache, EXPERIMENT.train_years)
            validation_data = PreparedCases(args.cache, EXPERIMENT.validation_years)
        year_partitions = training_year_partitions(train_data, validation_data)
        source = train_data._npz if train_data._npz is not None else train_data._zarr  # type: ignore[attr-defined]
        assert source is not None
        latitude = np.asarray(source["latitude"][:])
        spatial_fraction = prepared_spatial_fraction(source)
        normalization = train_data.metadata()  # type: ignore[attr-defined]

    weights = spatial_weights(latitude, spatial_fraction).to(device)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)

    model = TPProbUNet(
        in_channels=EXPERIMENT.in_channels,
        base_channels=EXPERIMENT.base_channels,
        dropout=EXPERIMENT.dropout,
        period_attention=args.period_attention,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
    ).to(device)
    optimizer = optimizer_for(model, EXPERIMENT.learning_rate, EXPERIMENT.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1.0e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    with torch.no_grad():
        raw_train_loss, raw_train_rps, raw_train_accuracy = run_epoch(
            model,
            train_loader,
            weights,
            device,
            None,
            None,
            EXPERIMENT.correction_penalty,
            EXPERIMENT.gradient_clip,
        )
        raw_validation_loss, raw_validation_rps, raw_validation_accuracy = run_epoch(
            model,
            validation_loader,
            weights,
            device,
            None,
            None,
            EXPERIMENT.correction_penalty,
            EXPERIMENT.gradient_clip,
        )
    history.append(
        {
            "epoch": 0.0,
            "train_loss": raw_train_loss,
            "train_rps": raw_train_rps,
            "train_weighted_modal_quintile_accuracy_diagnostic_not_acc": raw_train_accuracy,
            "validation_loss": raw_validation_loss,
            "validation_rps": raw_validation_rps,
            "validation_weighted_modal_quintile_accuracy_diagnostic_not_acc": raw_validation_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
    )
    # Epoch zero is exactly the FuXi p0 anchor. It remains deployable unless
    # a trained epoch demonstrates lower validation RPS.
    best_state: dict[str, torch.Tensor] | None = copy.deepcopy(model.state_dict())
    best_validation = raw_validation_rps
    best_validation_accuracy = raw_validation_accuracy
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    print(LICENSE_WARNING)
    print(f"device={device} train={len(train_data)} validation={len(validation_data)}")
    print(f"epoch=000 raw_anchor_validation_rps={raw_validation_rps:.6f}")

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_rps, train_accuracy = run_epoch(
            model,
            train_loader,
            weights,
            device,
            optimizer,
            scaler,
            EXPERIMENT.correction_penalty,
            EXPERIMENT.gradient_clip,
        )
        with torch.no_grad():
            validation_loss, validation_rps, validation_accuracy = run_epoch(
                model,
                validation_loader,
                weights,
                device,
                None,
                None,
                EXPERIMENT.correction_penalty,
                EXPERIMENT.gradient_clip,
            )
        scheduler.step(validation_rps)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_rps": train_rps,
            "train_weighted_modal_quintile_accuracy_diagnostic_not_acc": train_accuracy,
            "validation_loss": validation_loss,
            "validation_rps": validation_rps,
            "validation_weighted_modal_quintile_accuracy_diagnostic_not_acc": validation_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_rps={train_rps:.6f} "
            f"val_rps={validation_rps:.6f} "
            f"val_modal_accuracy_diagnostic_not_ACC={validation_accuracy:.6f} "
            f"lr={row['learning_rate']:.2e}"
        )
        if validation_rps < best_validation - 1.0e-7:
            best_validation = validation_rps
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed = time.monotonic() - started
    model_config = {
        "in_channels": EXPERIMENT.in_channels,
        "base_channels": EXPERIMENT.base_channels,
        "dropout": EXPERIMENT.dropout,
    }
    if args.period_attention:
        model_config.update(
            {
                "period_attention": True,
                "attention_heads": args.attention_heads,
                "attention_dropout": args.attention_dropout,
            }
        )
    checkpoint = {
        "model_state": best_state,
        "model_config": model_config,
        "normalization": normalization,
        "metadata": {
            "model_name": "TPProbUNet",
            "period_attention_enabled": bool(args.period_attention),
            "best_epoch": best_epoch,
            "best_validation_rps": best_validation,
            "best_validation_weighted_modal_quintile_accuracy_diagnostic_not_acc": best_validation_accuracy,
            "raw_validation_rps": raw_validation_rps,
            "raw_validation_weighted_modal_quintile_accuracy_diagnostic_not_acc": raw_validation_accuracy,
            "validation_rps_improvement": raw_validation_rps - best_validation,
            "selected_system": "raw_p0" if best_epoch == 0 else "trained_model",
            "seed": args.seed,
            "train_years": year_partitions["train"],
            "validation_years": year_partitions["validation"],
            "test_years": year_partitions["test"],
            "elapsed_seconds": elapsed,
            "fuxi_competition_use": "written_permission_required",
        },
    }
    torch.save(checkpoint, run_dir / "best.pt")
    with (run_dir / "history.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "epoch,train_loss,train_rps,"
            "train_weighted_modal_quintile_accuracy_diagnostic_not_acc,"
            "validation_loss,validation_rps,"
            "validation_weighted_modal_quintile_accuracy_diagnostic_not_acc,"
            "learning_rate\n"
        )
        for row in history:
            handle.write(
                f"{int(row['epoch'])},{row['train_loss']},{row['train_rps']},"
                f"{row['train_weighted_modal_quintile_accuracy_diagnostic_not_acc']},"
                f"{row['validation_loss']},{row['validation_rps']},"
                f"{row['validation_weighted_modal_quintile_accuracy_diagnostic_not_acc']},"
                f"{row['learning_rate']}\n"
            )
    provenance = normalization.get("provenance", {})
    curve_note = None
    if isinstance(provenance, dict) and not provenance.get(
        "official_ai_weather_quest_validation", True
    ):
        curve_note = "Exploratory FuXi/IMERG smoke — not official Quest validation"
    save_curve(
        history,
        run_dir / "figures" / "loss_curve.png",
        note=curve_note,
    )
    run_config = serializable_config()
    run_config["command"] = vars(args) | {"cache": str(args.cache), "run_dir": str(run_dir)}
    run_config["result"] = checkpoint["metadata"]
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"best_epoch={best_epoch} best_validation_rps={best_validation:.6f}")
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
