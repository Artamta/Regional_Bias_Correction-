#!/usr/bin/env python3
"""Train one frozen precipitation-adapter ablation with validation selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from precip_contract import SPLIT, contract_sha256
from precip_model import PrecipQuestAdapter, trainable_parameter_count


@dataclass(frozen=True)
class TrainConfig:
    variant: str = "full"
    width: int = 16
    seed: int = 42
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    correction_penalty: float = 1.0e-4
    gradient_clip: float = 1.0
    effective_batch_size: int = 8
    micro_batch_size: int = 2
    max_epochs: int = 12
    patience: int = 3
    dropout: float = 0.10
    attention_dropout: float = 0.10
    smoke_cases: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_sha256(config: TrainConfig, cache_contract: str) -> str:
    value = {"train": asdict(config), "cache_contract_sha256": cache_contract}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class CacheCases(Dataset):
    def __init__(
        self,
        cache: Path,
        years: tuple[int, ...],
        variant: str,
        *,
        limit_cases: int | None = None,
    ) -> None:
        import zarr

        self.group = zarr.open_group(str(cache), mode="r")
        if self.group.attrs.get("status") != "complete":
            raise RuntimeError("precipitation cache is not complete")
        if self.group.attrs.get("contract_sha256") != contract_sha256():
            raise ValueError("cache does not match the frozen precipitation contract")
        dates = np.asarray(self.group["init_yyyymmdd"][:], dtype=np.int32)
        case_years = dates // 10000
        self.indices = np.flatnonzero(np.isin(case_years, years))
        expected = 104 * len(years)
        if len(self.indices) != expected:
            raise ValueError(f"split has {len(self.indices)} cases, expected {expected}")
        if limit_cases is not None:
            if not 1 <= limit_cases <= expected:
                raise ValueError(f"limit_cases must be between 1 and {expected}")
            self.indices = self.indices[:limit_cases]
        if variant not in {"target_only", "tp_only", "full"}:
            raise ValueError(f"unsupported ablation {variant!r}")
        self.variant = variant

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        context = np.asarray(self.group["context_x"][index], dtype=np.float32)
        if self.variant == "target_only":
            context.fill(0.0)
        elif self.variant == "tp_only":
            # Dynamic 0:5 are TP quantiles, 5:15 are physical mean/std, and
            # 15:23 are coordinate/time/static maps.
            context[:, 5:15] = 0.0
        return (
            torch.from_numpy(context),
            torch.from_numpy(np.asarray(self.group["target_x"][index], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.group["p0"][index], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.group["target"][index], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.group["target_valid"][index], dtype=bool)),
            index,
        )


def spatial_weight(cache: Path) -> torch.Tensor:
    import zarr

    group = zarr.open_group(str(cache), mode="r")
    latitude = np.asarray(group["latitude"][:], dtype=np.float32)
    land = np.asarray(group["land_fraction"][:], dtype=np.float32)
    area = np.maximum(np.cos(np.deg2rad(latitude)), 0.0)[:, None]
    weight = area * (land >= 0.5)
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError("invalid land/area scoring weight")
    return torch.from_numpy(weight.astype(np.float32))


def rps_terms(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forecast_cdf = probabilities.cumsum(dim=2)[:, :, :4]
    boundary = torch.arange(4, device=target.device).view(1, 1, 4, 1, 1)
    observed_cdf = (target[:, :, None] <= boundary).to(probabilities.dtype)
    full_weight = valid[:, :, None].to(probabilities.dtype) * weight[None, None, None]
    numerator = ((forecast_cdf - observed_cdf).square() * full_weight).sum(
        dim=(0, 2, 3, 4)
    )
    denominator = full_weight.sum(dim=(0, 2, 3, 4))
    if torch.any(denominator <= 0):
        raise RuntimeError("RPS batch has a lead without valid weighted targets")
    return numerator, denominator, numerator / denominator


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weight: torch.Tensor,
    *,
    save_predictions: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    numerator = torch.zeros(2, dtype=torch.float64, device=device)
    denominator = torch.zeros_like(numerator)
    predictions: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for context, target_x, p0, target, valid, indices in loader:
            context = context.to(device)
            target_x = target_x.to(device)
            p0 = p0.to(device)
            target = target.to(device)
            valid = valid.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                probability = model(context, target_x, p0)
            batch_num, batch_den, _ = rps_terms(
                probability.float(), target, valid, weight
            )
            numerator += batch_num.double()
            denominator += batch_den.double()
            if save_predictions:
                predictions.append(
                    (np.asarray(indices), probability.float().cpu().numpy())
                )
    score = (numerator / denominator).cpu().numpy()
    if not save_predictions:
        return score, None
    ordered = sorted(predictions, key=lambda item: int(item[0][0]))
    return score, np.concatenate([item[1] for item in ordered], axis=0)


def save_checkpoint(
    path: Path,
    model: PrecipQuestAdapter,
    config: TrainConfig,
    cache_contract: str,
    configuration_hash: str,
    epoch: int,
    validation_rps: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "PrecipQuestAdapter",
            "model_kwargs": {
                "width": config.width,
                "dropout": config.dropout,
                "attention_dropout": config.attention_dropout,
                "use_context": config.variant != "target_only",
            },
            "train_config": asdict(config),
            "cache_contract_sha256": cache_contract,
            "configuration_sha256": configuration_hash,
            "epoch": epoch,
            "validation_rps_by_lead": validation_rps.tolist(),
            "created_utc": utc_now(),
        },
        temporary,
    )
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_history(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([int(row["epoch"]) for row in rows])
    pooled = np.asarray([float(row["validation_rps_pooled"]) for row in rows])
    lead_1 = np.asarray([float(row["validation_rps_d19_25"]) for row in rows])
    lead_2 = np.asarray([float(row["validation_rps_d26_32"]) for row in rows])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, pooled, marker="o", label="pooled")
    axis.plot(epochs, lead_1, alpha=0.75, label="D19-25")
    axis.plot(epochs, lead_2, alpha=0.75, label="D26-32")
    selected = next(int(row["epoch"]) for row in rows if row["selected"])
    axis.axvline(selected, linestyle="--", color="0.35", label=f"selected epoch {selected}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation RPS")
    axis.set_title("Precipitation adapter validation selection")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    config = TrainConfig(
        variant=args.variant,
        width=args.width,
        seed=args.seed,
        micro_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        smoke_cases=args.smoke_cases,
    )
    if config.effective_batch_size % config.micro_batch_size:
        raise ValueError("effective batch size must be divisible by micro batch size")
    accumulation_steps = config.effective_batch_size // config.micro_batch_size
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(config.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "checkpoints").mkdir()
    (args.run_dir / "predictions").mkdir()

    train_data = CacheCases(
        args.cache,
        SPLIT.train_years,
        config.variant,
        limit_cases=config.smoke_cases,
    )
    validation_data = CacheCases(
        args.cache,
        SPLIT.validation_years,
        config.variant,
        limit_cases=config.smoke_cases,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    weight = spatial_weight(args.cache).to(device)
    model = PrecipQuestAdapter(
        width=config.width,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        use_context=config.variant != "target_only",
    ).to(device)
    cache_contract = train_data.group.attrs["contract_sha256"]
    configuration_hash = config_sha256(config, cache_contract)

    epoch_zero, _ = evaluate(model, validation_loader, device, weight)
    best_score = float(epoch_zero.mean())
    best_epoch = 0
    best_path = args.run_dir / "checkpoints" / "best.pt"
    save_checkpoint(
        best_path, model, config, cache_contract, configuration_hash, 0, epoch_zero
    )
    # This separate immutable fallback proves the exact raw-p0 identity.
    save_checkpoint(
        args.run_dir / "checkpoints" / "epoch_zero_p0.pt",
        model,
        config,
        cache_contract,
        configuration_hash,
        0,
        epoch_zero,
    )
    rows = [
        {
            "epoch": 0,
            "train_objective": "",
            "validation_rps_d19_25": float(epoch_zero[0]),
            "validation_rps_d26_32": float(epoch_zero[1]),
            "validation_rps_pooled": best_score,
            "selected": True,
        }
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective_total = 0.0
        batches = 0
        for batch_index, (context, target_x, p0, target, valid, _indices) in enumerate(train_loader):
            context = context.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            p0 = p0.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                correction = model.forward_corrections(context, target_x)
                probability = torch.softmax(torch.log(p0.clamp_min(1.0e-8)) + correction, dim=2)
                _, _, lead_rps = rps_terms(probability.float(), target, valid, weight)
                objective = lead_rps.mean() + config.correction_penalty * correction.float().square().mean()
                scaled = objective / accumulation_steps
            scaled.backward()
            objective_total += float(objective.detach())
            batches += 1
            is_boundary = (batch_index + 1) % accumulation_steps == 0
            is_last = batch_index + 1 == len(train_loader)
            if is_boundary or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        validation_rps, _ = evaluate(model, validation_loader, device, weight)
        pooled = float(validation_rps.mean())
        improved = pooled < best_score - 1.0e-9
        if improved:
            best_score = pooled
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                best_path,
                model,
                config,
                cache_contract,
                configuration_hash,
                epoch,
                validation_rps,
            )
        else:
            stale += 1
        for row in rows:
            row["selected"] = int(row["epoch"]) == best_epoch
        rows.append(
            {
                "epoch": epoch,
                "train_objective": objective_total / max(batches, 1),
                "validation_rps_d19_25": float(validation_rps[0]),
                "validation_rps_d26_32": float(validation_rps[1]),
                "validation_rps_pooled": pooled,
                "selected": epoch == best_epoch,
            }
        )
        write_history(args.run_dir / "training_history.csv", rows)
        plot_history(args.run_dir / "rps_curves.png", rows)
        print(
            f"epoch={epoch} train={objective_total/max(batches,1):.7f} "
            f"validation={pooled:.7f} best={best_score:.7f}@{best_epoch}",
            flush=True,
        )
        if stale >= config.patience:
            break

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    selected_rps, prediction = evaluate(
        model, validation_loader, device, weight, save_predictions=True
    )
    np.testing.assert_allclose(selected_rps, payload["validation_rps_by_lead"], rtol=1e-6)
    np.save(args.run_dir / "predictions" / "validation.npy", prediction)
    summary = {
        "status": "smoke_complete" if config.smoke_cases is not None else "complete",
        "smoke_only": config.smoke_cases is not None,
        "training_cases": len(train_data),
        "validation_cases": len(validation_data),
        "variant": config.variant,
        "parameters": trainable_parameter_count(model),
        "best_epoch": int(payload["epoch"]),
        "validation_rps_by_lead": selected_rps.tolist(),
        "validation_rps_pooled": float(selected_rps.mean()),
        "epoch_zero_p0_rps_by_lead": epoch_zero.tolist(),
        "configuration_sha256": configuration_hash,
        "cache_contract_sha256": cache_contract,
        "completed_utc": utc_now(),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--variant", choices=("target_only", "tp_only", "full"), required=True)
    result.add_argument("--width", type=int, default=16)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--effective-batch-size", type=int, default=8)
    result.add_argument("--max-epochs", type=int, default=12)
    result.add_argument("--patience", type=int, default=3)
    result.add_argument("--dropout", type=float, default=0.10)
    result.add_argument("--attention-dropout", type=float, default=0.10)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument(
        "--smoke-cases",
        type=int,
        help="Use only the first N train and validation cases; marks artifacts smoke-only",
    )
    result.add_argument("--device", default="cuda")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
