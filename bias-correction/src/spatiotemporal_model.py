"""Identity-safe temporal refinement of the trained spatial rainfall U-Net."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fuxi_adapter.models import ResidualUNet
from fuxi_adapter.training import (
    SequenceDataset,
    masked_weighted_smooth_l1,
    set_deterministic_seed,
)


class IdentityGatedTemporalAttentionUNet(ResidualUNet):
    """Spatial U-Net with lead-week attention at its bottleneck.

    The temporal branch has a zero-initialized gate. After loading a trained
    :class:`ResidualUNet`, this model therefore gives exactly the same output
    before fine-tuning. Attention is applied independently at every spatial
    bottleneck location and only mixes the six lead-week tokens.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        dropout: float = 0.1,
        max_leads: int = 6,
    ) -> None:
        super().__init__(in_channels, base_channels, dropout)
        channels = 4 * base_channels
        if channels % 4:
            raise ValueError("four attention heads must divide bottleneck channels")

        self.max_leads = max_leads
        self.lead_position = nn.Parameter(torch.empty(1, max_leads, channels))
        nn.init.normal_(self.lead_position, mean=0.0, std=0.02)
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=2 * channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_gate = nn.Parameter(torch.zeros(()))
        self._spatial_frozen = False

    def load_spatial_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Load all spatial weights and leave only temporal parameters new."""

        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint does not contain a model state dictionary")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected spatial keys: {result.unexpected_keys}")
        temporal_prefixes = ("lead_position", "temporal_attention", "temporal_gate")
        if not result.missing_keys or not all(
            key.startswith(temporal_prefixes) for key in result.missing_keys
        ):
            raise ValueError(f"unexpected missing keys: {result.missing_keys}")

    def freeze_spatial_backbone(self) -> None:
        """Freeze the inherited U-Net so only temporal parameters are fitted."""

        temporal_prefixes = ("lead_position", "temporal_attention", "temporal_gate")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(temporal_prefixes)
        self._spatial_frozen = True

    def train(self, mode: bool = True) -> "IdentityGatedTemporalAttentionUNet":
        super().train(mode)
        if mode and self._spatial_frozen:
            for module in (
                self.encoder_1,
                self.encoder_2,
                self.bottleneck,
                self.up_2,
                self.decoder_2,
                self.up_1,
                self.decoder_1,
                self.residual_head,
            ):
                module.eval()
        return self

    def _transform_bottleneck(
        self, bottleneck: torch.Tensor, batch_size: int, leads: int
    ) -> torch.Tensor:
        if leads > self.max_leads:
            raise ValueError(f"received {leads} leads; max_leads={self.max_leads}")

        _, channels, height, width = bottleneck.shape
        tokens = bottleneck.reshape(batch_size, leads, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, leads, channels
        )
        positioned = tokens + self.lead_position[:, :leads]
        attended = self.temporal_attention(positioned)
        temporal_delta = attended - tokens
        refined = tokens + torch.tanh(self.temporal_gate) * temporal_delta
        return (
            refined.reshape(batch_size, height, width, leads, channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * leads, channels, height, width)
        )


@dataclass(frozen=True)
class FineTuneResult:
    best_epoch: int
    initial_validation_loss: float
    best_validation_loss: float
    history: pd.DataFrame
    elapsed_seconds: float
    temporal_gate: float


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    sample_count = 0

    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = masked_weighted_smooth_l1(prediction, target, weights, beta=1.0)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        batch_size = int(features.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    return total / sample_count


def fine_tune_from_spatial(
    model: IdentityGatedTemporalAttentionUNet,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    spatial_weights: np.ndarray,
    run_directory: Path,
    *,
    seed: int,
    device: str,
    batch_size: int = 16,
    max_epochs: int = 100,
    patience: int = 20,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
) -> FineTuneResult:
    """Fine-tune temporal attention, retaining epoch -1 as a valid checkpoint."""

    set_deterministic_seed(seed)
    target_device = torch.device(device)
    model.to(target_device)
    weights = torch.as_tensor(
        spatial_weights, dtype=torch.float32, device=target_device
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SequenceDataset(train_features, train_target),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_features, validation_target),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("the model has no trainable temporal parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1.0e-6
    )

    started = time.monotonic()
    with torch.no_grad():
        initial_loss = _run_epoch(
            model, validation_loader, weights, target_device, optimizer=None
        )
    best_loss = initial_loss
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    rows = [
        {
            "epoch": -1,
            "train_loss": np.nan,
            "validation_loss": initial_loss,
            "learning_rate": learning_rate,
            "temporal_gate": float(torch.tanh(model.temporal_gate).detach().cpu()),
            "improved": True,
        }
    ]

    for epoch in range(max_epochs):
        train_loss = _run_epoch(
            model, train_loader, weights, target_device, optimizer=optimizer
        )
        with torch.no_grad():
            validation_loss = _run_epoch(
                model, validation_loader, weights, target_device, optimizer=None
            )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1.0e-7
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "temporal_gate": float(
                    torch.tanh(model.temporal_gate).detach().cpu()
                ),
                "improved": improved,
            }
        )
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    logs = Path(run_directory) / "logs"
    checkpoints = Path(run_directory) / "checkpoints"
    logs.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    history.to_csv(logs / "training_history.csv", index=False)
    gate = float(torch.tanh(model.temporal_gate).detach().cpu())
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "initial_validation_loss": initial_loss,
            "best_validation_loss": best_loss,
            "seed": seed,
            "temporal_gate": gate,
            "architecture": "IdentityGatedTemporalAttentionUNet",
        },
        checkpoints / "best.pt",
    )
    return FineTuneResult(
        best_epoch=best_epoch,
        initial_validation_loss=initial_loss,
        best_validation_loss=best_loss,
        history=history,
        elapsed_seconds=time.monotonic() - started,
        temporal_gate=gate,
    )


__all__ = [
    "FineTuneResult",
    "IdentityGatedTemporalAttentionUNet",
    "fine_tune_from_spatial",
]
