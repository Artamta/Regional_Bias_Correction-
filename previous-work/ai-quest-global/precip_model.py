"""Compact eight-token precipitation probability adapter."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


N_CONTEXT = 6
N_TARGET = 2
N_CATEGORY = 5


def _groups(channels: int) -> int:
    for value in range(min(8, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class GlobalConv(nn.Module):
    """A 3x3 convolution with circular longitude and bounded latitude."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.block = nn.Sequential(
            GlobalConv(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            GlobalConv(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PrecipQuestAdapter(nn.Module):
    """Residual probability postprocessor with six context and two target tokens.

    ``context_x`` is ``[B,6,23,H,W]``, ``target_x`` is ``[B,2,18,H,W]`` and
    ``p0`` is ``[B,2,5,H,W]``.  Only the target tokens enter the decoder skip
    connections.  Context affects predictions solely through the shared
    eight-token Transformer at the two-pool bottleneck.
    """

    def __init__(
        self,
        width: int = 16,
        *,
        dropout: float = 0.10,
        attention_dropout: float = 0.10,
        context_channels: int = 23,
        target_channels: int = 18,
        attention_heads: int = 4,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        if width < 1 or 4 * width % attention_heads:
            raise ValueError("four-head attention must divide the bottleneck width")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= attention_dropout < 1.0:
            raise ValueError("dropout values must lie in [0,1)")
        self.width = int(width)
        self.context_channels = int(context_channels)
        self.target_channels = int(target_channels)
        self.use_context = bool(use_context)

        self.context_stem = ConvBlock(context_channels, width, dropout)
        self.target_stem = ConvBlock(target_channels, width, dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder_1 = ConvBlock(width, 2 * width, dropout)
        self.encoder_2 = ConvBlock(2 * width, 4 * width, dropout)

        self.token_position = nn.Parameter(torch.empty(1, 8, 4 * width))
        nn.init.normal_(self.token_position, mean=0.0, std=0.02)
        self.token_mixer = nn.TransformerEncoderLayer(
            d_model=4 * width,
            nhead=attention_heads,
            dim_feedforward=8 * width,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.up_2 = nn.Conv2d(4 * width, 2 * width, 1)
        self.decoder_2 = ConvBlock(4 * width, 2 * width, dropout)
        self.up_1 = nn.Conv2d(2 * width, width, 1)
        self.decoder_1 = ConvBlock(2 * width, width, dropout)
        self.correction_head = nn.Conv2d(width, N_CATEGORY, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _validate(
        self, context_x: torch.Tensor, target_x: torch.Tensor, p0: torch.Tensor | None
    ) -> tuple[int, int, int]:
        if context_x.ndim != 5 or tuple(context_x.shape[1:3]) != (
            N_CONTEXT,
            self.context_channels,
        ):
            raise ValueError("context_x must be [B,6,23,H,W]")
        if target_x.ndim != 5 or tuple(target_x.shape[1:3]) != (
            N_TARGET,
            self.target_channels,
        ):
            raise ValueError("target_x must be [B,2,18,H,W]")
        batch, _, _, height, width = context_x.shape
        if tuple(target_x.shape[:1]) != (batch,) or target_x.shape[-2:] != (height, width):
            raise ValueError("context_x and target_x batch/grid dimensions must match")
        if height < 4 or width < 4:
            raise ValueError("height and width must be at least four")
        if p0 is not None and tuple(p0.shape) != (batch, 2, 5, height, width):
            raise ValueError("p0 must be [B,2,5,H,W]")
        return batch, height, width

    def forward_corrections(
        self, context_x: torch.Tensor, target_x: torch.Tensor
    ) -> torch.Tensor:
        batch, height, width = self._validate(context_x, target_x, None)
        context_flat = context_x.reshape(
            batch * N_CONTEXT, self.context_channels, height, width
        )
        if not self.use_context:
            context_flat = torch.zeros_like(context_flat)
        target_flat = target_x.reshape(batch * N_TARGET, self.target_channels, height, width)

        context_stem = self.context_stem(context_flat)
        target_stem = self.target_stem(target_flat)
        stem = torch.cat(
            (
                context_stem.reshape(batch, N_CONTEXT, self.width, height, width),
                target_stem.reshape(batch, N_TARGET, self.width, height, width),
            ),
            dim=1,
        ).reshape(batch * 8, self.width, height, width)

        encoded_1 = self.encoder_1(self.pool(stem))
        encoded_2 = self.encoder_2(self.pool(encoded_1))
        bottleneck_height, bottleneck_width = encoded_2.shape[-2:]
        tokens = encoded_2.reshape(
            batch, 8, 4 * self.width, bottleneck_height, bottleneck_width
        )
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch * bottleneck_height * bottleneck_width, 8, 4 * self.width
        )
        mixed = self.token_mixer(tokens + self.token_position)
        mixed = mixed[:, N_CONTEXT:]
        mixed = (
            mixed.reshape(
                batch,
                bottleneck_height,
                bottleneck_width,
                N_TARGET,
                4 * self.width,
            )
            .permute(0, 3, 4, 1, 2)
            .reshape(batch * N_TARGET, 4 * self.width, bottleneck_height, bottleneck_width)
        )

        target_encoded_1 = encoded_1.reshape(
            batch, 8, 2 * self.width, encoded_1.shape[-2], encoded_1.shape[-1]
        )[:, N_CONTEXT:].reshape(
            batch * N_TARGET, 2 * self.width, encoded_1.shape[-2], encoded_1.shape[-1]
        )
        decoded = F.interpolate(
            mixed, size=target_encoded_1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.up_2(decoded)
        decoded = self.decoder_2(torch.cat((decoded, target_encoded_1), dim=1))
        decoded = F.interpolate(
            decoded, size=target_stem.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.up_1(decoded)
        decoded = self.decoder_1(torch.cat((decoded, target_stem), dim=1))
        correction = self.correction_head(decoded)
        return correction.reshape(batch, N_TARGET, N_CATEGORY, height, width)

    def forward(
        self, context_x: torch.Tensor, target_x: torch.Tensor, p0: torch.Tensor
    ) -> torch.Tensor:
        self._validate(context_x, target_x, p0)
        correction = self.forward_corrections(context_x, target_x)
        return torch.softmax(torch.log(p0.clamp_min(1.0e-8)) + correction, dim=2)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["PrecipQuestAdapter", "trainable_parameter_count"]
