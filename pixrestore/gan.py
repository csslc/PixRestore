"""Lightweight adversarial loss over frozen multi-layer DINO features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def as_feature_list(features) -> list[torch.Tensor]:
    return [features] if isinstance(features, torch.Tensor) else list(features)


def _as_tokens(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        batch, channels, height, width = feature.shape
        return feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    if feature.ndim == 3:
        return feature
    if feature.ndim == 2:
        return feature.unsqueeze(1)
    raise ValueError(f"Unsupported DINO feature shape: {tuple(feature.shape)}")


class MultiLayerDinoDiscriminator(nn.Module):
    """Applies one spectrally-normalized MLP head per DINO feature layer."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        *,
        hidden_ratio: float = 0.25,
        min_hidden: int = 64,
        max_hidden: int = 128,
        real_label: float = 0.8,
    ) -> None:
        super().__init__()
        hidden_dim = min(max(int(feature_dim * hidden_ratio), min_hidden), max_hidden)
        self.real_label = float(real_label)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    spectral_norm(nn.Linear(feature_dim, hidden_dim)),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Linear(hidden_dim, 1)),
                )
                for _ in range(num_layers)
            ]
        )
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        features,
        *,
        real: bool = True,
    ) -> torch.Tensor:
        features = as_feature_list(features)
        if len(features) != len(self.heads):
            raise ValueError(f"Expected {len(self.heads)} feature layers, got {len(features)}")
        target_value = self.real_label if real else 0.0
        losses = []
        for feature, head in zip(features, self.heads):
            logits = head(_as_tokens(feature).float()).squeeze(-1)
            target = torch.full_like(logits, target_value)
            losses.append(self.loss(logits, target).mean())
        return torch.stack(losses).sum()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
