"""Baseline CNNs. These are the reference points for the proposed multi-view model.

They use the SAME conv-stage design (two 3x3 conv-BN-ReLU + max-pool) and the
same width schedule as the proposed encoder, so differences in results come from
the multi-view design (shared weights + time-scale FiLM + cross-view fusion),
not from a stronger/weaker backbone.

  mode="merged": the four views are stacked along the channel axis and fed to one
                 CNN. This mirrors the merged-view idea of the original Gravity Spy
                 classifier family (a design reference, not a reimplementation).
  mode="single": one chosen view only (shows how much the other scales add).
"""
from __future__ import annotations

import torch
from torch import nn


class ConvStage(nn.Module):
    """Two 3x3 conv-BN-ReLU layers followed by 2x2 max-pooling."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


def stage_widths(base_channels: int, num_stages: int) -> list[int]:
    """base, 2*base, 4*base, 8*base, ... capped at 8*base."""
    return [base_channels * min(2 ** i, 8) for i in range(num_stages)]


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, num_views: int = 4,
                 mode: str = "merged", single_view_index: int = 1, base_channels: int = 32,
                 num_stages: int = 4, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        if mode not in ("merged", "single"):
            raise ValueError(f"mode must be merged|single, got {mode}")
        if mode == "single" and not 0 <= single_view_index < num_views:
            raise ValueError(f"single_view_index {single_view_index} out of range for {num_views} views")
        self.mode, self.view_idx, self.num_views = mode, single_view_index, num_views
        c0 = in_channels * (num_views if mode == "merged" else 1)
        widths = stage_widths(base_channels, num_stages)
        layers, c = [], c0
        for w in widths:
            layers.append(ConvStage(c, w))
            c = w
        self.encoder = nn.Sequential(*layers)
        self.embed = nn.Sequential(nn.Linear(c, embed_dim), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, C, H, W) -> embedding (B, embed_dim)."""
        if x.dim() != 5:
            raise ValueError(f"expected (B,V,C,H,W), got {tuple(x.shape)}")
        b, v, c, h, w = x.shape
        if v != self.num_views:
            raise ValueError(f"model built for {self.num_views} views, got {v}")
        x = x.reshape(b, v * c, h, w) if self.mode == "merged" else x[:, self.view_idx]
        f = self.encoder(x).mean(dim=(2, 3))   # global average pool (MPS-safe)
        return self.embed(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
