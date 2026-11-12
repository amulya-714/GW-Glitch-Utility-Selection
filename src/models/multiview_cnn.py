"""Proposed model: shared-weight multi-view CNN with time-scale FiLM and learned cross-view fusion.

Data flow for an event with V=4 views (0.5 s, 1 s, 2 s, 4 s):

    x (B,V,C,H,W) --reshape--> (B*V,C,H,W)
        |                           ^
        |   TimeScaleConditioner: one conditioning vector per view, computed from the
        |   view's duration (continuous log2(duration) MLP, or a learned per-view embedding)
        v
    SharedViewEncoder: ONE CNN whose BatchNorm affine parameters are replaced by
        FiLM(gamma, beta) generated from the time-scale vector, so the same filters
        specialise per time scale at 2*C_stage extra parameters per FiLM instead of 4 backbones.
        -> global average pool -> Linear -> view tokens (B,V,D)
        v
    (optional training-time view dropout)
        v
    CrossViewFusion: attention (1 pre-norm transformer layer over the V view tokens + learned
        attention pooling) | concat | mean
        v
    Classification head (LayerNorm, Dropout, Linear)

Every component is switchable from the config for ablations:
    film.enabled=false          shared encoder, no time-scale conditioning
    fusion.type=concat|mean     replaces the attention fusion
    view_dropout=0.0            off by default

Honest scope note: multi-view attention fusion is NOT claimed as novel. The hypothesis
under test is that a single shared encoder conditioned on time scale is a more
parameter-efficient and more shift-robust way to use the four views than a
channel-merged CNN. That has to be established by the ablations (and a literature check).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.baseline_cnn import ConvStage, stage_widths


# --------------------------------------------------------------------------- #
# Conditioning
# --------------------------------------------------------------------------- #
class TimeScaleConditioner(nn.Module):
    """Maps the V view durations to V conditioning vectors of size cond_dim.

    continuous: MLP over standardised log2(duration). The conditioning is a function of
        the physical time scale, not just an index.
    learned: nn.Embedding over the view index (ablation).
    """

    def __init__(self, durations: Sequence[float], cond_dim: int, mode: str = "continuous"):
        super().__init__()
        if mode not in ("continuous", "learned"):
            raise ValueError(f"film.mode must be continuous|learned, got {mode}")
        self.mode = mode
        d = torch.log2(torch.tensor(list(durations), dtype=torch.float32))
        self.register_buffer("log_d", ((d - d.mean()) / (d.std(unbiased=False) + 1e-6)).unsqueeze(-1))
        if mode == "continuous":
            self.net = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        else:
            self.emb = nn.Embedding(len(durations), cond_dim)

    def forward(self) -> torch.Tensor:  # (V, cond_dim)
        if self.mode == "continuous":
            return self.net(self.log_d)
        return self.emb(torch.arange(self.log_d.shape[0], device=self.log_d.device))


class FiLM(nn.Module):
    """Feature-wise linear modulation: x * (1 + gamma) + beta, zero-initialised (identity at start)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class FiLMConvStage(nn.Module):
    """conv-BN(no affine)-FiLM-ReLU, twice, then max-pool."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out, affine=False)
        self.film1 = FiLM(cond_dim, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out, affine=False)
        self.film2 = FiLM(cond_dim, c_out)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.film1(self.bn1(self.conv1(x)), cond))
        x = torch.relu(self.film2(self.bn2(self.conv2(x)), cond))
        return self.pool(x)


class SharedViewEncoder(nn.Module):
    """One CNN applied to every view. With cond=None it is a plain shared CNN (film ablation)."""

    def __init__(self, in_channels: int, base_channels: int, num_stages: int, cond_dim: int | None):
        super().__init__()
        self.use_film = cond_dim is not None
        widths = stage_widths(base_channels, num_stages)
        stages, c = [], in_channels
        for w in widths:
            stages.append(FiLMConvStage(c, w, cond_dim) if self.use_film else ConvStage(c, w))
            c = w
        self.stages = nn.ModuleList(stages)
        self.out_channels = c
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x, cond) if self.use_film else stage(x)
        return x.mean(dim=(2, 3))  # global average pool; MPS-safe


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _masked_softmax(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1)


class MeanFusion(nn.Module):
    def forward(self, tokens: torch.Tensor, keep: torch.Tensor):
        w = keep.float() / keep.float().sum(dim=1, keepdim=True)
        return (tokens * w.unsqueeze(-1)).sum(dim=1), w


class ConcatFusion(nn.Module):
    def __init__(self, dim: int, num_views: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_views * dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, tokens: torch.Tensor, keep: torch.Tensor):
        tokens = tokens * keep.unsqueeze(-1).float()
        w = keep.float() / keep.float().sum(dim=1, keepdim=True)
        return self.mlp(tokens.flatten(1)), w


class AttentionFusion(nn.Module):
    """Self-attention across the V view tokens, then learned attention pooling.

    Returns the pooled vector and the pooling weights (B, V), useful to inspect which
    time scale the model relies on per class.
    """

    def __init__(self, dim: int, num_views: int, num_heads: int = 4, num_layers: int = 1,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by fusion.num_heads {num_heads}")
        self.view_pos = nn.Parameter(torch.zeros(num_views, dim))
        nn.init.normal_(self.view_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_mult * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([_clone(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, keep: torch.Tensor):
        x = tokens + self.view_pos.unsqueeze(0)
        pad = ~keep
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)
        x = self.norm(x)
        w = _masked_softmax(self.score(x).squeeze(-1), keep)
        return (x * w.unsqueeze(-1)).sum(dim=1), w


class HierarchicalScaleInteraction(nn.Module):
    """Causal adjacent-scale residual message passing over duration-ordered tokens.

    The zero-initialized residual gates make this an exact identity at initialization,
    preserving the shared-CNN/FiLM/attention model as the starting baseline.
    """
    def __init__(self, dim: int, num_views: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(max(0, num_views - 1))])
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                                       for _ in range(max(0, num_views - 1))])
        self.gates = nn.Parameter(torch.zeros(max(0, num_views - 1)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = [tokens[:, 0]]
        for i, (norm, message) in enumerate(zip(self.norms, self.messages)):
            out.append(tokens[:, i + 1] + torch.tanh(self.gates[i]) * message(norm(out[-1])))
        return torch.stack(out, dim=1)


def _clone(module: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(module)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class MultiViewNet(nn.Module):
    def __init__(
        self, num_classes: int, in_channels: int = 1, view_durations: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        base_channels: int = 32, num_stages: int = 4, embed_dim: int = 128, dropout: float = 0.3,
        film_enabled: bool = True, film_cond_dim: int = 64, film_mode: str = "continuous",
        fusion_type: str = "attention", fusion_heads: int = 4, fusion_layers: int = 1,
        fusion_ff_mult: int = 2, view_dropout: float = 0.0,
        scale_interaction: bool = False,
    ):
        super().__init__()
        self.num_views = len(view_durations)
        self.view_dropout = float(view_dropout)
        if not 0.0 <= self.view_dropout < 1.0:
            raise ValueError("view_dropout must be in [0, 1)")
        self.conditioner = TimeScaleConditioner(view_durations, film_cond_dim, film_mode) if film_enabled else None
        self.encoder = SharedViewEncoder(in_channels, base_channels, num_stages, film_cond_dim if film_enabled else None)
        self.proj = nn.Sequential(nn.Linear(self.encoder.out_channels, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.scale_interaction = HierarchicalScaleInteraction(embed_dim, self.num_views) if scale_interaction else None
        if fusion_type == "attention":
            self.fusion = AttentionFusion(embed_dim, self.num_views, fusion_heads, fusion_layers, fusion_ff_mult)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(embed_dim, self.num_views)
        elif fusion_type == "mean":
            self.fusion = MeanFusion()
        else:
            raise ValueError(f"fusion.type must be attention|concat|mean, got '{fusion_type}'")
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    # -- view dropout ------------------------------------------------------- #
    def _keep_mask(self, b: int, v: int, device: torch.device) -> torch.Tensor:
        if not (self.training and self.view_dropout > 0.0):
            return torch.ones(b, v, dtype=torch.bool, device=device)
        keep = torch.rand(b, v, device=device) >= self.view_dropout
        empty = ~keep.any(dim=1)  # never drop every view
        if empty.any():
            forced = torch.randint(0, v, (int(empty.sum()),), device=device)
            keep[empty, forced] = True
        return keep

    # -- forward ------------------------------------------------------------ #
    def forward_features(self, x: torch.Tensor, return_extras: bool = False):
        if x.dim() != 5:
            raise ValueError(f"expected (B,V,C,H,W), got {tuple(x.shape)}")
        b, v, c, h, w = x.shape
        if v != self.num_views:
            raise ValueError(f"model built for {self.num_views} views, got {v}")
        cond = None
        if self.conditioner is not None:
            cond = self.conditioner().unsqueeze(0).expand(b, -1, -1).reshape(b * v, -1)
        f = self.encoder(x.reshape(b * v, c, h, w), cond)
        raw_tokens = self.proj(f).reshape(b, v, -1)
        tokens = raw_tokens
        if self.scale_interaction is not None:
            tokens = self.scale_interaction(raw_tokens)
        keep = self._keep_mask(b, v, x.device)
        fused, weights = self.fusion(tokens, keep)
        if return_extras:
            return fused, {
                "view_weights": weights,
                "tokens": tokens,
                "raw_tokens": raw_tokens,
            }
        return fused

    def forward(self, x: torch.Tensor, return_extras: bool = False):
        if return_extras:
            fused, extras = self.forward_features(x, True)
            logits = self.head(fused)
            return logits, {
                **extras,
                "features": fused,
                "view_logits": self.head(extras["raw_tokens"]),
            }
        return self.head(self.forward_features(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
