"""Losses and class-imbalance weighting.

Imbalance handling is configurable and has three independent pieces:
  * per-class caps (sampling.max_per_class_*, in splits.py)
  * loss weights (this file): none | inverse_freq | effective_number
  * a re-sampling sampler (dataset.make_sampler): none | sqrt | balanced

Using BOTH a weighted loss and a balancing sampler compensates twice; train.py warns.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def class_weights_from_counts(
    counts: np.ndarray, weighting: str = "none", power: float = 1.0, beta: float = 0.999,
    max_weight: float | None = 20.0,
) -> np.ndarray | None:
    """Per-class loss weights, normalised to mean 1 over classes that have training data.

    inverse_freq:      w_c ~ (1 / n_c) ** power
    effective_number:  w_c ~ (1 - beta) / (1 - beta ** n_c)   (Cui et al., CVPR 2019)
    Classes with zero training samples get weight 0 and a warning (they cannot be learned).
    """
    if weighting in (None, "none"):
        return None
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("No class has any training sample; cannot compute class weights.")
    w = np.zeros_like(counts)
    if weighting == "inverse_freq":
        w[present] = (1.0 / counts[present]) ** power
    elif weighting == "effective_number":
        if not 0.0 <= beta < 1.0:
            raise ValueError("effective_number_beta must be in [0, 1)")
        w[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    else:
        raise ValueError(f"imbalance.weighting must be none|inverse_freq|effective_number, got '{weighting}'")
    w[present] = w[present] / w[present].mean()
    if max_weight is not None:
        w = np.minimum(w, max_weight)
        w[present] = w[present] / w[present].mean()
    if (~present).any():
        log.warning("%d class(es) have zero training samples and get weight 0: indices %s",
                    int((~present).sum()), np.where(~present)[0].tolist())
    return w.astype(np.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss: -(1 - p_t)^gamma * log p_t, optional class weights / label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal_gamma must be >= 0")
        self.gamma, self.label_smoothing = float(gamma), float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(logp, target, reduction="none")
        if self.label_smoothing > 0:
            smooth = -logp.mean(dim=-1)
            ce = (1 - self.label_smoothing) * ce + self.label_smoothing * smooth
        pt = torch.exp(-F.nll_loss(logp, target, reduction="none"))
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum().clamp_min(1e-12)
        return loss.mean()


def build_loss(loss_cfg: dict[str, Any], class_weights: np.ndarray | None, device: torch.device) -> nn.Module:
    weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    name = loss_cfg["name"]
    ls = float(loss_cfg.get("label_smoothing", 0.0))
    if name == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
    if name == "focal":
        return FocalLoss(float(loss_cfg.get("focal_gamma", 2.0)), weight, ls).to(device)
    raise ValueError(f"loss.name must be ce|focal, got '{name}'")


def selective_scale_consistency(
    view_logits: torch.Tensor,
    fused_logits: torch.Tensor,
    confidence_threshold: float = 0.7,
    temperature: float = 1.0,
    return_stats: bool = False,
):
    """Asymmetric fused-to-view KL over confidently informative views."""
    if view_logits.ndim != 3 or fused_logits.ndim != 2 or view_logits.shape[0] != fused_logits.shape[0] or view_logits.shape[-1] != fused_logits.shape[-1]:
        raise ValueError("expected view_logits (B,V,K) and fused_logits (B,K)")
    if not 0.0 <= confidence_threshold <= 1.0 or temperature <= 0:
        raise ValueError("confidence_threshold must be in [0,1] and temperature must be positive")

    teacher = F.softmax(fused_logits.detach() / temperature, dim=-1)
    teacher_conf = teacher.max(dim=-1).values
    view_prob = F.softmax(view_logits.detach() / temperature, dim=-1)
    view_conf = view_prob.max(dim=-1).values
    mask = (teacher_conf[:, None] >= confidence_threshold) & (view_conf >= confidence_threshold)

    student_logp = F.log_softmax(view_logits / temperature, dim=-1)
    kl = F.kl_div(student_logp, teacher[:, None, :].expand_as(view_logits), reduction="none").sum(-1)
    weighted = (kl * mask.to(kl.dtype)).sum()
    loss = (weighted / mask.sum().clamp_min(1)) * (temperature ** 2)

    if return_stats:
        active_fraction = mask.float().mean().detach()
        return loss, active_fraction
    return loss
