"""Model factory. The model type is chosen from `cfg["model"]["type"]`."""
from __future__ import annotations

from typing import Any

from torch import nn

from src.models.baseline_cnn import BaselineCNN
from src.models.multiview_cnn import MultiViewNet, count_parameters

__all__ = ["BaselineCNN", "MultiViewNet", "build_model", "count_parameters"]


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    m, d = cfg["model"], cfg["data"]
    n_views = len(d["view_columns"])
    if len(d["view_durations"]) != n_views:
        raise ValueError("data.view_durations and data.view_columns must have the same length.")
    common = dict(base_channels=m["base_channels"], num_stages=m["num_stages"],
                  embed_dim=m["embed_dim"], dropout=m["dropout"])
    kind = m["type"]
    if kind == "multiview":
        return MultiViewNet(
            num_classes, in_channels=d["channels"], view_durations=d["view_durations"],
            film_enabled=m["film"]["enabled"], film_cond_dim=m["film"]["cond_dim"], film_mode=m["film"]["mode"],
            fusion_type=m["fusion"]["type"], fusion_heads=m["fusion"]["num_heads"],
            fusion_layers=m["fusion"]["num_layers"], fusion_ff_mult=m["fusion"]["ff_mult"],
            view_dropout=m["view_dropout"], scale_interaction=m.get("scale_interaction", {}).get("enabled", False), **common)
    if kind in ("baseline_merged", "baseline_single"):
        return BaselineCNN(
            num_classes, in_channels=d["channels"], num_views=n_views,
            mode="merged" if kind == "baseline_merged" else "single",
            single_view_index=m["single_view_index"], **common)
    raise ValueError(f"model.type must be multiview|baseline_merged|baseline_single, got '{kind}'")
