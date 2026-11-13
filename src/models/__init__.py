"""Model factory. The model type is chosen from `cfg["model"]["type"]`."""
from __future__ import annotations

from typing import Any

from torch import nn

from src.models.baseline_cnn import BaselineCNN
from src.models.multiview_cnn import MultiViewNet, count_parameters

__all__ = ["BaselineCNN", "MultiViewNet", "MODEL_TYPES", "build_model", "count_parameters"]

MODEL_TYPES = ("multiview", "baseline_merged", "baseline_single", "single_view")


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
    if kind == "single_view":
        # True single-view baseline: the DATASET delivers only view model.single_view_index
        # (see src.data.dataset.model_input_views), so the network is built for ONE input view.
        # Same conv backbone, width schedule, embedding and head as baseline_single.
        idx = m.get("single_view_index")
        if idx is None or not 0 <= int(idx) < n_views:
            raise ValueError(f"model.single_view_index={idx} out of range for {n_views} views.")
        return BaselineCNN(num_classes, in_channels=d["channels"], num_views=1, mode="single",
                           single_view_index=0, **common)
    raise ValueError(f"model.type must be multiview|baseline_merged|baseline_single|single_view, got '{kind}'")
