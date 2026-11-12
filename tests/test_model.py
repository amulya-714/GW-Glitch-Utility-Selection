"""Tests for the architectures, losses, metrics and device helpers."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.models import BaselineCNN, MultiViewNet, build_model, count_parameters
from src.models.evaluate import compute_metrics, expected_calibration_error
from src.models.losses import FocalLoss, build_loss, class_weights_from_counts, selective_scale_consistency
from src.models.multiview_cnn import (
    AttentionFusion, ConcatFusion, FiLM, HierarchicalScaleInteraction, MeanFusion, SharedViewEncoder, TimeScaleConditioner,
)
from src.utils import deep_update, load_config, project_root, select_device, set_seed

C, H, W, V, NC = 1, 32, 32, 4, 7
DUR = (0.5, 1.0, 2.0, 4.0)


def tiny_multiview(**kw):
    args = dict(num_classes=NC, in_channels=C, view_durations=DUR, base_channels=8, num_stages=3,
                embed_dim=32, film_cond_dim=16, fusion_heads=4)
    args.update(kw)
    return MultiViewNet(**args)


def batch(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, V, C, H, W, generator=g)


@pytest.fixture()
def cfg():
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c = deep_update(c, c["debug"])
    c["data"]["channels"] = 1            # these tests feed single-channel batches; the shipped baseline is RGB
    return c


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["multiview", "baseline_merged", "baseline_single"])
def test_factory_builds_and_outputs_logits(cfg, kind):
    cfg["model"]["type"] = kind
    m = build_model(cfg, NC)
    out = m(batch())
    assert out.shape == (6, NC) and torch.isfinite(out).all()


def test_factory_rejects_unknown_type(cfg):
    cfg["model"]["type"] = "resnet"
    with pytest.raises(ValueError, match="model.type"):
        build_model(cfg, NC)


def test_encoder_weights_are_shared_across_views():
    """Parameter count must not grow with the number of views (shared backbone)."""
    enc2 = tiny_multiview(view_durations=(0.5, 1.0)).encoder
    enc4 = tiny_multiview(view_durations=DUR).encoder
    assert count_parameters(enc2) == count_parameters(enc4)


def test_film_is_identity_at_init():
    enc = SharedViewEncoder(C, 8, 3, cond_dim=16).eval()
    x = torch.randn(5, C, H, W)
    a = enc(x, torch.zeros(5, 16))
    b = enc(x, torch.randn(5, 16))
    assert torch.allclose(a, b, atol=1e-6)


def test_film_applies_gamma_beta():
    f = FiLM(4, 3)
    nn.init.constant_(f.proj.bias, 0.0)
    with torch.no_grad():
        f.proj.bias[:3] = 1.0   # gamma = 1  -> scale by 2
        f.proj.bias[3:] = 0.5   # beta
    x = torch.ones(2, 3, 2, 2)
    out = f(x, torch.zeros(2, 4))
    assert torch.allclose(out, torch.full_like(out, 2.5))


@pytest.mark.parametrize("mode", ["continuous", "learned"])
def test_conditioning_differs_per_time_scale(mode):
    cond = TimeScaleConditioner(DUR, 16, mode)()
    assert cond.shape == (V, 16)
    assert (cond[0] - cond[3]).abs().sum() > 0


def test_conditioning_makes_views_distinguishable_once_trained():
    """Identical pixels in every view give identical tokens at init, but after FiLM weights move
    the tokens differ by time scale: the conditioning path is actually wired in."""
    m = tiny_multiview(fusion_type="mean").eval()
    x = torch.randn(3, 1, C, H, W).repeat(1, V, 1, 1, 1)
    _, ex0 = m.forward_features(x, True)
    assert torch.allclose(ex0["tokens"][:, 0], ex0["tokens"][:, 3], atol=1e-5)
    for mod in m.modules():
        if isinstance(mod, FiLM):
            nn.init.normal_(mod.proj.weight, std=0.5)
    _, ex1 = m.forward_features(x, True)
    assert not torch.allclose(ex1["tokens"][:, 0], ex1["tokens"][:, 3], atol=1e-3)


def test_hierarchical_interaction_is_identity_at_initialization():
    interaction = HierarchicalScaleInteraction(16, V)
    tokens = torch.randn(3, V, 16)
    assert torch.equal(interaction(tokens), tokens)
    assert torch.count_nonzero(interaction.gates) == 0


def test_selective_consistency_skips_unconfident_views_and_backprops_only_to_student():
    fused = torch.tensor([[8.0, -8.0]], requires_grad=True)
    views = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    loss = selective_scale_consistency(views, fused, confidence_threshold=0.8)
    loss.backward()
    assert loss.item() > 0
    assert fused.grad is None
    assert torch.allclose(views.grad[:, 1], torch.zeros_like(views.grad[:, 1]))
    assert views.grad[:, 0].abs().sum() > 0


def test_film_disabled_has_no_conditioning_parameters():
    m = tiny_multiview(film_enabled=False)
    assert m.conditioner is None
    assert not any(isinstance(x, FiLM) for x in m.modules())
    assert m(batch()).shape == (6, NC)
    assert count_parameters(m) < count_parameters(tiny_multiview(film_enabled=True))


@pytest.mark.parametrize("fusion", ["attention", "concat", "mean"])
def test_all_fusions_run_and_weights_sum_to_one(fusion):
    m = tiny_multiview(fusion_type=fusion).eval()
    logits, ex = m(batch(), return_extras=True)
    assert logits.shape == (6, NC) and ex["features"].shape == (6, 32)
    assert torch.allclose(ex["view_weights"].sum(1), torch.ones(6), atol=1e-5)


def test_bad_fusion_or_heads_raise():
    with pytest.raises(ValueError, match="fusion.type"):
        tiny_multiview(fusion_type="magic")
    with pytest.raises(ValueError, match="divisible"):
        tiny_multiview(embed_dim=30, fusion_heads=4)


@pytest.mark.parametrize("fusion_cls", [lambda: AttentionFusion(16, V, 4), lambda: ConcatFusion(16, V), MeanFusion])
def test_fusion_ignores_dropped_views(fusion_cls):
    f = fusion_cls().eval()
    tokens = torch.randn(5, V, 16)
    keep = torch.ones(5, V, dtype=torch.bool)
    keep[:, 3] = False
    a, w = f(tokens, keep)
    tokens2 = tokens.clone()
    tokens2[:, 3] = torch.randn(5, 16) * 10
    b, _ = f(tokens2, keep)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(w[:, 3], torch.zeros(5), atol=1e-6)


def test_view_dropout_train_only_and_never_drops_all():
    m = tiny_multiview(view_dropout=0.95)
    m.train()
    keep = m._keep_mask(512, V, torch.device("cpu"))
    assert keep.any(dim=1).all() and (~keep).any()
    m.eval()
    assert m._keep_mask(8, V, torch.device("cpu")).all()
    x = batch()
    assert torch.allclose(m(x), m(x))  # deterministic in eval


def test_gradients_reach_every_parameter():
    m = tiny_multiview(view_dropout=0.2)
    m.train()
    loss = nn.functional.cross_entropy(m(batch(8)), torch.randint(0, NC, (8,)))
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_wrong_input_shapes_give_clear_errors():
    mv, bl = tiny_multiview(), BaselineCNN(NC, C, V, base_channels=8, num_stages=3, embed_dim=32)
    for model in (mv, bl):
        with pytest.raises(ValueError, match="expected"):
            model(torch.randn(2, C, H, W))
        with pytest.raises(ValueError, match="views"):
            model(torch.randn(2, 3, C, H, W))


def test_baseline_merged_vs_single():
    merged = BaselineCNN(NC, C, V, "merged", base_channels=8, num_stages=3, embed_dim=32)
    single = BaselineCNN(NC, C, V, "single", single_view_index=2, base_channels=8, num_stages=3, embed_dim=32).eval()
    assert merged(batch()).shape == single(batch()).shape == (6, NC)
    x = batch()
    x2 = x.clone()
    x2[:, 0] = torch.randn_like(x2[:, 0])  # change a view the single-view model does not read
    assert torch.allclose(single(x), single(x2))
    with pytest.raises(ValueError):
        BaselineCNN(NC, C, V, "single", single_view_index=9)
    with pytest.raises(ValueError):
        BaselineCNN(NC, C, V, "bogus")


def test_seeded_initialisation_is_reproducible():
    set_seed(1)
    a = tiny_multiview().eval()(batch())
    set_seed(1)
    b = tiny_multiview().eval()(batch())
    assert torch.allclose(a, b)


def test_odd_image_sizes_work():
    m = tiny_multiview().eval()
    assert m(torch.randn(2, V, C, 37, 45)).shape == (2, NC)


def test_rgb_inputs_supported():
    m = MultiViewNet(NC, in_channels=3, view_durations=DUR, base_channels=8, num_stages=3, embed_dim=32, film_cond_dim=16)
    assert m(torch.randn(2, V, 3, H, W)).shape == (2, NC)


def test_model_can_overfit_a_tiny_batch():
    set_seed(0)
    m = tiny_multiview()
    x, y = batch(16, seed=1), torch.randint(0, NC, (16,), generator=torch.Generator().manual_seed(2))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(x), y)
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < 0.35 * first


def test_parameter_counts_reported(cfg):
    counts = {}
    for kind in ("multiview", "baseline_merged", "baseline_single"):
        cfg["model"]["type"] = kind
        counts[kind] = count_parameters(build_model(cfg, NC))
    assert all(v > 0 for v in counts.values())


# --------------------------------------------------------------------------- #
# Losses / imbalance
# --------------------------------------------------------------------------- #
def test_class_weights_normalised_and_ordered():
    counts = np.array([1000, 100, 10, 0])
    for kind in ("inverse_freq", "effective_number"):
        w = class_weights_from_counts(counts, kind, power=1.0, beta=0.999, max_weight=None)
        assert w[3] == 0.0
        assert np.isclose(w[:3].mean(), 1.0, atol=1e-5)
        assert w[0] < w[1] < w[2]
    assert class_weights_from_counts(counts, "none") is None


def test_max_weight_clips():
    w = class_weights_from_counts(np.array([10000, 10000, 2]), "inverse_freq", max_weight=3.0)
    assert w.max() <= 3.0 + 1e-6


def test_class_weights_errors():
    with pytest.raises(ValueError):
        class_weights_from_counts(np.array([1, 2]), "bogus")
    with pytest.raises(ValueError):
        class_weights_from_counts(np.array([0, 0]), "inverse_freq")


def test_focal_gamma_zero_equals_cross_entropy():
    logits, y = torch.randn(32, NC), torch.randint(0, NC, (32,))
    assert torch.allclose(FocalLoss(0.0)(logits, y), nn.functional.cross_entropy(logits, y), atol=1e-6)


def test_focal_downweights_easy_examples_and_accepts_weights():
    logits = torch.tensor([[8.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    y = torch.tensor([0, 0])
    f2 = FocalLoss(2.0)
    assert f2(logits, y) < nn.functional.cross_entropy(logits, y)
    w = torch.tensor([2.0, 1.0, 1.0])
    assert torch.isfinite(FocalLoss(2.0, w, 0.1)(logits, y))


def test_build_loss_variants():
    dev = torch.device("cpu")
    w = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert isinstance(build_loss({"name": "ce", "label_smoothing": 0.1}, w, dev), nn.CrossEntropyLoss)
    assert isinstance(build_loss({"name": "focal", "focal_gamma": 1.5}, None, dev), FocalLoss)
    with pytest.raises(ValueError):
        build_loss({"name": "hinge"}, None, dev)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_perfect_and_confusion_matrix():
    names = ["a", "b", "c"]
    y = np.array([0, 0, 1, 1, 2])
    m = compute_metrics(y, y, names)
    assert m["accuracy"] == 1.0 and m["macro_f1_present"] == 1.0
    assert np.array(m["confusion_matrix"]).sum() == 5


def test_macro_f1_present_ignores_absent_classes():
    names = ["a", "b", "c", "d"]
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred, names)
    assert m["macro_f1_present"] == 1.0
    assert m["macro_f1_all_classes"] == pytest.approx(0.5)
    assert m["num_classes_present"] == 2
    assert m["per_class"]["c"]["support"] == 0


def test_metrics_known_values():
    names = ["a", "b"]
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred, names)
    pc = m["per_class"]
    assert pc["a"]["precision"] == pytest.approx(1.0) and pc["a"]["recall"] == pytest.approx(2 / 3)
    assert pc["b"]["precision"] == pytest.approx(0.5) and pc["b"]["recall"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(0.75)


def test_metrics_empty_input_is_safe():
    assert compute_metrics(np.array([]), np.array([]), ["a"])["n"] == 0


def test_ece_bounds():
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(conf, np.array([1, 1, 1, 1], bool)) == pytest.approx(0.0)
    assert expected_calibration_error(conf, np.array([0, 0, 0, 0], bool)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def test_select_device_always_returns_usable_device():
    assert select_device("cpu").type == "cpu"
    for pref in ("auto", "cuda", "mps"):
        dev = select_device(pref)
        assert torch.zeros(2, device=dev).sum().item() == 0
    with pytest.raises(ValueError):
        select_device("tpu")


def test_deep_update_does_not_mutate_input():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_update(base, {"a": {"b": 9}})
    assert out["a"] == {"b": 9, "c": 2} and base["a"]["b"] == 1
