"""Tests for crop bounds, output dimensions, channels, alpha, normalisation and four-view alignment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import GlitchDataset, ImageStore, build_split_array, prepare_split
from src.data.preprocessing import (
    ALPHA_MODES, NORMALIZATION_MODES, SUPPORTED_RESOLUTIONS, CorruptImageError, ImageGeometryError, Normalizer,
    PreprocessConfig, compute_channel_stats, load_event_views, load_view, normalize_views, validate_crop_box,
)
from src.utils import load_config, project_root

VIEWS = ["url1", "url2", "url3", "url4"]
W, H = 800, 600
BOX = (115, 70, 685, 520)


def save_rgba(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), "RGBA").save(path)
    return path


def framed(inside_rgb, outside_rgb=(255, 0, 0), box=BOX, size_wh=(W, H)) -> np.ndarray:
    """Image whose plotting region is `inside_rgb` and whose surroundings are `outside_rgb`."""
    w, h = size_wh
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = outside_rgb
    a[..., 3] = 255
    l, t, r, b = box
    a[t:b, l:r, :3] = inside_rgb
    return a


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Crop bounds
# --------------------------------------------------------------------------- #
def test_valid_crop_box_is_accepted_and_normalised_to_ints():
    assert validate_crop_box([115, 70, 685, 520], (W, H)) == (115, 70, 685, 520)
    assert validate_crop_box((0, 0, W, H), (W, H)) == (0, 0, W, H)      # whole image is legal


@pytest.mark.parametrize("box", [
    [100, 50, 100, 400],      # zero width
    [200, 50, 100, 400],      # left > right
    [100, 400, 500, 400],     # zero height
    [-1, 0, 500, 500],        # negative
    [0, 0, W + 1, 500],       # beyond width
    [0, 0, 500, H + 1],       # beyond height
    [10, 10, 20, 400],        # narrower than the minimum side
    [10, 10, 400],            # wrong length
    [1.5, 0, 500, 500],       # non-integer
])
def test_invalid_crop_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        validate_crop_box(box, (W, H))


def test_enabled_crop_without_box_is_an_actionable_error():
    cfg = {"image_size": [96, 96], "channels": 1, "crop": {"enabled": True, "box": None}}
    with pytest.raises(ValueError, match="crop_preview"):
        PreprocessConfig.from_cfg(cfg)


def test_disabled_crop_ignores_the_box():
    cfg = {"image_size": 96, "channels": 3, "crop": {"enabled": False, "box": [1, 2, 3, 4]}}
    assert PreprocessConfig.from_cfg(cfg).crop_box is None


def test_box_is_checked_against_the_reference_size():
    with pytest.raises(ValueError):
        PreprocessConfig((96, 96), 1, crop_box=(0, 0, 500, 400), crop_reference_size=(400, 300))


def test_crop_extracts_exactly_the_requested_pixels(tmp_path):
    rng = np.random.default_rng(0)
    arr = np.concatenate([rng.integers(0, 256, (H, W, 3)), np.full((H, W, 1), 255)], axis=2).astype(np.uint8)
    p = save_rgba(tmp_path / "a.png", arr)
    l, t, r, b = BOX
    out = load_view(p, PreprocessConfig((b - t, r - l), 3, crop_box=BOX))   # identity resize -> exact pixels
    assert out.shape == (3, b - t, r - l)
    assert np.array_equal(np.transpose(out, (1, 2, 0)), arr[t:b, l:r, :3])


def test_crop_removes_everything_outside_the_box(tmp_path):
    p = save_rgba(tmp_path / "a.png", framed(inside_rgb=(0, 0, 255), outside_rgb=(255, 0, 0)))
    for size in [(64, 64), (128, 160)]:
        out = load_view(p, PreprocessConfig(size, 3, "area", crop_box=BOX))
        assert out[0].max() == 0 and out[2].min() == 255          # no red leaks in, all blue
    uncropped = load_view(p, PreprocessConfig((64, 64), 3, "area"))
    assert uncropped[0].max() > 0                                 # without the crop the red margin is visible


def test_wrong_image_size_for_the_crop_is_a_geometry_error(tmp_path):
    p = save_rgba(tmp_path / "small.png", framed((0, 0, 255), size_wh=(700, 600), box=(100, 70, 600, 520)))
    with pytest.raises(ImageGeometryError, match="700x600"):
        load_view(p, PreprocessConfig((64, 64), 3, crop_box=BOX))
    assert issubclass(ImageGeometryError, CorruptImageError)
    load_view(p, PreprocessConfig((64, 64), 3))                    # fine when no crop is requested


# --------------------------------------------------------------------------- #
# Dimensions and channels
# --------------------------------------------------------------------------- #
@pytest.fixture()
def png(tmp_path):
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.integers(0, 256, (H, W, 3)), np.full((H, W, 1), 255)], axis=2)
    return save_rgba(tmp_path / "x.png", a)


@pytest.mark.parametrize("res", SUPPORTED_RESOLUTIONS)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("crop", [None, BOX])
def test_output_dimensions_for_all_resolutions(png, res, channels, crop):
    out = load_view(png, PreprocessConfig((res, res), channels, crop_box=crop))
    assert out.shape == (channels, res, res) and out.dtype == np.uint8


def test_non_square_and_int_sizes(png):
    assert load_view(png, PreprocessConfig((96, 120), 3, crop_box=BOX)).shape == (3, 96, 120)
    cfg = PreprocessConfig.from_cfg({"image_size": 128, "channels": 1})
    assert cfg.image_size == (128, 128)
    assert PreprocessConfig.from_cfg({"image_size": [96, 120], "channels": 3}).image_size == (96, 120)
    with pytest.raises(ValueError):
        PreprocessConfig.from_cfg({"image_size": [96], "channels": 1})
    with pytest.raises(ValueError):
        PreprocessConfig.from_cfg({"image_size": 0, "channels": 1})


@pytest.mark.parametrize("mode", ["bilinear", "area", "bicubic", "lanczos"])
def test_resize_modes_preserve_shape_and_mean(png, mode):
    full = load_view(png, PreprocessConfig((H, W), 3))
    small = load_view(png, PreprocessConfig((96, 96), 3, mode))
    assert small.shape == (3, 96, 96)
    assert abs(float(small.mean()) - float(full.mean())) < 3.0


def test_unknown_options_raise():
    with pytest.raises(ValueError):
        PreprocessConfig((96, 96), 2)
    with pytest.raises(ValueError):
        PreprocessConfig((96, 96), 1, resize_mode="nearest-ish")
    with pytest.raises(ValueError):
        PreprocessConfig((96, 96), 1, alpha_background="grey")


def test_rgb_keeps_colour_and_gray_is_its_luma(tmp_path):
    a = framed((200, 30, 10), outside_rgb=(0, 0, 0))
    p = save_rgba(tmp_path / "c.png", a)
    rgb = load_view(p, PreprocessConfig((H, W), 3))
    gray = load_view(p, PreprocessConfig((H, W), 1))
    assert rgb[0].mean() > rgb[1].mean() > rgb[2].mean()          # colour information is preserved
    with Image.open(p) as im:
        luma = np.asarray(im.convert("RGB").convert("L"))
    assert np.array_equal(gray[0], luma)
    assert gray.shape == (1, H, W) and rgb.shape == (3, H, W)


def test_gray_input_png_is_supported_as_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.full((50, 60), 77, np.uint8), "L").save(p)
    out = load_view(p, PreprocessConfig((20, 20), 3))
    assert out.shape == (3, 20, 20) and (out == 77).all()


@pytest.mark.parametrize("mode,expected", [("white", (255, 255, 255)), ("black", (0, 0, 0)), ("drop", (10, 20, 30))])
def test_alpha_is_composited_over_the_chosen_background(tmp_path, mode, expected):
    a = np.zeros((40, 40, 4), np.uint8)
    a[..., :3] = (10, 20, 30)
    a[:, :20, 3] = 0           # transparent left half
    a[:, 20:, 3] = 255         # opaque right half
    p = save_rgba(tmp_path / "t.png", a)
    out = load_view(p, PreprocessConfig((40, 40), 3, "area", alpha_background=mode))
    assert tuple(out[:, 10, 5]) == expected
    assert tuple(out[:, 10, 30]) == (10, 20, 30)
    assert set(ALPHA_MODES) == {"white", "black", "drop"}


# --------------------------------------------------------------------------- #
# Cache key / configuration
# --------------------------------------------------------------------------- #
def test_cache_key_changes_with_every_geometry_option():
    base = PreprocessConfig((96, 96), 1)
    variants = [PreprocessConfig((128, 128), 1), PreprocessConfig((96, 96), 3), PreprocessConfig((96, 96), 1, "area"),
                PreprocessConfig((96, 96), 1, alpha_background="black"),
                PreprocessConfig((96, 96), 1, crop_box=BOX), PreprocessConfig((96, 96), 1, crop_box=(0, 0, 700, 500))]
    keys = [str(base.key())] + [str(v.key()) for v in variants]
    assert len(set(keys)) == len(keys)
    assert base.key() == PreprocessConfig((96, 96), 1).key()


def test_shipped_config_is_the_approved_baseline():
    cfg = load_config(project_root() / "configs" / "classifier.yaml")["data"]
    assert cfg["crop"] == {"enabled": True, "reference_size": [800, 600], "box": [101, 61, 674, 540],
                           "max_aspect_distortion": 0.02}
    assert cfg["channels"] == 3 and cfg["image_size"] == [128, 153]
    assert cfg["resize_mode"] == "bilinear" and cfg["alpha_background"] == "white"
    assert cfg["normalization"]["mode"] == "unit"
    assert cfg["view_columns"] == VIEWS and cfg["view_durations"] == [0.5, 1.0, 2.0, 4.0]
    pre = PreprocessConfig.from_cfg(cfg)
    assert pre.crop_box == (101, 61, 674, 540) and pre.crop_reference_size == (800, 600)
    assert pre.image_size == (128, 153) and pre.channels == 3
    assert Normalizer.from_cfg(cfg).mode == "unit"


def test_legacy_configs_keep_legacy_behaviour():
    legacy = {"image_size": [96, 96], "channels": 1, "resize_mode": "bilinear"}
    assert PreprocessConfig.from_cfg(legacy).crop_box is None
    assert Normalizer.from_cfg(legacy).mode == "per_view_standardize"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def two_view_batch(bright=(60, 120)):
    """(1, V=2, C=1, 8, 8): textured views whose mean intensities are `bright`; view 1 is brighter."""
    rng = np.random.default_rng(0)
    base = rng.integers(-20, 21, (8, 8))
    x = np.stack([np.clip(b + base, 0, 255) for b in bright])[None, :, None].astype(np.uint8)
    return torch.from_numpy(x)


def test_unit_is_x_over_255_and_preserves_relative_intensity():
    x = two_view_batch()
    out = Normalizer("unit")(x)
    assert torch.allclose(out, x.float() / 255.0)
    assert out.min() >= 0 and out.max() <= 1
    ratio = out[0, 1].mean() / out[0, 0].mean()
    assert ratio == pytest.approx(2.0, rel=0.05)                  # the 2x brightness difference survives


def test_per_view_standardise_removes_view_brightness_differences():
    out = Normalizer("per_view_standardize")(two_view_batch())
    for v in range(2):
        assert out[0, v].mean().abs() < 1e-4 and out[0, v].std(unbiased=False) == pytest.approx(1.0, abs=1e-3)
    assert torch.allclose(out[0, 0], out[0, 1], atol=1e-3)         # identical texture -> identical output
    assert torch.equal(normalize_views(two_view_batch()), out)     # legacy helper == this mode


def test_per_sample_standardise_keeps_relative_brightness_between_views():
    out = Normalizer("per_sample_standardize")(two_view_batch())
    assert out.mean().abs() < 1e-4 and out.std(unbiased=False) == pytest.approx(1.0, abs=1e-3)
    assert out[0, 1].mean() > out[0, 0].mean() + 1.0               # the brighter view stays clearly brighter


def test_fixed_normalisation_matches_formula_per_channel():
    x = torch.zeros(2, 4, 3, 5, 5, dtype=torch.uint8)
    x[:, :, 0], x[:, :, 1], x[:, :, 2] = 51, 102, 204              # 0.2, 0.4, 0.8
    n = Normalizer("fixed", mean=(0.1, 0.2, 0.4), std=(0.1, 0.2, 0.4))
    out = n(x)
    assert torch.allclose(out[..., 0, :, :], torch.full_like(out[..., 0, :, :], 1.0), atol=1e-5)
    assert torch.allclose(out[..., 1, :, :], torch.full_like(out[..., 1, :, :], 1.0), atol=1e-5)
    assert torch.allclose(out[..., 2, :, :], torch.full_like(out[..., 2, :, :], 1.0), atol=1e-5)
    scalar = Normalizer("fixed", mean=(0.5,), std=(0.25,))(x)
    assert torch.allclose(scalar[0, 0, 0, 0, 0], torch.tensor((51 / 255 - 0.5) / 0.25))


def test_fixed_normalisation_validation():
    with pytest.raises(ValueError, match="requires"):
        Normalizer("fixed")
    with pytest.raises(ValueError, match="strictly positive"):
        Normalizer("fixed", mean=(0.5,), std=(0.0,))
    with pytest.raises(ValueError, match="same length"):
        Normalizer("fixed", mean=(0.5, 0.5), std=(0.2,))
    with pytest.raises(ValueError, match="data.channels"):
        Normalizer.from_cfg({"channels": 3, "normalization": {"mode": "fixed", "mean": [0.5, 0.5], "std": [0.2, 0.2]}})
    assert Normalizer.from_cfg({"channels": 3, "normalization": {"mode": "fixed", "mean": 0.5, "std": 0.2}}).mean == (0.5,)
    with pytest.raises(ValueError, match="mode"):
        Normalizer("zscore")


@pytest.mark.parametrize("mode", [m for m in NORMALIZATION_MODES if m != "fixed"])
def test_constant_images_never_produce_nan(mode):
    x = torch.full((2, 4, 1, 8, 8), 90, dtype=torch.uint8)
    assert torch.isfinite(Normalizer(mode)(x)).all()


@pytest.mark.parametrize("mode", NORMALIZATION_MODES)
def test_normalisation_is_independent_of_other_samples(mode):
    """No dataset statistics: the same event gives the same output whatever else is in the batch."""
    a = torch.randint(0, 255, (1, 4, 1, 8, 8), dtype=torch.uint8)
    b = torch.randint(0, 255, (5, 4, 1, 8, 8), dtype=torch.uint8)
    b[2] = a[0]
    n = Normalizer(mode, mean=(0.4,), std=(0.2,)) if mode == "fixed" else Normalizer(mode)
    assert torch.allclose(n(a)[0], n(b)[2])


def test_compute_channel_stats_known_values():
    arr = np.zeros((6, 4, 3, 5, 5), np.uint8)
    arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = 51, 102, 204        # axis 2 is the channel axis
    arr[3:, :, 0] = 153                                            # channel 0: half the events 0.2, half 0.6
    mean, std = compute_channel_stats(arr)
    assert mean == pytest.approx([0.4, 0.4, 0.8], abs=1e-6)
    assert std == pytest.approx([0.2, 0.0, 0.0], abs=1e-6)
    m2, _ = compute_channel_stats(arr, max_samples=2, seed=1)
    assert len(m2) == 3


# --------------------------------------------------------------------------- #
# Four-view alignment, caching and original-file preservation
# --------------------------------------------------------------------------- #
def make_events(tmp_path, n=5, bad=None):
    """Each view's plotting region has value 20+30*e+7*v; margins are red. `bad` = {(event, view): size_wh}."""
    rows = []
    for e in range(n):
        urls = []
        for v in range(4):
            size = (bad or {}).get((e, v), (W, H))
            a = framed((20 + 30 * e + 7 * v,) * 3, size_wh=size, box=BOX if size == (W, H) else (100, 70, 600, 520))
            p = save_rgba(tmp_path / "src" / f"e{e}_v{v}.png", a)
            urls.append(f"file://{p}")
        rows.append({"gravityspy_id": f"evt{e:07d}", "event_time": 1e9 + e, "ifo": "H1", "ml_label": "Blip",
                     "ml_confidence": 0.9, "label": 0, **dict(zip(VIEWS, urls))})
    return pd.DataFrame(rows)


def store_for(tmp_path):
    return ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)


def test_four_views_stay_aligned_through_the_crop_pipeline(tmp_path):
    df = make_events(tmp_path, n=5)
    store = store_for(tmp_path)
    assert store.ensure_downloaded(df, VIEWS).empty
    pre = PreprocessConfig((64, 80), 3, "area", crop_box=BOX)
    path, kept, failures = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    arr = np.load(path)
    assert arr.shape == (5, 4, 3, 64, 80) and failures.empty
    for e in range(5):
        for v in range(4):
            assert np.all(arr[e, v] == 20 + 30 * e + 7 * v)       # exact: right event, right view, no red margin


def test_event_with_a_wrong_size_view_is_dropped_whole_and_alignment_holds(tmp_path):
    df = make_events(tmp_path, n=5, bad={(2, 3): (700, 600)})
    store = store_for(tmp_path)
    store.ensure_downloaded(df, VIEWS)
    pre = PreprocessConfig((32, 32), 1, "area", crop_box=BOX)
    path, kept, failures = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    arr = np.load(path)
    assert kept["gravityspy_id"].tolist() == ["evt0000000", "evt0000001", "evt0000003", "evt0000004"]
    assert arr.shape[0] == 4 and "700x600" in failures.iloc[0]["reason"]
    for row, e in enumerate([0, 1, 3, 4]):
        for v in range(4):
            assert np.all(arr[row, v] == 20 + 30 * e + 7 * v)


def test_original_pngs_are_never_modified_or_deleted(tmp_path):
    df = make_events(tmp_path, n=3, bad={(1, 0): (700, 600)})
    store = store_for(tmp_path)
    store.ensure_downloaded(df, VIEWS)
    cached = sorted((tmp_path / "img").rglob("*.png"))
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in cached}
    assert len(cached) == 12
    for pre in (PreprocessConfig((32, 32), 3, crop_box=BOX), PreprocessConfig((64, 64), 1), PreprocessConfig((96, 96), 3, crop_box=BOX)):
        build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    after = {p: (sha(p), p.stat().st_mtime_ns) for p in sorted((tmp_path / "img").rglob("*.png"))}
    assert after == before        # includes the valid-but-wrong-size PNG that was rejected by the crop


def test_changing_preprocessing_builds_a_new_array_instead_of_reusing_a_stale_one(tmp_path):
    df = make_events(tmp_path, n=3)
    store = store_for(tmp_path)
    store.ensure_downloaded(df, VIEWS)
    p1, _, _ = build_split_array(df, store, PreprocessConfig((32, 32), 1), VIEWS, tmp_path / "proc", "t")
    p2, _, _ = build_split_array(df, store, PreprocessConfig((32, 32), 1, crop_box=BOX), VIEWS, tmp_path / "proc", "t")
    p3, _, _ = build_split_array(df, store, PreprocessConfig((32, 32), 3, crop_box=BOX), VIEWS, tmp_path / "proc", "t")
    assert len({p1, p2, p3}) == 3
    assert np.load(p1).shape[2] == 1 and np.load(p3).shape[2] == 3
    assert np.load(p1)[0, 0].max() > np.load(p2)[0, 0].max()       # uncropped keeps the bright red margin


def test_dataset_applies_the_configured_normaliser_and_prepare_split_reads_the_config(tmp_path):
    df = make_events(tmp_path, n=3)
    cfg = {"data": {"view_columns": VIEWS, "image_size": [48, 61], "channels": 3, "resize_mode": "area",
                    "alpha_background": "white",
                    "crop": {"enabled": True, "reference_size": [W, H], "box": list(BOX)},
                    "normalization": {"mode": "unit"}},
           "paths": {"processed_dir": str(tmp_path / "proc")}}
    ds, kept, failures = prepare_split(df, "t", cfg, store_for(tmp_path))
    x, y = ds[1]
    assert x.shape == (4, 3, 48, 61) and failures.empty and len(ds) == 3
    assert x.min() >= 0 and x.max() <= 1
    assert torch.allclose(x[2], torch.full_like(x[2], (20 + 30 * 1 + 7 * 2) / 255.0), atol=1e-5)

    cfg["data"]["normalization"] = {"mode": "per_sample_standardize"}
    ds2, _, _ = prepare_split(df, "t", cfg, store_for(tmp_path))      # same cached array, different normaliser
    x2, _ = ds2[1]
    assert x2.mean().abs() < 1e-3 and not torch.allclose(x, x2)
    assert x2[3].mean() > x2[0].mean()                                # view brightness ordering kept (7*v offsets)


def test_dataset_constructor_default_normaliser_is_unit(tmp_path):
    np.save(tmp_path / "a.npy", np.full((2, 4, 1, 8, 8), 51, np.uint8))
    ds = GlitchDataset(tmp_path / "a.npy", None, ["a", "b"])
    assert torch.allclose(ds[0][0], torch.full((4, 1, 8, 8), 0.2))
