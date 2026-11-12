"""Tests pinning the approved baseline: crop [101, 61, 674, 540] on 800x600, RGB, 128x153 (HxW), unit, bilinear,
alpha over white, all four views. Synthetic 800x600 RGBA images only; no network, no O3a."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import ImageStore, array_cache_key, build_split_array, prepare_split
from src.data.preprocessing import (
    PreprocessConfig, aspect_distortion, aspect_preserving_size, box_size, load_event_views, load_view,
    preprocessing_signature,
)
from src.utils import load_config, project_root

VIEWS = ["url1", "url2", "url3", "url4"]
BOX = (101, 61, 674, 540)
SIZE = (128, 153)
W, H = 800, 600


@pytest.fixture()
def data_cfg():
    return copy.deepcopy(load_config(project_root() / "configs" / "classifier.yaml")["data"])


@pytest.fixture()
def approved(data_cfg):
    return PreprocessConfig.from_cfg(data_cfg)


def rgba(arr_rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([arr_rgb, np.full(arr_rgb.shape[:2] + (1,), 255, np.uint8)], axis=2).astype(np.uint8)


def save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path)
    return path


def framed(inside, outside=(255, 0, 0)):
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = outside
    l, t, r, b = BOX
    a[t:b, l:r] = inside
    return rgba(a)


# --------------------------------------------------------------------------- #
# Shape, channels, crop coordinates
# --------------------------------------------------------------------------- #
def test_approved_config_produces_exact_output_shape_and_channels(tmp_path, approved):
    rng = np.random.default_rng(0)
    p = save(tmp_path / "a.png", rgba(rng.integers(0, 256, (H, W, 3))))
    out = load_view(p, approved)
    assert out.shape == (3, 128, 153) and out.dtype == np.uint8          # (C, H, W): RGB, 128 high, 153 wide
    views = load_event_views([p] * 4, approved)
    assert views.shape == (4, 3, 128, 153)                               # four views kept
    assert not np.array_equal(out[0], out[2])                            # colour is not collapsed


def test_crop_coordinates_are_exactly_left_top_right_bottom(tmp_path):
    rng = np.random.default_rng(1)
    arr = rgba(rng.integers(0, 256, (H, W, 3)))
    p = save(tmp_path / "a.png", arr)
    l, t, r, b = BOX
    ident = load_view(p, PreprocessConfig((b - t, r - l), 3, crop_box=BOX))   # identity resize -> raw crop pixels
    assert ident.shape == (3, 479, 573) and box_size(BOX) == (573, 479)
    assert np.array_equal(np.transpose(ident, (1, 2, 0)), arr[61:540, 101:674, :3])
    assert np.array_equal(ident[:, 0, 0], arr[61, 101, :3]) and np.array_equal(ident[:, -1, -1], arr[539, 673, :3])


def test_pixels_one_step_outside_the_box_never_enter_the_output(tmp_path, approved):
    a = framed(inside=(0, 0, 255), outside=(255, 0, 0))
    a[60, :, :3] = a[540, :, :3] = (255, 0, 0)
    a[:, 100, :3] = a[:, 674, :3] = (255, 0, 0)                           # the ring just outside the box
    p = save(tmp_path / "a.png", a)
    out = load_view(p, approved)
    assert out[0].max() == 0 and (out[2] == 255).all()                    # all blue, no red at any edge
    l, t, r, b = BOX
    exact = load_view(p, PreprocessConfig((b - t, r - l), 3, crop_box=BOX))
    assert (exact[0] == 0).all() and (exact[2] == 255).all()


def test_the_crop_is_not_stretched(tmp_path, approved):
    """A disk of diameter 200 px must stay round: equal x and y extent after resizing."""
    a = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (BOX[0] + BOX[2]) // 2, (BOX[1] + BOX[3]) // 2
    a[(xx - cx) ** 2 + (yy - cy) ** 2 <= 100 ** 2] = 255
    p = save(tmp_path / "disk.png", rgba(a))

    def extent(img):
        ys, xs = np.where(img[0] > 127)
        return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

    wd, ht = extent(load_view(p, approved))
    assert abs(wd / ht - 1.0) < 0.04 and abs(wd - 200 * 153 / 573) <= 2     # round, and at the expected scale
    wd_sq, ht_sq = extent(load_view(p, PreprocessConfig((128, 128), 3, crop_box=BOX)))   # a square output
    assert wd_sq / ht_sq < 0.9                                              # ...visibly squashes the same disk


# --------------------------------------------------------------------------- #
# Aspect helpers and the stretch guard
# --------------------------------------------------------------------------- #
def test_aspect_helpers_and_the_approved_size_is_aspect_preserving():
    assert aspect_preserving_size((573, 479), height=128) == (128, 153)
    assert [aspect_preserving_size((573, 479), height=h)[1] for h in (96, 128, 160, 224)] == [115, 153, 191, 268]
    assert aspect_preserving_size((573, 479), width=153)[0] == 128
    assert aspect_distortion((573, 479), SIZE) < 0.001                      # 0.08%: integer rounding only
    assert aspect_distortion((573, 479), (128, 128)) == pytest.approx(0.164, abs=1e-3)
    with pytest.raises(ValueError):
        aspect_preserving_size((573, 479))
    with pytest.raises(ValueError):
        aspect_preserving_size((573, 479), height=1, width=1)


def test_stretching_configs_are_rejected_with_the_aspect_preserving_alternative(data_cfg):
    for bad in ([128, 128], [96, 96], [224, 224], [128, 170]):
        c = copy.deepcopy(data_cfg)
        c["image_size"] = bad
        with pytest.raises(ValueError, match="stretches"):
            PreprocessConfig.from_cfg(c)
    c = copy.deepcopy(data_cfg)
    c["image_size"] = [128, 128]
    with pytest.raises(ValueError, match=r"\[128, 153\]"):
        PreprocessConfig.from_cfg(c)
    c["crop"]["max_aspect_distortion"] = None                                # explicit opt-out
    assert PreprocessConfig.from_cfg(c).image_size == (128, 128)
    c2 = copy.deepcopy(data_cfg)
    c2["crop"]["enabled"] = False                                            # no crop, no check
    c2["image_size"] = [128, 128]
    assert PreprocessConfig.from_cfg(c2).crop_box is None


@pytest.mark.parametrize("h,w", [(96, 115), (128, 153), (160, 191), (224, 268)])
def test_all_preset_heights_pass_the_guard(data_cfg, h, w):
    data_cfg["image_size"] = [h, w]
    assert PreprocessConfig.from_cfg(data_cfg).image_size == (h, w)


# --------------------------------------------------------------------------- #
# Four-view consistency with the shipped config
# --------------------------------------------------------------------------- #
def make_events(tmp_path, n=4):
    rows = []
    for e in range(n):
        urls = []
        for v in range(4):
            val = 20 + 40 * e + 9 * v
            urls.append("file://" + str(save(tmp_path / "src" / f"e{e}_v{v}.png", framed((val, val, val)))))
        rows.append({"gravityspy_id": f"evt{e:07d}", "event_time": 1e9 + e, "ifo": "H1", "ml_label": "Blip",
                     "ml_confidence": 0.9, "label": 0, **dict(zip(VIEWS, urls))})
    return pd.DataFrame(rows)


def make_store(tmp_path):
    return ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)


def test_four_views_stay_consistent_with_the_shipped_config(tmp_path, data_cfg):
    cfg = {"data": data_cfg, "paths": {"processed_dir": str(tmp_path / "proc")}}
    df = make_events(tmp_path)
    ds, kept, failures = prepare_split(df, "t", cfg, make_store(tmp_path))
    assert failures.empty and len(ds) == 4
    for e in range(4):
        x, _ = ds[e]
        assert x.shape == (4, 3, 128, 153) and x.dtype == torch.float32
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0              # unit normalisation
        for v in range(4):
            assert torch.allclose(x[v], torch.full_like(x[v], (20 + 40 * e + 9 * v) / 255.0), atol=1e-6)


def test_view_order_follows_the_configured_columns(tmp_path, data_cfg):
    df = make_events(tmp_path, n=2)
    store = make_store(tmp_path)
    store.ensure_downloaded(df, VIEWS)
    pre = PreprocessConfig.from_cfg(data_cfg)
    p_fwd, _, _ = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    for v_idx in range(4):                      # slot v holds view v+1 (0.5 s, 1 s, 2 s, 4 s)
        assert abs(float(np.load(p_fwd)[0, v_idx].mean()) - (20 + 9 * v_idx)) < 1e-6
    # a different column order must never reuse this array
    assert array_cache_key(df["gravityspy_id"], pre, VIEWS) != array_cache_key(df["gravityspy_id"], pre, VIEWS[::-1])


# --------------------------------------------------------------------------- #
# Cache keys and signature
# --------------------------------------------------------------------------- #
CHANGES = {
    "crop box": lambda c: c["crop"].update(box=[102, 61, 674, 540]),
    "crop reference size": lambda c: c["crop"].update(reference_size=[801, 600]),
    "crop disabled": lambda c: c["crop"].update(enabled=False),
    "image size": lambda c: c.update(image_size=[96, 115]),
    "channels": lambda c: c.update(channels=1),
    "resize mode": lambda c: c.update(resize_mode="area"),
    "alpha background": lambda c: c.update(alpha_background="black"),
}


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_array_cache_key_changes_with_every_geometry_setting(data_cfg, name):
    ids = ["a", "b", "c"]
    base = array_cache_key(ids, PreprocessConfig.from_cfg(data_cfg), VIEWS)
    changed = copy.deepcopy(data_cfg)
    CHANGES[name](changed)
    assert array_cache_key(ids, PreprocessConfig.from_cfg(changed), VIEWS) != base


def test_array_cache_key_depends_on_ids_and_view_columns(data_cfg):
    pre = PreprocessConfig.from_cfg(data_cfg)
    k = array_cache_key(["a", "b"], pre, VIEWS)
    assert k == array_cache_key(["a", "b"], pre, list(VIEWS))                # stable
    assert k != array_cache_key(["b", "a"], pre, VIEWS)                       # event order
    assert k != array_cache_key(["a", "b"], pre, VIEWS[:3])                   # number of views
    assert k != array_cache_key(["a", "b"], pre, ["url2", "url1", "url3", "url4"])


def test_signature_covers_geometry_views_and_normalisation(data_cfg):
    base = preprocessing_signature(data_cfg)
    assert base["geometry"]["crop"] == list(BOX) and base["geometry"]["size"] == list(SIZE)
    assert base["views"] == VIEWS and base["normalization"]["mode"] == "unit" and len(base["hash"]) == 12
    assert preprocessing_signature(copy.deepcopy(data_cfg))["hash"] == base["hash"]
    variants = []
    for name, fn in CHANGES.items():
        c = copy.deepcopy(data_cfg)
        fn(c)
        c["crop"]["max_aspect_distortion"] = None
        variants.append(preprocessing_signature(c)["hash"])
    n = copy.deepcopy(data_cfg)
    n["normalization"]["mode"] = "per_sample_standardize"                    # not in the array key, but in the signature
    variants.append(preprocessing_signature(n)["hash"])
    v = copy.deepcopy(data_cfg)
    v["view_columns"] = ["url2", "url1", "url3", "url4"]
    variants.append(preprocessing_signature(v)["hash"])
    assert len({base["hash"], *variants}) == len(variants) + 1


def test_normalisation_change_reuses_the_array_but_changes_the_signature(tmp_path, data_cfg):
    cfg = {"data": data_cfg, "paths": {"processed_dir": str(tmp_path / "proc")}}
    df = make_events(tmp_path, n=2)
    store = make_store(tmp_path)
    ds1, _, _ = prepare_split(df, "t", cfg, store)
    sig1 = preprocessing_signature(cfg["data"])["hash"]
    cfg["data"]["normalization"] = {"mode": "per_view_standardize"}
    ds2, _, _ = prepare_split(df, "t", cfg, store)
    assert ds1.array_path == ds2.array_path                                   # same cached uint8 pixels
    assert preprocessing_signature(cfg["data"])["hash"] != sig1
    assert not torch.allclose(ds1[0][0], ds2[0][0])
