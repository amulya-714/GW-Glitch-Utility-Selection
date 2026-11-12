"""Tests for metadata cleaning, class mapping, temporal splits, downloading, alignment and datasets."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    GlitchDataset, ImageStore, build_split_array, class_counts, make_sampler, prepare_split,
)
from src.data.fake_data import FAKE_CLASSES, make_fake_dataset
from src.data.preprocessing import Augment, CorruptImageError, PreprocessConfig, load_view, normalize_views
from src.data.splits import (
    ClassMapping, TargetLeakageError, apply_class_mapping, apply_confidence_filter, assert_split_disjoint,
    build_class_mapping, build_source_splits, cap_per_class, clean_metadata, load_run_metadata, temporal_split,
)
from src.utils import load_config, project_root

VIEWS = ["url1", "url2", "url3", "url4"]


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_df(n=200, seed=0, t0=1.0e9, spacing=60.0, labels=("Blip", "Whistle")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gravityspy_id": [f"id{i:08d}" for i in range(n)],
        "event_time": t0 + np.cumsum(rng.exponential(spacing, n)),
        "ifo": "H1",
        "ml_label": rng.choice(labels, n),
        "ml_confidence": rng.uniform(0.3, 1.0, n),
        **{c: f"file:///nonexistent/{c}.png" for c in VIEWS},
    })


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_gw")
    make_fake_dataset(root, n_events=240, ifos=("H1",), seed=3)
    return root


@pytest.fixture()
def cfg(fake_root, tmp_path):
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c["paths"] = {"metadata_dir": str(fake_root / "metadata"), "image_dir": str(tmp_path / "img"),
                  "processed_dir": str(tmp_path / "proc")}
    c["split"].update(gap_seconds=1800, block_seconds=3600, min_val_per_class=1)
    c["data"]["image_size"] = [32, 32]
    c["data"]["channels"] = 1                       # these tests use small fake 140x170 images, grayscale
    c["data"]["crop"]["enabled"] = False            # (the shipped baseline crops 800x600 images)
    c["data"]["download"].update(workers=2, retries=1, backoff_s=0.0)
    c["sampling"] = {"max_per_class_train": None, "max_per_class_val": None}
    return c


def write_png(path: Path, value: int, size=(20, 24), mode="L"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(size, value, np.uint8)
    arr[::2] = np.clip(value + 20, 0, 255)  # texture: mean is exactly value + 10
    # Sparse +/-1 speckle keeps the PNG above the 200-byte corruption floor without moving the mean much.
    rng = np.random.default_rng(int(value))
    speckle = rng.integers(0, 2, size) * 2 - 1
    mask = np.zeros(size, bool)
    mask[1::4, ::2] = True
    arr = (arr.astype(int) + speckle * mask).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).convert(mode).save(path)
    return path


# --------------------------------------------------------------------------- #
# Cleaning / leakage guard / class mapping
# --------------------------------------------------------------------------- #
def test_exact_duplicate_is_removed_and_logged(caplog):
    df = make_df(50)
    dup = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with caplog.at_level("WARNING"):
        out = clean_metadata(dup, "H1", VIEWS, "t")
    assert len(out) == 50 and out["gravityspy_id"].is_unique
    assert "exact duplicate" in caplog.text and df.iloc[3]["gravityspy_id"] in caplog.text


def test_conflicting_duplicate_ids_are_dropped(caplog):
    df = make_df(30)
    conflict = df.iloc[[5]].copy()
    conflict["ml_label"] = "Chirp"
    with caplog.at_level("WARNING"):
        out = clean_metadata(pd.concat([df, conflict]), "H1", VIEWS, "t")
    assert df.iloc[5]["gravityspy_id"] not in set(out["gravityspy_id"])
    assert len(out) == 29


def test_wrong_ifo_raises():
    with pytest.raises(ValueError, match="ifo"):
        clean_metadata(make_df(20), "L1", VIEWS, "t")


def test_cleaned_data_is_time_sorted():
    df = make_df(100).sample(frac=1, random_state=1)
    out = clean_metadata(df, "H1", VIEWS)
    assert out["event_time"].is_monotonic_increasing


def test_target_run_cannot_be_loaded_for_training(tmp_path):
    with pytest.raises(TargetLeakageError):
        load_run_metadata(tmp_path, "H1", "O3a", VIEWS, target_runs=["O3a"], purpose="train")


def test_final_eval_requires_a_target_run(tmp_path):
    with pytest.raises(ValueError, match="target run"):
        load_run_metadata(tmp_path, "H1", "O2", VIEWS, target_runs=["O3a"], purpose="final_eval")


def test_missing_metadata_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="H1_O2.csv"):
        load_run_metadata(tmp_path, "H1", "O2", VIEWS, target_runs=["O3a"], purpose="train")


def test_training_never_needs_target_files(cfg, fake_root, tmp_path):
    """Remove O3a metadata entirely: the source pipeline must still work."""
    md = tmp_path / "md_only_o2"
    md.mkdir()
    (md / "H1_O2.csv").write_bytes((fake_root / "metadata" / "H1_O2.csv").read_bytes())
    cfg["paths"]["metadata_dir"] = str(md)
    train, val, mapping, _ = build_source_splits(cfg)
    assert len(train) > 0 and len(val) > 0


def test_class_mapping_explicit_and_logged(caplog):
    m = build_class_mapping({"names": ["A", "B", "C"], "exclude": ["C"], "unseen_in_source": ["X"]})
    assert m.names == ["A", "B"] and m.excluded == ["C"] and m.num_classes == 2
    df = make_df(10)
    df["ml_label"] = ["A", "B", "C", "X"] * 2 + ["A", "C"]
    with caplog.at_level("INFO"):
        mapped, other = apply_class_mapping(df, m, "t")
    assert set(mapped["label"]) <= {0, 1}
    assert set(other["ml_label"]) == {"C", "X"}
    assert "excluded by config" in caplog.text and "unseen-in-source" in caplog.text


def test_unknown_label_is_an_error_not_a_silent_drop():
    m = build_class_mapping({"names": ["A"], "exclude": [], "unseen_in_source": []})
    df = make_df(5)
    df["ml_label"] = "Mystery"
    with pytest.raises(ValueError, match="Mystery"):
        apply_class_mapping(df, m, "t")


def test_mapping_validation():
    with pytest.raises(ValueError):
        build_class_mapping({"names": ["A"], "exclude": ["B"]})
    with pytest.raises(ValueError):
        build_class_mapping({"names": ["A"], "unseen_in_source": ["A"]})


def test_class_mapping_roundtrip():
    m = ClassMapping(["A", "B"], ["C"], ["X"])
    m2 = ClassMapping.from_dict(m.to_dict())
    assert (m2.names, m2.excluded, m2.unseen_in_source) == (m.names, m.excluded, m.unseen_in_source)


# --------------------------------------------------------------------------- #
# Confidence / caps
# --------------------------------------------------------------------------- #
def test_confidence_filter_and_none_option():
    df = make_df(300)
    assert len(apply_confidence_filter(df, None)) == 300
    f = apply_confidence_filter(df, 0.7)
    assert (f["ml_confidence"] >= 0.7).all() and len(f) < 300


def test_cap_per_class_is_seeded_and_exact():
    df = make_df(500, labels=("A", "B", "C"))
    a, b = cap_per_class(df, 40, 1), cap_per_class(df, 40, 1)
    assert a["gravityspy_id"].tolist() == b["gravityspy_id"].tolist()
    assert (a["ml_label"].value_counts() <= 40).all()
    assert a["event_time"].is_monotonic_increasing
    assert cap_per_class(df, 40, 2)["gravityspy_id"].tolist() != a["gravityspy_id"].tolist()
    assert len(cap_per_class(df, None, 1)) == 500


# --------------------------------------------------------------------------- #
# Temporal splits
# --------------------------------------------------------------------------- #
def test_chronological_split_is_ordered_with_gap():
    df = make_df(2000)
    tr, va, pur = temporal_split(df, "chronological", 0.2, gap_seconds=600)
    assert tr["event_time"].max() < va["event_time"].min()
    assert va["event_time"].min() - tr["event_time"].max() >= 600
    assert len(tr) + len(va) + len(pur) == len(df)
    assert set(tr["gravityspy_id"]).isdisjoint(va["gravityspy_id"])
    assert 0.15 < len(va) / len(df) < 0.25
    assert len(pur) > 0


def test_split_is_deterministic():
    df = make_df(500)
    a = temporal_split(df, "blocked", 0.2, 300, 3000, seed=5)
    b = temporal_split(df, "blocked", 0.2, 300, 3000, seed=5)
    assert a[1]["gravityspy_id"].tolist() == b[1]["gravityspy_id"].tolist()


def test_blocked_split_respects_gap_both_directions():
    df = make_df(3000, spacing=30)
    gap = 600
    tr, va, pur = temporal_split(df, "blocked", 0.2, gap, block_seconds=7200, seed=1)
    tt, vt = tr["event_time"].to_numpy(), va["event_time"].to_numpy()
    nearest = np.abs(vt[:, None] - tt[None, :]).min(axis=1)
    assert nearest.min() >= gap
    assert 0.1 < len(va) / len(df) < 0.3
    assert len(tr) + len(va) + len(pur) == len(df)


def test_split_validation_errors():
    with pytest.raises(ValueError):
        temporal_split(make_df(100), "random", 0.2)
    with pytest.raises(ValueError):
        temporal_split(make_df(100), "chronological", 1.5)
    with pytest.raises(ValueError):
        temporal_split(make_df(5), "chronological", 0.2)


def test_overlap_detector_fires():
    df = make_df(20)
    with pytest.raises(AssertionError, match="both train and val"):
        assert_split_disjoint(df.iloc[:12], df.iloc[10:])


def test_split_boundary_independent_of_confidence_threshold(cfg):
    c1, c2 = copy.deepcopy(cfg), copy.deepcopy(cfg)
    c1["confidence"] = {"train": None, "val": None}
    c2["confidence"] = {"train": 0.9, "val": 0.9}
    _, _, _, r1 = build_source_splits(c1)
    _, _, _, r2 = build_source_splits(c2)
    assert r1["purged"]["gravityspy_id"].tolist() == r2["purged"]["gravityspy_id"].tolist()


def test_source_splits_report_counts_before_and_after(cfg):
    train, val, mapping, rep = build_source_splits(cfg)
    tab = rep["train_counts"]
    assert tab.shape[1] == 3  # split, confidence-filtered, final
    assert (tab.iloc[:, 0] >= tab.iloc[:, 1]).all()
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
    assert train["event_time"].max() < val["event_time"].min()
    assert (train["ml_confidence"] >= cfg["confidence"]["train"]).all()


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_load_view_shapes_channels_and_dtype(tmp_path):
    p = write_png(tmp_path / "a.png", 100, mode="RGB")
    g = load_view(p, PreprocessConfig((16, 12), 1))
    c = load_view(p, PreprocessConfig((16, 12), 3, "area"))
    assert g.shape == (1, 16, 12) and g.dtype == np.uint8
    assert c.shape == (3, 16, 12)


def test_corrupt_png_raises_named_error(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a png" * 50)
    with pytest.raises(CorruptImageError, match="bad.png"):
        load_view(p, PreprocessConfig())


def test_normalize_views_per_sample_per_view():
    x = torch.randint(0, 255, (3, 4, 1, 16, 16), dtype=torch.uint8)
    x[1, 2] = 77  # constant image -> zeros, not NaN
    n = normalize_views(x)
    assert torch.isfinite(n).all()
    assert torch.allclose(n[0, 0].mean(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(n[0, 0].std(unbiased=False), torch.tensor(1.0), atol=1e-3)
    assert torch.count_nonzero(n[1, 2]) == 0


def test_normalization_uses_no_dataset_statistics():
    a = torch.randint(0, 255, (1, 4, 1, 8, 8), dtype=torch.uint8)
    b = torch.randint(0, 255, (5, 4, 1, 8, 8), dtype=torch.uint8)
    b[2] = a[0]
    assert torch.allclose(normalize_views(a)[0], normalize_views(b)[2])  # same image, same output regardless of batch


def test_augment_preserves_shape_and_shifts_all_views_together():
    x = torch.randn(4, 1, 16, 20)
    aug = Augment(max_time_shift_frac=0.2, noise_std=0.0, seed=0)
    y = aug(x)
    assert y.shape == x.shape
    # identical views stay identical after a shared shift
    same = x[:1].repeat(4, 1, 1, 1)
    out = Augment(0.2, 0.0, seed=1)(same)
    assert torch.allclose(out[0], out[3])


# --------------------------------------------------------------------------- #
# Downloading / caching
# --------------------------------------------------------------------------- #
def event_df(tmp_path, n=3, bad_view=None):
    rows = []
    for e in range(n):
        urls = []
        for v in range(4):
            p = tmp_path / "src" / f"e{e}_v{v}.png"
            write_png(p, 10 * e + 2 * v + 5)
            urls.append(f"file://{p}")
        rows.append({"gravityspy_id": f"evt{e:07d}", "event_time": 1e9 + e, "ifo": "H1", "ml_label": "Blip",
                     "ml_confidence": 0.9, "label": 0, **dict(zip(VIEWS, urls))})
    return pd.DataFrame(rows)


def test_images_are_never_downloaded_twice(tmp_path):
    df = event_df(tmp_path)
    store = ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)
    assert store.ensure_downloaded(df, VIEWS).empty
    for p in (tmp_path / "src").glob("*.png"):
        p.unlink()  # sources vanish: a re-download would now fail
    assert store.ensure_downloaded(df, VIEWS).empty


class _Resp:
    def __init__(self, status, content=b""):
        self.status_code, self._c = status, content

    def iter_content(self, n):
        yield self._c


def test_retries_transient_errors_then_succeeds(tmp_path):
    png = write_png(tmp_path / "ok.png", 90).read_bytes()
    calls = []

    class Sess:
        def get(self, url, **kw):
            calls.append(url)
            return _Resp(503) if len(calls) < 3 else _Resp(200, png)

    store = ImageStore(tmp_path / "img", min_bytes=1, retries=4, backoff_s=0.0)
    store._session = lambda: Sess()
    ok, reason = store.fetch("https://example.org/x.png", tmp_path / "img" / "a" / "x.png")
    assert ok and len(calls) == 3 and reason == "downloaded"


def test_client_errors_are_not_retried(tmp_path):
    calls = []

    class Sess:
        def get(self, url, **kw):
            calls.append(1)
            return _Resp(404)

    store = ImageStore(tmp_path / "img", min_bytes=1, retries=4, backoff_s=0.0)
    store._session = lambda: Sess()
    ok, reason = store.fetch("https://example.org/x.png", tmp_path / "img" / "x.png")
    assert not ok and "404" in reason and len(calls) == 1


def test_missing_source_is_reported_as_failure(tmp_path):
    df = event_df(tmp_path, n=2)
    df.loc[1, "url3"] = "file:///definitely/not/here.png"
    store = ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)
    fail = store.ensure_downloaded(df, VIEWS)
    assert len(fail) == 1 and fail.iloc[0]["gravityspy_id"] == "evt0000001" and fail.iloc[0]["view"] == 3


def test_event_with_one_bad_view_is_dropped_whole(tmp_path):
    df = event_df(tmp_path, n=3)
    Path(df.loc[1, "url2"][len("file://"):]).write_bytes(b"garbage" * 100)
    cfg = {"data": {"view_columns": VIEWS, "image_size": [16, 16], "channels": 1, "resize_mode": "bilinear"},
           "paths": {"processed_dir": str(tmp_path / "proc")}}
    store = ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)
    ds, kept, failures = prepare_split(df, "t", cfg, store)
    assert len(ds) == 2 and "evt0000001" not in set(kept["gravityspy_id"])
    assert len(failures) >= 1
    assert ds.ids == kept["gravityspy_id"].tolist()


def test_views_stay_aligned_to_their_event(tmp_path):
    """Each view's mean intensity encodes (event, view). Verify the array honours it."""
    df = event_df(tmp_path, n=4)
    store = ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)
    assert store.ensure_downloaded(df, VIEWS).empty
    pre = PreprocessConfig((20, 24), 1)
    path, kept, _ = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    arr = np.load(path)
    assert arr.shape == (4, 4, 1, 20, 24)
    for e in range(4):
        for v in range(4):
            base = 10 * e + 2 * v + 5
            assert abs(arr[e, v].mean() - (base + 10)) < 1.0  # texture adds +20 to half the rows


def test_array_cache_is_reused(tmp_path):
    df = event_df(tmp_path, n=3)
    store = ImageStore(tmp_path / "img", min_bytes=1, workers=2, retries=0, backoff_s=0)
    store.ensure_downloaded(df, VIEWS)
    pre = PreprocessConfig((16, 16), 1)
    p1, _, _ = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    m1 = p1.stat().st_mtime_ns
    p2, kept, _ = build_split_array(df, store, pre, VIEWS, tmp_path / "proc", "t")
    assert p1 == p2 and p2.stat().st_mtime_ns == m1 and len(kept) == 3


# --------------------------------------------------------------------------- #
# Dataset / sampler on fake Gravity-Spy-like data
# --------------------------------------------------------------------------- #
def test_dataset_items_and_label_alignment(cfg):
    train, val, mapping, _ = build_source_splits(cfg)
    store = ImageStore.from_cfg(cfg, cfg["paths"]["image_dir"])
    ds, kept, failures = prepare_split(train.iloc[:30], "t", cfg, store)
    x, y = ds[0]
    assert x.shape == (4, 1, 32, 32) and x.dtype == torch.float32 and y.dtype == torch.long
    assert len(failures) == 0 and len(ds) == 30
    assert [mapping.names[int(ds[i][1])] for i in range(30)] == kept["ml_label"].tolist()


def test_dataset_rejects_misaligned_inputs(tmp_path):
    arr = np.zeros((5, 4, 1, 8, 8), np.uint8)
    np.save(tmp_path / "a.npy", arr)
    with pytest.raises(ValueError, match="Misaligned"):
        GlitchDataset(tmp_path / "a.npy", np.zeros(4, int), [f"i{i}" for i in range(5)])


def test_balanced_sampler_equalises_classes():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    s = make_sampler(labels, "balanced", 3, seed=0)
    drawn = labels[list(s)]
    frac = class_counts(drawn, 3) / len(drawn)
    assert np.all(np.abs(frac - 1 / 3) < 0.06)
    assert make_sampler(labels, "none", 3, 0) is None
    with pytest.raises(ValueError):
        make_sampler(labels, "weird", 3, 0)
