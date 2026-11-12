"""Tests for the crop-suggestion / preview tools.

They run on a SYNTHETIC 800x600 RGBA layout whose plotting region is known. Recovering that box shows
the algorithm does what it claims on that layout; it says nothing about the real Gravity Spy figures,
which is why the tools always require a visual check before a box goes into the config.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from src.data import crop_preview
from src.data.dataset import prepare_split
from src.data.fake_data import FAKE_CLASSES, LAYOUT_PLOT_BOX, make_fake_run
from src.data.preprocessing import (
    SUPPORTED_RESOLUTIONS, PreprocessConfig, aspect_distortion, aspect_preserving_size, load_view,
)
from src.data.splits import TargetLeakageError
from src.utils import load_config, project_root

VIEWS = ["url1", "url2", "url3", "url4"]


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("layout_fake")
    make_fake_run(root, "H1", "O2", 30, 1.164e9, FAKE_CLASSES, seed=4, style="layout800", transparent_corner=True)
    return root


@pytest.fixture()
def cfg(fake_root, tmp_path):
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c["paths"] = {"metadata_dir": str(fake_root / "metadata"), "image_dir": str(tmp_path / "img"),
                  "processed_dir": str(tmp_path / "proc")}
    c["data"]["download"].update(workers=2, retries=0, backoff_s=0.0)
    return c


def sha_tree(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.png"))}


# --------------------------------------------------------------------------- #
# Box suggestion
# --------------------------------------------------------------------------- #
def test_longest_run():
    assert crop_preview.longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (4, 7)
    assert crop_preview.longest_run(np.array([1, 1, 1], bool)) == (0, 3)
    assert crop_preview.longest_run(np.zeros(4, bool)) is None


def test_suggest_box_finds_dense_region_and_ignores_sparse_text():
    std = np.zeros((600, 800))
    rng = np.random.default_rng(0)
    std[70:520, 115:685] = rng.uniform(6, 9, (450, 570))      # plotting region: dense variation
    std[18:38, 270:530] = 90.0                                 # title text that varies, but is sparse and thin
    std[300:303, 20:60] = 50.0                                 # a stray varying tick label
    assert crop_preview.suggest_box(std) == (115, 70, 685, 520)
    assert crop_preview.suggest_box(std, inset=5) == (120, 75, 680, 515)


def test_suggest_box_fails_clearly_without_variation():
    with pytest.raises(ValueError, match="No pixel varies"):
        crop_preview.suggest_box(np.zeros((100, 100)))


def test_suggest_recovers_the_known_box_on_the_synthetic_layout(cfg, tmp_path):
    out = tmp_path / "sug"
    s = crop_preview.suggest(cfg, "H1", "O2", out, events_per_class=2, max_events=12)
    got, true = s["suggested_box_ltrb"], list(LAYOUT_PLOT_BOX)
    assert max(abs(a - b) for a, b in zip(got, true)) <= 2
    assert s["image_size_wh"] == [800, 600] and s["n_images"] == 48
    assert all(vb and max(abs(a - b) for a, b in zip(got, vb)) <= 3 for vb in s["per_view_boxes_ltrb"])
    assert s["max_per_view_disagreement_px"] <= 3
    assert "SUGGESTION ONLY" in s["status"]
    assert f"box: {got}" in s["config_snippet"] and "enabled: true" in s["config_snippet"]
    for f in ("suggestion_maps.png", "suggestion_overlay_events.png"):
        assert (out / f).exists() and (out / f).stat().st_size > 1000
    assert (out / "crop_suggestion.json").exists()
    assert json.loads((out / "crop_suggestion.json").read_text())["suggested_box_ltrb"] == got


def test_suggest_reports_transparency(cfg, tmp_path):
    s = crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s", max_events=8)
    assert s["min_alpha"] == 0 and 0 < s["max_fraction_non_opaque"] < 0.001
    assert any("not fully opaque" in w for w in s["warnings"])


def test_suggest_warns_when_reference_size_does_not_match(cfg, tmp_path):
    cfg["data"]["crop"]["reference_size"] = [640, 480]
    s = crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s", max_events=6)
    assert any("reference_size" in w for w in s["warnings"])
    assert "reference_size: [800, 600]" in s["config_snippet"]       # snippet uses the real image size


def test_suggest_refuses_images_of_different_sizes(cfg, fake_root, tmp_path):
    import pandas as pd
    md = tmp_path / "md"
    md.mkdir()
    df = pd.read_csv(fake_root / "metadata" / "H1_O2.csv")
    odd = tmp_path / "odd.png"
    Image.fromarray(np.full((500, 700, 4), 255, np.uint8), "RGBA").save(odd)
    df.loc[0, "url2"] = f"file://{odd}"
    df.to_csv(md / "H1_O2.csv", index=False)
    cfg["paths"]["metadata_dir"] = str(md)
    with pytest.raises(ValueError, match="different sizes"):
        crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s", events_per_class=500, max_events=500)


# --------------------------------------------------------------------------- #
# Target data is never touched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd", [["suggest"], ["preview", "--box", "115", "70", "685", "520"]])
def test_target_run_is_refused_with_no_override(fake_root, tmp_path, cmd):
    args = [cmd[0], "--ifo", "H1", "--run", "O3a", *cmd[1:], "--metadata-dir", str(fake_root / "metadata"),
            "--image-dir", str(tmp_path / "img"), "--out-dir", str(tmp_path / "o")]
    with pytest.raises(TargetLeakageError):
        crop_preview.main(args)
    assert not (tmp_path / "img").exists()                 # nothing was downloaded


def test_there_is_no_flag_to_override_the_target_guard():
    import argparse
    parser_flags = set()
    orig = argparse.ArgumentParser.add_argument

    def spy(self, *a, **k):
        parser_flags.update(x for x in a if isinstance(x, str) and x.startswith("--"))
        return orig(self, *a, **k)

    argparse.ArgumentParser.add_argument = spy
    try:
        with pytest.raises(SystemExit):
            crop_preview.main(["suggest", "--help"])
    finally:
        argparse.ArgumentParser.add_argument = orig
    assert parser_flags and not any("target" in f or "allow" in f for f in parser_flags)


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
def test_preview_sizes_are_aspect_preserving_by_default():
    region = (570, 450)                                   # the synthetic layout's plot box
    sizes = crop_preview.preview_sizes(region, SUPPORTED_RESOLUTIONS)
    assert sizes == [(96, 122), (128, 162), (160, 203), (224, 284)]
    assert all(aspect_distortion(region, z) < 0.01 for z in sizes)
    square = crop_preview.preview_sizes(region, SUPPORTED_RESOLUTIONS, aspect_preserving=False)
    assert square == [(r, r) for r in SUPPORTED_RESOLUTIONS]
    assert all(aspect_distortion(region, z) > 0.2 for z in square)       # squares stretch this crop by ~27%
    # the approved baseline crop: 573x479 at 128 high is 153 wide
    assert crop_preview.preview_sizes((573, 479), [128])[0] == (128, 153)


def test_configured_size_is_listed_first_without_duplication():
    sizes = crop_preview.preview_sizes((573, 479), [96, 128, 160, 224], configured_size=(128, 153))
    assert sizes[0] == (128, 153) and sizes.count((128, 153)) == 1 and len(sizes) == 4
    assert crop_preview.preview_sizes((573, 479), [96], configured_size=(128, 153)) == [(128, 153), (96, 115)]


def test_render_variants_shapes_match_the_training_pipeline(cfg, tmp_path):
    events, store = crop_preview._load_events(cfg, "H1", "O2", 1, 1, 0)
    path = store.path_for(events.iloc[0]["gravityspy_id"], 0)
    sizes = [(96, 122), (128, 162), (128, 128)]
    var = crop_preview.render_variants(path, LAYOUT_PLOT_BOX, sizes)
    assert var["original"].shape == (600, 800, 3)
    assert var["crop_rgb"].shape == (450, 570, 3) and var["crop_gray"].shape == (450, 570)
    for h, w in sizes:
        assert var[f"rgb_{h}x{w}"].shape == (h, w, 3)
    assert not np.array_equal(var["crop_rgb"][..., 0], var["crop_rgb"][..., 2])
    full = crop_preview.render_variants(path, None, [(60, 80)])         # no crop = full image
    assert full["crop_rgb"].shape == (600, 800, 3) and full["rgb_60x80"].shape == (60, 80, 3)
    ref = load_view(path, PreprocessConfig((128, 162), 3, "bilinear", "white", LAYOUT_PLOT_BOX, (800, 600)))
    assert np.array_equal(var["rgb_128x162"], np.transpose(ref, (1, 2, 0)))   # preview == training pipeline


def test_preview_writes_one_sheet_per_event_with_aspect_preserving_variants(cfg, tmp_path):
    out = tmp_path / "prev"
    s = crop_preview.preview(cfg, "H1", "O2", out, list(LAYOUT_PLOT_BOX), events_per_class=1, max_events=3)
    assert len(s["sheets"]) == 3 and s["box_ltrb"] == list(LAYOUT_PLOT_BOX) and s["region_wh"] == [570, 450]
    assert s["sizes_hw"] == [[96, 122], [128, 162], [160, 203], [224, 284]] and s["aspect_preserving"] is True
    assert max(s["stretch_by_size"].values()) < 0.01
    for p in s["sheets"]:
        with Image.open(p) as im:
            assert im.width > 1200 and im.height > 600
    assert json.loads((out / "preview_summary.json").read_text())["image_size_wh"] == [800, 600]


def test_preview_square_option_is_available_and_reports_its_stretch(cfg, tmp_path):
    s = crop_preview.preview(cfg, "H1", "O2", tmp_path / "sq", list(LAYOUT_PLOT_BOX), max_events=1,
                             aspect_preserving=False)
    assert s["sizes_hw"] == [[r, r] for r in SUPPORTED_RESOLUTIONS]
    assert min(s["stretch_by_size"].values()) > 0.2


def test_preview_supports_no_crop_custom_heights_and_rejects_bad_boxes(cfg, tmp_path):
    s = crop_preview.preview(cfg, "H1", "O2", tmp_path / "a", None, max_events=1, resolutions=[64, 128])
    assert s["box_ltrb"] is None and s["region_wh"] == [800, 600]
    assert s["sizes_hw"] == [[64, 85], [128, 171]]                       # 4:3 full image, aspect preserved
    with pytest.raises(ValueError):
        crop_preview.preview(cfg, "H1", "O2", tmp_path / "b", [0, 0, 900, 500], max_events=1)


def test_preview_cli_from_config_shows_the_configured_size_first(fake_root, tmp_path):
    out = tmp_path / "o"
    crop_preview.main(["preview", "--from-config", "--ifo", "H1", "--metadata-dir", str(fake_root / "metadata"),
                       "--image-dir", str(tmp_path / "img"), "--out-dir", str(out), "--max-events", "1"])
    s = json.loads((out / "preview_summary.json").read_text())
    assert s["box_ltrb"] == [101, 61, 674, 540] and s["sizes_hw"][0] == [128, 153]
    assert s["stretch_by_size"]["128x153"] < 0.001
    assert len(list(out.glob("preview_*.png"))) == 1


def test_preview_cli_from_config_requires_an_enabled_box(fake_root, tmp_path):
    import yaml
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c["data"]["crop"] = {"enabled": False, "box": None, "reference_size": [800, 600]}
    cfg_path = tmp_path / "nocrop.yaml"
    cfg_path.write_text(yaml.safe_dump(c))
    with pytest.raises(SystemExit, match="enabled"):
        crop_preview.main(["preview", "--from-config", "--config", str(cfg_path), "--ifo", "H1",
                           "--metadata-dir", str(fake_root / "metadata"), "--image-dir", str(tmp_path / "img"),
                           "--out-dir", str(tmp_path / "o")])
    crop_preview.main(["preview", "--box", *map(str, LAYOUT_PLOT_BOX), "--max-events", "1", "--resolutions", "96",
                       "--ifo", "H1", "--metadata-dir", str(fake_root / "metadata"),
                       "--image-dir", str(tmp_path / "img"), "--out-dir", str(tmp_path / "o2")])
    assert len(list((tmp_path / "o2").glob("preview_*.png"))) == 1


def test_describe_prints_the_effective_preprocessing_without_reading_data(capsys):
    crop_preview.main(["describe"])
    info = json.loads(capsys.readouterr().out)
    assert info["crop_box_ltrb"] == [101, 61, 674, 540] and info["crop_size_wh"] == [573, 479]
    assert info["output_size_hw"] == [128, 153] and info["model_input_shape_per_event"] == [4, 3, 128, 153]
    assert info["aspect_distortion"] < 0.001 and info["stored_bytes_per_event"] == 4 * 3 * 128 * 153
    assert info["normalization"]["mode"] == "unit" and info["views"] == VIEWS


def test_tools_never_modify_original_pngs(cfg, fake_root, tmp_path):
    crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s", max_events=6)
    src_before, cache_before = sha_tree(fake_root / "png"), sha_tree(tmp_path / "img")
    assert cache_before
    crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s2", max_events=6)
    crop_preview.preview(cfg, "H1", "O2", tmp_path / "p", list(LAYOUT_PLOT_BOX), max_events=2)
    assert sha_tree(fake_root / "png") == src_before
    assert {k: v for k, v in sha_tree(tmp_path / "img").items() if k in cache_before} == cache_before


# --------------------------------------------------------------------------- #
# Suggested box used end to end
# --------------------------------------------------------------------------- #
def test_suggested_box_feeds_the_pipeline_at_every_aspect_preserving_height(cfg, tmp_path):
    s = crop_preview.suggest(cfg, "H1", "O2", tmp_path / "s", events_per_class=2, max_events=6)
    box = s["suggested_box_ltrb"]
    region = (box[2] - box[0], box[3] - box[1])
    events, store = crop_preview._load_events(cfg, "H1", "O2", 2, 6, 0)
    events = events.assign(label=0)
    for res in SUPPORTED_RESOLUTIONS:
        h, w = aspect_preserving_size(region, height=res)
        cfg["data"].update(image_size=[h, w], channels=3)
        cfg["data"]["crop"] = {"enabled": True, "reference_size": [800, 600], "box": box}
        ds, kept, failures = prepare_split(events, f"r{res}", cfg, store)
        assert failures.empty and len(ds) == len(events)
        assert ds[0][0].shape == (4, 3, h, w)
    cfg["data"]["image_size"] = [96, 96]                    # a square would stretch this crop: refused at startup
    with pytest.raises(ValueError, match="stretches"):
        prepare_split(events, "sq", cfg, store)
