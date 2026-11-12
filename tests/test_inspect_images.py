"""Tests for the image-inspection utility (offline, on fake PNGs)."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from PIL import Image

from src.data.fake_data import FAKE_CLASSES, make_fake_dataset
from src.data.inspect_images import inspect, sample_events
from src.data.splits import clean_metadata
from src.data import inspect_images
from src.utils import load_config, project_root

VIEWS = ["url1", "url2", "url3", "url4"]


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("inspect_fake")
    make_fake_dataset(root, n_events=240, ifos=("H1",), seed=5)
    return root


@pytest.fixture()
def cfg(fake_root, tmp_path):
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c["paths"] = {"metadata_dir": str(fake_root / "metadata"), "image_dir": str(tmp_path / "img"),
                  "processed_dir": str(tmp_path / "proc")}
    c["data"]["download"].update(workers=2, retries=0, backoff_s=0.0)
    return c


def test_contact_sheets_show_four_views_per_event(cfg, tmp_path):
    out = tmp_path / "insp"
    s = inspect(cfg, "H1", "O2", out, events_per_class=2, max_events=8, seed=1)
    assert s["n_events"] == 8 and not s["is_target_run"]
    sheets = sorted(out.glob("contact_sheet_*.png"))
    assert len(sheets) == 2 or len(sheets) == 1       # 6 events per sheet
    for p in sheets:
        with Image.open(p) as im:
            assert im.width > 400 and im.height > 100
    info = pd.read_csv(out / "image_info.csv")
    assert len(info) == 8 * 4 and set(info["view"]) == {1, 2, 3, 4}
    assert (info.groupby("gravityspy_id")["view"].count() == 4).all()
    events = pd.read_csv(out / "events.csv")
    assert set(events["ml_label"]) <= set(FAKE_CLASSES)
    assert (out / "inspection_summary.json").exists()


def test_summary_reports_native_image_format_not_config(cfg, tmp_path):
    out = tmp_path / "insp2"
    s = inspect(cfg, "H1", "O2", out, events_per_class=1, max_events=6)
    assert s["image_sizes_wh"] == ["140x170"]          # the fake PNGs' native size, untouched by preprocessing
    assert s["image_modes"] == ["RGB"]
    assert s["configured_image_size_hw"] == cfg["data"]["image_size"]


def test_sampling_is_stratified_seeded_and_bounded(fake_root):
    df = clean_metadata(pd.read_csv(fake_root / "metadata" / "H1_O2.csv"), "H1", VIEWS)
    a = sample_events(df, 2, 100, seed=3)
    b = sample_events(df, 2, 100, seed=3)
    assert a["gravityspy_id"].tolist() == b["gravityspy_id"].tolist()
    assert a["ml_label"].value_counts().max() <= 2 and a["ml_label"].nunique() == len(FAKE_CLASSES)
    small = sample_events(df, 5, 4, seed=3)
    assert len(small) == 4 and small["ml_label"].nunique() == 4      # round-robin keeps class diversity
    only = sample_events(df, 3, 50, seed=3, labels=["Blip"])
    assert set(only["ml_label"]) == {"Blip"} and len(only) == 3
    hi = sample_events(df, 50, 500, seed=3, min_confidence=0.9)
    assert (hi["ml_confidence"] >= 0.9).all()


def test_images_are_cached_and_reused(cfg, tmp_path):
    inspect(cfg, "H1", "O2", tmp_path / "a", events_per_class=1, max_events=4, seed=2)
    n_files = len(list((tmp_path / "img").rglob("*.png")))
    inspect(cfg, "H1", "O2", tmp_path / "b", events_per_class=1, max_events=4, seed=2)
    assert len(list((tmp_path / "img").rglob("*.png"))) == n_files == 16


def test_target_run_refused_without_explicit_flag(cfg, tmp_path):
    with pytest.raises(SystemExit, match="allow-target-inspection"):
        inspect(cfg, "H1", "O3a", tmp_path / "t")
    assert not (tmp_path / "t").exists()


def test_target_inspection_is_recorded_when_allowed(cfg, tmp_path):
    out = tmp_path / "t2"
    s = inspect(cfg, "H1", "O3a", out, events_per_class=1, max_events=4, allow_target_inspection=True)
    assert s["is_target_run"]
    log = json.loads((out / "inspection_log.json").read_text())
    assert log["allow_target_inspection"] is True and log["run"] == "O3a"


def test_missing_view_skips_event_and_is_logged(cfg, tmp_path, fake_root, caplog):
    df = pd.read_csv(fake_root / "metadata" / "H1_O2.csv")
    victim = df.iloc[0]["gravityspy_id"]
    md = tmp_path / "md"
    md.mkdir()
    df.loc[df["gravityspy_id"] == victim, "url3"] = "file:///does/not/exist.png"
    df.to_csv(md / "H1_O2.csv", index=False)
    cfg["paths"]["metadata_dir"] = str(md)
    with caplog.at_level("WARNING"):
        s = inspect(cfg, "H1", "O2", tmp_path / "m", events_per_class=500, max_events=500)
    assert victim in caplog.text
    ev = pd.read_csv(tmp_path / "m" / "events.csv")
    assert victim not in set(ev["gravityspy_id"]) and s["n_events"] == len(ev)
    assert len(pd.read_csv(tmp_path / "m" / "download_failures.csv")) == 1


def test_cli_entry_point(fake_root, tmp_path):
    inspect_images.main(["--ifo", "H1", "--run", "O2", "--events-per-class", "1", "--max-events", "3",
                         "--metadata-dir", str(fake_root / "metadata"), "--image-dir", str(tmp_path / "img"),
                         "--out-dir", str(tmp_path / "cli")])
    assert (tmp_path / "cli" / "contact_sheet_01.png").exists()
