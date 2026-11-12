"""End-to-end tests on synthetic Gravity-Spy-like data: train -> artifacts -> val eval -> gated target eval."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.fake_data import FAKE_CLASSES, LEGACY, UNSEEN, make_fake_dataset, make_fake_run
from src.data.preprocessing import preprocessing_signature
from src.models import evaluate as ev
from src.models import train as tr


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    make_fake_dataset(root, n_events=500, ifos=("H1",), seed=11)
    return root


def cli(env, *extra, name="smoke"):
    return ["--debug", "--device", "cpu", "--name", name,
            "--metadata-dir", str(env / "metadata"), "--image-dir", str(env / "img"),
            "--processed-dir", str(env / "proc"), "--output-dir", str(env / "runs"),
            "--set", "data.crop.enabled=false",      # fake images are 140x170; the real baseline crops 800x600
            "--set", "split.gap_seconds=1800", "--set", "split.block_seconds=3600",
            "--set", "split.min_val_per_class=1", *extra]


@pytest.fixture(scope="module")
def run_dir(env):
    tr.main(cli(env, name="smoke"))
    return env / "runs" / "smoke_debug"


def test_training_writes_all_artifacts(run_dir):
    expected = [
        "config.yaml", "class_mapping.json", "history.csv", "summary.json", "training_curves.png",
        "image_failures.csv", "train.log",
        "splits/train_manifest.csv", "splits/val_manifest.csv", "splits/train_class_counts.csv",
        "splits/val_class_counts.csv", "splits/purged_events.csv",
        "checkpoints/best.pt", "checkpoints/last.pt",
        "eval_val_best/metrics.json", "eval_val_best/per_class.csv", "eval_val_best/confusion_matrix.csv",
    ]
    for rel in expected:
        assert (run_dir / rel).exists(), f"missing {rel}"
    assert len(pd.read_csv(run_dir / "history.csv")) == 2


def test_class_mapping_is_explicit(run_dir):
    m = json.loads((run_dir / "class_mapping.json").read_text())
    assert "None_of_the_Above" in m["excluded_from_training"]
    assert "None_of_the_Above" not in m["class_to_idx"]
    assert set(UNSEEN) == set(m["unseen_in_source_run"])


def test_saved_manifests_are_temporally_disjoint(run_dir):
    tr_m = pd.read_csv(run_dir / "splits" / "train_manifest.csv")
    va_m = pd.read_csv(run_dir / "splits" / "val_manifest.csv")
    assert set(tr_m["gravityspy_id"]).isdisjoint(va_m["gravityspy_id"])
    assert tr_m["event_time"].max() < va_m["event_time"].min()
    assert tr_m["event_time"].max() < 1.2e9  # O2-era GPS times only; no O3a (>1.238e9) event in training
    assert va_m["event_time"].max() < 1.2e9


def test_checkpoint_selection_used_source_val_only(run_dir):
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["best_epoch"] in (1, 2)
    assert 0.0 <= s["val_metrics"]["macro_f1_present"] <= 1.0


def test_evaluate_val_reproduces_training_metrics(run_dir):
    ev.main(["--run-dir", str(run_dir), "--split", "val", "--device", "cpu"])
    again = json.loads((run_dir / "eval_val_best" / "metrics.json").read_text())
    s = json.loads((run_dir / "summary.json").read_text())
    assert again["macro_f1_present"] == pytest.approx(s["val_metrics"]["macro_f1_present"], abs=1e-6)
    assert again["accuracy"] == pytest.approx(s["val_metrics"]["accuracy"], abs=1e-6)


def test_target_evaluation_refuses_without_final_flag(run_dir):
    with pytest.raises(SystemExit, match="--final"):
        ev.main(["--run-dir", str(run_dir), "--split", "target", "--device", "cpu"])
    assert not (run_dir / "eval_target_O3a").exists()


def test_final_target_evaluation_then_rerun_is_blocked(run_dir):
    ev.main(["--run-dir", str(run_dir), "--split", "target", "--final", "--device", "cpu"])
    root = run_dir / "eval_target_O3a"
    totals = {}
    for tau in ("none", "0.7", "0.9"):
        rep = json.loads((root / f"conf_{tau}" / "report.json").read_text())
        sc = rep["sample_counts"]
        assert sc["total"] == sc["shared_closed_set"] + sc["legacy_excluded"] + sc["new_o3_classes"]
        assert set(rep["legacy_excluded_labels"]["label_counts"]) <= set(LEGACY)
        assert set(rep["new_o3_classes"]["label_counts"]) <= set(UNSEEN)
        assert sc["new_o3_classes"] > 0
        assert "accuracy_outside_label_space_as_incorrect" in rep["strict"]
        for f in ("closed_set_metrics.json", "closed_set_per_class.csv", "closed_set_confusion_matrix.csv"):
            assert (root / f"conf_{tau}" / f).exists()
        totals[tau] = sc["total"]
    assert totals["0.9"] <= totals["0.7"] <= totals["none"]
    preds = pd.read_csv(root / "predictions.csv")
    assert preds["gravityspy_id"].is_unique  # the injected exact duplicate was removed
    assert set(preds["category"]) <= {"shared_closed_set", "legacy_excluded", "new_o3_class"}
    summ = json.loads((run_dir / "target_summary.json").read_text())["O3a"]["events"]
    assert summ["cap_applied"] is True and summ["cap_per_class"] == 20  # debug run only
    log = json.loads((run_dir / "final_eval_log.json").read_text())
    assert log["checkpoint_sha256"] and log["config_hash"] and log["debug_run"] is True
    with pytest.raises(RuntimeError, match="already had its final"):
        ev.main(["--run-dir", str(run_dir), "--split", "target", "--final", "--device", "cpu"])
    ev.main(["--run-dir", str(run_dir), "--split", "target", "--final", "--allow-rerun", "--device", "cpu"])
    assert json.loads((run_dir / "final_eval_log.json").read_text())["allow_rerun"] is True


def test_full_target_evaluation_uses_every_eligible_event(env):
    """Debug-sized training, but the target cap is null: every deduplicated O3a event is scored."""
    tr.main(cli(env, "--set", "evaluate.max_per_class_target=null", "--set", "train.epochs=1", name="full"))
    run_dir = env / "runs" / "full_debug"
    ev.main(["--run-dir", str(run_dir), "--split", "target", "--final", "--device", "cpu"])
    summ = json.loads((run_dir / "target_summary.json").read_text())["O3a"]
    e = summ["events"]
    n_csv = len(pd.read_csv(env / "metadata" / "H1_O3a.csv"))
    assert e["rows_in_csv"] == n_csv == 251                    # 250 events + 1 injected duplicate row
    assert e["after_cleaning"] == e["evaluated_pool"] == e["scored"] == 250
    assert e["cap_applied"] is False and e["cap_per_class"] is None
    assert e["dropped_for_image_failures"] == 0
    rep = summ["thresholds"]["none"]
    assert rep["sample_counts"]["total"] == 250
    assert rep["sample_counts"]["legacy_excluded"] > 0 and rep["sample_counts"]["new_o3_classes"] > 0
    table = pd.read_csv(run_dir / "eval_target_O3a" / "threshold_summary.csv")
    assert table["confidence_threshold"].tolist() == ["none", "0.7", "0.9"]
    assert table["n_total"].is_monotonic_decreasing
    assert (table["n_total"] == table["n_shared"] + table["n_legacy_excluded"] + table["n_new_o3"]).all()


def test_training_refuses_to_use_a_target_run_as_source(env):
    with pytest.raises(ValueError, match="target_runs"):
        tr.main(cli(env, "--set", "protocol.source_run=O3a", name="leak"))


def test_existing_run_dir_needs_overwrite(env, run_dir):
    with pytest.raises(FileExistsError):
        tr.main(cli(env, name="smoke"))


def test_baseline_model_trains_end_to_end(env):
    tr.main(cli(env, "--model", "baseline_merged", "--set", "train.epochs=1", name="base"))
    assert (env / "runs" / "base_debug" / "checkpoints" / "best.pt").exists()


def test_no_confidence_filter_option(env):
    tr.main(cli(env, "--set", "confidence.train=null", "--set", "confidence.val=null",
                "--set", "train.epochs=1", name="nofilter"))
    counts = pd.read_csv(env / "runs" / "nofilter_debug" / "splits" / "train_class_counts.csv", index_col=0)
    assert (counts.iloc[:, 0] >= counts.iloc[:, 1]).all()


# --------------------------------------------------------------------------- #
# Approved baseline preprocessing through the real training entry point
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def layout_env(tmp_path_factory):
    """Synthetic 800x600 RGBA O2 events. The metadata directory holds ONLY H1_O2.csv: no O3a file exists."""
    root = tmp_path_factory.mktemp("layout_pipeline")
    make_fake_run(root, "H1", "O2", 30, 1.164e9, FAKE_CLASSES, seed=4, style="layout800")
    assert sorted(p.name for p in (root / "metadata").iterdir()) == ["H1_O2.csv"]
    return root


def layout_cli(root, name, *extra):
    return ["--debug", "--device", "cpu", "--name", name, "--metadata-dir", str(root / "metadata"),
            "--image-dir", str(root / "img"), "--processed-dir", str(root / "proc" / name),
            "--output-dir", str(root / "runs"), "--set", "split.gap_seconds=300", "--set", "split.val_fraction=0.3",
            "--set", "split.min_val_per_class=1", "--set", "confidence.train=null", "--set", "confidence.val=null",
            "--set", "train.epochs=1", *extra]


def tree_hashes(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.png"))}


def test_training_uses_the_approved_baseline_dimensions_end_to_end(layout_env):
    root = layout_env
    src_before = tree_hashes(root / "png")
    tr.main(layout_cli(root, "base128", "--set", "data.image_size=[128,153]"))     # approved size (debug shrinks it)
    run_dir = root / "runs" / "base128_debug"

    cfg = json.loads(json.dumps(__import__("yaml").safe_load((run_dir / "config.yaml").read_text())))
    assert cfg["data"]["crop"]["enabled"] is True and cfg["data"]["crop"]["box"] == [101, 61, 674, 540]
    assert cfg["data"]["crop"]["reference_size"] == [800, 600] and cfg["data"]["channels"] == 3
    assert cfg["data"]["image_size"] == [128, 153] and cfg["data"]["normalization"]["mode"] == "unit"
    assert cfg["data"]["resize_mode"] == "bilinear" and cfg["data"]["alpha_background"] == "white"

    arrays = sorted((root / "proc" / "base128" / "arrays").glob("*.npy"))
    assert arrays and all(np.load(a, mmap_mode="r").shape[1:] == (4, 3, 128, 153) for a in arrays)   # 4 views, RGB

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["input_shape_per_event"] == [4, 3, 128, 153]
    sig = json.loads((run_dir / "preprocessing.json").read_text())
    assert sig["hash"] == preprocessing_signature(cfg["data"])["hash"] == summary["preprocessing"]["hash"]
    assert sig["geometry"]["crop"] == [101, 61, 674, 540] and sig["views"] == ["url1", "url2", "url3", "url4"]
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["preprocessing"]["hash"] == sig["hash"]
    assert (run_dir / "image_failures.csv").read_text().strip().count("\n") == 0      # header only: no failures

    # originals preserved: source PNGs unchanged, and every cached copy is byte-identical to its source
    assert tree_hashes(root / "png") == src_before
    cached = list(tree_hashes(root / "img").values())
    assert cached and set(cached) <= set(src_before.values())          # cached names differ (<id>_v1.png); compare content
    # O3a never touched: no file, and nothing in the log mentions it
    assert "O3a" not in (run_dir / "train.log").read_text()


def test_debug_size_is_aspect_preserving_and_trains(layout_env):
    root = layout_env
    tr.main(layout_cli(root, "dbg"))
    summary = json.loads((root / "runs" / "dbg_debug" / "summary.json").read_text())
    assert summary["input_shape_per_event"] == [4, 3, 64, 77]
    cfg = tr.build_cfg(__import__("argparse").Namespace(
        config=str(__import__("src.utils", fromlist=["project_root"]).project_root() / "configs" / "classifier.yaml"),
        debug=True, ifo=None, model=None, seed=None, device=None, metadata_dir=None, image_dir=None,
        processed_dir=None, output_dir=None, set=None, name=None))
    from src.data.preprocessing import aspect_distortion
    assert aspect_distortion((573, 479), tuple(cfg["data"]["image_size"])) < 0.01


def test_stretching_size_fails_at_startup_before_any_download(layout_env):
    root = layout_env
    with pytest.raises(ValueError, match="stretches"):
        tr.main(layout_cli(root, "stretch", "--set", "data.image_size=[128,128]"))
    assert not (root / "runs" / "stretch_debug").exists()
