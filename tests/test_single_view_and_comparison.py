"""Single-view baselines, the unified comparison pipeline and the read-only O3a diagnostic renderer."""
from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data.dataset import GlitchDataset, model_input_views
from src.data.fake_data import make_fake_dataset
from src.evaluation import compare_models as cmp
from src.evaluation import o3a_diagnostic as o3a
from src.evaluation.metrics import (
    evaluate_predictions, per_class_table, reliability_bins, risk_coverage_curve, threshold_table,
)
from src.models import BaselineCNN, build_model, count_parameters
from src.models import evaluate as ev
from src.models import train as tr
from src.models import train_single_view as tsv
from src.models.evaluate import compute_metrics, expected_calibration_error
from src.utils import deep_update, file_sha256, load_config, project_root

NAMES = ["A", "B", "C", "D"]


@pytest.fixture()
def cfg():
    c = load_config(project_root() / "configs" / "classifier.yaml")
    c = deep_update(c, c["debug"])
    c["data"]["channels"] = 1
    return c


# --------------------------------------------------------------------------- #
# Dataset: view selection
# --------------------------------------------------------------------------- #
def _view_array(tmp_path, n=3, v=4):
    arr = np.zeros((n, v, 1, 4, 5), np.uint8)
    for i in range(n):
        for k in range(v):
            arr[i, k] = 10 * k + i          # every pixel encodes (event, view)
    np.save(tmp_path / "a.npy", arr)
    return tmp_path / "a.npy"


def test_dataset_returns_only_the_selected_view(tmp_path):
    path = _view_array(tmp_path)
    ids = ["e0", "e1", "e2"]
    full = GlitchDataset(path, np.array([0, 1, 2]), ids)
    assert full[1][0].shape == (4, 1, 4, 5)
    for k in range(4):
        ds = GlitchDataset(path, np.array([0, 1, 2]), ids, view_indices=[k])
        x, y = ds[2]
        assert x.shape == (1, 1, 4, 5) and int(y) == 2
        assert torch.allclose(x * 255.0, torch.full_like(x, 10 * k + 2))      # unit normalisation: x/255
        assert torch.equal(x[0], full[2][0][k])                               # identical pixels to the 4-view input


@pytest.mark.parametrize("bad", [[4], [-1], [], [1, 1]])
def test_dataset_rejects_invalid_view_indices(tmp_path, bad):
    with pytest.raises(ValueError, match="view_indices"):
        GlitchDataset(_view_array(tmp_path), None, ["e0", "e1", "e2"], view_indices=bad)


def test_model_input_views_follows_model_type(cfg):
    assert model_input_views(cfg) is None                          # multiview: all views
    for t in ("baseline_merged", "baseline_single"):
        cfg["model"]["type"] = t
        assert model_input_views(cfg) is None                      # legacy types unchanged
    cfg["model"]["type"] = "single_view"
    for k in range(4):
        cfg["model"]["single_view_index"] = k
        assert model_input_views(cfg) == [k]
    cfg["model"]["single_view_index"] = 4
    with pytest.raises(ValueError, match="single_view_index"):
        model_input_views(cfg)


# --------------------------------------------------------------------------- #
# Model / checkpoint
# --------------------------------------------------------------------------- #
def test_single_view_forward_takes_exactly_one_view(cfg):
    cfg["model"]["type"] = "single_view"
    cfg["model"]["single_view_index"] = 2
    m = build_model(cfg, 5).eval()
    x = torch.randn(3, 1, 1, 32, 32)
    out = m(x)
    assert out.shape == (3, 5) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="built for 1 views"):
        m(torch.randn(3, 4, 1, 32, 32))


def test_single_view_reuses_the_baseline_backbone(cfg):
    cfg["model"]["type"] = "single_view"
    sv = build_model(cfg, 5)
    cfg["model"]["type"] = "baseline_single"
    legacy = build_model(cfg, 5)
    assert isinstance(sv, BaselineCNN)
    assert count_parameters(sv) == count_parameters(legacy)
    assert {k: v.shape for k, v in sv.state_dict().items()} == {k: v.shape for k, v in legacy.state_dict().items()}


def test_single_view_matches_legacy_baseline_on_the_same_view(cfg):
    """With identical weights, feeding only view k == legacy model reading view k out of four."""
    cfg["model"]["type"] = "baseline_single"
    cfg["model"]["single_view_index"] = 3
    legacy = build_model(cfg, 5).eval()
    cfg["model"]["type"] = "single_view"
    sv = build_model(cfg, 5).eval()
    sv.load_state_dict(legacy.state_dict())
    x = torch.randn(4, 4, 1, 32, 32)
    assert torch.allclose(sv(x[:, 3:4]), legacy(x), atol=1e-6)


def test_checkpoint_roundtrip_and_validation(cfg, tmp_path, caplog):
    from src.data.preprocessing import preprocessing_signature
    from src.data.splits import ClassMapping
    cfg["model"]["type"] = "single_view"
    cfg["model"]["single_view_index"] = 1
    mapping = ClassMapping(NAMES)
    torch.manual_seed(0)
    m = build_model(cfg, len(NAMES)).eval()
    sig = preprocessing_signature(cfg["data"])
    path = tmp_path / "ck" / "best.pt"
    tr.save_checkpoint(path, m, cfg, mapping, 3, {"accuracy": 0.5}, sig)
    loaded, cfg2, mapping2 = ev.load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(2, 1, 1, 32, 32)
    assert torch.allclose(loaded(x), m(x)) and mapping2.names == NAMES
    assert cfg2["model"]["type"] == "single_view" and cfg2["model"]["single_view_index"] == 1

    bad = dict(cfg, model=dict(cfg["model"], type="multiview"))           # architecture does not match weights
    tr.save_checkpoint(tmp_path / "bad.pt", m, bad, mapping, 3, {}, sig)
    with pytest.raises(RuntimeError, match="does not match the architecture"):
        ev.load_checkpoint(tmp_path / "bad.pt", torch.device("cpu"))

    tr.save_checkpoint(tmp_path / "hash.pt", m, cfg, mapping, 3, {}, {**sig, "hash": "000000000000"})
    with caplog.at_level(logging.WARNING):
        ev.load_checkpoint(tmp_path / "hash.pt", torch.device("cpu"))
    assert "preprocessing hash" in caplog.text


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _probs(pred, conf, c=4):
    p = np.full((len(pred), c), 0.0)
    for i, (k, s) in enumerate(zip(pred, conf)):
        p[i] = (1 - s) / (c - 1)
        p[i, k] = s
    return p


def test_metrics_match_sklearn_and_keep_all_classes():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2])            # class D (3) absent from the evaluated set
    pred = np.array([0, 0, 1, 1, 1, 2, 0, 0])
    conf = np.array([.9, .8, .6, .95, .7, .55, .4, .65])
    e = evaluate_predictions(y, _probs(pred, conf), NAMES)
    o = e["overall"]
    assert o["accuracy"] == pytest.approx(5 / 8)
    assert o["macro_f1"] == pytest.approx(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    assert o["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted", zero_division=0))
    assert o["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert o["ece"] == pytest.approx(expected_calibration_error(conf, pred == y))
    pc = e["per_class"]
    assert pc["class"].tolist() == NAMES                                  # mapping order, nothing dropped
    d = pc.set_index("class").loc["D"]
    assert d["support"] == 0 and np.isnan(d["recall"]) and np.isnan(d["f1"]) and "absent" in d["note"]
    assert pc["support"].sum() == len(y)


def test_never_predicted_class_has_undefined_precision_but_zero_f1():
    m = compute_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 0, 0]), NAMES[:3])
    c = per_class_table(m, NAMES[:3]).set_index("class").loc["C"]
    assert c["predicted"] == 0 and np.isnan(c["precision"]) and c["recall"] == 0 and c["f1"] == 0
    assert "never predicted" in c["note"]


def test_confusion_matrix_uses_fixed_class_order():
    y, pred = np.array([2, 2, 0]), np.array([2, 0, 0])
    e = evaluate_predictions(y, _probs(pred, [.9, .9, .9]), NAMES)
    cm = e["confusion_matrix"]
    assert cm.shape == (4, 4)
    assert cm[2, 2] == 1 and cm[2, 0] == 1 and cm[0, 0] == 1 and cm.sum() == 3
    assert cm[1].sum() == 0 and cm[3].sum() == 0                          # absent classes kept as empty rows


def test_threshold_coverage_and_retained_metrics():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    pred = np.array([0, 1, 2, 3, 1, 0, 2, 0, 0, 2])
    conf = np.array([.95, .91, .75, .72, .69, .55, .51, .45, .3, .2])
    t = threshold_table(y, pred, conf, NAMES, conf, "model_confidence").set_index("threshold")
    assert t.loc["none", "n_retained"] == 10 and t.loc["none", "coverage"] == 1.0
    assert (t.loc["0.5", "n_retained"], t.loc["0.7", "n_retained"], t.loc["0.9", "n_retained"]) == (7, 4, 2)
    assert t.loc["0.7", "coverage"] == pytest.approx(0.4)
    assert t.loc["0.7", "accuracy_retained"] == 1.0 and t.loc["0.9", "accuracy_retained"] == 1.0
    assert t.loc["0.5", "accuracy_retained"] == pytest.approx(5 / 7)
    assert t.loc["0.5", "selective_risk_retained"] == pytest.approx(2 / 7)
    assert np.allclose(t["accuracy_all_samples"], 0.6)                  # all-sample column never changes
    assert t.loc["0.9", "num_classes_present_retained"] == 2
    lab = threshold_table(y, pred, conf, NAMES, np.full(10, 0.6), "label_confidence").set_index("threshold")
    assert lab.loc["0.5", "n_retained"] == 10 and lab.loc["0.7", "n_retained"] == 0
    assert set(lab["threshold_type"]) == {"label_confidence"}


def test_empty_retained_subset_is_reported_not_dropped():
    t = threshold_table(np.array([0, 1]), np.array([0, 1]), np.array([.3, .4]), NAMES, np.array([.3, .4]), "m")
    row = t.set_index("threshold").loc["0.9"]
    assert row["n_retained"] == 0 and row["coverage"] == 0 and np.isnan(row["accuracy_retained"])


def test_risk_coverage_and_reliability():
    conf = np.array([.9, .8, .7, .6])
    correct = np.array([True, False, True, False])
    cov, risk, aurc = risk_coverage_curve(conf, correct)
    assert cov.tolist() == [.25, .5, .75, 1.0]
    assert risk == pytest.approx([0, .5, 1 / 3, .5])
    assert aurc == pytest.approx(np.mean([0, .5, 1 / 3, .5]))
    bins = reliability_bins(conf, correct)
    assert bins["count"].sum() == 4
    ece = (bins["count"] / 4 * (bins["accuracy"] - bins["mean_confidence"]).abs()).sum()
    assert ece == pytest.approx(expected_calibration_error(conf, correct))


def test_fast_bootstrap_scores_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    acc, mf1, bal = cmp._fast_scores(y, p, 6)                              # class 5 absent
    assert acc == pytest.approx((y == p).mean())
    assert mf1 == pytest.approx(f1_score(y, p, labels=range(5), average="macro"))
    assert bal == pytest.approx(balanced_accuracy_score(y, p))


def test_paired_bootstrap_identical_models_show_no_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    p = np.where(rng.random(400) < .7, y, rng.integers(0, 3, 400))
    t = np.sort(rng.uniform(0, 30 * 86400, 400))
    d = cmp.paired_bootstrap(y, {"ref": p, "same": p.copy(), "perfect": y.copy()}, "ref", t, 3, 200, 86400, 0)
    same = d[d.model == "same"]
    assert (same["difference_model_minus_reference"] == 0).all() and same["verdict"].str.contains("no clear").all()
    assert (d[d.model == "perfect"]["verdict"] == "higher than reference").all()


def test_paired_bootstrap_refuses_with_too_few_time_blocks():
    y = np.array([0, 1, 2] * 20)
    t = np.linspace(0, 3600, 60)                                          # all events inside one day
    d = cmp.paired_bootstrap(y, {"ref": y.copy(), "bad": np.zeros(60, int)}, "ref", t, 3, 100, 86400, 0)
    assert d["ci95_low"].isna().all() and d["verdict"].str.startswith("not assessed").all()
    assert (d["difference_model_minus_reference"] < 0).all()             # point estimate still reported


def test_population_report_flags_differences():
    def fake(label, split, kept):
        return {"spec": cmp.RunSpec(__import__("pathlib").Path(label), label, "x", "", None),
                "val_ids": split, "kept": pd.DataFrame({"gravityspy_id": kept})}
    same = cmp.population_report([fake("a", ["1", "2", "3"], ["1", "2", "3"]), fake("b", ["1", "2", "3"], ["1", "2", "3"])])
    assert same["identical_split"] and same["identical_scored_population"] and same["n_common"] == 3
    diff = cmp.population_report([fake("a", ["1", "2", "3"], ["1", "2", "3"]), fake("b", ["1", "2", "3"], ["1", "3"])])
    assert diff["identical_split"] and not diff["identical_scored_population"]
    assert diff["n_common"] == 2 and diff["excluded"]["gravityspy_id"].tolist() == ["2"]
    assert "image failure" in diff["excluded"]["reason"].iloc[0]


# --------------------------------------------------------------------------- #
# Single-view CLI helpers
# --------------------------------------------------------------------------- #
def test_view_parsing_and_naming():
    assert tsv.parse_views(["all"]) == [0.5, 1.0, 2.0, 4.0]
    assert tsv.parse_views(["0.5", "2s"]) == [0.5, 2.0]
    assert tsv.view_index_for(2, [0.5, 1.0, 2.0, 4.0]) == 2
    with pytest.raises(ValueError):
        tsv.view_index_for(3, [0.5, 1.0, 2.0, 4.0])
    c = {"data": {"ifo": "H1"}, "protocol": {"source_run": "O2"}}
    assert [tsv.default_run_name(c, v) for v in (0.5, 1.0, 2.0, 4.0)] == [
        "H1_O2_single_view_0.5s", "H1_O2_single_view_1s", "H1_O2_single_view_2s", "H1_O2_single_view_4s"]


def test_protocol_diff_classifies_differences():
    ref = {"split": {"gap_seconds": 21600}, "model": {"type": "multiview"}, "train": {"epochs": 30}}
    new = {"split": {"gap_seconds": 3600}, "model": {"type": "single_view"}, "train": {"epochs": 10}}
    d = tsv.protocol_diff(new, ref)
    keys = {r["key"]: r for r in d["all"]}
    assert keys["model.type"]["expected"] and not keys["model.type"]["protocol_critical"]
    assert keys["split.gap_seconds"]["protocol_critical"]
    assert not keys["train.epochs"]["expected"] and not keys["train.epochs"]["protocol_critical"]
    assert [r["key"] for r in d["protocol_critical"]] == ["split.gap_seconds"]


# --------------------------------------------------------------------------- #
# End to end on synthetic data: multi-view reference -> single-view runs -> comparison -> O3a renderer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("single_view_e2e")
    make_fake_dataset(root, n_events=400, ifos=("H1",), seed=5)
    common = ["--metadata-dir", str(root / "metadata"), "--image-dir", str(root / "img"),
              "--processed-dir", str(root / "proc"), "--output-dir", str(root / "runs")]
    tr.main(["--debug", "--device", "cpu", "--name", "mv_ref", *common, "--set", "data.crop.enabled=false",
             "--set", "split.gap_seconds=1800", "--set", "split.min_val_per_class=1", "--set", "train.epochs=3"])
    ref = root / "runs" / "mv_ref_debug"
    ref_hash = file_sha256(ref / "checkpoints" / "best.pt", n=64)
    tsv.main(["--view", "0.5", "4", "--debug", "--device", "cpu", "--reference-run", str(ref)])
    return {"root": root, "ref": ref, "ref_hash": ref_hash,
            "sv": [root / "runs" / "H1_O2_single_view_0.5s_debug", root / "runs" / "H1_O2_single_view_4s_debug"]}


def test_single_view_runs_inherit_protocol_and_see_one_view(e2e):
    ref_val = pd.read_csv(e2e["ref"] / "splits" / "val_manifest.csv")
    ref_train = pd.read_csv(e2e["ref"] / "splits" / "train_manifest.csv")
    for run, idx in zip(e2e["sv"], (0, 3)):
        s = json.loads((run / "summary.json").read_text())
        assert s["input_shape_per_event"][0] == 1                              # ONE view per event
        cfg = load_config(run / "config.yaml")
        assert cfg["model"]["type"] == "single_view" and cfg["model"]["single_view_index"] == idx
        assert cfg["train"]["epochs"] == 3 and cfg["experiment"]["debug"] is True   # reference values kept
        assert cfg["provenance"]["input_view_column"] == f"url{idx + 1}"
        pc = json.loads((run / "protocol_check.json").read_text())
        assert pc["verified_against_reference"] and pc["differences"]["protocol_critical"] == []
        assert {r["key"] for r in pc["differences"]["all"]} <= tsv.EXPECTED_DIFFS
        for name, ref_df in (("val_manifest.csv", ref_val), ("train_manifest.csv", ref_train)):
            pd.testing.assert_frame_equal(pd.read_csv(run / "splits" / name), ref_df)   # identical events
        ck = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        assert ck["config"]["model"]["type"] == "single_view"
    assert file_sha256(e2e["ref"] / "checkpoints" / "best.pt", n=64) == e2e["ref_hash"]


def test_single_view_refuses_existing_run_and_skip_existing_works(e2e):
    with pytest.raises(FileExistsError):
        tsv.main(["--view", "0.5", "--debug", "--device", "cpu", "--reference-run", str(e2e["ref"])])
    tsv.main(["--view", "0.5", "--debug", "--device", "cpu", "--reference-run", str(e2e["ref"]), "--skip-existing"])


def test_single_view_refuses_protocol_critical_change(e2e, tmp_path):
    ns = ["--view", "1", "--device", "cpu", "--reference-run", str(e2e["ref"]), "--output-dir", str(tmp_path),
          "--set", "split.gap_seconds=60"]
    with pytest.raises(ValueError, match="protocol-critical"):
        tsv.main(ns)


def test_single_view_training_never_loads_the_target_run(e2e, tmp_path, monkeypatch):
    import src.data.splits as splits
    calls = []
    real = splits.load_run_metadata

    def spy(*a, **k):
        calls.append(a[2])
        return real(*a, **k)
    monkeypatch.setattr(splits, "load_run_metadata", spy)
    tsv.main(["--view", "2", "--debug", "--device", "cpu", "--reference-run", str(e2e["ref"]),
              "--output-dir", str(tmp_path), "--epochs", "1"])
    assert calls and set(calls) == {"O2"}


@pytest.fixture(scope="module")
def comparison(e2e):
    out = e2e["root"] / "comparison"
    cmp.main(["--multiview-run", str(e2e["ref"]), "--single-view-runs", *map(str, e2e["sv"]),
              "--out", str(out), "--device", "cpu", "--bootstrap", "50"])
    return out


def test_comparison_outputs_and_consistency(e2e, comparison):
    out = comparison
    for rel in ["comparison_table.csv", "summary.json", "README.md", "thresholds.csv", "all_metrics_long.csv",
                "per_class_f1.csv", "per_class_recall.csv", "per_class_precision.csv",
                "paired_differences_vs_multiview.csv", "population/population_by_model.csv",
                "population/excluded_samples.csv", "population/class_distribution_common.csv"]:
        assert (out / rel).exists(), rel
    for stem in ["overall_metric_comparison", "per_class_f1_comparison", "per_class_recall_comparison",
                 "reliability_diagrams", "confidence_distributions", "coverage_vs_threshold",
                 "accuracy_vs_threshold", "selective_risk_vs_coverage"]:
        assert (out / "plots" / f"{stem}.png").exists() and (out / "plots" / f"{stem}.pdf").exists(), stem
    t = pd.read_csv(out / "comparison_table.csv")
    assert t["Model"].tolist() == ["Multi-view (4 views)", "Single-view 0.5 s", "Single-view 4 s"]
    assert t["N evaluated"].nunique() == 1
    # numbers reproduce what train.py recorded for each model on the same events
    assert np.allclose(t["Accuracy"], t["Val accuracy in summary.json"], atol=1e-9)
    s = json.loads((out / "summary.json").read_text())
    assert s["population"]["identical_split"] and s["population"]["identical_scored_population"]
    assert pd.read_csv(out / "population" / "excluded_samples.csv").empty
    th = pd.read_csv(out / "thresholds.csv")
    assert set(th["threshold_type"]) == {"model_confidence", "label_confidence"}
    assert th.groupby(["model", "threshold_type"]).size().eq(4).all()            # none, 0.5, 0.7, 0.9
    assert (th["n_retained"] <= th["n_total"]).all()
    pf1 = pd.read_csv(out / "per_class_f1.csv")
    assert pf1["class"].tolist() == s["class_order"]
    cms = out / "plots" / "confusion_matrices"
    for suffix in ("counts", "normalised"):     # dotted run name (0.5s) must not collapse the two files into one
        assert (cms / f"H1_O2_single_view_0.5s_debug_{suffix}.png").exists()
        assert (cms / f"H1_O2_single_view_0.5s_debug_{suffix}.pdf").exists()
    for run in [e2e["ref"], *e2e["sv"]]:
        d = run / cmp.PER_MODEL_DIRNAME
        assert (d / "metrics.json").exists() and (d / "plots" / "confusion_matrix_normalised.pdf").exists()
    assert file_sha256(e2e["ref"] / "checkpoints" / "best.pt", n=64) == e2e["ref_hash"]


def test_comparison_refuses_to_overwrite(e2e, comparison):
    with pytest.raises(FileExistsError):
        cmp.main(["--multiview-run", str(e2e["ref"]), "--single-view-runs", *map(str, e2e["sv"]),
                  "--out", str(comparison), "--device", "cpu", "--bootstrap", "0"])


def test_comparison_autodiscovers_single_view_runs(e2e, tmp_path):
    found = cmp.discover_single_view_runs(e2e["root"] / "runs", "H1", "O2", include_debug=True)
    assert [p.name for p in found][:1] == ["H1_O2_single_view_0.5s_debug"] and e2e["sv"][1] in found
    assert e2e["ref"] not in found
    assert cmp.discover_single_view_runs(e2e["root"] / "runs", "H1", "O2", include_debug=False) == []


def test_o3a_renderer_is_read_only_and_matches_existing_evaluator(e2e, tmp_path, monkeypatch):
    diag = tmp_path / "diag"
    shutil.copytree(e2e["ref"], diag, ignore=shutil.ignore_patterns(cmp.PER_MODEL_DIRNAME))
    ev.main(["--run-dir", str(diag), "--split", "target", "--final", "--device", "cpu"])   # the existing guarded path
    from src.utils import setup_logging
    setup_logging()          # a fresh CLI process: detach evaluate.py's file handler from the diagnostic run
    before = o3a._snapshot(diag)

    import src.data.dataset as dataset
    import src.data.splits as splits

    def forbidden(*a, **k):
        raise AssertionError("the O3a renderer must not load metadata or download images")
    monkeypatch.setattr(splits, "load_run_metadata", forbidden)
    monkeypatch.setattr(dataset.ImageStore, "fetch", forbidden)
    res = o3a.render_report(diag)
    assert o3a._snapshot(diag) == before
    out = diag / o3a.OUT_DIRNAME / "eval_target_O3a"
    for f in ["README.txt", "label_confidence_thresholds.csv", "shared_per_class.csv",
              "plots/shared_confusion_matrix_normalised.png", "plots/model_confidence_by_category.png",
              "plots/outside_label_space_predicted_as.png", "plots/shared_selective_risk_vs_coverage.pdf"]:
        assert (out / f).exists(), f
    assert "PRELIMINARY" in (out / "README.txt").read_text()
    lab = pd.read_csv(out / "label_confidence_thresholds.csv").set_index("label_confidence_threshold")
    old = pd.read_csv(diag / "eval_target_O3a" / "threshold_summary.csv").set_index("confidence_threshold")
    for tau in ("none", "0.7", "0.9"):
        assert lab.loc[tau, "n_total_events"] == old.loc[tau, "n_total"]
        assert lab.loc[tau, "n_shared_class"] == old.loc[tau, "n_shared"]
        assert lab.loc[tau, "shared_accuracy"] == pytest.approx(old.loc[tau, "closed_set_accuracy"])
        assert lab.loc[tau, "shared_macro_f1"] == pytest.approx(old.loc[tau, "closed_set_macro_f1"])
        assert lab.loc[tau, "strict_accuracy_all_events"] == pytest.approx(old.loc[tau, "strict_accuracy"])
    r = res["results"]["eval_target_O3a"]
    assert r["n_events"] == r["n_shared"] + r["n_legacy_excluded"] + r["n_new_o3_class"] and r["n_new_o3_class"] > 0
    with pytest.raises(FileExistsError):
        o3a.render_report(diag)


def test_o3a_renderer_skips_cleanly_without_predictions(tmp_path):
    (tmp_path / "empty_run").mkdir()
    assert o3a.render_report(tmp_path / "empty_run") is None
    assert not (tmp_path / "empty_run" / o3a.OUT_DIRNAME).exists()
