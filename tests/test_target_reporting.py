"""Tests for the revised target-run reporting and the full-target-evaluation policy."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.data.splits import ClassMapping
from src.models.evaluate import (
    CAT_LEGACY, CAT_NEW, CAT_SHARED, assign_category, build_target_report, resolve_target_cap,
)
from src.utils import deep_update, load_config, project_root

MAPPING = ClassMapping(["A", "B", "C"], ["None_of_the_Above"], ["Blip_Low_Frequency", "Fast_Scattering"])
NAMES = MAPPING.names


def make_sel():
    """Hand-built scored events with known answers.

    shared (6):  A->A, A->A, A->B, B->B, B->B, C->A     => 4 correct
    legacy (3):  None_of_the_Above -> A, A, B
    new (5):     Blip_Low_Frequency -> A, B ; Fast_Scattering -> C, C, C
    total 14, strict accuracy 4/14
    """
    rows = []

    def add(lbl, pred, conf=0.9):
        rows.append({"ml_label": lbl, "label": MAPPING.class_to_idx.get(lbl, -1), "pred": MAPPING.class_to_idx[pred],
                     "pred_name": pred, "pred_conf": conf, "ml_confidence": conf})

    for lbl, pred in [("A", "A"), ("A", "A"), ("A", "B"), ("B", "B"), ("B", "B"), ("C", "A")]:
        add(lbl, pred)
    for pred in ["A", "A", "B"]:
        add("None_of_the_Above", pred)
    for pred in ["A", "B"]:
        add("Blip_Low_Frequency", pred)
    for pred in ["C", "C", "C"]:
        add("Fast_Scattering", pred)
    df = pd.DataFrame(rows)
    df["category"] = assign_category(df["ml_label"], MAPPING)
    return df


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #
def test_categories_are_separate():
    df = make_sel()
    assert set(df.loc[df.ml_label == "None_of_the_Above", "category"]) == {CAT_LEGACY}
    assert set(df.loc[df.ml_label.isin(["Blip_Low_Frequency", "Fast_Scattering"]), "category"]) == {CAT_NEW}
    assert set(df.loc[df.ml_label.isin(NAMES), "category"]) == {CAT_SHARED}


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="Mystery"):
        assign_category(pd.Series(["A", "Mystery"]), MAPPING)


# --------------------------------------------------------------------------- #
# Report content
# --------------------------------------------------------------------------- #
def test_sample_counts_distinguish_all_categories():
    rep, _ = build_target_report(make_sel(), MAPPING, None)
    sc = rep["sample_counts"]
    assert (sc["total"], sc["shared_closed_set"], sc["legacy_excluded"], sc["new_o3_classes"]) == (14, 6, 3, 5)
    assert sc["by_label"]["None_of_the_Above"] == 3 and sc["by_label"]["Fast_Scattering"] == 3


def test_legacy_and_new_distributions_are_not_merged():
    rep, _ = build_target_report(make_sel(), MAPPING, None)
    leg, new = rep["legacy_excluded_labels"], rep["new_o3_classes"]
    assert leg["label_counts"] == {"None_of_the_Above": 3}
    assert new["label_counts"] == {"Fast_Scattering": 3, "Blip_Low_Frequency": 2}
    assert set(leg["per_label"]) == {"None_of_the_Above"}
    assert set(new["per_label"]) == {"Blip_Low_Frequency", "Fast_Scattering"}
    pa = leg["per_label"]["None_of_the_Above"]["predicted_as"]
    assert pa["A"]["count"] == 2 and pa["A"]["fraction"] == pytest.approx(2 / 3, abs=1e-3) and pa["B"]["count"] == 1
    assert new["per_label"]["Fast_Scattering"]["predicted_as"]["C"]["fraction"] == 1.0
    assert leg["labels"] == ["None_of_the_Above"] and new["labels"] == ["Blip_Low_Frequency", "Fast_Scattering"]


def test_closed_set_metrics_ignore_outside_label_space_events():
    full, closed_full = build_target_report(make_sel(), MAPPING, None)
    only_shared = make_sel().query("category == @CAT_SHARED")
    alone, closed_alone = build_target_report(only_shared, MAPPING, None)
    cs = full["shared_closed_set"]
    assert cs["n"] == 6 and cs["accuracy"] == pytest.approx(4 / 6)
    for k in ("n", "accuracy", "macro_f1_present", "closed_set_macro_f1"):
        assert cs[k] == pytest.approx(alone["shared_closed_set"][k])
    assert closed_full["confusion_matrix"] == closed_alone["confusion_matrix"]
    expected = f1_score([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 1, 0], labels=[0, 1, 2], average="macro", zero_division=0)
    assert cs["closed_set_macro_f1"] == pytest.approx(expected)
    assert sum(v["support"] for v in closed_full["per_class"].values()) == 6


def test_strict_accuracy_counts_outside_events_as_wrong():
    rep, _ = build_target_report(make_sel(), MAPPING, None)
    st = rep["strict"]
    assert st["n_correct"] == 4 and st["n_events"] == 14
    assert st["accuracy_outside_label_space_as_incorrect"] == pytest.approx(4 / 14)
    assert st["accuracy_outside_label_space_as_incorrect"] < rep["shared_closed_set"]["accuracy"]


def test_report_has_definitions_and_threshold_recorded():
    rep, _ = build_target_report(make_sel(), MAPPING, 0.7)
    assert rep["confidence_threshold"] == 0.7
    assert {"shared_closed_set", "legacy_excluded_labels", "new_o3_classes", "strict"} <= set(rep["definitions"])


def test_threshold_subsets_report_their_own_counts():
    df = make_sel()
    df["ml_confidence"] = np.linspace(0.5, 1.0, len(df))
    totals = []
    for tau in (None, 0.7, 0.9):
        sel = df if tau is None else df[df["ml_confidence"] >= tau]
        rep, _ = build_target_report(sel, MAPPING, tau)
        sc = rep["sample_counts"]
        assert sc["total"] == len(sel) == sc["shared_closed_set"] + sc["legacy_excluded"] + sc["new_o3_classes"]
        totals.append(sc["total"])
    assert totals[0] == 14 and totals[0] > totals[1] > totals[2] > 0


def test_empty_subset_does_not_crash():
    df = make_sel().iloc[0:0]
    rep, closed = build_target_report(df, MAPPING, 0.99)
    assert rep["sample_counts"]["total"] == 0 and closed["n"] == 0
    assert np.isnan(rep["strict"]["accuracy_outside_label_space_as_incorrect"])
    assert rep["legacy_excluded_labels"]["n"] == 0 and rep["new_o3_classes"]["n"] == 0


def test_outside_events_never_change_shared_when_no_shared_present():
    only_outside = make_sel().query("category != @CAT_SHARED")
    rep, closed = build_target_report(only_outside, MAPPING, None)
    assert closed["n"] == 0 and rep["strict"]["n_correct"] == 0 and rep["strict"]["n_events"] == 8
    assert rep["strict"]["accuracy_outside_label_space_as_incorrect"] == 0.0


# --------------------------------------------------------------------------- #
# Full-target-evaluation policy
# --------------------------------------------------------------------------- #
def test_default_config_evaluates_every_target_event():
    cfg = load_config(project_root() / "configs" / "classifier.yaml")
    assert cfg["evaluate"]["max_per_class_target"] is None
    assert cfg["experiment"]["debug"] is False
    assert resolve_target_cap(cfg) is None
    assert cfg["evaluate"]["target_confidence_thresholds"] == [None, 0.7, 0.9]


def test_cap_is_only_allowed_in_debug():
    cfg = load_config(project_root() / "configs" / "classifier.yaml")
    bad = deep_update(cfg, {"evaluate": {"max_per_class_target": 500}})
    with pytest.raises(ValueError, match="every eligible event"):
        resolve_target_cap(bad)
    dbg = deep_update(cfg, cfg["debug"])
    dbg["experiment"]["debug"] = True
    assert resolve_target_cap(dbg) == 20


def test_debug_override_block_still_defines_a_cap():
    cfg = load_config(project_root() / "configs" / "classifier.yaml")
    assert cfg["debug"]["evaluate"]["max_per_class_target"] == 20
