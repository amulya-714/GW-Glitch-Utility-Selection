"""Evaluation metrics shared by the model-comparison pipeline and the O3a diagnostic report.

Builds on src.models.evaluate.compute_metrics / expected_calibration_error (the functions train.py already uses
for model selection), so every number here is computed the same way as the numbers in each run's summary.json.

Two different "confidence" axes exist in this project and are NEVER mixed:
  model_confidence  max softmax probability of OUR classifier. Used for calibration (ECE, reliability),
                    selective prediction (coverage, selective risk) and model-confidence thresholds.
  label_confidence  Gravity Spy `ml_confidence`, the confidence of the reference LABEL. Filtering on it changes
                    which events are evaluated (label-quality filter), not which predictions are trusted.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.evaluate import compute_metrics, expected_calibration_error

DEFAULT_THRESHOLDS: tuple[float | None, ...] = (None, 0.5, 0.7, 0.9)
ECE_BINS = 15   # identical to expected_calibration_error's default, which train.py reports

METRIC_DEFINITIONS = {
    "accuracy": "Fraction of evaluated events whose predicted class equals the reference label.",
    "macro_f1": "Unweighted mean of per-class F1 over classes with >=1 reference event in the evaluated set "
                "(same as macro_f1_present in train.py). A class that is present but never predicted counts as F1=0.",
    "macro_f1_all_classes": "Mean F1 over ALL trained classes; classes absent from the evaluated set count as 0.",
    "weighted_f1": "Per-class F1 weighted by class support.",
    "balanced_accuracy": "Mean per-class recall over classes with >=1 reference event (sklearn definition).",
    "ece": f"Expected calibration error of the max-softmax confidence, {ECE_BINS} equal-width bins.",
    "aurc": "Area under the selective risk-coverage curve (mean risk over all coverage levels when events are "
            "accepted in order of decreasing model confidence). Lower is better.",
    "coverage": "n_retained / n_total for a confidence threshold.",
    "selective_risk": "1 - accuracy on the retained subset.",
    "precision_undefined": "Precision is undefined (NaN) for a class the model never predicts.",
    "recall_undefined": "Recall and F1 are undefined (NaN) for a class with zero reference events.",
}


def tau_name(tau: float | None) -> str:
    return "none" if tau is None else f"{tau:g}"


# --------------------------------------------------------------------------- #
# Headline / per-class
# --------------------------------------------------------------------------- #
def headline(m: dict[str, Any]) -> dict[str, Any]:
    """Rename compute_metrics keys into the reporting vocabulary (no recomputation)."""
    if not m.get("n"):
        return {"n": 0, "accuracy": np.nan, "macro_f1": np.nan, "macro_f1_all_classes": np.nan,
                "weighted_f1": np.nan, "balanced_accuracy": np.nan, "ece": np.nan, "num_classes_present": 0}
    return {
        "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1_present"],
        "macro_f1_all_classes": m["macro_f1_all_classes"], "weighted_f1": m["weighted_f1"],
        "balanced_accuracy": m["balanced_accuracy_present"], "ece": m.get("ece", np.nan),
        "num_classes_present": m["num_classes_present"],
    }


def per_class_table(m: dict[str, Any], class_names: Sequence[str]) -> pd.DataFrame:
    """One row per trained class in mapping order, INCLUDING classes with zero support or zero predictions.

    Undefined quantities are NaN (not 0) and explained in `note`.
    """
    rows = []
    for name in class_names:
        pc = m["per_class"][name] if m.get("n") else {"precision": 0, "recall": 0, "f1": 0, "support": 0, "predicted": 0}
        support, predicted = int(pc["support"]), int(pc["predicted"])
        notes = []
        precision, recall, f1 = float(pc["precision"]), float(pc["recall"]), float(pc["f1"])
        if predicted == 0:
            precision = np.nan
            notes.append("never predicted")
        if support == 0:
            recall, f1 = np.nan, np.nan
            notes.append("absent from evaluated set")
        rows.append({"class": name, "support": support, "predicted": predicted, "precision": precision,
                     "recall": recall, "f1": f1, "note": "; ".join(notes)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calibration / selective prediction
# --------------------------------------------------------------------------- #
def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> pd.DataFrame:
    """Per-bin count, mean confidence and accuracy, with the same binning as expected_calibration_error."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "count": int(mask.sum()),
                     "mean_confidence": float(conf[mask].mean()) if mask.any() else np.nan,
                     "accuracy": float(correct[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Accept events in order of decreasing confidence. Returns (coverage, selective_risk, AURC).

    Ties are broken by original order (stable sort), so the curve is deterministic.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n = len(conf)
    if n == 0:
        return np.zeros(0), np.zeros(0), float("nan")
    order = np.argsort(-conf, kind="mergesort")
    k = np.arange(1, n + 1)
    risk = 1.0 - np.cumsum(correct[order]) / k
    return k / n, risk, float(risk.mean())


def threshold_sweep(conf: np.ndarray, correct: np.ndarray, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Coverage and retained-subset accuracy for a dense grid of model-confidence thresholds (for line plots)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    grid = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, float)
    n = max(len(conf), 1)
    rows = []
    for t in grid:
        keep = conf >= t
        rows.append({"threshold": float(t), "n_retained": int(keep.sum()), "coverage": keep.sum() / n,
                     "accuracy": float(correct[keep].mean()) if keep.any() else np.nan})
    return pd.DataFrame(rows)


def threshold_table(y_true: np.ndarray, y_pred: np.ndarray, model_conf: np.ndarray, class_names: Sequence[str],
                    filter_values: np.ndarray, threshold_type: str,
                    thresholds: Sequence[float | None] = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Metrics on the subset with filter_values >= tau, for every tau.

    `filter_values` is model confidence (selective prediction) or label confidence (label-quality filter),
    named by `threshold_type`. Columns ending in _retained describe ONLY the retained subset; the *_all_samples
    columns repeat the unfiltered numbers so the two are never confused.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    model_conf, filter_values = np.asarray(model_conf, float), np.asarray(filter_values, float)
    full = headline(compute_metrics(y_true, y_pred, class_names, model_conf))
    n_total = len(y_true)
    rows = []
    for tau in thresholds:
        keep = np.ones(n_total, bool) if tau is None else filter_values >= float(tau)
        h = headline(compute_metrics(y_true[keep], y_pred[keep], class_names, model_conf[keep]))
        rows.append({
            "threshold_type": threshold_type, "threshold": tau_name(tau),
            "n_total": n_total, "n_retained": int(keep.sum()), "n_rejected": int(n_total - keep.sum()),
            "coverage": keep.sum() / n_total if n_total else np.nan,
            "accuracy_retained": h["accuracy"], "macro_f1_retained": h["macro_f1"],
            "balanced_accuracy_retained": h["balanced_accuracy"], "weighted_f1_retained": h["weighted_f1"],
            "selective_risk_retained": 1.0 - h["accuracy"] if h["n"] else np.nan,
            "num_classes_present_retained": h["num_classes_present"],
            "accuracy_all_samples": full["accuracy"], "macro_f1_all_samples": full["macro_f1"],
            "balanced_accuracy_all_samples": full["balanced_accuracy"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# One complete evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str],
                         label_conf: np.ndarray | None = None,
                         thresholds: Sequence[float | None] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Everything the report needs from (labels, softmax probabilities)."""
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, float)
    if probs.ndim != 2 or probs.shape[0] != len(y_true) or probs.shape[1] != len(class_names):
        raise ValueError(f"probs shape {probs.shape} does not match {len(y_true)} labels x {len(class_names)} classes")
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    correct = y_pred == y_true
    m = compute_metrics(y_true, y_pred, class_names, conf)
    cov, risk, aurc = risk_coverage_curve(conf, correct)
    out = {
        "overall": {**headline(m), "aurc": aurc},
        "per_class": per_class_table(m, class_names),
        "confusion_matrix": np.asarray(m["confusion_matrix"]) if m.get("n") else np.zeros((len(class_names),) * 2, int),
        "reliability": reliability_bins(conf, correct),
        "model_confidence_thresholds": threshold_table(y_true, y_pred, conf, class_names, conf,
                                                       "model_confidence", thresholds),
        "sweep": threshold_sweep(conf, correct),
        "risk_coverage": (cov, risk),
        "conf": conf, "correct": correct, "y_pred": y_pred,
    }
    if label_conf is not None:
        out["label_confidence_thresholds"] = threshold_table(
            y_true, y_pred, conf, class_names, np.asarray(label_conf, float), "label_confidence", thresholds)
    # self-check: our ECE is exactly the function train.py logs
    assert np.isclose(out["overall"]["ece"], expected_calibration_error(conf, correct), equal_nan=True)
    return out


def class_distribution(labels: np.ndarray, class_names: Sequence[str]) -> pd.Series:
    counts = np.bincount(np.asarray(labels, int), minlength=len(class_names))
    return pd.Series(counts, index=list(class_names), name="n_events")
