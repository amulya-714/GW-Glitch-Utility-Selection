"""Visualise the EXISTING preliminary sampled H1 O3a diagnostic from its saved predictions. Read-only.

    python -m src.evaluation.o3a_diagnostic \
        --run-dir experiments/runs/H1_O2_colab_baseline_v4_sample_diagnostic

Strict scope
  * Reads only <run>/eval_target_*/predictions.csv (and class_mapping.json / config.yaml for the label space).
    It never loads O3a metadata or images, never downloads anything, never runs a model, and never calls
    src.models.evaluate (whose one-shot --final guard is untouched).
  * Writes ONLY to <run>/preliminary_o3a_visualisation/. Every pre-existing file in the run directory is hashed
    before and after; any change raises an error.
  * Every figure and table is labelled PRELIMINARY / sampled. These are not complete O3a results.
  * Shared trained classes, legacy-excluded labels (None_of_the_Above) and new O3 classes are kept separate.
    Events whose label the model was never trained on are NEVER scored as correct: they are shown only as
    counts and as "where the model sends them".
  * Two threshold axes are reported separately:
      label confidence (Gravity Spy ml_confidence)  - the axis of the existing preliminary table and of
                                                       evaluate.py's target thresholds;
      model confidence (classifier max softmax)     - selective prediction, coverage, calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import plots
from src.evaluation.metrics import (
    DEFAULT_THRESHOLDS, headline, per_class_table, reliability_bins, risk_coverage_curve, tau_name,
    threshold_sweep, threshold_table,
)
from src.utils import file_sha256, load_config, resolve_path, save_json, setup_logging

log = logging.getLogger("o3a_diagnostic")

OUT_DIRNAME = "preliminary_o3a_visualisation"
CAT_SHARED, CAT_LEGACY, CAT_NEW = "shared_closed_set", "legacy_excluded", "new_o3_class"
BANNER = "PRELIMINARY sampled O3a diagnostic"
CAVEAT = ("PRELIMINARY. Computed from a SAMPLED subset of H1 O3a events scored by the frozen O2 model in a separate "
          "diagnostic run. This is NOT a complete O3a evaluation and must not be reported as one. O3a was not used "
          "for training, preprocessing, tuning or model selection.")


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): file_sha256(p, n=64) for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] != OUT_DIRNAME}


def load_label_space(run_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """(trained classes in index order, legacy-excluded labels, unseen-in-source labels)."""
    cm = run_dir / "class_mapping.json"
    if cm.exists():
        d = json.loads(cm.read_text())
        return list(d["trained_classes"]), list(d.get("excluded_from_training", [])), list(d.get("unseen_in_source_run", []))
    cfg_p = run_dir / "config.yaml"
    if cfg_p.exists():
        c = load_config(cfg_p)["classes"]
        excl = list(c.get("exclude") or [])
        return [n for n in c["names"] if n not in excl], excl, list(c.get("unseen_in_source") or [])
    raise FileNotFoundError(f"{run_dir}: need class_mapping.json or config.yaml to know the trained label space.")


def find_prediction_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("eval_target_*/predictions.csv"))


def prepare_predictions(df: pd.DataFrame, names: list[str], legacy: list[str], unseen: list[str]) -> pd.DataFrame:
    df = df.copy()
    need = {"ml_label", "ml_confidence"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv lacks required columns {sorted(missing)}")
    idx = {c: i for i, c in enumerate(names)}
    if "pred" not in df.columns:
        if "pred_name" not in df.columns:
            raise ValueError("predictions.csv needs 'pred' or 'pred_name'")
        df["pred"] = df["pred_name"].map(idx)
    if "pred_name" not in df.columns:
        df["pred_name"] = [names[int(i)] for i in df["pred"]]
    if (df["pred"] < 0).any() or (df["pred"] >= len(names)).any() or df["pred"].isna().any():
        raise ValueError("predicted class index outside the trained label space")
    df["pred"] = df["pred"].astype(int)
    df["label"] = df["ml_label"].map(idx).fillna(-1).astype(int)
    expected_cat = np.where(df["label"] >= 0, CAT_SHARED,
                            np.where(df["ml_label"].isin(legacy), CAT_LEGACY,
                                     np.where(df["ml_label"].isin(unseen), CAT_NEW, "unknown")))
    if "category" in df.columns and not (df["category"].astype(str).to_numpy() == expected_cat).all():
        raise ValueError("saved 'category' column disagrees with the run's class mapping")
    df["category"] = expected_cat
    unknown = sorted(df.loc[df["category"] == "unknown", "ml_label"].unique())
    if unknown:
        raise ValueError(f"labels {unknown} are neither trained, legacy-excluded nor unseen_in_source")
    return df


def label_confidence_summary(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per label-confidence threshold: counts by category, shared-class metrics, strict accuracy."""
    rows = []
    n_all = len(df)
    shared_all = df[df["category"] == CAT_SHARED]
    for tau in DEFAULT_THRESHOLDS:
        sel = df if tau is None else df[df["ml_confidence"] >= float(tau)]
        sh = sel[sel["category"] == CAT_SHARED]
        from src.models.evaluate import compute_metrics
        h = headline(compute_metrics(sh["label"].to_numpy(), sh["pred"].to_numpy(), names,
                                     sh["pred_conf"].to_numpy() if "pred_conf" in sh else None))
        n_correct = int((sh["label"] == sh["pred"]).sum())
        rows.append({
            "label_confidence_threshold": tau_name(tau),
            "n_total_events": len(sel), "coverage_of_all_events": len(sel) / n_all if n_all else np.nan,
            "n_shared_class": len(sh), "n_legacy_excluded": int((sel["category"] == CAT_LEGACY).sum()),
            "n_new_o3_class": int((sel["category"] == CAT_NEW).sum()),
            "coverage_of_shared_events": len(sh) / len(shared_all) if len(shared_all) else np.nan,
            "shared_accuracy": h["accuracy"], "shared_macro_f1": h["macro_f1"],
            "shared_balanced_accuracy": h["balanced_accuracy"], "shared_weighted_f1": h["weighted_f1"],
            "shared_ece": h["ece"], "shared_classes_present": h["num_classes_present"],
            "strict_accuracy_all_events": n_correct / len(sel) if len(sel) else np.nan,
        })
    return pd.DataFrame(rows)


def render_one(pred_path: Path, run_dir: Path, out: Path) -> dict[str, Any]:
    names, legacy, unseen = load_label_space(run_dir)
    df = prepare_predictions(pd.read_csv(pred_path), names, legacy, unseen)
    has_conf = "pred_conf" in df.columns
    target = pred_path.parent.name.replace("eval_target_", "")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{BANNER} ({target}, n={len(df)})"

    lab = label_confidence_summary(df, names)
    lab.to_csv(out / "label_confidence_thresholds.csv", index=False)
    sh = df[df["category"] == CAT_SHARED].reset_index(drop=True)
    y, p = sh["label"].to_numpy(), sh["pred"].to_numpy()
    from src.models.evaluate import compute_metrics
    m = compute_metrics(y, p, names, sh["pred_conf"].to_numpy() if has_conf else None)
    pc = per_class_table(m, names)
    pc.to_csv(out / "shared_per_class.csv", index=False)
    cm = np.asarray(m["confusion_matrix"]) if m.get("n") else np.zeros((len(names),) * 2, int)
    pd.DataFrame(cm, index=names, columns=names).rename_axis("reference \\ predicted").to_csv(
        out / "shared_confusion_matrix.csv")

    outside: dict[str, dict[str, int]] = {}
    for lbl, g in df[df["category"] != CAT_SHARED].groupby("ml_label"):
        outside[lbl] = g["pred_name"].value_counts().to_dict()
    save_json(outside, out / "outside_label_space_predicted_as.json")

    pdir = out / "plots"
    plots.plot_confusion(cm, names, pdir / "shared_confusion_matrix", f"{BANNER}: shared classes", False,
                         f"{target}, shared-class events only (n={len(sh)})")
    plots.plot_confusion(cm, names, pdir / "shared_confusion_matrix_normalised",
                         f"{BANNER}: shared classes (row-normalised)", True,
                         f"{target}, shared-class events only (n={len(sh)})")
    if outside:
        plots.plot_predicted_as(outside, pdir / "outside_label_space_predicted_as",
                                f"{BANNER}: labels outside the O2 training taxonomy\n"
                                "(not classifiable by this model; shown only as where they are sent)")
    correct = (y == p)
    lab_sweep = threshold_sweep(sh["ml_confidence"].to_numpy(float), correct)
    plots.plot_sweep([{"label": "shared-class events", "index": 0, "sweep": lab_sweep}], "accuracy",
                     pdir / "shared_accuracy_vs_label_confidence", "Accuracy on retained shared-class events",
                     f"{BANNER}", f"accuracy vs Gravity Spy label-confidence threshold ({target})",
                     xlabel="Label-confidence threshold τ (keep events with ml_confidence ≥ τ)")
    plots.plot_sweep([{"label": "shared-class events", "index": 0, "sweep": lab_sweep}], "coverage",
                     pdir / "shared_coverage_vs_label_confidence", "Fraction of shared-class events retained",
                     f"{BANNER}", f"coverage vs Gravity Spy label-confidence threshold ({target})",
                     xlabel="Label-confidence threshold τ (keep events with ml_confidence ≥ τ)")

    result: dict[str, Any] = {
        "target": target, "predictions_file": str(pred_path), "predictions_sha256": file_sha256(pred_path, n=64),
        "n_events": len(df), "n_shared": len(sh), "n_legacy_excluded": int((df["category"] == CAT_LEGACY).sum()),
        "n_new_o3_class": int((df["category"] == CAT_NEW).sum()),
        "label_counts_outside_taxonomy": df.loc[df["category"] != CAT_SHARED, "ml_label"].value_counts().to_dict(),
        "label_confidence_thresholds": lab.to_dict("records"),
        "shared_headline_no_threshold": headline(m), "status": "PRELIMINARY_SAMPLED_DIAGNOSTIC",
    }
    if has_conf:
        conf = sh["pred_conf"].to_numpy(float)
        mod = threshold_table(y, p, conf, names, conf, "model_confidence")
        mod.to_csv(out / "shared_model_confidence_thresholds.csv", index=False)
        rel = reliability_bins(conf, correct)
        rel.to_csv(out / "shared_reliability_bins.csv", index=False)
        cov, risk, aurc = risk_coverage_curve(conf, correct)
        sweep = threshold_sweep(conf, correct)
        e = {"label": "shared-class events", "index": 0}
        plots.plot_reliability([{**e, "bins": rel, "ece": headline(m)["ece"]}], pdir / "shared_reliability_diagram",
                               f"{BANNER}: reliability ({target})")
        plots.plot_sweep([{**e, "sweep": sweep}], "coverage", pdir / "shared_coverage_vs_model_confidence",
                         "Coverage of shared-class events", BANNER, f"coverage vs model-confidence threshold ({target})")
        plots.plot_sweep([{**e, "sweep": sweep}], "accuracy", pdir / "shared_accuracy_vs_model_confidence",
                         "Accuracy on retained shared-class events", BANNER,
                         f"accuracy vs model-confidence threshold ({target})")
        plots.plot_risk_coverage([{**e, "coverage": cov, "risk": risk, "aurc": aurc}],
                                 pdir / "shared_selective_risk_vs_coverage", f"{BANNER}: selective risk ({target})")
        groups = [("shared, correct", conf[correct]), ("shared, incorrect", conf[~correct]),
                  ("legacy-excluded label", df.loc[df["category"] == CAT_LEGACY, "pred_conf"].to_numpy(float)),
                  ("new O3 class (untrainable)", df.loc[df["category"] == CAT_NEW, "pred_conf"].to_numpy(float))]
        plots.plot_confidence_by_category(groups, pdir / "model_confidence_by_category",
                                          f"{BANNER}: model confidence by event category ({target})")
        result["shared_model_confidence_thresholds"] = mod.to_dict("records")
        result["shared_aurc"] = aurc
    else:
        log.warning("%s has no pred_conf column: model-confidence plots skipped.", pred_path)

    lines = [CAVEAT, "", f"Source: {pred_path}", f"Events: {len(df)} total = {len(sh)} shared-class + "
             f"{result['n_legacy_excluded']} legacy-excluded + {result['n_new_o3_class']} new-O3-class.", "",
             "Label-confidence (Gravity Spy ml_confidence) thresholds - same axis as the existing preliminary table:",
             lab[["label_confidence_threshold", "n_total_events", "n_shared_class", "coverage_of_shared_events",
                  "shared_accuracy", "shared_macro_f1", "shared_balanced_accuracy",
                  "strict_accuracy_all_events"]].to_string(index=False), "",
             "shared_* metrics use shared-class events only. strict_accuracy counts every legacy/new-class event as "
             "wrong. New O3 classes cannot be predicted by an O2-trained model; they are never counted as correct."]
    (out / "README.txt").write_text("\n".join(lines) + "\n")
    log.info("\n%s", "\n".join(lines))
    return result


def render_report(run_dir: Path, overwrite: bool = False) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        log.warning("O3a diagnostic run %s not found; skipped.", run_dir)
        return None
    files = find_prediction_files(run_dir)
    if not files:
        log.warning("No eval_target_*/predictions.csv in %s; O3a diagnostic visualisation skipped "
                    "(nothing is recomputed or downloaded to replace it).", run_dir)
        return None
    out_root = run_dir / OUT_DIRNAME
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_root} exists. Pass --overwrite to regenerate it (only that folder).")
        import shutil
        shutil.rmtree(out_root)
    before = _snapshot(run_dir)
    results = {f.parent.name: render_one(f, run_dir, out_root / f.parent.name) for f in files}
    after = _snapshot(run_dir)
    changed = sorted(k for k in before if after.get(k) != before[k])
    if changed:
        raise RuntimeError(f"Existing diagnostic artifacts changed: {changed[:10]}")
    summary = {"status": "PRELIMINARY_SAMPLED_DIAGNOSTIC", "caveat": CAVEAT, "run_dir": str(run_dir),
               "output_dir": str(out_root), "existing_files_verified_unchanged": len(before), "results": results}
    save_json(summary, out_root / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="The existing sampled O3a diagnostic run directory.")
    ap.add_argument("--overwrite", action="store_true", help=f"Regenerate {OUT_DIRNAME}/ (nothing else is touched).")
    args = ap.parse_args(argv)
    setup_logging()
    if render_report(resolve_path(args.run_dir), args.overwrite) is None:
        raise SystemExit("Nothing rendered: no eval_target_*/predictions.csv found.")


if __name__ == "__main__":
    main()
