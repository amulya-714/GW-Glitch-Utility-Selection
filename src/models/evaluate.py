"""Evaluation: metrics, source-validation evaluation, and the gated final target-run evaluation.

CLI
---
Source-run validation (safe, repeatable):
    python -m src.models.evaluate --run-dir experiments/runs/<run> --split val

FINAL cross-run evaluation on the unseen target run (O3a). Run ONCE per frozen model:
    python -m src.models.evaluate --run-dir experiments/runs/<run> --split target --final

Target-run guard rails:
  * `--final` is mandatory for --split target.
  * The checkpoint is the one chosen on source-val only (best.pt); nothing is tuned here.
  * Thresholds come from config (evaluate.target_confidence_thresholds), all are reported.
  * A final_eval_log.json is written; a second target run is refused unless --allow-rerun
    is passed, so repeated peeking is visible in the record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ImageStore, make_loader, prepare_split
from src.data.splits import (
    ClassMapping, apply_class_mapping, build_source_splits, cap_per_class, clean_metadata,
    load_run_metadata,
)
from src.models import build_model
from src.utils import (
    file_sha256, load_config, resolve_path, save_json, select_device, set_seed, setup_logging,
    stable_hash,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
                    conf: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class P/R/F1/support and the confusion matrix.

    macro_f1_present averages only over classes that have >=1 true sample in this
    evaluation set (absent classes would otherwise count as F1=0). macro_f1_all_classes
    averages over every class and is reported alongside it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    c = len(class_names)
    labels = list(range(c))
    if len(y_true) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1_present": float("nan")}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = s > 0
    predicted = cm.sum(axis=0)
    correct = (y_true == y_pred)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy_present": float(r[present].mean()),
        "macro_f1_present": float(f[present].mean()),
        "macro_precision_present": float(p[present].mean()),
        "macro_recall_present": float(r[present].mean()),
        "macro_f1_all_classes": float(f.mean()),
        "weighted_f1": float((f * s).sum() / s.sum()),
        "num_classes_present": int(present.sum()),
        "per_class": {
            class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                             "support": int(s[i]), "predicted": int(predicted[i])}
            for i in range(c)
        },
        "confusion_matrix": cm.tolist(),
    }
    if conf is not None:
        out["ece"] = expected_calibration_error(np.asarray(conf), correct)
    return out


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            criterion: nn.Module | None = None) -> dict[str, np.ndarray | float]:
    """Run the model; returns y_true, y_pred, probs (N,C) and mean loss if criterion given."""
    model.eval()
    ys, probs, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if criterion is not None and (y >= 0).all():
            loss_sum += float(criterion(logits, y.to(device))) * len(y)
        n += len(y)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        ys.append(y.numpy())
    probs_a = np.concatenate(probs) if probs else np.zeros((0, 0))
    return {"y_true": np.concatenate(ys) if ys else np.zeros(0, int), "probs": probs_a,
            "y_pred": probs_a.argmax(1) if len(probs_a) else np.zeros(0, int),
            "loss": loss_sum / max(n, 1) if criterion is not None else float("nan")}


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, class_names: Sequence[str],
                    criterion: nn.Module | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pred = predict(model, loader, device, criterion)
    conf = pred["probs"].max(1) if len(pred["probs"]) else None
    m = compute_metrics(pred["y_true"], pred["y_pred"], class_names, conf)
    m["loss"] = pred["loss"]
    return m, pred


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    keep = np.where((cm.sum(1) > 0) | (cm.sum(0) > 0))[0]
    if len(keep) == 0:
        return
    sub = cm[np.ix_(keep, keep)]
    norm = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(keep) + 3),) * 2)
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(keep)), [class_names[i] for i in keep], rotation=90, fontsize=7)
    ax.set_yticks(range(len(keep)), [class_names[i] for i in keep], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], out_dir: str | Path, class_names: Sequence[str], prefix: str = "") -> None:
    """Write <prefix>metrics.json, per_class.csv, confusion_matrix.csv and a PNG."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out / f"{prefix}metrics.json")
    if "per_class" in metrics:
        pd.DataFrame(metrics["per_class"]).T.rename_axis("class").to_csv(out / f"{prefix}per_class.csv")
        cm = np.asarray(metrics["confusion_matrix"])
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / f"{prefix}confusion_matrix.csv")
        plot_confusion_matrix(cm, class_names, out / f"{prefix}confusion_matrix.png", prefix.rstrip("_"))


def format_summary(m: dict[str, Any]) -> str:
    if not m.get("n"):
        return "no samples"
    return (f"n={m['n']} acc={m['accuracy']:.4f} macroF1(present)={m['macro_f1_present']:.4f} "
            f"macroF1(all)={m['macro_f1_all_classes']:.4f} bal.acc={m['balanced_accuracy_present']:.4f}"
            + (f" ECE={m['ece']:.4f}" if "ece" in m else ""))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], ClassMapping]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model_state", "config", "class_mapping"):
        if k not in ckpt:
            raise KeyError(f"Checkpoint {path} is missing '{k}'. Was it written by train.py?")
    cfg, mapping = ckpt["config"], ClassMapping.from_dict(ckpt["class_mapping"])
    # The stored preprocessing signature must match the stored config, otherwise the model would be fed
    # pixels it was not trained on. Old checkpoints without a signature are accepted unchanged.
    stored = ckpt.get("preprocessing")
    if stored and stored.get("hash"):
        from src.data.preprocessing import preprocessing_signature
        now = preprocessing_signature(cfg["data"])["hash"]
        if now != stored["hash"]:
            log.warning("Checkpoint %s: preprocessing hash stored at training time (%s) differs from the hash "
                        "recomputed from its config now (%s). Either the preprocessing code changed since training "
                        "or the checkpoint was edited. Verify the input pipeline before trusting these results.",
                        path, stored["hash"], now)
    model = build_model(cfg, mapping.num_classes)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint {path} does not match the architecture built from its own config "
                           f"(model.type={cfg['model'].get('type')}): {exc}") from exc
    return model.to(device).eval(), cfg, mapping


# --------------------------------------------------------------------------- #
# Source validation
# --------------------------------------------------------------------------- #
def evaluate_source_val(run_dir: Path, cfg: dict, model: nn.Module, mapping: ClassMapping,
                        device: torch.device, ckpt_name: str) -> dict[str, Any]:
    _, val_df, _, _ = build_source_splits(cfg)
    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    ds, kept, _ = prepare_split(val_df, f"{cfg['data']['ifo']}_val", cfg, store)
    loader = make_loader(ds, cfg["train"]["batch_size"], False, num_workers=0)
    m, pred = evaluate_loader(model, loader, device, mapping.names, nn.CrossEntropyLoss())
    out = run_dir / f"eval_val_{ckpt_name}"
    save_metrics(m, out, mapping.names)
    log.info("SOURCE VAL [%s]: %s", ckpt_name, format_summary(m))
    return m


# --------------------------------------------------------------------------- #
# Final target-run evaluation (O3a)
# --------------------------------------------------------------------------- #
def _tau_name(tau: float | None) -> str:
    return "none" if tau is None else f"{tau:g}"


CAT_SHARED = "shared_closed_set"       # label is one of the trained classes
CAT_LEGACY = "legacy_excluded"         # label is in classes.exclude (e.g. None_of_the_Above)
CAT_NEW = "new_o3_class"               # label is in classes.unseen_in_source (Blip_Low_Frequency, Fast_Scattering)

REPORT_DEFINITIONS = {
    "shared_closed_set": "Events whose ml_label is a trained class. Standard closed-set metrics; "
                         "macro-F1 averages over classes with >=1 true event at this threshold.",
    "legacy_excluded_labels": "Events carrying a label that exists in the source scheme but was excluded from "
                              "training (classes.exclude). Reported as counts and where the model sends them.",
    "new_o3_classes": "Events carrying a label that does not exist in the source run (classes.unseen_in_source). "
                      "The model cannot predict them. Reported as counts and where the model sends them.",
    "strict": "Correct shared-class predictions divided by ALL events at this threshold (shared + legacy + new). "
              "Every legacy/new event counts as incorrect.",
}


def resolve_target_cap(cfg: dict[str, Any]) -> int | None:
    """Per-class cap for the target run. None (default) means every eligible event is evaluated.

    A cap is only accepted on debug runs; otherwise it is an error rather than a silent subsample.
    """
    cap = cfg["evaluate"].get("max_per_class_target")
    if cap is None:
        return None
    if not cfg["experiment"].get("debug", False):
        raise ValueError(
            f"evaluate.max_per_class_target={cap} on a non-debug run. The final target evaluation must use "
            "every eligible event: set evaluate.max_per_class_target to null.")
    return int(cap)


def assign_category(labels: pd.Series, mapping: ClassMapping) -> pd.Series:
    """Map each ml_label to shared / legacy-excluded / new-O3 category. Unknown labels are an error."""
    legacy, new = set(mapping.excluded), set(mapping.unseen_in_source)

    def cat(lbl: str) -> str:
        if lbl in mapping.class_to_idx:
            return CAT_SHARED
        if lbl in legacy:
            return CAT_LEGACY
        if lbl in new:
            return CAT_NEW
        raise ValueError(f"Label '{lbl}' is not trained, excluded or unseen_in_source; add it to the config.")
    return labels.map(cat)


def _outside_block(df: pd.DataFrame) -> dict[str, Any]:
    """Counts per label and, per label, where the model sends those events."""
    block: dict[str, Any] = {"n": int(len(df)), "label_counts": df["ml_label"].value_counts().to_dict(),
                             "per_label": {}}
    for lbl, g in df.groupby("ml_label"):
        vc = g["pred_name"].value_counts()
        block["per_label"][lbl] = {
            "n": int(len(g)),
            "predicted_as": {c: {"count": int(n), "fraction": round(float(n) / len(g), 4)} for c, n in vc.items()},
        }
    return block


def build_target_report(sel: pd.DataFrame, mapping: ClassMapping, tau: float | None
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the structured target report for ONE confidence-threshold subset.

    `sel` needs columns: ml_label, label, pred, pred_name, pred_conf, category.
    Returns (report, closed_set_metrics_full). Legacy-excluded and new-O3 events are never
    merged with each other and never enter the closed-set metrics.
    """
    shared = sel[sel["category"] == CAT_SHARED]
    legacy = sel[sel["category"] == CAT_LEGACY]
    new = sel[sel["category"] == CAT_NEW]
    closed = compute_metrics(shared["label"].to_numpy(), shared["pred"].to_numpy(), mapping.names,
                             shared["pred_conf"].to_numpy())
    n_correct = int((shared["label"] == shared["pred"]).sum())
    n_total = int(len(sel))
    headline = {k: v for k, v in closed.items() if k not in ("per_class", "confusion_matrix")}
    if closed.get("n"):
        headline["closed_set_macro_f1"] = closed["macro_f1_present"]
    report = {
        "confidence_threshold": tau,
        "sample_counts": {
            "total": n_total,
            "shared_closed_set": int(len(shared)),
            "legacy_excluded": int(len(legacy)),
            "new_o3_classes": int(len(new)),
            "by_label": sel["ml_label"].value_counts().to_dict(),
        },
        "shared_closed_set": headline,
        "legacy_excluded_labels": {"labels": list(mapping.excluded), **_outside_block(legacy)},
        "new_o3_classes": {"labels": list(mapping.unseen_in_source), **_outside_block(new)},
        "strict": {
            "accuracy_outside_label_space_as_incorrect": n_correct / n_total if n_total else float("nan"),
            "n_correct": n_correct,
            "n_events": n_total,
        },
        "definitions": REPORT_DEFINITIONS,
    }
    return report, closed


def evaluate_target_final(run_dir: Path, cfg: dict, model: nn.Module, mapping: ClassMapping,
                          device: torch.device, ckpt_path: Path, allow_rerun: bool = False) -> dict[str, Any]:
    """Evaluate on every protocol target run for the configured detector. Never trains or tunes."""
    log_path = run_dir / "final_eval_log.json"
    if log_path.exists() and not allow_rerun:
        raise RuntimeError(
            f"{log_path} exists: this model already had its final target evaluation. "
            "Re-evaluating the same model on the target is a form of peeking. "
            "Pass --allow-rerun only if you are sure (it will be recorded).")

    d, proto, ev = cfg["data"], cfg["protocol"], cfg["evaluate"]
    ifo, seed = d["ifo"], cfg["experiment"]["seed"]
    cap = resolve_target_cap(cfg)   # None = every eligible event
    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    results: dict[str, Any] = {}
    for run in proto["target_runs"]:
        name = f"{ifo}_{run}"
        raw = load_run_metadata(resolve_path(cfg["paths"]["metadata_dir"]), ifo, run, d["view_columns"],
                                target_runs=proto["target_runs"], purpose="final_eval")
        clean = clean_metadata(raw, ifo, d["view_columns"], name)
        # Cap is None on real runs. When set (debug only) it is applied to the UNFILTERED target set,
        # and confidence thresholds are then taken as subsets of it.
        capped = cap_per_class(clean, cap, seed + 7)
        if cap is None:
            log.info("%s: evaluating ALL %d eligible events (%d images to fetch/cache)",
                     name, len(capped), len(capped) * len(d["view_columns"]))
        else:
            log.warning("%s: DEBUG per-class cap %d active: %d of %d events. Not a valid final evaluation.",
                        name, cap, len(capped), len(clean))
        mapped, other = apply_class_mapping(capped, mapping, name)
        full = pd.concat([mapped, other.assign(label=-1)], ignore_index=True)
        full = full.sort_values(["event_time", "gravityspy_id"], kind="mergesort").reset_index(drop=True)

        ds, kept, failures = prepare_split(full, f"{name}_target", cfg, store)
        n_image_failed = len(full) - len(kept)
        if n_image_failed:
            failures.to_csv(run_dir / f"target_{name}_image_failures.csv", index=False)
            log.warning("%s: %d of %d target events (%.3f%%) could not be evaluated because an image view "
                        "failed to download/decode. They are listed in target_%s_image_failures.csv.",
                        name, n_image_failed, len(full), 100.0 * n_image_failed / max(len(full), 1), name)
        loader = make_loader(ds, cfg["train"]["batch_size"], False, num_workers=0)
        pred = predict(model, loader, device)
        kept = kept.assign(pred=pred["y_pred"], pred_conf=pred["probs"].max(1))
        kept["pred_name"] = [mapping.names[i] for i in kept["pred"]]
        kept["category"] = assign_category(kept["ml_label"], mapping)
        out_root = run_dir / f"eval_target_{run}"
        out_root.mkdir(parents=True, exist_ok=True)
        kept[["gravityspy_id", "event_time", "ml_label", "category", "ml_confidence", "label", "pred",
              "pred_name", "pred_conf"]].to_csv(out_root / "predictions.csv", index=False)

        per_tau: dict[str, Any] = {}
        rows = []
        for tau in ev["target_confidence_thresholds"]:
            sel = kept if tau is None else kept[kept["ml_confidence"] >= float(tau)]
            report, closed = build_target_report(sel, mapping, tau)
            tdir = out_root / f"conf_{_tau_name(tau)}"
            save_json(report, tdir / "report.json")
            save_metrics(closed, tdir, mapping.names, prefix="closed_set_")
            per_tau[_tau_name(tau)] = report
            sc, cs, st = report["sample_counts"], report["shared_closed_set"], report["strict"]
            rows.append({"confidence_threshold": _tau_name(tau), "n_total": sc["total"],
                         "n_shared": sc["shared_closed_set"], "n_legacy_excluded": sc["legacy_excluded"],
                         "n_new_o3": sc["new_o3_classes"], "closed_set_macro_f1": cs.get("macro_f1_present"),
                         "closed_set_accuracy": cs.get("accuracy"),
                         "n_classes_present": cs.get("num_classes_present"),
                         "strict_accuracy": st["accuracy_outside_label_space_as_incorrect"]})
            log.info("TARGET %s conf>=%s | shared n=%d macroF1=%s acc=%s | legacy n=%d %s | new n=%d %s | strict acc=%.4f",
                     name, _tau_name(tau), sc["shared_closed_set"],
                     f"{cs['macro_f1_present']:.4f}" if cs.get("n") else "n/a",
                     f"{cs['accuracy']:.4f}" if cs.get("n") else "n/a",
                     sc["legacy_excluded"], report["legacy_excluded_labels"]["label_counts"],
                     sc["new_o3_classes"], report["new_o3_classes"]["label_counts"],
                     st["accuracy_outside_label_space_as_incorrect"])
        pd.DataFrame(rows).to_csv(out_root / "threshold_summary.csv", index=False)
        results[run] = {
            "events": {"rows_in_csv": int(len(raw)), "after_cleaning": int(len(clean)),
                       "evaluated_pool": int(len(capped)), "cap_per_class": cap, "cap_applied": cap is not None,
                       "dropped_for_image_failures": int(n_image_failed), "scored": int(len(kept))},
            "thresholds": per_tau,
        }
    save_json(results, run_dir / "target_summary.json")
    save_json({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path), "checkpoint_sha256": file_sha256(ckpt_path),
        "config_hash": stable_hash(cfg), "allow_rerun": allow_rerun,
        "thresholds": ev["target_confidence_thresholds"], "target_cap_per_class": cap,
        "debug_run": bool(cfg["experiment"].get("debug", False)),
    }, log_path)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Directory produced by train.py")
    ap.add_argument("--split", choices=["val", "target"], default="val")
    ap.add_argument("--checkpoint", default="best", help="'best' | 'last' | path to a .pt file")
    ap.add_argument("--final", action="store_true", help="Required to evaluate on the target run.")
    ap.add_argument("--allow-rerun", action="store_true", help="Allow a second target evaluation (recorded).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    setup_logging(run_dir / "evaluate.log")
    ckpt_path = Path(args.checkpoint) if args.checkpoint.endswith(".pt") else run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    device = select_device(args.device or "auto")
    model, cfg, mapping = load_checkpoint(ckpt_path, device)
    set_seed(cfg["experiment"]["seed"])
    log.info("Loaded %s on %s", ckpt_path, device)

    if args.split == "val":
        evaluate_source_val(run_dir, cfg, model, mapping, device, ckpt_path.stem)
        return
    if not args.final:
        raise SystemExit(
            "Refusing to evaluate on the target run without --final.\n"
            "Target data is reserved for ONE final evaluation of a frozen model. "
            "Re-run with --final when you are ready.")
    evaluate_target_final(run_dir, cfg, model, mapping, device, ckpt_path, args.allow_rerun)


if __name__ == "__main__":
    main()
