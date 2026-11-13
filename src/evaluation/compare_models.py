"""Unified O2-validation evaluation and comparison: multi-view model vs. single-view baselines.

    python -m src.evaluation.compare_models \
        --multiview-run experiments/runs/H1_O2_colab_baseline_v4 \
        --runs-root experiments/runs \
        --out experiments/comparisons/H1_O2_single_vs_multiview \
        --o3a-diagnostic-run experiments/runs/H1_O2_colab_baseline_v4_sample_diagnostic \
        --device auto

What it does
  1. Finds checkpoints: the multi-view run, every finished single-view run under --runs-root
     (model.type=single_view, same detector and source run; *_debug runs only with --include-debug),
     plus any --extra-run.
  2. Loads each best.pt with its OWN saved config (architecture, preprocessing, split) and verifies the
     checkpoint file is byte-identical before and after (frozen checkpoints are never written).
  3. Rebuilds each model's O2 validation split from its config, scores it, and checks that every model saw
     the SAME events. If not, the differences are written to population/ and the comparison table is computed
     on the common events only (each model's own-population numbers are kept in its run directory).
  4. Writes per-model reports to <run>/eval_val_comparison/ and the cross-model comparison to --out.
  5. Optionally renders the PRELIMINARY sampled O3a diagnostic from its existing predictions.csv
     (read-only; see src.evaluation.o3a_diagnostic). O3a is never loaded, downloaded or scored here.

Nothing here trains, tunes, or touches the target run; the one-shot guard in src.models.evaluate is unchanged.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import plots
from src.evaluation.metrics import (
    DEFAULT_THRESHOLDS, METRIC_DEFINITIONS, class_distribution, evaluate_predictions,
)
from src.utils import file_sha256, load_config, project_root, resolve_path, save_json, setup_logging

log = logging.getLogger("compare_models")

PER_MODEL_DIRNAME = "eval_val_comparison"


# --------------------------------------------------------------------------- #
# Run discovery / description
# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    run_dir: Path
    label: str
    kind: str
    input_desc: str
    view_seconds: float | None
    index: int = 0

    @property
    def slug(self) -> str:
        return self.run_dir.name


def describe(cfg: dict[str, Any]) -> tuple[str, str, float | None, str]:
    """(kind, input description, view seconds or None, default display label) from a run config."""
    m, d = cfg["model"], cfg["data"]
    kind = m["type"]
    durs = [float(x) for x in d["view_durations"]]
    all_views = " + ".join(f"{x:g} s" for x in durs)
    if kind == "multiview":
        return kind, f"{all_views} (all {len(durs)} views)", None, f"Multi-view ({len(durs)} views)"
    if kind == "single_view":
        s = durs[int(m["single_view_index"])]
        return kind, f"{s:g} s view only", s, f"Single-view {s:g} s"
    if kind == "baseline_single":
        s = durs[int(m["single_view_index"])]
        return kind, f"{s:g} s (legacy: receives all views, uses one)", s, f"Baseline-single {s:g} s (legacy)"
    if kind == "baseline_merged":
        return kind, f"{all_views} (channel-stacked)", None, "Merged-view CNN"
    return kind, "?", None, kind


def discover_single_view_runs(runs_root: Path, ifo: str, source_run: str, include_debug: bool) -> list[Path]:
    found = []
    if not runs_root.exists():
        return found
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not ((d / "checkpoints" / "best.pt").exists() and (d / "config.yaml").exists()):
            continue
        if d.name.endswith("_debug") and not include_debug:
            continue
        try:
            cfg = load_config(d / "config.yaml")
        except Exception as exc:  # pragma: no cover - corrupt config
            log.warning("skipping %s: unreadable config.yaml (%s)", d, exc)
            continue
        if cfg.get("model", {}).get("type") != "single_view":
            continue
        if cfg["data"]["ifo"] != ifo or cfg["protocol"]["source_run"] != source_run:
            continue
        found.append(d)
    durs = lambda p: float(load_config(p / "config.yaml")["data"]["view_durations"][  # noqa: E731
        int(load_config(p / "config.yaml")["model"]["single_view_index"])])
    return sorted(found, key=lambda p: (durs(p), p.name))


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    dirs: list[Path] = []
    ref_cfg = None
    if args.multiview_run:
        mv = resolve_path(args.multiview_run)
        if not (mv / "checkpoints" / f"{args.checkpoint}.pt").exists():
            raise FileNotFoundError(f"--multiview-run {mv}: checkpoints/{args.checkpoint}.pt not found.")
        dirs.append(mv)
        ref_cfg = load_config(mv / "config.yaml") if (mv / "config.yaml").exists() else None
    if args.single_view_runs:
        dirs += [resolve_path(p) for p in args.single_view_runs]
    else:
        ifo = args.ifo or (ref_cfg["data"]["ifo"] if ref_cfg else "H1")
        src = ref_cfg["protocol"]["source_run"] if ref_cfg else "O2"
        auto = discover_single_view_runs(resolve_path(args.runs_root), ifo, src, args.include_debug)
        log.info("Discovered %d single-view run(s) in %s: %s", len(auto), args.runs_root, [p.name for p in auto])
        dirs += auto
    dirs += [resolve_path(p) for p in (args.extra_run or [])]
    seen, specs = set(), []
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        ck = d / "checkpoints" / f"{args.checkpoint}.pt"
        if not ck.exists():
            log.warning("skipping %s: no checkpoints/%s.pt (training not finished?)", d, args.checkpoint)
            continue
        cfg_path = d / "config.yaml"
        cfg = load_config(cfg_path) if cfg_path.exists() else None
        if cfg is None:
            import torch
            cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
        kind, desc, secs, label = describe(cfg)
        specs.append(RunSpec(d, label, kind, desc, secs))
    labels = [s.label for s in specs]
    for s in specs:   # disambiguate duplicates (e.g. two seeds of the same view)
        if labels.count(s.label) > 1:
            s.label = f"{s.label} [{s.slug}]"
    for i, s in enumerate(specs):
        s.index = i
    if not specs:
        raise SystemExit("No checkpoints found to evaluate.")
    if len(specs) > len(plots.PALETTE):
        raise SystemExit(f"{len(specs)} runs requested; comparison figures support at most {len(plots.PALETTE)}.")
    return specs


# --------------------------------------------------------------------------- #
# Integrity helpers
# --------------------------------------------------------------------------- #
def tree_snapshot(root: Path, exclude_dirs: tuple[str, ...] = ()) -> dict[str, str]:
    """sha256 of every file under root except inside the named first-level subdirectories."""
    snap = {}
    if not root.exists():
        return snap
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in exclude_dirs:
            continue
        snap[str(rel)] = file_sha256(p, n=64)
    return snap


def assert_unchanged(before: dict[str, str], after: dict[str, str], root: Path) -> None:
    changed = sorted(k for k in before if after.get(k) != before[k])
    if changed:
        raise RuntimeError(f"Pre-existing files under {root} changed during evaluation: {changed[:10]}")


def prepare_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already exists and is not empty. Pass --overwrite to regenerate it "
                                  "(only this generated directory is replaced).")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root(), capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Scoring one run on its O2 validation split
# --------------------------------------------------------------------------- #
def score_run(spec: RunSpec, args: argparse.Namespace, device) -> dict[str, Any]:
    import torch
    from src.data.dataset import ImageStore, make_loader, prepare_split
    from src.data.splits import build_source_splits
    from src.models import count_parameters
    from src.models.evaluate import load_checkpoint, predict
    from src.utils import set_seed

    ckpt_path = spec.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    sha_before = file_sha256(ckpt_path, n=64)
    model, cfg, mapping = load_checkpoint(ckpt_path, device)
    ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(cfg)
    for key, val in (("metadata_dir", args.metadata_dir), ("image_dir", args.image_dir),
                     ("processed_dir", args.processed_dir)):
        if val:
            cfg["paths"][key] = val
    if cfg["protocol"]["source_run"] in cfg["protocol"]["target_runs"]:
        raise ValueError(f"{spec.slug}: source run is listed as a target run; refusing.")
    set_seed(cfg["experiment"]["seed"])
    log.info("[%s] %s | %s | checkpoint epoch %s", spec.label, spec.slug, spec.input_desc, ckpt_meta.get("epoch"))

    _, val_df, _, _ = build_source_splits(cfg)
    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    ds, kept, failures = prepare_split(val_df, f"{cfg['data']['ifo']}_val", cfg, store)
    bs = args.batch_size or cfg["train"]["batch_size"]
    loader = make_loader(ds, bs, False, num_workers=args.workers, seed=cfg["experiment"]["seed"])
    pred = predict(model, loader, device)
    if not np.array_equal(pred["y_true"], kept["label"].to_numpy()):
        raise AssertionError(f"{spec.slug}: loader labels are misaligned with the kept manifest.")
    input_shape = list(ds[0][0].shape)
    n_params = count_parameters(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if file_sha256(ckpt_path, n=64) != sha_before:
        raise RuntimeError(f"{ckpt_path} changed during evaluation. This must never happen.")

    summary = {}
    if (spec.run_dir / "summary.json").exists():
        import json
        summary = json.loads((spec.run_dir / "summary.json").read_text())
    if spec.kind == "single_view" and input_shape[0] != 1:
        raise AssertionError(f"{spec.slug}: single-view model received {input_shape[0]} views.")

    return {
        "spec": spec, "cfg": cfg, "mapping": mapping, "val_ids": val_df["gravityspy_id"].astype(str).tolist(),
        "kept": kept.reset_index(drop=True), "probs": pred["probs"], "failures": failures,
        "summary": summary, "input_shape": input_shape, "n_params": n_params,
        "checkpoint": {"path": str(ckpt_path), "sha256": sha_before, "epoch": ckpt_meta.get("epoch"),
                       "val_metrics_at_save": ckpt_meta.get("val_metrics")},
    }


# --------------------------------------------------------------------------- #
# Population checks
# --------------------------------------------------------------------------- #
def population_report(scored: list[dict[str, Any]]) -> dict[str, Any]:
    split_sets = {s["spec"].label: set(s["val_ids"]) for s in scored}
    kept_sets = {s["spec"].label: set(s["kept"]["gravityspy_id"].astype(str)) for s in scored}
    common = set.intersection(*kept_sets.values())
    union = set.union(*kept_sets.values())
    split_union = set.union(*split_sets.values())
    rows = []
    for s in scored:
        lab = s["spec"].label
        rows.append({"model": lab, "run": s["spec"].slug, "n_val_split": len(split_sets[lab]),
                     "n_dropped_image_failures": len(split_sets[lab]) - len(kept_sets[lab]),
                     "n_scored": len(kept_sets[lab]), "n_in_common_population": len(common),
                     "n_scored_but_excluded_from_common": len(kept_sets[lab] - common),
                     "n_in_other_models_but_missing_here": len(union - kept_sets[lab])})
    excluded = []
    for gid in sorted(union - common):
        missing = [lab for lab, ks in kept_sets.items() if gid not in ks]
        reasons = ["not in this model's val split" if gid not in split_sets[lab] else "image failure for this model"
                   for lab in missing]
        excluded.append({"gravityspy_id": gid, "missing_from": "; ".join(missing), "reason": "; ".join(reasons)})
    split_identical = all(v == next(iter(split_sets.values())) for v in split_sets.values())
    return {
        "identical_split": split_identical,
        "identical_scored_population": len(common) == len(union),
        "n_union": len(union), "n_common": len(common), "n_split_union": len(split_union),
        "per_model": pd.DataFrame(rows), "excluded": pd.DataFrame(excluded, columns=["gravityspy_id", "missing_from",
                                                                                    "reason"]),
        "common_ids": common,
    }


def restrict(s: dict[str, Any], ids: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    mask = s["kept"]["gravityspy_id"].astype(str).isin(ids).to_numpy()
    kept = s["kept"].loc[mask].reset_index(drop=True)
    probs = s["probs"][mask]
    order = np.argsort(kept["gravityspy_id"].astype(str).to_numpy(), kind="mergesort")   # same order for all models
    return kept.iloc[order].reset_index(drop=True), probs[order]


# --------------------------------------------------------------------------- #
# Paired block bootstrap
# --------------------------------------------------------------------------- #
def _fast_scores(y: np.ndarray, p: np.ndarray, c: int) -> tuple[float, float, float]:
    cm = np.bincount(y * c + p, minlength=c * c).reshape(c, c).astype(float)
    tp, sup, prd = np.diag(cm), cm.sum(1), cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(prd > 0, tp / prd, 0.0)
        rec = np.where(sup > 0, tp / sup, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = sup > 0
    return tp.sum() / max(len(y), 1), f1[present].mean(), rec[present].mean()


def paired_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], reference: str, times: np.ndarray, c: int,
                     n_boot: int, block_seconds: float, seed: int, min_blocks: int = 10) -> pd.DataFrame:
    """CI of (other - reference) for accuracy, macro-F1, balanced accuracy. Blocks of `block_seconds` are
    resampled with replacement (glitches are correlated in time, so per-event resampling is too optimistic).

    With fewer than `min_blocks` blocks the interval is meaningless (one block -> zero width), so no CI and no
    verdict are produced: the row says "not assessed" instead of claiming a difference."""
    rng = np.random.default_rng(seed)
    blocks = np.floor((times - times.min()) / block_seconds).astype(int)
    uniq, inv = np.unique(blocks, return_inverse=True)
    members = [np.where(inv == b)[0] for b in range(len(uniq))]
    point = {k: _fast_scores(y, v, c) for k, v in preds.items()}
    if len(members) < min_blocks:
        log.warning("Paired bootstrap not assessed: only %d time block(s) of %.0fs in the evaluated events "
                    "(need >= %d for a usable interval).", len(members), block_seconds, min_blocks)
        rows = []
        for k in preds:
            if k == reference:
                continue
            for j, metric in enumerate(("accuracy", "macro_f1", "balanced_accuracy")):
                rows.append({"model": k, "reference": reference, "metric": metric, "model_value": point[k][j],
                             "reference_value": point[reference][j],
                             "difference_model_minus_reference": point[k][j] - point[reference][j],
                             "ci95_low": np.nan, "ci95_high": np.nan,
                             "verdict": f"not assessed: only {len(members)} time block(s), need >= {min_blocks}",
                             "n_blocks": len(members), "n_bootstrap": 0, "block_seconds": block_seconds})
        return pd.DataFrame(rows)
    draws = {k: [] for k in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[b] for b in pick])
        for k, v in preds.items():
            draws[k].append(_fast_scores(y[idx], v[idx], c))
    ref = np.asarray(draws[reference])
    rows = []
    for k in preds:
        if k == reference:
            continue
        d = np.asarray(draws[k]) - ref
        for j, metric in enumerate(("accuracy", "macro_f1", "balanced_accuracy")):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            diff = point[k][j] - point[reference][j]
            verdict = ("higher than reference" if lo > 0 else "lower than reference" if hi < 0
                       else "no clear difference (95% CI includes 0)")
            rows.append({"model": k, "reference": reference, "metric": metric,
                         "model_value": point[k][j], "reference_value": point[reference][j],
                         "difference_model_minus_reference": diff, "ci95_low": lo, "ci95_high": hi,
                         "verdict": verdict, "n_blocks": len(members), "n_bootstrap": n_boot,
                         "block_seconds": block_seconds})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def ev_to_json(ev: dict[str, Any]) -> dict[str, Any]:
    out = {"overall": ev["overall"], "per_class": ev["per_class"].to_dict("records"),
           "confusion_matrix": ev["confusion_matrix"].tolist(),
           "model_confidence_thresholds": ev["model_confidence_thresholds"].to_dict("records"),
           "reliability_bins": ev["reliability"].to_dict("records")}
    if "label_confidence_thresholds" in ev:
        out["label_confidence_thresholds"] = ev["label_confidence_thresholds"].to_dict("records")
    return out


def model_plots(ev: dict[str, Any], spec: RunSpec, names: list[str], out: Path, population_note: str) -> None:
    pdir = out / "plots"
    one = [{"label": spec.label, "index": spec.index}]
    plots.plot_confusion(ev["confusion_matrix"], names, pdir / "confusion_matrix", f"{spec.label}: confusion matrix",
                         False, population_note)
    plots.plot_confusion(ev["confusion_matrix"], names, pdir / "confusion_matrix_normalised",
                         f"{spec.label}: row-normalised confusion matrix", True, population_note)
    plots.plot_reliability([{**one[0], "bins": ev["reliability"], "ece": ev["overall"]["ece"]}],
                           pdir / "reliability_diagram", f"Reliability, {population_note}")
    plots.plot_confidence_distribution([{**one[0], "conf": ev["conf"], "correct": ev["correct"]}],
                                       pdir / "confidence_distribution", f"Confidence distribution, {population_note}")
    plots.plot_sweep([{**one[0], "sweep": ev["sweep"]}], "coverage", pdir / "coverage_vs_threshold",
                     "Coverage (fraction retained)", f"Coverage vs model-confidence threshold, {population_note}")
    plots.plot_sweep([{**one[0], "sweep": ev["sweep"]}], "accuracy", pdir / "accuracy_vs_threshold",
                     "Accuracy on retained events", f"Accuracy vs model-confidence threshold, {population_note}")
    cov, risk = ev["risk_coverage"]
    plots.plot_risk_coverage([{**one[0], "coverage": cov, "risk": risk, "aurc": ev["overall"]["aurc"]}],
                             pdir / "selective_risk_vs_coverage", f"Selective risk vs coverage, {population_note}")


def write_model_report(s: dict[str, Any], ev: dict[str, Any], kept: pd.DataFrame, probs: np.ndarray,
                       out: Path, population_note: str, overwrite: bool) -> None:
    spec, names = s["spec"], s["mapping"].names
    prepare_out_dir(out, overwrite)
    ev["per_class"].to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(ev["confusion_matrix"], index=names, columns=names).rename_axis("reference \\ predicted").to_csv(
        out / "confusion_matrix.csv")
    thr = [ev["model_confidence_thresholds"]] + ([ev["label_confidence_thresholds"]]
                                                 if "label_confidence_thresholds" in ev else [])
    pd.concat(thr, ignore_index=True).to_csv(out / "threshold_table.csv", index=False)
    ev["reliability"].to_csv(out / "reliability_bins.csv", index=False)
    pred = kept[["gravityspy_id", "event_time", "ml_label", "ml_confidence", "label"]].copy()
    pred["pred"] = ev["y_pred"]
    pred["pred_name"] = [names[i] for i in ev["y_pred"]]
    pred["pred_conf"] = ev["conf"]
    pred["correct"] = ev["correct"]
    pred.to_csv(out / "predictions.csv", index=False)
    np.save(out / "probs.npy", probs.astype(np.float32))
    save_json({"run": spec.slug, "model": spec.label, "input": spec.input_desc, "population": population_note,
               "class_order": names, "checkpoint": s["checkpoint"], "metrics": ev_to_json(ev),
               "definitions": METRIC_DEFINITIONS}, out / "metrics.json")
    model_plots(ev, spec, names, out, population_note)


def comparison_table(scored, evs, pop, boot_ok) -> pd.DataFrame:
    rows = []
    for s in scored:
        spec, ev, summ = s["spec"], evs[s["spec"].label], s["summary"]
        vm = summ.get("val_metrics", {})
        own_acc = vm.get("accuracy")
        rows.append({
            "Model": spec.label, "Input": spec.input_desc, "Run": spec.slug, "N evaluated": ev["overall"]["n"],
            "Accuracy": ev["overall"]["accuracy"], "Macro-F1": ev["overall"]["macro_f1"],
            "Weighted-F1": ev["overall"]["weighted_f1"], "Balanced Accuracy": ev["overall"]["balanced_accuracy"],
            "ECE": ev["overall"]["ece"], "AURC": ev["overall"]["aurc"],
            "Parameters": s["n_params"], "Input shape per event": "x".join(map(str, s["input_shape"])),
            "Best epoch": summ.get("best_epoch", s["checkpoint"]["epoch"]),
            "Epochs run": summ.get("epochs_run"),
            "Best val macro-F1 (training log)": vm.get("macro_f1_present"),
            "Training time (h)": (summ["train_seconds"] / 3600.0) if summ.get("train_seconds") else None,
            "Val accuracy in summary.json": own_acc,
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else f"{v:.4f}"
        return "n/a" if v is None else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def conclusion_lines(boot: pd.DataFrame | None, reference: str | None) -> list[str]:
    if boot is None or boot.empty or reference is None:
        return ["No paired comparison was computed (needs a multi-view reference run and --bootstrap > 0)."]
    out = []
    for (model, metric), g in boot.groupby(["model", "metric"], sort=False):
        r = g.iloc[0]
        ci = (f"95% CI {r['ci95_low']:+.4f} to {r['ci95_high']:+.4f}" if np.isfinite(r["ci95_low"])
              else "no interval")
        out.append(f"- {model} vs {reference}, {metric}: {r['difference_model_minus_reference']:+.4f} "
                   f"({ci}) -> {r['verdict']}.")
    out.append("")
    out.append("These intervals reflect evaluation-sample uncertainty only (day-block bootstrap over the O2 "
               "validation events). Each model was trained once with one seed, so training variance is NOT included; "
               "a difference that is not clear here should not be reported as one model being better.")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Path:
    from src.utils import select_device

    out = resolve_path(args.out)
    prepare_out_dir(out, args.overwrite)
    setup_logging(out / "compare.log")
    device = select_device(args.device)
    specs = build_specs(args)
    log.info("Evaluating %d model(s) on %s: %s", len(specs), device, [s.label for s in specs])

    for s in specs:   # fail before any (slow) inference if a report would be overwritten
        d = s.run_dir / PER_MODEL_DIRNAME
        if d.exists() and any(d.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{d} already exists. Pass --overwrite to regenerate the per-model reports.")
    snapshots = {s.run_dir: tree_snapshot(s.run_dir, (PER_MODEL_DIRNAME,)) for s in specs}
    scored = [score_run(s, args, device) for s in specs]

    names = scored[0]["mapping"].names
    for s in scored[1:]:
        if s["mapping"].names != names:
            raise ValueError(f"{s['spec'].slug}: class mapping/order differs from {scored[0]['spec'].slug}; "
                             "models with different label spaces cannot be compared on one table.")

    pop = population_report(scored)
    if not pop["identical_split"]:
        log.error("The O2 validation SPLITS differ between models (different split/confidence/cap settings). "
                  "The comparison table uses only the %d common events; see population/.", pop["n_common"])
    elif not pop["identical_scored_population"]:
        log.warning("Same split, but %d event(s) were scored by some models and not others (image failures). "
                    "The comparison uses the %d common events.", pop["n_union"] - pop["n_common"], pop["n_common"])
    else:
        log.info("All models were scored on the identical %d O2 validation events.", pop["n_common"])
    if pop["n_common"] == 0:
        raise RuntimeError("No event is common to all models; nothing can be compared.")

    common_note = (f"O2 validation, common population (n={pop['n_common']})" if not pop["identical_scored_population"]
                   else f"O2 validation (n={pop['n_common']}, identical for all models)")

    # per-model reports (own population) + common-population evaluations
    evs_common, kept_common, preds_common = {}, None, {}
    for s in scored:
        spec = s["spec"]
        own_ev = evaluate_predictions(s["kept"]["label"].to_numpy(), s["probs"], names,
                                      label_conf=s["kept"]["ml_confidence"].to_numpy())
        own_note = f"O2 validation, this model's scored events (n={len(s['kept'])})"
        write_model_report(s, own_ev, s["kept"], s["probs"], spec.run_dir / PER_MODEL_DIRNAME, own_note,
                           args.overwrite)
        kc, pc = restrict(s, pop["common_ids"])
        if kept_common is None:
            kept_common = kc
        elif not (kc["gravityspy_id"].astype(str).to_numpy() == kept_common["gravityspy_id"].astype(str).to_numpy()).all() \
                or not (kc["label"].to_numpy() == kept_common["label"].to_numpy()).all():
            raise AssertionError("Common-population events or labels differ between models.")
        evs_common[spec.label] = evaluate_predictions(kc["label"].to_numpy(), pc, names,
                                                      label_conf=kc["ml_confidence"].to_numpy())
        preds_common[spec.label] = evs_common[spec.label]["y_pred"]

    for s in scored:
        assert_unchanged(snapshots[s["spec"].run_dir], tree_snapshot(s["spec"].run_dir, (PER_MODEL_DIRNAME,)),
                         s["spec"].run_dir)

    # ---- population artifacts --------------------------------------------- #
    (out / "population").mkdir()
    pop["per_model"].to_csv(out / "population" / "population_by_model.csv", index=False)
    pop["excluded"].to_csv(out / "population" / "excluded_samples.csv", index=False)
    dist = class_distribution(kept_common["label"].to_numpy(), names)
    dist_df = dist.rename_axis("class").reset_index()
    dist_df["fraction"] = dist_df["n_events"] / max(dist_df["n_events"].sum(), 1)
    dist_df.to_csv(out / "population" / "class_distribution_common.csv", index=False)
    for s in scored:
        f = s["failures"]
        if len(f):
            f.to_csv(out / "population" / f"image_failures_{s['spec'].slug}.csv", index=False)

    # ---- tables ------------------------------------------------------------ #
    table = comparison_table(scored, evs_common, pop, True)
    table.to_csv(out / "comparison_table.csv", index=False)
    per_class = {k: pd.DataFrame({lab: evs_common[lab]["per_class"].set_index("class")[k] for lab in evs_common})
                 for k in ("precision", "recall", "f1")}
    support = evs_common[scored[0]["spec"].label]["per_class"].set_index("class")["support"]
    for k, df in per_class.items():
        df.insert(0, "support", support)
        df.rename_axis("class").to_csv(out / f"per_class_{k}.csv")
    thr = []
    for lab, ev in evs_common.items():
        for key in ("model_confidence_thresholds", "label_confidence_thresholds"):
            if key in ev:
                thr.append(ev[key].assign(model=lab))
    thresholds = pd.concat(thr, ignore_index=True)
    thresholds = thresholds[["model"] + [c for c in thresholds.columns if c != "model"]]
    thresholds.to_csv(out / "thresholds.csv", index=False)

    long_rows = []
    for lab, ev in evs_common.items():
        for k, v in ev["overall"].items():
            long_rows.append({"model": lab, "group": "overall", "class": "", "threshold_type": "", "threshold": "",
                              "metric": k, "value": v})
        for _, r in ev["per_class"].iterrows():
            for k in ("support", "predicted", "precision", "recall", "f1"):
                long_rows.append({"model": lab, "group": "per_class", "class": r["class"], "threshold_type": "",
                                  "threshold": "", "metric": k, "value": r[k]})
    for _, r in thresholds.iterrows():
        for k in ("n_retained", "coverage", "accuracy_retained", "macro_f1_retained", "balanced_accuracy_retained",
                  "weighted_f1_retained", "selective_risk_retained"):
            long_rows.append({"model": r["model"], "group": "threshold", "class": "",
                              "threshold_type": r["threshold_type"], "threshold": r["threshold"], "metric": k,
                              "value": r[k]})
    pd.DataFrame(long_rows).assign(population=common_note).to_csv(out / "all_metrics_long.csv", index=False)

    # ---- paired bootstrap --------------------------------------------------- #
    reference = next((s["spec"].label for s in scored if s["spec"].kind == "multiview"), None)
    boot = None
    if reference is not None and args.bootstrap > 0 and len(scored) > 1:
        block = float(scored[0]["cfg"]["split"].get("block_seconds", 86400))
        boot = paired_bootstrap(kept_common["label"].to_numpy(), preds_common, reference,
                                kept_common["event_time"].to_numpy(float), len(names), args.bootstrap, block,
                                args.seed, args.min_blocks)
        boot.to_csv(out / "paired_differences_vs_multiview.csv", index=False)

    # ---- plots ---------------------------------------------------------------- #
    pdir = out / "plots"
    entries = [{"label": s["spec"].label, "index": s["spec"].index} for s in scored]
    plots.plot_overall_comparison(table, pdir / "overall_metric_comparison", f"Overall metrics, {common_note}")
    plots.plot_per_class_comparison(per_class["f1"].drop(columns="support"), support, pdir / "per_class_f1_comparison",
                                    "F1 score", f"Per-class F1, {common_note}")
    plots.plot_per_class_comparison(per_class["recall"].drop(columns="support"), support,
                                    pdir / "per_class_recall_comparison", "Recall", f"Per-class recall, {common_note}")
    plots.plot_reliability([{**e, "bins": evs_common[e["label"]]["reliability"], "ece": evs_common[e["label"]]["overall"]["ece"]}
                            for e in entries], pdir / "reliability_diagrams", f"Reliability diagrams, {common_note}")
    plots.plot_confidence_distribution([{**e, "conf": evs_common[e["label"]]["conf"],
                                         "correct": evs_common[e["label"]]["correct"]} for e in entries],
                                       pdir / "confidence_distributions", f"Model-confidence distributions, {common_note}")
    sweeps = [{**e, "sweep": evs_common[e["label"]]["sweep"]} for e in entries]
    plots.plot_sweep(sweeps, "coverage", pdir / "coverage_vs_threshold", "Coverage (fraction of events retained)",
                     "Coverage vs model-confidence threshold", common_note)
    plots.plot_sweep(sweeps, "accuracy", pdir / "accuracy_vs_threshold", "Accuracy on retained events",
                     "Accuracy vs model-confidence threshold", common_note)
    plots.plot_risk_coverage([{**e, "coverage": evs_common[e["label"]]["risk_coverage"][0],
                               "risk": evs_common[e["label"]]["risk_coverage"][1],
                               "aurc": evs_common[e["label"]]["overall"]["aurc"]} for e in entries],
                             pdir / "selective_risk_vs_coverage", f"Selective risk vs coverage, {common_note}")
    for s in scored:
        ev = evs_common[s["spec"].label]
        plots.plot_confusion(ev["confusion_matrix"], names, pdir / "confusion_matrices" / f"{s['spec'].slug}_counts",
                             f"{s['spec'].label}: confusion matrix", False, common_note)
        plots.plot_confusion(ev["confusion_matrix"], names, pdir / "confusion_matrices" / f"{s['spec'].slug}_normalised",
                             f"{s['spec'].label}: row-normalised confusion matrix", True, common_note)

    # ---- O3a preliminary diagnostic (read-only) ------------------------------ #
    o3a = None
    if args.o3a_diagnostic_run:
        from src.evaluation.o3a_diagnostic import render_report
        o3a = render_report(resolve_path(args.o3a_diagnostic_run), overwrite=args.overwrite)

    # ---- summary.json + README ------------------------------------------------- #
    save_json({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "git_commit": git_commit(),
        "device": str(device), "checkpoint": args.checkpoint, "class_order": names,
        "population": {"identical_split": pop["identical_split"],
                       "identical_scored_population": pop["identical_scored_population"],
                       "n_common": pop["n_common"], "n_union": pop["n_union"],
                       "per_model": pop["per_model"].to_dict("records"),
                       "class_distribution_common": dist.to_dict(), "note": common_note},
        "thresholds_evaluated": [None if t is None else t for t in DEFAULT_THRESHOLDS],
        "models": [{"label": s["spec"].label, "run": s["spec"].slug, "run_dir": str(s["spec"].run_dir),
                    "kind": s["spec"].kind, "input": s["spec"].input_desc, "input_shape_per_event": s["input_shape"],
                    "parameters": s["n_params"], "checkpoint": s["checkpoint"],
                    "training_summary": {k: s["summary"].get(k) for k in ("best_epoch", "epochs_run", "train_seconds",
                                                                         "n_train", "n_val", "device")},
                    "protocol_check": _protocol_check(s["spec"].run_dir),
                    "metrics_common_population": ev_to_json(evs_common[s["spec"].label])} for s in scored],
        "paired_bootstrap": None if boot is None else boot.to_dict("records"),
        "o3a_preliminary_diagnostic": o3a,
        "definitions": METRIC_DEFINITIONS,
    }, out / "summary.json")
    cols = ["Model", "Input", "N evaluated", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced Accuracy", "ECE"]
    extra = ["Model", "Parameters", "Best epoch", "Epochs run", "Best val macro-F1 (training log)", "Training time (h)",
             "Val accuracy in summary.json"]
    readme = [f"# O2 validation comparison: multi-view vs single-view", "",
              f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}. Population: {common_note}.", "",
              "## Headline metrics", "", markdown_table(table, cols), "",
              "## Training facts", "", markdown_table(table, extra), "",
              "`Val accuracy in summary.json` is the number recorded at training time on that model's own validation "
              "events; it should equal `Accuracy` when the population is identical.", "",
              "## Paired differences (day-block bootstrap, 95% CI)", "", *conclusion_lines(boot, reference), "",
              "## Population", "",
              f"- identical validation split for all models: {pop['identical_split']}",
              f"- identical scored events for all models: {pop['identical_scored_population']}",
              f"- events in common population: {pop['n_common']} (union {pop['n_union']})", "",
              markdown_table(pop["per_model"], list(pop["per_model"].columns)), "",
              "Model-confidence thresholds use the classifier's max softmax; label-confidence thresholds filter on "
              "Gravity Spy `ml_confidence`. See thresholds.csv (column threshold_type).", ""]
    if o3a:
        readme += ["## O3a (PRELIMINARY sampled diagnostic, not a full O3a evaluation)", "",
                   f"Rendered to `{o3a['output_dir']}` from existing predictions only.", ""]
    (out / "README.md").write_text("\n".join(readme))
    log.info("\n%s", markdown_table(table, cols))
    for line in conclusion_lines(boot, reference):
        if line:
            log.info(line)
    log.info("Comparison written to %s", out)
    return out


def _protocol_check(run_dir: Path) -> dict | None:
    p = run_dir / "protocol_check.json"
    if not p.exists():
        return None
    import json
    d = json.loads(p.read_text())
    diffs = d.get("differences") or {}
    return {"verified_against_reference": d.get("verified_against_reference"),
            "unexpected_differences": diffs.get("unexpected", []),
            "protocol_critical_differences": diffs.get("protocol_critical", [])}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multiview-run", help="Frozen multi-view run dir (reference for paired differences).")
    ap.add_argument("--single-view-runs", nargs="+", help="Explicit single-view run dirs (default: auto-discover).")
    ap.add_argument("--runs-root", default="experiments/runs", help="Where to auto-discover single-view runs.")
    ap.add_argument("--extra-run", action="append", help="Any other run dir to include (repeatable).")
    ap.add_argument("--include-debug", action="store_true", help="Also auto-discover *_debug runs.")
    ap.add_argument("--ifo", choices=["H1", "L1"], help="Detector for auto-discovery (default: from the multi-view run).")
    ap.add_argument("--out", default="experiments/comparisons/H1_O2_single_vs_multiview")
    ap.add_argument("--checkpoint", default="best", help="Checkpoint name inside each run's checkpoints/ (default best).")
    ap.add_argument("--o3a-diagnostic-run", help="Existing sampled O3a diagnostic run dir to visualise (read-only).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, help="Inference batch size (default: each run's train.batch_size).")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for inference.")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Paired block-bootstrap resamples (0 = off).")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    ap.add_argument("--min-blocks", type=int, default=10,
                    help="Minimum number of day-blocks for the bootstrap to report an interval.")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing comparison/per-model output dirs.")
    ap.add_argument("--metadata-dir")
    ap.add_argument("--image-dir")
    ap.add_argument("--processed-dir")
    args = ap.parse_args(argv)
    setup_logging()
    run(args)


if __name__ == "__main__":
    main()
