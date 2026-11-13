"""Single-view CNN baselines: one model per Omega-scan duration, each receiving ONLY that view.

    # one view (recommended: inherit the exact protocol of the frozen multi-view run)
    python -m src.models.train_single_view --ifo H1 --view 0.5 --device cuda \
        --reference-run experiments/runs/H1_O2_colab_baseline_v4

    # all four, sequentially (skips views whose run already finished)
    python -m src.models.train_single_view --ifo H1 --view all --device cuda \
        --reference-run experiments/runs/H1_O2_colab_baseline_v4 --skip-existing

Design
------
* Model: model.type=single_view -> the existing BaselineCNN conv backbone (same ConvStage design, width
  schedule, embedding and head as the other baselines), built for ONE input view.
* Input: the dataset hands the model only view `model.single_view_index` -> tensor (B, 1, C, H, W).
  Events are still decoded from the same aligned four-view array, so the train/val event populations are
  identical to the multi-view run (an event with ANY failed view is dropped for every model).
* Protocol: with --reference-run, the reference run's saved config.yaml is the base config, so the
  chronological O2 split, purge gap, confidence filtering, per-class caps, class weighting, loss, optimiser,
  schedule, epochs, early stopping and seed are inherited unchanged. Only these are changed:
      experiment.name / debug / device, model.type=single_view, model.single_view_index,
      loss.scale_consistency.weight=0 (a multi-view-only regulariser)
  plus anything you override explicitly (--epochs, --batch-size, --lr, --workers, --set ...).
  Every difference from the reference is written to <run>/protocol_check.json and logged.
* Training itself is src.models.train.run (unchanged): same loop, checkpointing, early stopping on source-val
  macro-F1, artifacts. O3a is never read (TargetLeakageError in splits.load_run_metadata).
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models import train as train_mod
from src.utils import file_sha256, load_config, project_root, resolve_path, save_json, setup_logging

log = logging.getLogger("train_single_view")

# Keys that are EXPECTED to differ from the multi-view reference run.
EXPECTED_DIFFS = {
    "experiment.name", "experiment.debug", "experiment.device",
    "model.type", "model.single_view_index", "loss.scale_consistency.weight", "provenance",
}
# Keys whose difference would make the comparison unfair (data population / labels / selection protocol).
PROTOCOL_CRITICAL_PREFIXES = (
    "protocol.", "data.", "classes.", "confidence.", "split.", "sampling.", "experiment.seed",
)


def view_index_for(view: float, durations: list[float]) -> int:
    for i, d in enumerate(durations):
        if abs(float(d) - float(view)) < 1e-9:
            return i
    raise ValueError(f"--view {view:g} is not one of data.view_durations {durations}")


def default_run_name(cfg: dict[str, Any], view: float) -> str:
    return f"{cfg['data']['ifo']}_{cfg['protocol']['source_run']}_single_view_{view:g}s"


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def protocol_diff(cfg: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Every dotted key whose value differs between the single-view config and the reference config."""
    a, b = flatten(cfg), flatten(reference)
    rows = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("debug.") or k.startswith("provenance"):
            continue
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if va != vb:
            expected = k in EXPECTED_DIFFS
            critical = (not expected) and k.startswith(PROTOCOL_CRITICAL_PREFIXES)
            rows.append({"key": k, "single_view": va, "reference": vb, "expected": expected,
                         "protocol_critical": critical})
    return {
        "n_differences": len(rows),
        "unexpected": [r for r in rows if not r["expected"]],
        "protocol_critical": [r for r in rows if r["protocol_critical"]],
        "all": rows,
    }


def build_single_view_cfg(args: argparse.Namespace, view: float) -> tuple[dict[str, Any], dict[str, Any] | None, Path]:
    """Return (cfg, reference_cfg_or_None, base_config_path)."""
    if args.reference_run:
        ref_dir = resolve_path(args.reference_run)
        base_path = ref_dir / "config.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"--reference-run {ref_dir}: config.yaml not found. Point it at the multi-view run directory "
                "produced by train.py (it contains config.yaml and checkpoints/best.pt).")
        reference = load_config(base_path)
    else:
        base_path = resolve_path(args.config)
        reference = None
    base = load_config(base_path)
    idx = view_index_for(view, base["data"]["view_durations"])

    sets: list[str] = []
    if args.epochs is not None:
        sets.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        sets.append(f"train.batch_size={args.batch_size}")
    if args.lr is not None:
        sets.append(f"train.lr={args.lr}")
    if args.workers is not None:
        sets.append(f"train.num_workers={args.workers}")
    if args.patience is not None:
        sets.append(f"train.patience={'null' if args.patience < 0 else args.patience}")
    sets += list(args.set or [])
    # forced last: these define the single-view experiment and cannot be overridden by --set
    sets += [f"model.single_view_index={idx}", "loss.scale_consistency.weight=0.0"]

    ifo = args.ifo or base["data"]["ifo"]
    name = args.name or default_run_name({**base, "data": {**base["data"], "ifo": ifo}}, view)
    # A reference run trained with --debug already carries the debug overrides (plus any of its own --set values);
    # re-applying the debug block would silently undo the latter, so it is only applied to non-debug bases.
    base_is_debug = bool(base.get("experiment", {}).get("debug", False))
    ns = argparse.Namespace(
        config=str(base_path), debug=args.debug and not base_is_debug, ifo=args.ifo, model="single_view", seed=args.seed,
        device=args.device, metadata_dir=args.metadata_dir, image_dir=args.image_dir,
        processed_dir=args.processed_dir, output_dir=args.output_dir, set=sets, name=name)
    cfg = train_mod.build_cfg(ns)
    if base_is_debug:
        cfg["experiment"]["debug"] = True
        if not cfg["experiment"]["name"].endswith("_debug"):
            cfg["experiment"]["name"] += "_debug"
    if cfg["model"]["type"] != "single_view" or cfg["model"]["single_view_index"] != idx:
        raise AssertionError("single-view configuration was not applied")
    cfg["provenance"] = {
        "experiment_kind": "single_view_baseline",
        "input_view_index": idx,
        "input_view_column": cfg["data"]["view_columns"][idx],
        "input_view_seconds": float(cfg["data"]["view_durations"][idx]),
        "base_config": str(base_path),
        "base_config_sha256": file_sha256(base_path),
        "reference_run": str(resolve_path(args.reference_run)) if args.reference_run else None,
    }
    return cfg, reference, base_path


def run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "checkpoints" / "best.pt").exists() and (run_dir / "summary.json").exists()


def train_one_view(args: argparse.Namespace, view: float) -> Path | None:
    cfg, reference, base_path = build_single_view_cfg(args, view)
    run_dir = resolve_path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    if args.skip_existing and run_is_complete(run_dir):
        log.info("[%gs] %s already complete (best.pt + summary.json); skipping.", view, run_dir)
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        hint = ("it looks incomplete (no summary.json): re-run with --overwrite to restart it"
                if not run_is_complete(run_dir) else "use --skip-existing to keep it or --overwrite to replace it")
        raise FileExistsError(f"{run_dir} already exists; {hint}.")

    diff = protocol_diff(cfg, reference) if reference is not None else None
    if reference is None:
        log.warning("[%gs] No --reference-run given: protocol taken from %s. Consistency with the multi-view run "
                    "is NOT verified; pass --reference-run <multi-view run dir> for a like-for-like baseline.",
                    view, base_path)
    else:
        for r in diff["unexpected"]:
            (log.error if r["protocol_critical"] else log.warning)(
                "[%gs] differs from reference run: %s = %r (reference %r)%s", view, r["key"], r["single_view"],
                r["reference"], "  <-- PROTOCOL-CRITICAL" if r["protocol_critical"] else "")
        if diff["protocol_critical"] and not (args.allow_protocol_change or args.debug):
            raise ValueError(
                f"{len(diff['protocol_critical'])} protocol-critical setting(s) differ from the reference run "
                f"({[r['key'] for r in diff['protocol_critical']]}). The comparison would not be like-for-like. "
                "Pass --allow-protocol-change only if this is intentional.")

    log.info("[%gs] training %s: input view %s (%s), base config %s", view, cfg["experiment"]["name"],
             cfg["provenance"]["input_view_index"], cfg["provenance"]["input_view_column"], base_path)
    out = train_mod.run(cfg, overwrite=args.overwrite)
    save_json({
        "run": cfg["experiment"]["name"], "provenance": cfg["provenance"],
        "reference_run": cfg["provenance"]["reference_run"],
        "verified_against_reference": reference is not None,
        "differences": diff,
    }, out / "protocol_check.json")
    return out


def parse_views(values: list[str]) -> list[float]:
    if len(values) == 1 and values[0].lower() == "all":
        return [0.5, 1.0, 2.0, 4.0]
    out = []
    for v in values:
        try:
            out.append(float(v.rstrip("s")))
        except ValueError as exc:
            raise SystemExit(f"--view expects durations like 0.5 1 2 4 or 'all', got '{v}'") from exc
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", nargs="+", required=True,
                    help="Duration(s) in seconds: 0.5 1 2 4, or 'all'. Several values train sequentially.")
    ap.add_argument("--config", default=str(project_root() / "configs" / "classifier.yaml"),
                    help="Base config when --reference-run is not given.")
    ap.add_argument("--reference-run", help="Multi-view run directory whose config.yaml defines the protocol "
                                            "(e.g. experiments/runs/H1_O2_colab_baseline_v4). Strongly recommended.")
    ap.add_argument("--ifo", choices=["H1", "L1"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="With --device cuda, continue on CPU if no GPU is visible (default: stop).")
    ap.add_argument("--name", help="Run name (only with a single --view). Default {ifo}_{run}_single_view_{v}s")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int, help="DataLoader workers (train.num_workers). 2 is a good Colab value.")
    ap.add_argument("--patience", type=int, help="Early-stopping patience; negative disables it.")
    ap.add_argument("--debug", action="store_true", help="Apply the config's `debug:` overrides (tiny and fast).")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing run directory.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip views whose run already has best.pt + summary.json.")
    ap.add_argument("--allow-protocol-change", action="store_true",
                    help="Allow data/split/label/confidence settings that differ from the reference run.")
    ap.add_argument("--metadata-dir")
    ap.add_argument("--image-dir")
    ap.add_argument("--processed-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="Override any config value, e.g. --set train.weight_decay=0.02 (repeatable).")
    args = ap.parse_args(argv)
    setup_logging()

    views = parse_views(args.view)
    if args.name and len(views) > 1:
        raise SystemExit("--name can only be used with a single --view.")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.allow_cpu_fallback:
        raise SystemExit("--device cuda requested but no CUDA GPU is visible. On Colab: Runtime > Change runtime "
                         "type > GPU. Pass --allow-cpu-fallback to train on CPU anyway.")

    done, failed = [], []
    for view in views:
        try:
            done.append((view, train_one_view(args, view)))
        except Exception as exc:  # keep going so one failure does not lose a whole Colab session
            log.exception("[%gs] FAILED: %s", view, exc)
            failed.append((view, f"{type(exc).__name__}: {exc}"))
            if len(views) == 1:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log.info("Single-view summary: %d finished, %d failed", len(done), len(failed))
    for v, p in done:
        log.info("  %gs -> %s", v, p)
    for v, why in failed:
        log.error("  %gs FAILED: %s", v, why)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
