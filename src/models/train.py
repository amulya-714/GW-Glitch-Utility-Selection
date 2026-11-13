"""Train a glitch classifier on SOURCE-run (O2) data only.

    python -m src.models.train --config configs/classifier.yaml --ifo H1 --model multiview
    python -m src.models.train --config configs/classifier.yaml --ifo H1 --debug        # tiny/fast

Protocol guarantees (see also splits.py):
  * only the source run is ever loaded (target runs raise TargetLeakageError);
  * train/val are a temporal split with a purge gap;
  * model selection (best epoch, early stopping) uses source-val macro-F1 only;
  * no detector or run identifier is a model input.
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.data.dataset import ImageStore, class_counts, make_loader, make_sampler, prepare_split
from src.data.preprocessing import Augment, Normalizer, PreprocessConfig, preprocessing_signature
from src.data.splits import build_source_splits
from src.models import build_model, count_parameters
from src.models.evaluate import evaluate_loader, format_summary, save_metrics
from src.models.losses import build_loss, class_weights_from_counts, selective_scale_consistency
from src.utils import (
    deep_update, load_config, parse_override, project_root, resolve_path, save_json,
    select_device, set_by_dotted_key, set_seed, setup_logging,
)

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.debug:
        cfg = deep_update(cfg, cfg.get("debug", {}))
    if args.ifo:
        cfg["data"]["ifo"] = args.ifo
    if args.model:
        cfg["model"]["type"] = args.model
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.device:
        cfg["experiment"]["device"] = args.device
    if args.metadata_dir:
        cfg["paths"]["metadata_dir"] = args.metadata_dir
    if args.image_dir:
        cfg["paths"]["image_dir"] = args.image_dir
    if args.processed_dir:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    for item in args.set or []:
        key, value = parse_override(item)
        set_by_dotted_key(cfg, key, value)
    cfg["experiment"]["debug"] = bool(args.debug)
    base = args.name or cfg["experiment"].get("name") or \
        f"{cfg['data']['ifo']}_{cfg['model']['type']}_seed{cfg['experiment']['seed']}"
    if args.debug and not base.endswith("_debug"):
        base += "_debug"
    cfg["experiment"]["name"] = base
    validate_cfg(cfg)
    return cfg


def validate_cfg(cfg: dict[str, Any]) -> None:
    proto = cfg["protocol"]
    if proto["source_run"] in proto["target_runs"]:
        raise ValueError(f"protocol.source_run '{proto['source_run']}' is also listed in protocol.target_runs.")
    if cfg["data"]["ifo"] not in ("H1", "L1"):
        raise ValueError(f"data.ifo must be H1 or L1, got '{cfg['data']['ifo']}'")
    if len(cfg["data"]["view_columns"]) != len(cfg["data"]["view_durations"]):
        raise ValueError("data.view_columns and data.view_durations must have equal length.")
    consistency = cfg["loss"].get("scale_consistency", {})
    if float(consistency.get("weight", 0.0)) < 0:
        raise ValueError("loss.scale_consistency.weight must be non-negative")
    if float(consistency.get("weight", 0.0)) and not (
        cfg["model"]["type"] == "multiview" and cfg["model"].get("scale_interaction", {}).get("enabled", False)
    ):
        raise ValueError("scale consistency requires model.type=multiview and model.scale_interaction.enabled=true")
    if cfg["model"]["type"] == "single_view":
        idx = cfg["model"].get("single_view_index")
        if idx is None or not 0 <= int(idx) < len(cfg["data"]["view_columns"]):
            raise ValueError(f"model.single_view_index={idx} out of range for data.view_columns.")
    # Fail fast on bad crop / resolution / normalisation settings, before any download.
    PreprocessConfig.from_cfg(cfg["data"])
    Normalizer.from_cfg(cfg["data"])
    imb = cfg["imbalance"]
    if imb["weighting"] != "none" and imb["sampler"] != "none":
        log.warning("Both imbalance.weighting=%s and imbalance.sampler=%s are active: classes are "
                    "compensated twice. Usually pick one.", imb["weighting"], imb["sampler"])


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def make_scheduler(opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, mapping, epoch: int, val_metrics: dict,
                    preprocessing: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "config": cfg, "class_mapping": mapping.to_dict(),
        "epoch": epoch, "preprocessing": preprocessing,
        "val_metrics": {k: v for k, v in val_metrics.items() if k not in ("per_class", "confusion_matrix")},
    }, path)


def train_one_epoch(model, loader, criterion, opt, sched, device, grad_clip, log_every, epoch,
                    consistency_cfg: dict | None = None) -> tuple[float, float]:
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    consistency_sum, active_sum = 0.0, 0.0
    fused_conf_sum, view_conf_sum = 0.0, 0.0
    threshold_sums = {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}

    for step, (x, y) in enumerate(loader):
        if len(y) < 2:
            continue

        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        scale_weight = float((consistency_cfg or {}).get("weight", 0.0))

        if scale_weight:
            logits, extras = model(x, return_extras=True)
        else:
            logits = model(x)

        loss = criterion(logits, y)

        if scale_weight:
            consistency_loss, active_fraction = selective_scale_consistency(
                extras["view_logits"], logits,
                float(consistency_cfg.get("confidence_threshold", 0.7)),
                float(consistency_cfg.get("temperature", 1.0)),
                return_stats=True,
            )
            loss = loss + scale_weight * consistency_loss
            consistency_sum += consistency_loss.detach().item() * len(y)
            active_sum += active_fraction.item() * len(y)

            with torch.no_grad():
                fused_conf = torch.softmax(logits.detach(), dim=-1).max(dim=-1).values
                view_conf = torch.softmax(extras["view_logits"].detach(), dim=-1).max(dim=-1).values
                fused_conf_sum += fused_conf.mean().item() * len(y)
                view_conf_sum += view_conf.mean().item() * len(y)

                for threshold in threshold_sums:
                    passed = (
                        (fused_conf[:, None] >= threshold) &
                        (view_conf >= threshold)
                    )
                    threshold_sums[threshold] += passed.float().mean().item() * len(y)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch} step {step}. Try a lower train.lr, "
                "check class weights (imbalance.max_weight) or inputs."
            )

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        loss_sum += loss.detach().item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        n += len(y)

        if log_every and (step + 1) % log_every == 0:
            log.info("  epoch %d step %d/%d loss=%.4f",
                     epoch, step + 1, len(loader), loss_sum / max(n, 1))

    if scale_weight and n:
        log.info(
            "  epoch %d consistency loss=%.5f | configured gate active=%.1f%%",
            epoch, consistency_sum / n, 100.0 * active_sum / n
        )
        log.info(
            "  confidence: fused mean=%.3f | individual-view mean=%.3f",
            fused_conf_sum / n, view_conf_sum / n
        )
        log.info(
            "  gate activation: 0.3=%.1f%% | 0.5=%.1f%% | 0.7=%.1f%%",
            *(100.0 * threshold_sums[t] / n for t in (0.3, 0.5, 0.7))
        )

    return loss_sum / max(n, 1), correct / max(n, 1)

def plot_curves(history: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_title("loss")
    ax[1].plot(history["epoch"], history["train_acc"], label="train")
    ax[1].plot(history["epoch"], history["val_acc"], label="val")
    ax[1].set_title("accuracy")
    ax[2].plot(history["epoch"], history["val_macro_f1"], color="C2")
    ax[2].set_title("val macro-F1 (present classes)")
    ax[3].plot(history["epoch"], history["lr"], color="C3")
    ax[3].set_title("learning rate")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any], overwrite: bool = False) -> Path:
    exp = cfg["experiment"]
    run_dir = resolve_path(exp["output_dir"]) / exp["name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists. Use --overwrite or choose --name.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    set_seed(exp["seed"])
    device = select_device(exp["device"])
    log.info("Run: %s | device: %s | seed: %d", exp["name"], device, exp["seed"])
    prep_sig = preprocessing_signature(cfg["data"])
    save_json(prep_sig, run_dir / "preprocessing.json")
    log.info("Preprocessing [%s]: crop=%s ref=%s size(HxW)=%s channels=%d resize=%s alpha=%s norm=%s views=%s",
             prep_sig["hash"], prep_sig["geometry"]["crop"], prep_sig["geometry"]["crop_ref"],
             prep_sig["geometry"]["size"], prep_sig["geometry"]["channels"], prep_sig["geometry"]["resize"],
             prep_sig["geometry"]["alpha"], prep_sig["normalization"]["mode"], prep_sig["views"])

    # ---- data ------------------------------------------------------------- #
    train_df, val_df, mapping, reports = build_source_splits(cfg)
    save_json(mapping.to_dict(), run_dir / "class_mapping.json")
    (run_dir / "splits").mkdir(exist_ok=True)
    reports["train_counts"].to_csv(run_dir / "splits" / "train_class_counts.csv")
    reports["val_counts"].to_csv(run_dir / "splits" / "val_class_counts.csv")
    reports["purged"][["gravityspy_id", "event_time", "ml_label"]].to_csv(run_dir / "splits" / "purged_events.csv", index=False)
    reports["outside_label_space"][["gravityspy_id", "event_time", "ml_label"]].to_csv(
        run_dir / "splits" / "excluded_label_events.csv", index=False)

    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    aug_cfg = cfg["augment"]
    augment = Augment(aug_cfg["max_time_shift_frac"], aug_cfg["noise_std"], exp["seed"]) if aug_cfg["enabled"] else None
    ifo = cfg["data"]["ifo"]
    train_ds, train_kept, f_tr = prepare_split(train_df, f"{ifo}_train", cfg, store, augment=augment)
    val_ds, val_kept, f_va = prepare_split(val_df, f"{ifo}_val", cfg, store)
    failures = pd.concat([f_tr.assign(split="train"), f_va.assign(split="val")], ignore_index=True)
    failures.to_csv(run_dir / "image_failures.csv", index=False)
    cols = ["gravityspy_id", "event_time", "ml_label", "ml_confidence", "label"]
    train_kept[cols].to_csv(run_dir / "splits" / "train_manifest.csv", index=False)
    val_kept[cols].to_csv(run_dir / "splits" / "val_manifest.csv", index=False)
    log.info("Final sizes: train=%d val=%d (image failures: %d)", len(train_ds), len(val_ds), len(failures))

    tr_counts = class_counts(train_ds.labels, mapping.num_classes)
    log.info("Train class counts (after drops): %s", {mapping.names[i]: int(c) for i, c in enumerate(tr_counts)})
    empty = [mapping.names[i] for i, c in enumerate(tr_counts) if c == 0]
    if empty:
        log.warning("Classes with NO training samples (cannot be learned): %s", empty)

    # ---- model / loss / optimiser ------------------------------------------ #
    model = build_model(cfg, mapping.num_classes).to(device)
    log.info("Model %s: %s trainable parameters", cfg["model"]["type"], f"{count_parameters(model):,}")
    imb, tr = cfg["imbalance"], cfg["train"]
    weights = class_weights_from_counts(tr_counts, imb["weighting"], imb["weight_power"],
                                        imb["effective_number_beta"], imb["max_weight"])
    if weights is not None:
        log.info("Class weights: %s", {mapping.names[i]: round(float(w), 3) for i, w in enumerate(weights)})
    criterion = build_loss(cfg["loss"], weights, device)
    val_criterion = nn.CrossEntropyLoss()  # unweighted so val loss is comparable across settings

    sampler = make_sampler(train_ds.labels, imb["sampler"], mapping.num_classes, exp["seed"])
    train_loader = make_loader(train_ds, tr["batch_size"], True, sampler, tr["num_workers"], exp["seed"],
                               drop_last=len(train_ds) > tr["batch_size"])
    val_loader = make_loader(val_ds, tr["batch_size"], False, num_workers=tr["num_workers"], seed=exp["seed"])

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    steps_per_epoch = max(1, len(train_loader))
    sched = make_scheduler(opt, tr["epochs"] * steps_per_epoch, int(tr["warmup_epochs"] * steps_per_epoch))

    # ---- loop -------------------------------------------------------------- #
    history: list[dict] = []
    best = {"f1": -1.0, "loss": float("inf"), "epoch": -1}
    bad_epochs = 0
    t_start = time.time()
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt, sched, device,
                                          tr["grad_clip"], tr["log_every"], epoch,
                                          cfg["loss"].get("scale_consistency", {}))
        vm, _ = evaluate_loader(model, val_loader, device, mapping.names, val_criterion)
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": vm["loss"],
               "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1_present"],
               "val_macro_f1_all": vm["macro_f1_all_classes"], "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - t0}
        history.append(row)
        log.info("epoch %02d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f macroF1 %.4f | %.0fs",
                 epoch, tr["epochs"], tr_loss, tr_acc, vm["loss"], vm["accuracy"], vm["macro_f1_present"], row["seconds"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, cfg, mapping, epoch, vm, prep_sig)
        improved = (vm["macro_f1_present"] > best["f1"] + 1e-6) or (
            abs(vm["macro_f1_present"] - best["f1"]) <= 1e-6 and vm["loss"] < best["loss"])
        if improved:
            best = {"f1": vm["macro_f1_present"], "loss": vm["loss"], "epoch": epoch}
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, cfg, mapping, epoch, vm, prep_sig)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if tr["patience"] is not None and bad_epochs >= tr["patience"]:
                log.info("Early stopping: no val macro-F1 improvement for %d epochs.", tr["patience"])
                break

    hist = pd.DataFrame(history)
    plot_curves(hist, run_dir / "training_curves.png")

    # ---- final source-val evaluation with the selected checkpoint ----------- #
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    vm, _ = evaluate_loader(model, val_loader, device, mapping.names, val_criterion)
    save_metrics(vm, run_dir / "eval_val_best", mapping.names)
    log.info("BEST (epoch %d) SOURCE VAL: %s", best["epoch"], format_summary(vm))
    save_json({
        "run": exp["name"], "device": str(device), "best_epoch": best["epoch"],
        "epochs_run": len(history), "train_seconds": time.time() - t_start,
        "parameters": count_parameters(model), "n_train": len(train_ds), "n_val": len(val_ds),
        "preprocessing": prep_sig, "input_shape_per_event": list(train_ds[0][0].shape),
        "val_metrics": {k: v for k, v in vm.items() if k not in ("per_class", "confusion_matrix")},
    }, run_dir / "summary.json")
    log.info("Done. Artifacts in %s", run_dir)
    log.info("NEXT (when the model is frozen): python -m src.models.evaluate --run-dir %s --split target --final",
             run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(project_root() / "configs" / "classifier.yaml"))
    ap.add_argument("--ifo", choices=["H1", "L1"])
    ap.add_argument("--model", choices=["multiview", "baseline_merged", "baseline_single", "single_view"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--name", help="Run name (default: {ifo}_{model}_seed{seed})")
    ap.add_argument("--debug", action="store_true", help="Apply the config's `debug:` overrides (tiny and fast).")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing run directory.")
    ap.add_argument("--metadata-dir")
    ap.add_argument("--image-dir")
    ap.add_argument("--processed-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="Override any config value, e.g. --set train.lr=0.0005 (repeatable).")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = build_cfg(args)
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
