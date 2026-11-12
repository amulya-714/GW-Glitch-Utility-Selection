"""Inspect raw Gravity Spy spectrograms: download a few events, show all four views side by side.

    python -m src.data.inspect_images --ifo H1 --run O2 --events-per-class 2 --max-events 24

For each sampled event the contact sheet shows the four views (0.5 s, 1 s, 2 s, 4 s) at their
ORIGINAL resolution (no resizing, no grayscale, no normalisation), labelled with event ID,
ml_label and ml_confidence. It also writes image_info.csv with every PNG's width, height, mode and
byte size, which is the evidence needed to confirm the preprocessing assumptions
(data.image_size, data.channels) against the real files. Preprocessing is not touched here.

Protocol: source runs are inspected freely. A target run (O3a) is refused unless
--allow-target-inspection is passed; that choice is recorded in inspection_log.json. Inspect
the target only to check file format, never to tune anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import ImageStore
from src.data.splits import cap_per_class, clean_metadata, load_run_metadata
from src.utils import load_config, project_root, resolve_path, save_json, setup_logging

log = logging.getLogger(__name__)


def sample_events(df: pd.DataFrame, events_per_class: int, max_events: int, seed: int,
                  labels: Sequence[str] | None = None, min_confidence: float | None = None) -> pd.DataFrame:
    """Seeded, class-stratified sample (at most `events_per_class` per label, `max_events` overall)."""
    if labels:
        missing = [lbl for lbl in labels if lbl not in set(df["ml_label"])]
        if missing:
            log.warning("labels not present in this run, skipped: %s", missing)
        df = df[df["ml_label"].isin(labels)]
    if min_confidence is not None:
        df = df[df["ml_confidence"] >= min_confidence]
    out = cap_per_class(df.reset_index(drop=True), events_per_class, seed)
    if len(out) > max_events:
        # keep class diversity: take round-robin across labels rather than the first rows
        rng = np.random.default_rng(seed)
        groups = {lbl: list(rng.permutation(g.index.to_numpy())) for lbl, g in out.groupby("ml_label")}
        picked: list[int] = []
        while len(picked) < max_events and any(groups.values()):
            for lbl in sorted(groups):
                if groups[lbl] and len(picked) < max_events:
                    picked.append(groups[lbl].pop())
        out = out.loc[sorted(picked)]
    return out.reset_index(drop=True)


def image_info(store: ImageStore, df: pd.DataFrame, n_views: int) -> pd.DataFrame:
    rows = []
    for gid in df["gravityspy_id"]:
        for k in range(n_views):
            p = store.path_for(gid, k)
            if not p.exists():
                continue
            with Image.open(p) as im:
                rows.append({"gravityspy_id": gid, "view": k + 1, "width": im.width, "height": im.height,
                             "mode": im.mode, "format": im.format, "bytes": p.stat().st_size})
    return pd.DataFrame(rows, columns=["gravityspy_id", "view", "width", "height", "mode", "format", "bytes"])


def make_contact_sheets(events: pd.DataFrame, store: ImageStore, durations: Sequence[float], out_dir: Path,
                        rows_per_sheet: int = 6, title: str = "") -> list[Path]:
    """One PNG per `rows_per_sheet` events; each row = the four views of one event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(durations)
    paths: list[Path] = []
    for s, start in enumerate(range(0, len(events), rows_per_sheet), start=1):
        chunk = events.iloc[start:start + rows_per_sheet]
        fig, axes = plt.subplots(len(chunk), n_views, figsize=(3.0 * n_views, 3.0 * len(chunk) + 0.6), squeeze=False)
        for r, (_, ev) in enumerate(chunk.iterrows()):
            for v in range(n_views):
                ax = axes[r][v]
                ax.set_xticks([])
                ax.set_yticks([])
                with Image.open(store.path_for(ev["gravityspy_id"], v)) as im:
                    ax.imshow(np.asarray(im.convert("RGB")))
                if r == 0:
                    ax.set_title(f"{durations[v]:g} s", fontsize=10)
            axes[r][0].set_ylabel(f"{ev['gravityspy_id']}\n{ev['ml_label']}\nconf={ev['ml_confidence']:.3f}",
                                  fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        path = out_dir / f"contact_sheet_{s:02d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def inspect(cfg: dict[str, Any], ifo: str, run: str, out_dir: Path, events_per_class: int = 2,
            max_events: int = 24, labels: Sequence[str] | None = None, min_confidence: float | None = None,
            seed: int = 0, allow_target_inspection: bool = False) -> dict[str, Any]:
    d, proto = cfg["data"], cfg["protocol"]
    is_target = run in set(proto["target_runs"])
    if is_target and not allow_target_inspection:
        raise SystemExit(
            f"'{run}' is a protocol target run. Refusing to look at its images without "
            "--allow-target-inspection (use only to check file format, never to tune anything).")
    raw = load_run_metadata(resolve_path(cfg["paths"]["metadata_dir"]), ifo, run, d["view_columns"],
                            target_runs=proto["target_runs"], purpose="final_eval" if is_target else "train")
    clean = clean_metadata(raw, ifo, d["view_columns"], f"{ifo}_{run}")
    events = sample_events(clean, events_per_class, max_events, seed, labels, min_confidence)
    if events.empty:
        raise ValueError("No events selected. Check --labels / --min-confidence.")
    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    failures = store.ensure_downloaded(events, d["view_columns"], desc="inspect-download")
    bad = set(failures["gravityspy_id"])
    if bad:
        log.warning("%d event(s) skipped because a view failed to download: %s", len(bad), sorted(bad))
        events = events[~events["gravityspy_id"].isin(bad)].reset_index(drop=True)
    if events.empty:
        raise RuntimeError("Every sampled event failed to download; see failures above.")

    out_dir.mkdir(parents=True, exist_ok=True)
    info = image_info(store, events, len(d["view_columns"]))
    info.to_csv(out_dir / "image_info.csv", index=False)
    events[["gravityspy_id", "event_time", "ml_label", "ml_confidence"]].to_csv(out_dir / "events.csv", index=False)
    failures.to_csv(out_dir / "download_failures.csv", index=False)
    sheets = make_contact_sheets(events, store, d["view_durations"], out_dir,
                                 title=f"{ifo} {run}: raw views (original resolution)")
    summary = {
        "ifo": ifo, "run": run, "is_target_run": is_target, "n_events": int(len(events)),
        "labels": events["ml_label"].value_counts().to_dict(),
        "image_sizes_wh": sorted({f"{w}x{h}" for w, h in zip(info["width"], info["height"])}),
        "image_modes": sorted(info["mode"].unique().tolist()),
        "bytes_min_max": [int(info["bytes"].min()), int(info["bytes"].max())],
        "configured_image_size_hw": d["image_size"], "configured_channels": d["channels"],
        "contact_sheets": [str(p) for p in sheets],
    }
    save_json(summary, out_dir / "inspection_summary.json")
    if is_target:
        save_json({"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "run": run,
                   "allow_target_inspection": True, "n_events": int(len(events))}, out_dir / "inspection_log.json")
    log.info("Sizes (WxH): %s | modes: %s | sheets: %s", summary["image_sizes_wh"], summary["image_modes"],
             [p.name for p in sheets])
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(project_root() / "configs" / "classifier.yaml"))
    ap.add_argument("--ifo", choices=["H1", "L1"])
    ap.add_argument("--run", help="Default: protocol.source_run")
    ap.add_argument("--events-per-class", type=int, default=2)
    ap.add_argument("--max-events", type=int, default=24)
    ap.add_argument("--labels", nargs="+", help="Restrict to these ml_labels")
    ap.add_argument("--min-confidence", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", help="Default: docs/image_inspection/<ifo>_<run>")
    ap.add_argument("--allow-target-inspection", action="store_true")
    ap.add_argument("--metadata-dir")
    ap.add_argument("--image-dir")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if args.metadata_dir:
        cfg["paths"]["metadata_dir"] = args.metadata_dir
    if args.image_dir:
        cfg["paths"]["image_dir"] = args.image_dir
    ifo = args.ifo or cfg["data"]["ifo"]
    run = args.run or cfg["protocol"]["source_run"]
    out = Path(args.out_dir) if args.out_dir else project_root() / "docs" / "image_inspection" / f"{ifo}_{run}"
    inspect(cfg, ifo, run, out, args.events_per_class, args.max_events, args.labels, args.min_confidence,
            args.seed, args.allow_target_inspection)


if __name__ == "__main__":
    main()
