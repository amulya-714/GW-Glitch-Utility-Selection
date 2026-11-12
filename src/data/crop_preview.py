"""Find, check and compare crops of the spectrogram plotting region. SOURCE RUNS ONLY.

    # 1. Propose a crop box from pixel statistics of real downloaded events (a SUGGESTION, not a fact)
    python -m src.data.crop_preview suggest --ifo H1 --run O2

    # 2. Look at it: original (with box), cropped RGB, cropped gray, and aspect-preserving resized variants
    python -m src.data.crop_preview preview --box L T R B --max-events 3
    python -m src.data.crop_preview preview --from-config      # the shipped baseline, configured size first

    # 3. Print what a config will do (reads no images)
    python -m src.data.crop_preview describe

How `suggest` works: titles, tick labels, axis frames and the colour bar are (nearly) identical from one
image to the next, while the plotted data is different in every image. So the per-pixel standard
deviation across many events is high inside the plotting region and low in the furniture. The box is the
longest dense block of high-variance rows and columns. This assumption has to be CONFIRMED by eye on the
overlays it writes. If the figure contains event-specific text or the plot is unusually smooth, the
box can be off, so never paste it into the config without running `preview` first.

The tools only read PNGs from the image cache (downloading missing ones for the sampled events) and
never modify them. Target runs are refused outright: there is no override flag.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.data.dataset import ImageStore
from src.data.inspect_images import sample_events
from src.data.preprocessing import (
    ALPHA_MODES, SUPPORTED_RESOLUTIONS, Normalizer, PreprocessConfig, _RESAMPLE, _to_rgb, aspect_distortion,
    aspect_preserving_size, box_size, load_view, preprocessing_signature, validate_crop_box,
)
from src.data.splits import clean_metadata, load_run_metadata
from src.utils import load_config, project_root, resolve_path, save_json, setup_logging

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Statistics and box suggestion
# --------------------------------------------------------------------------- #
class PixelStats:
    """Running per-pixel mean/std of luminance (RGBA composited over white) for same-sized images."""

    def __init__(self) -> None:
        self.n = 0
        self.size_wh: tuple[int, int] | None = None
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None

    def add(self, lum: np.ndarray) -> None:
        if self._sum is None:
            self._sum = np.zeros_like(lum, dtype=np.float64)
            self._sumsq = np.zeros_like(lum, dtype=np.float64)
            self.size_wh = (lum.shape[1], lum.shape[0])
        self._sum += lum
        self._sumsq += lum.astype(np.float64) ** 2
        self.n += 1

    @property
    def mean(self) -> np.ndarray:
        return self._sum / self.n

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self._sumsq / self.n - self.mean ** 2, 0.0))


def longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """(start, end_exclusive) of the longest run of True values, or None."""
    best, start = None, None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def suggest_box(std_map: np.ndarray, rel_thresh: float = 0.3, density: float = 0.5,
                inset: int = 0) -> tuple[int, int, int, int]:
    """Propose [left, top, right, bottom] (exclusive) of the high-variance plotting region.

    A pixel is 'active' if its across-image std exceeds rel_thresh x the median std of all
    pixels that vary at all. The box is the longest run of rows, and of columns, in which at least
    `density` x the best row/column's fraction of pixels is active, refined alternately.
    """
    varying = std_map[std_map > 1e-3]
    if varying.size == 0:
        raise ValueError("No pixel varies across the sampled images; cannot locate the plotting region.")
    active = std_map > rel_thresh * float(np.median(varying))

    rows = active.mean(axis=1)
    cols = active.mean(axis=0)
    r = longest_run(rows > density * rows.max())
    c = longest_run(cols > density * cols.max())
    if r is None or c is None:
        raise ValueError("Could not find a dense high-variance region; try a smaller --rel-thresh/--density.")
    for _ in range(2):  # refine each axis using only the other axis' current extent
        cols = active[r[0]:r[1], :].mean(axis=0)
        c = longest_run(cols > density * cols.max()) or c
        rows = active[:, c[0]:c[1]].mean(axis=1)
        r = longest_run(rows > density * rows.max()) or r
    h, w = std_map.shape
    return validate_crop_box((c[0] + inset, r[0] + inset, c[1] - inset, r[1] - inset), (w, h))


def _luminance(path: Path, background: str = "white") -> tuple[np.ndarray, tuple[int, int], float, float]:
    """(luminance float32 HxW, (w, h), min alpha, fraction of non-opaque pixels) for one PNG."""
    with Image.open(path) as im:
        im.load()
        size = im.size
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            alpha = np.asarray(im.convert("RGBA"))[..., 3]
            amin, frac = float(alpha.min()), float((alpha < 255).mean())
        else:
            amin, frac = 255.0, 0.0
        lum = np.asarray(_to_rgb(im, background).convert("L"), dtype=np.float32)
    return lum, size, amin, frac


def draw_box(ax, box: Sequence[int], color: str = "red") -> None:
    import matplotlib.patches as patches
    l, t, r, b = box
    ax.add_patch(patches.Rectangle((l - 0.5, t - 0.5), r - l, b - t, fill=False, edgecolor=color, linewidth=1.6))


# --------------------------------------------------------------------------- #
# suggest
# --------------------------------------------------------------------------- #
def _load_events(cfg: dict[str, Any], ifo: str, run: str, events_per_class: int, max_events: int, seed: int):
    """Sample source-run events and make sure all their PNGs are cached. Target runs raise TargetLeakageError."""
    d, proto = cfg["data"], cfg["protocol"]
    raw = load_run_metadata(resolve_path(cfg["paths"]["metadata_dir"]), ifo, run, d["view_columns"],
                            target_runs=proto["target_runs"], purpose="train")
    clean = clean_metadata(raw, ifo, d["view_columns"], f"{ifo}_{run}")
    events = sample_events(clean, events_per_class, max_events, seed)
    store = ImageStore.from_cfg(cfg, resolve_path(cfg["paths"]["image_dir"]))
    failures = store.ensure_downloaded(events, d["view_columns"], desc="crop-download")
    bad = set(failures["gravityspy_id"])
    if bad:
        log.warning("%d event(s) skipped (download failed): %s", len(bad), sorted(bad))
        events = events[~events["gravityspy_id"].isin(bad)].reset_index(drop=True)
    if events.empty:
        raise RuntimeError("No events available; check metadata and network.")
    return events, store


def suggest(cfg: dict[str, Any], ifo: str, run: str, out_dir: Path, events_per_class: int = 2,
            max_events: int = 24, seed: int = 0, rel_thresh: float = 0.3, density: float = 0.5,
            inset: int = 0) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg["data"]
    n_views = len(d["view_columns"])
    ref_wh = tuple(int(v) for v in d.get("crop", {}).get("reference_size", (800, 600)))
    events, store = _load_events(cfg, ifo, run, events_per_class, max_events, seed)

    per_view = [PixelStats() for _ in range(n_views)]
    pooled = PixelStats()
    sizes: dict[tuple[int, int], int] = {}
    alpha_min, alpha_frac = 255.0, 0.0
    n_images = 0
    for gid in events["gravityspy_id"]:
        for v in range(n_views):
            lum, size, amin, frac = _luminance(store.path_for(gid, v), d.get("alpha_background", "white"))
            sizes[size] = sizes.get(size, 0) + 1
            if len(sizes) > 1:
                raise ValueError(f"Images have different sizes {sizes}; cannot compute per-pixel statistics. "
                                 "Inspect them with `python -m src.data.inspect_images` first.")
            per_view[v].add(lum)
            pooled.add(lum)
            alpha_min, alpha_frac, n_images = min(alpha_min, amin), max(alpha_frac, frac), n_images + 1

    box = suggest_box(pooled.std, rel_thresh, density, inset)
    view_boxes = []
    for v in range(n_views):
        try:
            view_boxes.append(suggest_box(per_view[v].std, rel_thresh, density, inset))
        except ValueError:
            view_boxes.append(None)
    disagreement = max((max(abs(a - b) for a, b in zip(box, vb)) for vb in view_boxes if vb), default=0)

    warnings: list[str] = []
    if pooled.size_wh != ref_wh:
        warnings.append(f"Image size {pooled.size_wh} differs from data.crop.reference_size {list(ref_wh)}; "
                        "update reference_size or the box will be rejected at training time.")
    if disagreement > 5:
        warnings.append(f"Per-view boxes disagree with the pooled box by up to {disagreement}px; check each view.")
    w_img, h_img = pooled.size_wh
    frac_area = (box[2] - box[0]) * (box[3] - box[1]) / (w_img * h_img)
    if frac_area < 0.2 or frac_area > 0.95:
        warnings.append(f"Box covers {frac_area:.0%} of the image, which is unusual; inspect the overlays closely.")
    if alpha_min < 255:
        warnings.append(f"{alpha_frac:.2%} of pixels are not fully opaque (min alpha {alpha_min:.0f}); "
                        "data.alpha_background decides what colour they become.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(pooled.mean, cmap="gray")
    axes[0].set_title(f"mean image ({pooled.n} images)")
    im1 = axes[1].imshow(pooled.std, cmap="magma")
    axes[1].set_title("per-pixel std across images")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for a in axes:
        draw_box(a, box)
        a.set_xlabel("x (px)")
        a.set_ylabel("y (px)")
    fig.suptitle(f"{ifo} {run}: SUGGESTED crop box {list(box)} (left, top, right, bottom)")
    fig.tight_layout()
    fig.savefig(out_dir / "suggestion_maps.png", dpi=110)
    plt.close(fig)

    show = events.iloc[:2]
    fig, axes = plt.subplots(len(show), n_views, figsize=(3.6 * n_views, 2.8 * len(show)), squeeze=False)
    for r, (_, ev) in enumerate(show.iterrows()):
        for v in range(n_views):
            with Image.open(store.path_for(ev["gravityspy_id"], v)) as im:
                axes[r][v].imshow(np.asarray(_to_rgb(im, "white")))
            draw_box(axes[r][v], box)
            axes[r][v].set_xticks([])
            axes[r][v].set_yticks([])
            if r == 0:
                axes[r][v].set_title(f"{d['view_durations'][v]:g} s")
        axes[r][0].set_ylabel(f"{ev['gravityspy_id']}\n{ev['ml_label']}", fontsize=8)
    fig.suptitle("Suggested box on real events: does it contain only the plotting region?")
    fig.tight_layout()
    fig.savefig(out_dir / "suggestion_overlay_events.png", dpi=100)
    plt.close(fig)

    snippet = (f"crop:\n  enabled: true\n  reference_size: [{w_img}, {h_img}]\n  box: {list(box)}")
    summary = {
        "ifo": ifo, "run": run, "n_events": int(len(events)), "n_images": n_images,
        "image_size_wh": list(pooled.size_wh), "suggested_box_ltrb": list(box),
        "per_view_boxes_ltrb": [list(b) if b else None for b in view_boxes],
        "max_per_view_disagreement_px": int(disagreement), "box_area_fraction": round(frac_area, 4),
        "min_alpha": alpha_min, "max_fraction_non_opaque": alpha_frac,
        "params": {"rel_thresh": rel_thresh, "density": density, "inset": inset, "seed": seed},
        "warnings": warnings, "config_snippet": snippet,
        "status": "SUGGESTION ONLY: verify with `crop_preview preview` before use.",
    }
    save_json(summary, out_dir / "crop_suggestion.json")
    log.info("Suggested box (left, top, right, bottom): %s | image %s | area %.0f%%",
             list(box), list(pooled.size_wh), 100 * frac_area)
    for w in warnings:
        log.warning(w)
    log.info("Paste into configs/classifier.yaml ONLY after checking the overlays in %s:\n%s", out_dir, snippet)
    return summary


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def preview_sizes(region_wh: tuple[int, int], heights: Sequence[int], aspect_preserving: bool = True,
                  configured_size: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Output sizes (H, W) shown in the preview.

    aspect_preserving (default): each height h gives (h, round(h * region_w / region_h)), so the crop is never
    stretched. Otherwise square (h, h), which stretches a non-square crop. `configured_size` (the size in the
    config) is placed first when given, so the sheet shows exactly what training will use.
    """
    sizes = [aspect_preserving_size(region_wh, height=h) if aspect_preserving else (int(h), int(h)) for h in heights]
    if configured_size is not None:
        cs = (int(configured_size[0]), int(configured_size[1]))
        sizes = [cs] + [z for z in sizes if z != cs]
    return sizes


def render_variants(path: Path, box: tuple[int, int, int, int] | None, sizes: Sequence[tuple[int, int]],
                    alpha_background: str = "white", resize_mode: str = "bilinear") -> dict[str, np.ndarray]:
    """Display-ready arrays produced by the SAME load_view used in training.

    'original' is the composited full image, 'crop_rgb'/'crop_gray' are the crop at native resolution
    (identity resize), then 'rgb_<H>x<W>' for each (H, W) in `sizes`.
    """
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        original = np.asarray(_to_rgb(im, alpha_background))
    ref = (w, h)
    use_box = box if box is not None else (0, 0, w, h)
    ch, cw = use_box[3] - use_box[1], use_box[2] - use_box[0]

    def cfg(size, channels):
        return PreprocessConfig(size, channels, resize_mode, alpha_background, use_box, ref)

    out = {"original": original,
           "crop_rgb": np.transpose(load_view(path, cfg((ch, cw), 3)), (1, 2, 0)),
           "crop_gray": load_view(path, cfg((ch, cw), 1))[0]}
    for (sh, sw) in sizes:
        out[f"rgb_{sh}x{sw}"] = np.transpose(load_view(path, cfg((sh, sw), 3)), (1, 2, 0))
    return out


def make_preview_sheets(events, store: ImageStore, box: tuple[int, int, int, int] | None, durations: Sequence[float],
                        sizes: Sequence[tuple[int, int]], out_dir: Path, alpha_background: str = "white",
                        resize_mode: str = "bilinear", configured_size: tuple[int, int] | None = None) -> list[Path]:
    """One sheet per event: rows = the four views, columns = original(+box), crop RGB, crop gray, RGB at each size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n_cols = 3 + len(sizes)
    names = ["original", "crop_rgb", "crop_gray"] + [f"rgb_{h}x{w}" for h, w in sizes]
    for _, ev in events.iterrows():
        fig, axes = plt.subplots(len(durations), n_cols, figsize=(2.6 * n_cols, 2.4 * len(durations) + 0.8),
                                 squeeze=False)
        for v, dur in enumerate(durations):
            var = render_variants(store.path_for(ev["gravityspy_id"], v), box, sizes, alpha_background, resize_mode)
            for c, name in enumerate(names):
                ax = axes[v][c]
                arr = var[name]
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0 if arr.ndim == 2 else None,
                          vmax=255 if arr.ndim == 2 else None, interpolation="nearest")
                if name == "original" and box is not None:
                    draw_box(ax, box)
                ax.set_xticks([])
                ax.set_yticks([])
                if v == 0:
                    shape = f"{arr.shape[1]}x{arr.shape[0]}"           # width x height
                    fixed = {"original": "original + box", "crop_rgb": "crop RGB", "crop_gray": "crop gray"}
                    if name in fixed:
                        label = fixed[name]
                    else:
                        sh, sw = arr.shape[0], arr.shape[1]
                        region = box_size(box) if box is not None else (var["original"].shape[1], var["original"].shape[0])
                        label = f"RGB {sh}x{sw} (HxW)\nstretch {aspect_distortion(region, (sh, sw)):.1%}"
                        if configured_size is not None and (sh, sw) == tuple(configured_size):
                            label = "CONFIG " + label
                    ax.set_title(f"{label}\n{shape}" if name in fixed else label, fontsize=8)
            axes[v][0].set_ylabel(f"{dur:g} s", fontsize=9)
        fig.suptitle(f"{ev['gravityspy_id']} | {ev['ml_label']} | conf={ev['ml_confidence']:.3f}"
                     + ("" if box is not None else "  (NO CROP: full image)"), fontsize=10)
        fig.tight_layout()
        p = out_dir / f"preview_{ev['gravityspy_id']}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def preview(cfg: dict[str, Any], ifo: str, run: str, out_dir: Path, box: Sequence[int] | None,
            events_per_class: int = 1, max_events: int = 3, seed: int = 0,
            resolutions: Sequence[int] = SUPPORTED_RESOLUTIONS, alpha_background: str | None = None,
            resize_mode: str | None = None, aspect_preserving: bool = True,
            configured_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """`resolutions` are output HEIGHTS; widths follow the crop's aspect ratio unless aspect_preserving=False."""
    d = cfg["data"]
    alpha_background = alpha_background or d.get("alpha_background", "white")
    resize_mode = resize_mode or d.get("resize_mode", "bilinear")
    events, store = _load_events(cfg, ifo, run, events_per_class, max_events, seed)
    first = store.path_for(events.iloc[0]["gravityspy_id"], 0)
    with Image.open(first) as im:
        size_wh = im.size
    use_box = None if box is None else validate_crop_box(box, size_wh)
    region = box_size(use_box) if use_box is not None else size_wh
    sizes = preview_sizes(region, resolutions, aspect_preserving, configured_size)
    sheets = make_preview_sheets(events, store, use_box, d["view_durations"], sizes, out_dir,
                                 alpha_background, resize_mode, configured_size)
    summary = {"ifo": ifo, "run": run, "image_size_wh": list(size_wh),
               "box_ltrb": None if use_box is None else list(use_box), "region_wh": list(region),
               "sizes_hw": [list(z) for z in sizes], "aspect_preserving": aspect_preserving,
               "stretch_by_size": {f"{h}x{w}": round(aspect_distortion(region, (h, w)), 5) for h, w in sizes},
               "alpha_background": alpha_background, "resize_mode": resize_mode,
               "events": events["gravityspy_id"].tolist(), "sheets": [str(p) for p in sheets]}
    save_json(summary, out_dir / "preview_summary.json")
    log.info("Wrote %d preview sheet(s) to %s | sizes (HxW): %s", len(sheets), out_dir, [tuple(z) for z in sizes])
    return summary


def describe_preprocessing(cfg: dict[str, Any]) -> dict[str, Any]:
    """What the training pipeline will do with a config, derived without reading any image or metadata."""
    d = cfg["data"]
    pre = PreprocessConfig.from_cfg(d)
    norm = Normalizer.from_cfg(d)
    region = box_size(pre.crop_box) if pre.crop_box else tuple(pre.crop_reference_size)
    h, w = pre.image_size
    n_views = len(d["view_columns"])
    return {
        "crop_enabled": pre.crop_box is not None, "crop_box_ltrb": list(pre.crop_box) if pre.crop_box else None,
        "reference_size_wh": list(pre.crop_reference_size), "crop_size_wh": list(region),
        "crop_aspect_w_over_h": round(region[0] / region[1], 5),
        "output_size_hw": [h, w], "output_aspect_w_over_h": round(w / h, 5),
        "x_scale": round(w / region[0], 5), "y_scale": round(h / region[1], 5),
        "aspect_distortion": round(aspect_distortion(region, (h, w)), 5),
        "channels": pre.channels, "resize_mode": pre.resize_mode, "alpha_background": pre.alpha_background,
        "normalization": {"mode": norm.mode, "mean": norm.mean, "std": norm.std},
        "views": list(d["view_columns"]), "view_durations_s": list(d["view_durations"]),
        "model_input_shape_per_event": [n_views, pre.channels, h, w],
        "stored_bytes_per_event": n_views * pre.channels * h * w,
        "signature_hash": preprocessing_signature(d)["hash"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", default=str(project_root() / "configs" / "classifier.yaml"))
        p.add_argument("--ifo", choices=["H1", "L1"])
        p.add_argument("--run", help="Source run only (default: protocol.source_run). Target runs are refused.")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--metadata-dir")
        p.add_argument("--image-dir")
        p.add_argument("--out-dir")

    s = sub.add_parser("suggest", help="propose a crop box from pixel statistics")
    common(s)
    s.add_argument("--events-per-class", type=int, default=2)
    s.add_argument("--max-events", type=int, default=24)
    s.add_argument("--rel-thresh", type=float, default=0.3)
    s.add_argument("--density", type=float, default=0.5)
    s.add_argument("--inset", type=int, default=0, help="shrink the suggested box by N px on every side")

    p = sub.add_parser("preview", help="contact sheets: original, crop RGB, crop gray, resized variants")
    common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--box", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    g.add_argument("--from-config", action="store_true", help="use data.crop.box from the config")
    g.add_argument("--no-crop", action="store_true", help="show the full image (baseline for comparison)")
    p.add_argument("--events-per-class", type=int, default=1)
    p.add_argument("--max-events", type=int, default=3)
    p.add_argument("--resolutions", nargs="+", type=int, default=list(SUPPORTED_RESOLUTIONS),
                   help="output HEIGHTS; widths follow the crop's aspect ratio (default) so nothing is stretched")
    p.add_argument("--square", action="store_true",
                   help="use square outputs (R x R) instead; this STRETCHES a non-square crop")
    p.add_argument("--alpha-background", choices=ALPHA_MODES)
    p.add_argument("--resize-mode", choices=sorted(_RESAMPLE))

    ds = sub.add_parser("describe", help="print the effective preprocessing of a config (reads no images)")
    ds.add_argument("--config", default=str(project_root() / "configs" / "classifier.yaml"))

    args = ap.parse_args(argv)
    setup_logging()
    if args.cmd == "describe":
        info = describe_preprocessing(load_config(args.config))
        print(json.dumps(info, indent=2))
        return
    cfg = load_config(args.config)
    if args.metadata_dir:
        cfg["paths"]["metadata_dir"] = args.metadata_dir
    if args.image_dir:
        cfg["paths"]["image_dir"] = args.image_dir
    ifo = args.ifo or cfg["data"]["ifo"]
    run = args.run or cfg["protocol"]["source_run"]
    base = Path(args.out_dir) if args.out_dir else project_root() / "docs" / "preprocessing_preview" / f"{ifo}_{run}"
    if args.cmd == "suggest":
        suggest(cfg, ifo, run, base, args.events_per_class, args.max_events, args.seed, args.rel_thresh,
                args.density, args.inset)
        return
    if args.from_config:
        crop = cfg["data"].get("crop") or {}
        if not crop.get("enabled") or crop.get("box") is None:
            raise SystemExit("--from-config needs data.crop.enabled: true and a box in the config.")
        box = crop["box"]
    else:
        box = None if args.no_crop else args.box
    configured = tuple(PreprocessConfig.from_cfg(cfg["data"]).image_size) if args.from_config else None
    preview(cfg, ifo, run, base, box, args.events_per_class, args.max_events, args.seed, args.resolutions,
            args.alpha_background, args.resize_mode, not args.square, configured)


if __name__ == "__main__":
    main()
