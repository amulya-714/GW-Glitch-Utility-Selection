"""Synthetic stand-in for the Gravity Spy metadata + PNGs, for offline tests and smoke runs.

NOT scientific data. It only mimics the file layout (same CSV columns, four PNG views per
event, `file://` URLs) so the full pipeline can be exercised without network access.
Each class gets a distinct, learnable pattern; the four views are time-axis zooms.

    python -m src.data.fake_data --out /tmp/fake_gw --events 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FAKE_CLASSES = ["Blip", "Whistle", "Koi_Fish", "Scattered_Light", "Low_Frequency_Burst", "1080Lines"]
UNSEEN = ["Blip_Low_Frequency", "Fast_Scattering"]
LEGACY = ["None_of_the_Above"]   # present in the fake O3a like the real data (small count)
VIEW_SECONDS = [0.5, 1.0, 2.0, 4.0]


def _pattern(cls: str, view: int, rng: np.random.Generator, h: int = 170, w: int = 140) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = rng.normal(0.25, 0.05, (h, w))
    zoom = 1.0 / (1.0 + view)  # longer views show the glitch narrower in time
    cx, cy = w / 2 + rng.normal(0, 3), h / 2 + rng.normal(0, 6)
    sx = max(2.0, 18 * zoom)
    if cls in ("Blip", "Blip_Low_Frequency"):
        img += 0.8 * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - (cy if cls == "Blip" else h * 0.75)) / 10) ** 2))
    elif cls == "Whistle":
        img += 0.8 * np.exp(-(((yy - (h - 1 - 0.9 * (xx - cx) - h / 2 + cy)) / 4) ** 2)) * (np.abs(xx - cx) < 40)
    elif cls == "Koi_Fish":
        img += 0.8 * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / 18) ** 2)) + \
               0.5 * np.exp(-(((xx - cx) / (2 * sx)) ** 2 + ((yy - cy + 25) / 6) ** 2))
    elif cls in ("Scattered_Light", "Fast_Scattering"):
        img += 0.7 * np.exp(-((yy - h * 0.85 + 0.01 * (xx - cx) ** 2 * (1 if cls == "Scattered_Light" else 3)) / 6) ** 2)
    elif cls == "Low_Frequency_Burst":
        img += 0.8 * np.exp(-(((xx - cx) / (3 * sx)) ** 2 + ((yy - h * 0.9) / 8) ** 2))
    elif cls == "1080Lines":
        img += 0.7 * np.exp(-((yy - h * 0.4) / 2.5) ** 2)
    return np.clip(img, 0, 1)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 * img - 0.4, 0, 1)
    g = np.clip(1.5 * img, 0, 1) * 0.9
    b = np.clip(0.9 - 1.2 * img, 0, 1) * 0.8 + 0.1
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


# --- SYNTHETIC 800x600 RGBA layout used only to test the crop tools -------------------------------
# This is NOT the real Gravity Spy figure layout. It mimics the *kinds* of furniture the real PNGs
# were reported to have (title, axis frame, tick labels, colour bar, margins) around a data region
# whose position is known, so crop detection can be checked against ground truth.
LAYOUT_SIZE_WH = (800, 600)
LAYOUT_PLOT_BOX = (115, 70, 685, 520)    # left, top, right, bottom (exclusive) of the data region


def render_layout_image(rng: np.random.Generator, cls: str, view: int, transparent_corner: bool = False) -> np.ndarray:
    """uint8 (600, 800, 4). White margins, varying title, static frame/ticks/colour bar, data inside the box."""
    w, h = LAYOUT_SIZE_WH
    left, top, right, bottom = LAYOUT_PLOT_BOX
    img = np.full((h, w, 4), 255, np.uint8)
    img[top:bottom, left:right, :3] = _to_rgb(_pattern(cls, view, rng, h=bottom - top, w=right - left))
    img[top - 2:top, left - 2:right + 2, :3] = 0          # frame (static)
    img[bottom:bottom + 2, left - 2:right + 2, :3] = 0
    img[top:bottom, left - 2:left, :3] = 0
    img[top:bottom, right:right + 2, :3] = 0
    for y in range(top, bottom, 60):                       # y tick labels (static)
        img[y:y + 8, left - 40:left - 8, :3] = 40
    for x in range(left, right, 90):                       # x tick labels (static)
        img[bottom + 8:bottom + 18, x:x + 30, :3] = 40
    grad = np.linspace(1.0, 0.0, bottom - top)[:, None]    # colour bar (static)
    img[top:bottom, 715:745, :3] = _to_rgb(np.repeat(grad, 30, axis=1))
    tw = int(rng.integers(120, 380))                       # title: varies per image, like event-specific text
    img[18:38, 150:150 + tw, :3] = 30
    if transparent_corner:
        img[0:6, 0:6, 3] = 0
    return img


def make_fake_run(out_dir: str | Path, ifo: str, run: str, n_events: int, t0: float,
                  classes: list[str], seed: int, conf_low_frac: float = 0.2, style: str = "small",
                  transparent_corner: bool = False) -> Path:
    """Write <out>/metadata/<ifo>_<run>.csv and PNGs under <out>/png/. Returns the CSV path.

    style: "small" (140x170 RGB) or "layout800" (synthetic 800x600 RGBA with plot furniture).
    """
    out = Path(out_dir)
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    t = t0
    for i in range(n_events):
        t += rng.exponential(300.0)  # ~5 min between events
        cls = classes[rng.integers(len(classes))]
        gid = f"{ifo[0]}{run[-2:]}{i:07d}"[:10].ljust(10, "x")
        urls = []
        for v in range(4):
            p = out / "png" / f"{gid}_{v}.png"
            if style == "layout800":
                Image.fromarray(render_layout_image(rng, cls, v, transparent_corner), "RGBA").save(p)
            else:
                Image.fromarray(_to_rgb(_pattern(cls, v, rng))).save(p)
            urls.append(f"file://{p}")
        conf = float(rng.uniform(0.35, 0.69) if rng.random() < conf_low_frac else rng.uniform(0.7, 1.0))
        rows.append({"event_time": t, "ifo": ifo, "gravityspy_id": gid, "ml_label": cls,
                     "ml_confidence": conf, "url1": urls[0], "url2": urls[1], "url3": urls[2], "url4": urls[3]})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # real CSVs are not time sorted
    path = out / "metadata" / f"{ifo}_{run}.csv"
    df.to_csv(path, index=False)
    return path


def make_fake_dataset(out_dir: str | Path, n_events: int = 600, ifos: tuple[str, ...] = ("H1",), seed: int = 0) -> Path:
    """O2 (source classes) and O3a (source classes + two new labels + None_of_the_Above + one exact duplicate row)."""
    out = Path(out_dir)
    for k, ifo in enumerate(ifos):
        make_fake_run(out, ifo, "O2", n_events, 1.164e9, FAKE_CLASSES, seed + k)
        p = make_fake_run(out, ifo, "O3a", n_events // 2, 1.238e9, FAKE_CLASSES + UNSEEN + LEGACY, seed + 100 + k)
        df = pd.read_csv(p)
        pd.concat([df, df.iloc[[0]]]).to_csv(p, index=False)  # exact duplicate, like L1_O3a
    return out / "metadata"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--ifos", nargs="+", default=["H1"])
    a = ap.parse_args()
    print("metadata dir:", make_fake_dataset(a.out, a.events, tuple(a.ifos)))
