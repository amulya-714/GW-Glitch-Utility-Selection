"""Image download/caching, aligned four-view dataset, sampler and loaders.

Pipeline for a split DataFrame (one row per event, columns url1..url4):

  ImageStore.ensure_downloaded  -> raw PNGs in data/raw/images/<id[:2]>/<id>_v<k>.png
                                   (retries, timeouts, atomic writes, never re-downloaded)
  build_split_array             -> one uint8 array (N, V, C, H, W) per split, cached by content key
  GlitchDataset                 -> memory-mapped reads, per-sample normalisation

An event is kept only if ALL four views download and decode; otherwise the whole
event is dropped and logged, so views are always aligned to the same event.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.data.preprocessing import (
    Augment, CorruptImageError, ImageGeometryError, Normalizer, PreprocessConfig, load_event_views,
)
from src.utils import save_json, seed_worker, stable_hash

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
class ImageStore:
    """On-disk cache of raw spectrogram PNGs."""

    def __init__(self, image_dir: str | Path, workers: int = 8, timeout_s: float = 30.0,
                 retries: int = 4, backoff_s: float = 1.5, min_bytes: int = 200):
        self.image_dir = Path(image_dir)
        self.workers = max(1, int(workers))
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self.min_bytes = int(min_bytes)
        self._local = threading.local()

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], image_dir: str | Path) -> "ImageStore":
        dl = cfg["data"]["download"]
        return cls(image_dir, dl["workers"], dl["timeout_s"], dl["retries"], dl["backoff_s"], dl["min_bytes"])

    # -- paths -------------------------------------------------------------- #
    def path_for(self, gravityspy_id: str, view_idx: int) -> Path:
        gid = str(gravityspy_id)
        return self.image_dir / gid[:2] / f"{gid}_v{view_idx + 1}.png"

    def event_paths(self, gravityspy_id: str, n_views: int) -> list[Path]:
        return [self.path_for(gravityspy_id, k) for k in range(n_views)]

    # -- single file -------------------------------------------------------- #
    def _session(self):
        import requests
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def _is_cached(self, dest: Path) -> bool:
        return dest.exists() and dest.stat().st_size >= self.min_bytes

    def fetch(self, url: str, dest: Path) -> tuple[bool, str]:
        """Download `url` to `dest` unless cached. Returns (ok, reason)."""
        if self._is_cached(dest):
            return True, "cached"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".part{threading.get_ident()}")
        last = "unknown"
        for attempt in range(self.retries + 1):
            try:
                if url.startswith("file://"):
                    shutil.copyfile(url[len("file://"):], tmp)
                else:
                    import requests
                    r = self._session().get(url, timeout=(10.0, self.timeout_s), stream=True)
                    if r.status_code in _RETRYABLE_STATUS:
                        last = f"HTTP {r.status_code}"
                        raise requests.RequestException(last)
                    if r.status_code != 200:
                        return False, f"HTTP {r.status_code} (not retried)"
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 16):
                            fh.write(chunk)
                size = tmp.stat().st_size
                if size < self.min_bytes:
                    last = f"file too small ({size} bytes)"
                    raise CorruptImageError(last)
                from PIL import Image
                with Image.open(tmp) as im:
                    im.verify()
                tmp.replace(dest)
                return True, "downloaded"
            except FileNotFoundError as exc:
                tmp.unlink(missing_ok=True)
                return False, f"local file missing: {exc}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                tmp.unlink(missing_ok=True)
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (2 ** attempt))
        return False, f"failed after {self.retries + 1} attempts: {last}"

    # -- many files --------------------------------------------------------- #
    def ensure_downloaded(self, df: pd.DataFrame, view_columns: Sequence[str], desc: str = "download") -> pd.DataFrame:
        """Download every view of every event. Returns a failure table
        (gravityspy_id, view, url, reason); empty when everything succeeded."""
        jobs = []
        for gid, *urls in df[["gravityspy_id", *view_columns]].itertuples(index=False, name=None):
            for k, url in enumerate(urls):
                dest = self.path_for(gid, k)
                if not self._is_cached(dest):
                    jobs.append((gid, k, str(url), dest))
        n_total = len(df) * len(view_columns)
        log.info("%s: %d images needed, %d cached, %d to download", desc, n_total, n_total - len(jobs), len(jobs))
        failures: list[dict] = []
        if jobs:
            def run(job):
                gid, k, url, dest = job
                ok, reason = self.fetch(url, dest)
                return job, ok, reason
            with ThreadPoolExecutor(self.workers) as ex:
                for job, ok, reason in tqdm(ex.map(run, jobs), total=len(jobs), desc=desc, mininterval=2.0):
                    if not ok:
                        failures.append({"gravityspy_id": job[0], "view": job[1] + 1, "url": job[2], "reason": reason})
        if failures:
            log.warning("%s: %d image(s) failed to download (their events will be dropped)", desc, len(failures))
        return pd.DataFrame(failures, columns=["gravityspy_id", "view", "url", "reason"])


# --------------------------------------------------------------------------- #
# Array cache
# --------------------------------------------------------------------------- #
def array_cache_key(ids: Sequence[str], pre_cfg: PreprocessConfig, view_columns: Sequence[str]) -> str:
    """Content key of a cached array: the ordered event IDs, EVERY geometry setting (crop box and reference size,
    channels, size, resize filter, alpha background) and the ORDERED view columns (which view fills which slot).
    Normalisation is applied on the fly to the uint8 array, so it is deliberately not part of the key."""
    return stable_hash({"ids": list(ids), "pre": pre_cfg.key(), "views": list(view_columns)})


def build_split_array(
    df: pd.DataFrame, store: ImageStore, pre_cfg: PreprocessConfig, view_columns: Sequence[str],
    processed_dir: str | Path, name: str, chunk: int = 1024, workers: int = 4,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Decode all views into one uint8 array (N_kept, V, C, H, W).

    Returns (array_path, kept_df, failures). kept_df is df minus events with any
    missing/corrupt view, in the same row order as the array. Corrupt cached PNGs
    are deleted so the next run re-downloads them.
    """
    n_views = len(view_columns)
    key = array_cache_key(df["gravityspy_id"].tolist(), pre_cfg, view_columns)
    out_dir = Path(processed_dir) / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_path, meta_path = out_dir / f"{name}_{key}.npy", out_dir / f"{name}_{key}.meta.json"

    if arr_path.exists() and meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        kept_ids = meta["kept_ids"]
        kept = df.set_index("gravityspy_id").loc[kept_ids].reset_index()[df.columns]
        log.info("%s: using cached array %s (%d events)", name, arr_path.name, len(kept))
        return arr_path, kept, pd.DataFrame(meta["failures"], columns=["gravityspy_id", "view", "url", "reason"])

    h, w = pre_cfg.image_size
    n = len(df)
    tmp_path = arr_path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(n, n_views, pre_cfg.channels, h, w))
    ids = df["gravityspy_id"].tolist()
    kept_rows: list[int] = []
    failures: list[dict] = []

    def load_one(i: int):
        paths = store.event_paths(ids[i], n_views)
        try:
            return i, load_event_views(paths, pre_cfg), None
        except CorruptImageError as exc:
            return i, None, (str(exc), paths, isinstance(exc, ImageGeometryError))

    k = 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for start in tqdm(range(0, n, chunk), desc=f"decode[{name}]", mininterval=2.0):
            for i, views, err in ex.map(load_one, range(start, min(start + chunk, n))):
                if err is not None:
                    reason, paths, wrong_size_only = err
                    failures.append({"gravityspy_id": ids[i], "view": 0, "url": "", "reason": reason})
                    if wrong_size_only:
                        continue  # the PNG is valid: never touch the original download
                    for p in paths:  # delete only files that genuinely fail to decode, so a rerun re-downloads
                        try:
                            from PIL import Image
                            with Image.open(p) as im:
                                im.load()
                        except Exception:
                            Path(p).unlink(missing_ok=True)
                    continue
                arr[k] = views
                kept_rows.append(i)
                k += 1
    arr.flush()
    del arr

    if k < n:
        log.warning("%s: %d/%d events dropped (corrupt/undecodable views)", name, n - k, n)
        src = np.load(tmp_path, mmap_mode="r")
        dst = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(k, n_views, pre_cfg.channels, h, w))
        for s in range(0, k, 4096):
            dst[s:s + 4096] = src[s:min(s + 4096, k)]
        dst.flush()
        del src, dst
        tmp_path.unlink()
    else:
        tmp_path.replace(arr_path)

    kept = df.iloc[kept_rows].reset_index(drop=True)
    fail_df = pd.DataFrame(failures, columns=["gravityspy_id", "view", "url", "reason"])
    save_json({"kept_ids": kept["gravityspy_id"].tolist(), "failures": fail_df.to_dict("records"),
               "pre": pre_cfg.key()}, meta_path)
    return arr_path, kept, fail_df


# --------------------------------------------------------------------------- #
# Dataset / sampler / loaders
# --------------------------------------------------------------------------- #
class GlitchDataset(Dataset):
    """Aligned four-view events. __getitem__ -> (x float32 [V,C,H,W], y int64).

    `labels` may be None for unlabeled use; then y = -1.
    """

    def __init__(self, array_path: str | Path, labels: np.ndarray | None, ids: Sequence[str],
                 augment: Augment | None = None, normalizer: Normalizer | None = None,
                 view_indices: Sequence[int] | None = None):
        self.array_path = str(array_path)
        self._x: np.ndarray | None = None
        shape = np.load(self.array_path, mmap_mode="r").shape
        n = shape[0]
        if len(ids) != n or (labels is not None and len(labels) != n):
            raise ValueError(f"Array has {n} rows but got {len(ids)} ids and "
                             f"{None if labels is None else len(labels)} labels. Misaligned inputs.")
        # view_indices=None returns every stored view (multi-view / merged models). A list selects views from
        # the aligned (N, V, C, H, W) array, so a single-view model receives (1, C, H, W) per event while the
        # event population (and the cached array) stays identical to the multi-view run.
        if view_indices is not None:
            view_indices = [int(v) for v in view_indices]
            n_views = shape[1] if len(shape) > 1 else 0
            if not view_indices or len(set(view_indices)) != len(view_indices) or \
                    not all(0 <= v < n_views for v in view_indices):
                raise ValueError(f"view_indices {view_indices} invalid for an array with {n_views} views.")
        self.view_indices = view_indices
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.ids = list(ids)
        self.augment = augment
        self.normalizer = normalizer if normalizer is not None else Normalizer("unit")

    def __len__(self) -> int:
        return len(self.ids)

    def _array(self) -> np.ndarray:
        if self._x is None:  # lazy so DataLoader workers each open their own memmap
            self._x = np.load(self.array_path, mmap_mode="r")
        return self._x

    def __getitem__(self, i: int):
        raw = self._array()[i] if self.view_indices is None else self._array()[i, self.view_indices]
        x = self.normalizer(torch.from_numpy(np.array(raw)))
        if self.augment is not None:
            x = self.augment(x)
        y = -1 if self.labels is None else int(self.labels[i])
        return x, torch.tensor(y, dtype=torch.long)


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)


def make_sampler(labels: np.ndarray, kind: str, num_classes: int, seed: int):
    """none -> None (shuffle instead) | sqrt -> w~1/sqrt(n_c) | balanced -> w~1/n_c."""
    if kind in (None, "none"):
        return None
    counts = np.maximum(class_counts(labels, num_classes), 1).astype(np.float64)
    if kind == "balanced":
        cw = 1.0 / counts
    elif kind == "sqrt":
        cw = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"imbalance.sampler must be none|sqrt|balanced, got '{kind}'")
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(cw[labels], dtype=torch.double), len(labels), True, generator=gen)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, sampler=None, num_workers: int = 0,
                seed: int = 0, drop_last: bool = False) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=num_workers, worker_init_fn=seed_worker if num_workers else None,
        generator=gen, drop_last=drop_last, persistent_workers=bool(num_workers),
    )


def model_input_views(cfg: dict[str, Any]) -> list[int] | None:
    """Which stored views the configured model receives. None = all of them.

    model.type=single_view feeds exactly ONE view (model.single_view_index, 0..V-1 in data.view_columns order).
    Every other model type (multiview, baseline_merged, legacy baseline_single) receives all views.
    """
    m = cfg.get("model", {})
    if m.get("type") != "single_view":
        return None
    n_views = len(cfg["data"]["view_columns"])
    idx = m.get("single_view_index")
    if idx is None or not 0 <= int(idx) < n_views:
        raise ValueError(f"model.single_view_index={idx} out of range for {n_views} views.")
    return [int(idx)]


def prepare_split(
    df: pd.DataFrame, name: str, cfg: dict[str, Any], store: ImageStore,
    *, augment: Augment | None = None, has_labels: bool = True,
) -> tuple[GlitchDataset, pd.DataFrame, pd.DataFrame]:
    """Download, decode and wrap one split. Returns (dataset, kept_df, failures).

    `failures` aggregates download and decode failures for logging.
    """
    from src.utils import resolve_path
    d = cfg["data"]
    pre = PreprocessConfig.from_cfg(d)
    dl_fail = store.ensure_downloaded(df, d["view_columns"], desc=f"download[{name}]")
    if len(dl_fail):
        bad = set(dl_fail["gravityspy_id"])
        df = df.loc[~df["gravityspy_id"].isin(bad)].reset_index(drop=True)
    path, kept, dec_fail = build_split_array(
        df, store, pre, d["view_columns"], resolve_path(cfg["paths"]["processed_dir"]), name)
    failures = pd.concat([dl_fail, dec_fail], ignore_index=True)
    if len(kept) == 0:
        raise RuntimeError(
            f"No usable events left in split '{name}'. Check network/URLs and the failures log; if the reasons "
            "mention image size, data.crop.reference_size does not match the downloaded PNGs.")
    labels = kept["label"].to_numpy() if has_labels else None
    view_indices = model_input_views(cfg)
    if view_indices is not None:
        log.info("%s: model input restricted to view(s) %s (%s)", name, view_indices,
                 [d["view_columns"][v] for v in view_indices])
    ds = GlitchDataset(path, labels, kept["gravityspy_id"].tolist(), augment, Normalizer.from_cfg(d),
                       view_indices=view_indices)
    return ds, kept, failures
