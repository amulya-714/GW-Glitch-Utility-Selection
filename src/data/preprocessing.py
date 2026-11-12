"""Image preprocessing: crop -> alpha composite -> (grayscale) -> resize, then a configurable normaliser.

Two stages with different lifetimes:

  1. GEOMETRY (PreprocessConfig, applied once, cached as uint8 arrays):
       optional crop of the plotting region (pixel box on the ORIGINAL image)
       -> alpha handling (RGBA -> RGB over a chosen background)
       -> optional grayscale -> resize to the target resolution.
     Changing any of these changes the array-cache key, so stale caches are never reused.

  2. NORMALISATION (Normalizer, applied on the fly in the Dataset):
       unit | per_view_standardize | per_sample_standardize | fixed
     It is not part of the cache, so switching it needs no re-decoding.

Nothing here computes dataset statistics implicitly. The only data-derived option is `fixed`,
whose constants come from the config; `compute_channel_stats` exists to produce them from SOURCE
TRAIN data only, and that is the caller's responsibility.

Original downloaded PNGs are only ever read here, never modified or deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

SUPPORTED_RESOLUTIONS = (96, 128, 160, 224)   # presets used by the preview tools; any positive size works
ALPHA_MODES = ("white", "black", "drop")
NORMALIZATION_MODES = ("unit", "per_view_standardize", "per_sample_standardize", "fixed")
MIN_CROP_SIDE = 16

_RESAMPLE = {"bilinear": Image.BILINEAR, "area": Image.BOX, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


class CorruptImageError(RuntimeError):
    """Raised when a spectrogram cannot be decoded or is implausibly small."""


class ImageGeometryError(CorruptImageError):
    """The file decodes fine but does not match the size the crop box was defined for.

    The PNG is valid, so it is never deleted from the cache; the event is dropped and logged.
    """


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _parse_size(size: Any) -> tuple[int, int]:
    """int -> (n, n); [h, w] -> (h, w)."""
    if isinstance(size, (int, np.integer)):
        h = w = int(size)
    else:
        try:
            h, w = (int(v) for v in size)
        except Exception as exc:
            raise ValueError(f"data.image_size must be an int or [height, width], got {size!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"data.image_size must be positive, got {size!r}")
    return h, w


def validate_crop_box(box: Sequence[int], reference_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Check [left, top, right, bottom] (right/bottom exclusive) against the reference (width, height)."""
    if box is None or len(box) != 4:
        raise ValueError(f"crop box must be [left, top, right, bottom], got {box!r}")
    if any(isinstance(v, bool) or int(v) != v for v in box):
        raise ValueError(f"crop box must contain integers, got {box!r}")
    left, top, right, bottom = (int(v) for v in box)
    width, height = reference_size
    if not (0 <= left < right <= width):
        raise ValueError(f"crop box x-range [{left}, {right}) must satisfy 0 <= left < right <= {width}")
    if not (0 <= top < bottom <= height):
        raise ValueError(f"crop box y-range [{top}, {bottom}) must satisfy 0 <= top < bottom <= {height}")
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        raise ValueError(f"crop box {box!r} is smaller than {MIN_CROP_SIDE}px on a side; check the coordinates")
    return left, top, right, bottom


def box_size(box: Sequence[int]) -> tuple[int, int]:
    """(width, height) in pixels of a [left, top, right, bottom] box."""
    return int(box[2]) - int(box[0]), int(box[3]) - int(box[1])


def aspect_preserving_size(region_wh: Sequence[int], height: int | None = None,
                           width: int | None = None) -> tuple[int, int]:
    """(H, W) with the same aspect ratio as a region of `region_wh` = (width, height), given EXACTLY ONE of
    `height` / `width`. The other side is rounded to the nearest pixel, so the residual stretch is under half a
    pixel per side (see `aspect_distortion`)."""
    if (height is None) == (width is None):
        raise ValueError("give exactly one of height / width")
    rw, rh = int(region_wh[0]), int(region_wh[1])
    if height is not None:
        return int(height), max(1, round(int(height) * rw / rh))
    return max(1, round(int(width) * rh / rw)), int(width)


def aspect_distortion(region_wh: Sequence[int], image_size: tuple[int, int]) -> float:
    """Relative anisotropic stretch when a region (width, height) is resized to image_size = (H, W):
    |x_scale / y_scale - 1|. 0 means the aspect ratio is preserved; a square output for a 573x479 crop gives 0.164."""
    rw, rh = int(region_wh[0]), int(region_wh[1])
    h, w = image_size
    return abs((w / rw) / (h / rh) - 1.0)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: tuple[int, int] = (96, 96)        # (H, W) of the cached arrays
    channels: int = 1                             # 1 grayscale | 3 RGB
    resize_mode: str = "bilinear"                 # bilinear | area | bicubic | lanczos
    alpha_background: str = "white"               # white | black | drop (drop = ignore alpha)
    crop_box: tuple[int, int, int, int] | None = None   # [left, top, right, bottom] on the original image
    crop_reference_size: tuple[int, int] = (800, 600)   # (width, height) the box coordinates refer to

    def __post_init__(self) -> None:
        if self.channels not in (1, 3):
            raise ValueError(f"data.channels must be 1 or 3, got {self.channels}")
        if self.resize_mode not in _RESAMPLE:
            raise ValueError(f"data.resize_mode must be one of {sorted(_RESAMPLE)}, got {self.resize_mode}")
        if self.alpha_background not in ALPHA_MODES:
            raise ValueError(f"data.alpha_background must be one of {ALPHA_MODES}, got {self.alpha_background}")
        if self.crop_box is not None:
            validate_crop_box(self.crop_box, self.crop_reference_size)

    @classmethod
    def from_cfg(cls, data_cfg: dict[str, Any]) -> "PreprocessConfig":
        crop = data_cfg.get("crop") or {}
        ref = tuple(int(v) for v in crop.get("reference_size", (800, 600)))
        box = None
        if crop.get("enabled", False):
            if crop.get("box") is None:
                raise ValueError(
                    "data.crop.enabled is true but data.crop.box is null. Run "
                    "`python -m src.data.crop_preview suggest ...`, verify the overlay with "
                    "`crop_preview preview`, then paste the verified box into the config.")
            box = validate_crop_box(crop["box"], ref)
        size = _parse_size(data_cfg["image_size"])
        max_d = crop.get("max_aspect_distortion", 0.02)
        if box is not None and max_d is not None:
            d = aspect_distortion(box_size(box), size)
            if d > float(max_d):
                bw, bh = box_size(box)
                raise ValueError(
                    f"data.image_size {list(size)} (HxW) stretches the {bw}x{bh} crop by {d:.1%} "
                    f"(limit data.crop.max_aspect_distortion={float(max_d):.1%}). Aspect-preserving choices: "
                    f"{list(aspect_preserving_size((bw, bh), height=size[0]))} or "
                    f"{list(aspect_preserving_size((bw, bh), width=size[1]))}. "
                    "Set data.crop.max_aspect_distortion: null only if stretching is intended.")
        return cls(size, int(data_cfg["channels"]),
                   data_cfg.get("resize_mode", "bilinear"), data_cfg.get("alpha_background", "white"), box, ref)

    def key(self) -> dict:
        """Everything that changes the cached pixels. Used in the array-cache key."""
        return {"size": list(self.image_size), "channels": self.channels, "resize": self.resize_mode,
                "alpha": self.alpha_background, "crop": list(self.crop_box) if self.crop_box else None,
                "crop_ref": list(self.crop_reference_size) if self.crop_box else None}


def _to_rgb(im: Image.Image, background: str) -> Image.Image:
    """Any PNG mode -> RGB. RGBA is composited over `background` unless background == 'drop'."""
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    if not has_alpha:
        return im.convert("RGB")
    rgba = im.convert("RGBA")
    if background == "drop":
        return rgba.convert("RGB")
    fill = (255, 255, 255, 255) if background == "white" else (0, 0, 0, 255)
    return Image.alpha_composite(Image.new("RGBA", rgba.size, fill), rgba).convert("RGB")


def load_view(path: str | Path, cfg: PreprocessConfig) -> np.ndarray:
    """Decode one PNG to uint8 (C, H, W) following the geometry config.

    Order: validate size (if cropping) -> alpha composite -> crop -> grayscale -> resize.
    Raises CorruptImageError (undecodable) or ImageGeometryError (wrong size for the crop box),
    always naming the file.
    """
    path = Path(path)
    try:
        with Image.open(path) as im:
            im.load()
            if cfg.crop_box is not None and im.size != tuple(cfg.crop_reference_size):
                raise ImageGeometryError(
                    f"{path}: image is {im.size[0]}x{im.size[1]} but the crop box is defined for "
                    f"{cfg.crop_reference_size[0]}x{cfg.crop_reference_size[1]} (width x height)")
            img = _to_rgb(im, cfg.alpha_background)
            if cfg.crop_box is not None:
                img = img.crop(cfg.crop_box)
            if cfg.channels == 1:
                img = img.convert("L")
            h, w = cfg.image_size
            img = img.resize((w, h), _RESAMPLE[cfg.resize_mode])
            arr = np.asarray(img, dtype=np.uint8)
    except CorruptImageError:
        raise
    except Exception as exc:  # PIL raises many different types
        raise CorruptImageError(f"cannot decode {path}: {exc}") from exc
    arr = arr[None, :, :] if arr.ndim == 2 else np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr)


def load_event_views(paths: Sequence[str | Path], cfg: PreprocessConfig) -> np.ndarray:
    """Load the four views of ONE event, in order, -> uint8 (V, C, H, W).

    All paths come from the same metadata row, so views stay aligned to the event.
    """
    return np.stack([load_view(p, cfg) for p in paths], axis=0)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _as_channel_tuple(v: Any, name: str) -> tuple[float, ...]:
    vals = (float(v),) if isinstance(v, (int, float)) else tuple(float(x) for x in v)
    if not vals:
        raise ValueError(f"normalization.{name} must not be empty")
    return vals


@dataclass(frozen=True)
class Normalizer:
    """uint8/float (..., V, C, H, W) -> float32.

    unit                    x / 255 in [0, 1]. Fixed scale: absolute and cross-view intensity is preserved.
    per_view_standardize    each view image standardised on its own (mean 0, std 1 over C,H,W). Removes
                            absolute brightness and the brightness ratio between time scales.
    per_sample_standardize  one mean/std per event over all views jointly. Removes event-level offset/scale
                            but keeps the relative brightness between the four views.
    fixed                   (x/255 - mean) / std with constants from the config (scalar or per channel).
                            Constants must come from SOURCE TRAIN data only (see compute_channel_stats).
    Constant images map to zeros (no NaN) in the standardising modes.
    """
    mode: str = "unit"
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in NORMALIZATION_MODES:
            raise ValueError(f"normalization.mode must be one of {NORMALIZATION_MODES}, got '{self.mode}'")
        if self.mode == "fixed":
            if self.mean is None or self.std is None:
                raise ValueError("normalization.mode='fixed' requires normalization.mean and normalization.std")
            if len(self.mean) != len(self.std):
                raise ValueError("normalization.mean and normalization.std must have the same length")
            if any(s <= 0 for s in self.std):
                raise ValueError("normalization.std must be strictly positive")

    @classmethod
    def from_cfg(cls, data_cfg: dict[str, Any]) -> "Normalizer":
        norm = data_cfg.get("normalization")
        if norm is None:  # configs/checkpoints written before this option existed used this behaviour
            return cls("per_view_standardize")
        mode = norm.get("mode", "unit")
        mean = None if norm.get("mean") is None else _as_channel_tuple(norm["mean"], "mean")
        std = None if norm.get("std") is None else _as_channel_tuple(norm["std"], "std")
        out = cls(mode, mean, std)
        if mode == "fixed":
            ch = int(data_cfg["channels"])
            for name, vals in (("mean", out.mean), ("std", out.std)):
                if len(vals) not in (1, ch):
                    raise ValueError(f"normalization.{name} has {len(vals)} values but data.channels={ch}; "
                                     f"give a scalar or exactly {ch} values")
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        if self.mode == "unit":
            return x
        if self.mode == "fixed":
            mean = torch.tensor(self.mean, dtype=x.dtype).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=x.dtype).view(-1, 1, 1)
            return (x - mean) / std
        dims = (-3, -2, -1) if self.mode == "per_view_standardize" else (-4, -3, -2, -1)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)


def normalize_views(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Legacy per-view standardisation (kept for backwards compatibility)."""
    return Normalizer("per_view_standardize", eps=eps)(x)


def compute_channel_stats(arr: np.ndarray, max_samples: int | None = None, seed: int = 0,
                          chunk: int = 512) -> tuple[list[float], list[float]]:
    """Per-channel mean/std in [0, 1] units of a uint8 array (N, V, C, H, W) or (N, C, H, W).

    For use with normalization.mode='fixed'. Call it ONLY on a SOURCE TRAIN array: using
    validation or target-run images here would leak information into preprocessing.
    """
    n = arr.shape[0]
    idx = np.arange(n)
    if max_samples is not None and max_samples < n:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=max_samples, replace=False))
    c_axis = arr.ndim - 3
    c = arr.shape[c_axis]
    s = np.zeros(c, np.float64)
    ss = np.zeros(c, np.float64)
    count = 0
    for start in range(0, len(idx), chunk):
        block = np.asarray(arr[idx[start:start + chunk]], dtype=np.float64) / 255.0
        block = np.moveaxis(block, c_axis, 0).reshape(c, -1)
        s += block.sum(axis=1)
        ss += (block ** 2).sum(axis=1)
        count += block.shape[1]
    mean = s / count
    std = np.sqrt(np.maximum(ss / count - mean ** 2, 0.0))
    return mean.tolist(), std.tolist()


def preprocessing_signature(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Every setting that determines the model's input pixels, plus a short hash.

    geometry      crop, channels, size, filter, alpha background (these also key the array cache)
    views         the ORDERED metadata columns, i.e. which view feeds which model slot
    normalization mode and constants (applied on the fly, so not part of the array cache)
    Stored in each run's summary and checkpoint so a result can always be tied to its exact preprocessing.
    """
    from src.utils import stable_hash
    norm = Normalizer.from_cfg(data_cfg)
    body = {"geometry": PreprocessConfig.from_cfg(data_cfg).key(), "views": list(data_cfg["view_columns"]),
            "view_durations": [float(v) for v in data_cfg["view_durations"]],
            "normalization": {"mode": norm.mode, "mean": norm.mean, "std": norm.std}}
    return {**body, "hash": stable_hash(body)}


# --------------------------------------------------------------------------- #
# Augmentation (unchanged; operates on the already-normalised tensor)
# --------------------------------------------------------------------------- #
class Augment:
    """Optional light augmentation on a (V, C, H, W) float tensor.

    * time shift: the SAME random horizontal shift is applied to all four views
      (they share the event time axis in pixel terms only approximately, so the
      shift is kept small), edge-replicated rather than wrapped.
    * Gaussian noise (std is in normalised units, so its meaning depends on normalization.mode).
    Disabled by default so the baseline is plain real-data training.
    """

    def __init__(self, max_time_shift_frac: float = 0.0, noise_std: float = 0.0, seed: int | None = None):
        self.max_shift = float(max_time_shift_frac)
        self.noise_std = float(noise_std)
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift > 0:
            w = x.shape[-1]
            m = int(round(self.max_shift * w))
            if m > 0:
                s = int(torch.randint(-m, m + 1, (1,), generator=self.gen))
                if s != 0:
                    pad = torch.nn.functional.pad(
                        x.reshape(-1, *x.shape[-2:]).unsqueeze(0), (m, m, 0, 0), mode="replicate"
                    ).squeeze(0).reshape(*x.shape[:-1], w + 2 * m)
                    x = pad[..., m - s: m - s + w]
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn(x.shape, generator=self.gen)
        return x
