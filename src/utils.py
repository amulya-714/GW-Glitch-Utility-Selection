"""Shared helpers: seeding, device selection, logging, config IO, hashing.

Nothing in here touches data, so it is safe to import from anywhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Must be set before torch initialises MPS so unsupported ops fall back to CPU
# instead of crashing. Harmless on other platforms.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def project_root() -> Path:
    """Repository root (the directory that contains `src/` and `configs/`)."""
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # DataLoader worker_init_fn
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(preference: str = "auto") -> torch.device:
    """Pick a device. 'auto' tries CUDA, then Apple MPS, then CPU.

    An explicit request that is unavailable falls back to CPU with a warning,
    never an exception, so the same config runs on every teammate's laptop.
    """
    log = logging.getLogger(__name__)
    pref = (preference or "auto").lower()
    if pref not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device '{preference}'. Use auto|cpu|cuda|mps.")
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("auto", "mps") and has_mps:
        try:  # smoke-test the backend; some macOS/torch combos report available but fail
            torch.zeros(1, device="mps") + 1
            return torch.device("mps")
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.warning("MPS reported available but failed (%s); using CPU.", exc)
            return torch.device("cpu")
    if pref in ("cuda", "mps"):
        log.warning("Requested device '%s' unavailable; falling back to CPU.", pref)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# --------------------------------------------------------------------------- #
# Config / IO
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at top level.")
    return cfg


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge `updates` into a deep copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    """cfg['a']['b'] = value for 'a.b'. Intermediate dicts are created."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def parse_override(item: str) -> tuple[str, Any]:
    """'train.lr=0.01' -> ('train.lr', 0.01). Value parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"Override '{item}' must look like key.subkey=value")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(to_jsonable(obj), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def file_sha256(path: str | Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
