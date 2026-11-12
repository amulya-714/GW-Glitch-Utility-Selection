"""Metadata loading, cleaning, class mapping, confidence filtering and temporal splits.

Design rules enforced here:
  * Target-run (O3a) metadata can only be loaded with purpose="final_eval".
  * Splits are time-based (never random per-event) with a purge gap.
  * Class exclusions / unseen labels are explicit and always logged.
  * The split boundary is computed BEFORE confidence filtering, so it does not
    depend on the threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

META_COLUMNS = ["gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence"]


class TargetLeakageError(RuntimeError):
    """Raised when training/validation code tries to touch target-run data."""


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def metadata_path(metadata_dir: str | Path, ifo: str, run: str) -> Path:
    return Path(metadata_dir) / f"{ifo}_{run}.csv"


def load_run_metadata(
    metadata_dir: str | Path,
    ifo: str,
    run: str,
    view_columns: Sequence[str],
    *,
    target_runs: Sequence[str],
    purpose: str,
) -> pd.DataFrame:
    """Read the CSV for (ifo, run), only the columns we need.

    purpose must be 'train' (source-run use) or 'final_eval' (target-run use).
    Loading a target run for any other purpose raises TargetLeakageError.
    """
    if purpose not in {"train", "final_eval"}:
        raise ValueError(f"purpose must be 'train' or 'final_eval', got '{purpose}'")
    if run in set(target_runs) and purpose != "final_eval":
        raise TargetLeakageError(
            f"Run '{run}' is a protocol target run and cannot be loaded for purpose='{purpose}'. "
            "Target data is reserved for `evaluate.py --split target --final`."
        )
    if purpose == "final_eval" and run not in set(target_runs):
        raise ValueError(f"final_eval expects a target run, got '{run}' (targets: {list(target_runs)}).")
    path = metadata_path(metadata_dir, ifo, run)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Put {ifo}_{run}.csv in {metadata_dir}."
        )
    cols = META_COLUMNS + list(view_columns)
    try:
        df = pd.read_csv(path, usecols=cols)
    except ValueError as exc:
        raise ValueError(f"{path} is missing required columns {cols}: {exc}") from exc
    df["run"] = run  # bookkeeping only; never a model input
    return df


def clean_metadata(df: pd.DataFrame, ifo: str, view_columns: Sequence[str], name: str = "") -> pd.DataFrame:
    """Dedupe, validate and type-fix. Every removal is logged."""
    n0 = len(df)
    df = df.copy()
    df["gravityspy_id"] = df["gravityspy_id"].astype(str)
    df["ml_label"] = df["ml_label"].astype(str)
    df["event_time"] = pd.to_numeric(df["event_time"], errors="coerce")
    df["ml_confidence"] = pd.to_numeric(df["ml_confidence"], errors="coerce")

    wrong_ifo = df["ifo"].astype(str) != ifo
    if wrong_ifo.any():
        raise ValueError(f"{name}: {int(wrong_ifo.sum())} rows have ifo != {ifo}. Wrong file?")

    bad = df["event_time"].isna() | df["ml_confidence"].isna() | df[list(view_columns)].isna().any(axis=1)
    if bad.any():
        log.warning("%s: dropping %d rows with missing time/confidence/URL", name, int(bad.sum()))
        df = df.loc[~bad]

    # Exact duplicates first, then ID duplicates that differ in content (flagged loudly).
    dup_exact = df.duplicated(keep="first")
    if dup_exact.any():
        ids = sorted(df.loc[dup_exact, "gravityspy_id"].unique())
        log.warning("%s: removed %d exact duplicate row(s): %s", name, int(dup_exact.sum()), ids[:10])
        df = df.loc[~dup_exact]
    dup_id = df["gravityspy_id"].duplicated(keep=False)
    if dup_id.any():
        ids = sorted(df.loc[dup_id, "gravityspy_id"].unique())
        log.warning(
            "%s: %d gravityspy_id(s) occur with DIFFERENT row content; dropping all of them: %s",
            name, len(ids), ids[:10],
        )
        df = df.loc[~dup_id]

    df = df.sort_values(["event_time", "gravityspy_id"], kind="mergesort").reset_index(drop=True)
    log.info("%s: %d -> %d events after cleaning", name, n0, len(df))
    return df


# --------------------------------------------------------------------------- #
# Class mapping
# --------------------------------------------------------------------------- #
@dataclass
class ClassMapping:
    names: list[str]                       # trained classes, index = class id
    excluded: list[str] = field(default_factory=list)
    unseen_in_source: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate class names in mapping.")
        self.class_to_idx = {c: i for i, c in enumerate(self.names)}

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_to_idx": self.class_to_idx,
            "trained_classes": self.names,
            "excluded_from_training": self.excluded,
            "unseen_in_source_run": self.unseen_in_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClassMapping":
        return cls(d["trained_classes"], d.get("excluded_from_training", []), d.get("unseen_in_source_run", []))


def build_class_mapping(classes_cfg: dict[str, Any]) -> ClassMapping:
    names = list(classes_cfg["names"])
    excluded = list(classes_cfg.get("exclude") or [])
    unseen = list(classes_cfg.get("unseen_in_source") or [])
    unknown = [c for c in excluded if c not in names]
    if unknown:
        raise ValueError(f"classes.exclude contains labels not in classes.names: {unknown}")
    overlap = [c for c in unseen if c in names]
    if overlap:
        raise ValueError(f"classes.unseen_in_source overlaps classes.names: {overlap}")
    return ClassMapping([c for c in names if c not in excluded], excluded, unseen)


def apply_class_mapping(
    df: pd.DataFrame, mapping: ClassMapping, name: str = "", *, strict_unknown: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (mapped, other). `mapped` gets an integer `label` column.

    `other` holds events with excluded / unseen / unknown labels. Training code
    discards `other` (after logging); final evaluation keeps it for open-set analysis.
    """
    df = df.copy()
    df["label"] = df["ml_label"].map(mapping.class_to_idx)
    in_space = df["label"].notna()
    other = df.loc[~in_space].drop(columns=["label"])
    mapped = df.loc[in_space].copy()
    mapped["label"] = mapped["label"].astype(int)
    if len(other):
        counts = other["ml_label"].value_counts()
        known = set(mapping.excluded) | set(mapping.unseen_in_source)
        unknown = [c for c in counts.index if c not in known]
        for lbl, n in counts.items():
            kind = "excluded by config" if lbl in mapping.excluded else (
                "unseen-in-source" if lbl in mapping.unseen_in_source else "UNKNOWN label")
            log.info("%s: %d events with label '%s' are outside the trained label space (%s)", name, n, lbl, kind)
        if unknown and strict_unknown:
            raise ValueError(
                f"{name}: labels {unknown} are neither trained, excluded nor listed in "
                "classes.unseen_in_source. Add them to the config explicitly."
            )
    return mapped.reset_index(drop=True), other.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Confidence filtering, caps and count tables
# --------------------------------------------------------------------------- #
def apply_confidence_filter(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return df.copy()
    return df.loc[df["ml_confidence"] >= float(threshold)].reset_index(drop=True)


def cap_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int, label_col: str = "ml_label") -> pd.DataFrame:
    """Seeded random cap per class. The kept rows stay in chronological order."""
    if max_per_class is None:
        return df.copy()
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for _, idx in df.groupby(label_col).indices.items():
        idx = np.asarray(idx)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    if not keep:
        return df.iloc[0:0].copy()
    kept = np.sort(np.concatenate(keep))
    return df.iloc[kept].reset_index(drop=True)


def class_count_table(stages: dict[str, pd.DataFrame], label_col: str = "ml_label") -> pd.DataFrame:
    """Rows = classes, columns = stage names, values = event counts."""
    tab = pd.DataFrame({k: v[label_col].value_counts() for k, v in stages.items()}).fillna(0).astype(int)
    return tab.sort_values(tab.columns[0], ascending=False)


# --------------------------------------------------------------------------- #
# Temporal splitting
# --------------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame,
    mode: str = "chronological",
    val_fraction: float = 0.2,
    gap_seconds: float = 21600.0,
    block_seconds: float = 86400.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by event_time into (train, val, purged).

    chronological: earliest (1-val_fraction) of events -> train, latest -> val,
        events within `gap_seconds` before the boundary are purged.
    blocked: the timeline is cut into `block_seconds` blocks; blocks are assigned
        to val at random (seeded) until ~val_fraction of events are covered;
        events within `gap_seconds` of a block of the other split are purged.
        Gives better rare-class coverage in val but is a weaker shift proxy.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(df) < 10:
        raise ValueError(f"Too few events ({len(df)}) to split.")
    df = df.sort_values(["event_time", "gravityspy_id"], kind="mergesort").reset_index(drop=True)
    t = df["event_time"].to_numpy()

    if mode == "chronological":
        cut = int(round(len(df) * (1.0 - val_fraction)))
        t_cut = t[min(cut, len(df) - 1)]
        is_val = t >= t_cut
        is_train = t < (t_cut - gap_seconds)
    elif mode == "blocked":
        t0 = t.min()
        block = np.floor((t - t0) / block_seconds).astype(int)
        ublocks, counts = np.unique(block, return_counts=True)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ublocks))
        target = val_fraction * len(df)
        val_blocks: set[int] = set()
        acc = 0
        for o in order:
            if acc >= target:
                break
            val_blocks.add(int(ublocks[o]))
            acc += int(counts[o])
        is_val_block = np.isin(block, list(val_blocks))
        # distance from each event to the nearest event of the other split
        val_times = t[is_val_block]
        tr_times = t[~is_val_block]
        d_to_val = _nearest_distance(t, val_times)
        d_to_tr = _nearest_distance(t, tr_times)
        is_val = is_val_block & (d_to_tr >= gap_seconds)
        is_train = (~is_val_block) & (d_to_val >= gap_seconds)
    else:
        raise ValueError(f"Unknown split.mode '{mode}' (use chronological|blocked)")

    purged = ~(is_train | is_val)
    train, val, pur = df[is_train].reset_index(drop=True), df[is_val].reset_index(drop=True), df[purged].reset_index(drop=True)
    log.info("split[%s]: train=%d val=%d purged=%d (gap=%.0fs)", mode, len(train), len(val), len(pur), gap_seconds)
    assert_split_disjoint(train, val, gap_seconds if mode == "chronological" else 0.0, mode)
    return train, val, pur


def _nearest_distance(query: np.ndarray, ref_sorted_or_not: np.ndarray) -> np.ndarray:
    if len(ref_sorted_or_not) == 0:
        return np.full(len(query), np.inf)
    ref = np.sort(ref_sorted_or_not)
    pos = np.searchsorted(ref, query)
    left = ref[np.clip(pos - 1, 0, len(ref) - 1)]
    right = ref[np.clip(pos, 0, len(ref) - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right))


def assert_split_disjoint(train: pd.DataFrame, val: pd.DataFrame, min_gap: float = 0.0, mode: str = "chronological") -> None:
    """Hard checks that run on every split: ID disjointness and (for chronological) time ordering."""
    overlap = set(train["gravityspy_id"]) & set(val["gravityspy_id"])
    if overlap:
        raise AssertionError(f"{len(overlap)} events appear in both train and val, e.g. {sorted(overlap)[:5]}")
    if len(train) and len(val):
        if mode == "chronological":
            gap = val["event_time"].min() - train["event_time"].max()
            if gap < min_gap:
                raise AssertionError(f"train/val time gap {gap:.1f}s is smaller than required {min_gap:.1f}s")
        else:  # blocked: check nearest cross-split distance
            d = _nearest_distance(val["event_time"].to_numpy(), train["event_time"].to_numpy()).min()
            if d < 0:
                raise AssertionError("negative time distance")


def warn_small_val_classes(val: pd.DataFrame, mapping: ClassMapping, min_per_class: int) -> list[str]:
    counts = val["ml_label"].value_counts()
    weak = [c for c in mapping.names if counts.get(c, 0) < min_per_class]
    for c in weak:
        log.warning("val class '%s' has only %d event(s) (< %d): its metrics will be unreliable%s",
                    c, int(counts.get(c, 0)), min_per_class, " (ABSENT)" if counts.get(c, 0) == 0 else "")
    return weak


# --------------------------------------------------------------------------- #
# End-to-end source split used by train.py
# --------------------------------------------------------------------------- #
def build_source_splits(
    cfg: dict[str, Any], *, ifo: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, ClassMapping, dict[str, pd.DataFrame]]:
    """Load O2 metadata for one detector and return (train, val, mapping, reports).

    reports holds the class-count tables (before/after each stage) and the purged events.
    """
    from src.utils import resolve_path  # local import keeps this module light

    d, proto = cfg["data"], cfg["protocol"]
    ifo = ifo or d["ifo"]
    run = proto["source_run"]
    seed = cfg["experiment"]["seed"]
    mapping = build_class_mapping(cfg["classes"])

    raw = load_run_metadata(
        resolve_path(cfg["paths"]["metadata_dir"]), ifo, run, d["view_columns"],
        target_runs=proto["target_runs"], purpose="train",
    )
    name = f"{ifo}_{run}"
    clean = clean_metadata(raw, ifo, d["view_columns"], name)
    mapped, other = apply_class_mapping(clean, mapping, name)

    sp = cfg["split"]
    train, val, purged = temporal_split(
        mapped, sp["mode"], sp["val_fraction"], sp["gap_seconds"], sp["block_seconds"], seed
    )
    warn_small_val_classes(val, mapping, sp["min_val_per_class"])

    train_f = apply_confidence_filter(train, cfg["confidence"]["train"])
    val_f = apply_confidence_filter(val, cfg["confidence"]["val"])
    samp = cfg["sampling"]
    train_c = cap_per_class(train_f, samp["max_per_class_train"], seed)
    val_c = cap_per_class(val_f, samp["max_per_class_val"], seed + 1)

    reports = {
        "train_counts": class_count_table(
            {"split": train, f"conf>={cfg['confidence']['train']}": train_f, "final(cap)": train_c}),
        "val_counts": class_count_table(
            {"split": val, f"conf>={cfg['confidence']['val']}": val_f, "final(cap)": val_c}),
        "purged": purged,
        "outside_label_space": other,
    }
    log.info("TRAIN class counts (before/after confidence filter and cap):\n%s", reports["train_counts"].to_string())
    log.info("VAL class counts (before/after confidence filter and cap):\n%s", reports["val_counts"].to_string())
    if len(train_c) == 0 or len(val_c) == 0:
        raise ValueError("Empty train or val set after filtering. Relax confidence/cap settings.")
    train_c["split"], val_c["split"] = "train", "val"
    return train_c, val_c, mapping, reports
