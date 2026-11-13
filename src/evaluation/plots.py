"""Publication-quality matplotlib figures for the model comparison and the O3a diagnostic.

Conventions
  * Every figure is saved as a 300-dpi PNG and a vector PDF with the same stem.
  * Colours follow the MODEL, never its rank (fixed order below, a colour-blind-checked categorical palette);
    every line also carries its own marker so identity is never colour-alone.
  * Class order is always the trained class order from the checkpoint's class mapping. No class is dropped;
    zero-support rows are shaded grey and labelled "n=0".
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.WARNING)   # PDF font subsetting is very chatty at INFO
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Categorical slots, fixed order (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6 on white).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5",
                                                          "#1c5cab", "#0d366b"])
THRESHOLD_MARKS = (0.5, 0.7, 0.9)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.6,
    })


def save(fig: plt.Figure, stem: str | Path, pdf: bool = True) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # append, never with_suffix(): run names such as H1_O2_single_view_0.5s contain a dot
    out = [stem.parent / f"{stem.name}.png"]
    fig.savefig(out[0])
    if pdf:
        out.append(stem.parent / f"{stem.name}.pdf")
        fig.savefig(out[1])
    plt.close(fig)
    return out


def style_for(i: int) -> dict:
    if i >= len(PALETTE):
        raise ValueError(f"At most {len(PALETTE)} models per comparison figure (got index {i}); "
                         "split the comparison instead of generating extra colours.")
    return {"color": PALETTE[i], "marker": MARKERS[i]}


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #
def plot_confusion(cm: np.ndarray, class_names: Sequence[str], stem: str | Path, title: str,
                   normalise: bool, subtitle: str = "") -> list[Path]:
    """All classes, mapping order. normalise=True -> row-normalised recall; zero-support rows are grey/NaN."""
    apply_style()
    cm = np.asarray(cm, float)
    c = len(class_names)
    support = cm.sum(1)
    if normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.where(support[:, None] > 0, cm / support[:, None], np.nan)
        vmax, cbar_label = 1.0, "fraction of true class (row-normalised)"
    else:
        data, vmax, cbar_label = cm, max(1.0, cm.max()), "number of events"
    size = max(6.5, 0.36 * c + 2.8)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    cmap = SEQ_BLUE.with_extremes(bad="#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax.grid(False)
    ylabels = [f"{n} (n={int(s)})" for n, s in zip(class_names, support)]
    ax.set_xticks(range(c), class_names, rotation=90)
    ax.set_yticks(range(c), ylabels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class (Gravity Spy label)")
    if (support == 0).any():
        subtitle = (subtitle + "; " if subtitle else "") + "grey rows: class has no reference events in this set"
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    if c <= 30:
        fs = 6 if c > 15 else 7
        for i in range(c):
            for j in range(c):
                v = data[i, j]
                if np.isnan(v) or v == 0:
                    continue
                txt = f"{v:.2f}" if normalise else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                        color="white" if v > 0.55 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    return save(fig, stem)


# --------------------------------------------------------------------------- #
# Per-class and overall comparisons
# --------------------------------------------------------------------------- #
def plot_per_class_comparison(values: pd.DataFrame, support: pd.Series, stem: str | Path, metric_label: str,
                              title: str) -> list[Path]:
    """values: index = classes (mapping order), columns = model labels (fixed order). NaN = undefined."""
    apply_style()
    classes, models = list(values.index), list(values.columns)
    n_c, n_m = len(classes), len(models)
    bar_h = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.3 * n_c * max(1, n_m / 3) + 1.4)))
    y = np.arange(n_c)
    all_nan = values.isna().all(axis=1).to_numpy()
    for k, model in enumerate(models):
        st = style_for(k)
        v = values[model].to_numpy(float)
        pos = y - 0.4 + bar_h * (k + 0.5)
        ax.barh(pos, np.nan_to_num(v, nan=0.0), height=bar_h * 0.92, color=st["color"],
                label=model, edgecolor="white", linewidth=0.4)
        for yi in np.where(v == 0)[0]:          # a real zero must not look like a missing bar
            ax.text(0.004, pos[yi], "0", va="center", fontsize=5.5, color=st["color"], fontweight="bold")
        for yi in np.where(np.isnan(v) & ~all_nan)[0]:
            ax.text(0.004, pos[yi], "undefined", va="center", fontsize=5.5, color=MUTED)
    for yi in np.where(all_nan)[0]:
        ax.text(0.01, y[yi], "no reference events in this set (undefined)", va="center", fontsize=6.5,
                color=MUTED, style="italic")
    ax.set_yticks(y, [f"{c} (n={int(support.get(c, 0))})" for c in classes])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0 + 0.6 / max(n_c, 1)), ncol=min(n_m, 3))
    fig.tight_layout()
    return save(fig, stem)


def plot_overall_comparison(table: pd.DataFrame, stem: str | Path, title: str) -> list[Path]:
    """table: one row per model (column 'Model'), metric columns. Left: higher-is-better; right: ECE."""
    apply_style()
    good = [c for c in ("Accuracy", "Macro-F1", "Weighted-F1", "Balanced Accuracy") if c in table]
    models = table["Model"].tolist()
    n_m = len(models)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [4, 1.3]})
    x = np.arange(len(good))
    w = 0.8 / n_m
    for k, model in enumerate(models):
        st = style_for(k)
        vals = table.loc[table["Model"] == model, good].to_numpy(float).ravel()
        bars = ax.bar(x - 0.4 + w * (k + 0.5), vals, width=w * 0.92, color=st["color"], label=model,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90, color=INK)
        if "ECE" in table:
            e = float(table.loc[table["Model"] == model, "ECE"].iloc[0])
            ax2.bar(k, e, color=st["color"], width=0.8, edgecolor="white", linewidth=0.4)
            ax2.text(k, e, f"{e:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, good)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(n_m, 3))
    ax2.set_xticks(range(n_m), [""] * n_m)
    ax2.set_ylabel("ECE (lower is better)")
    ax2.set_title("Calibration error")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, stem)


# --------------------------------------------------------------------------- #
# Calibration / confidence
# --------------------------------------------------------------------------- #
def _grid(n: int, w: float = 3.3, h: float = 3.2):
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_reliability(entries: Sequence[dict], stem: str | Path, title: str) -> list[Path]:
    """entries: {label, bins (reliability DataFrame), ece, index}. One diagram per model, with a separate
    events-per-bin panel underneath (so the counts never hide the accuracy bars)."""
    apply_style()
    n = len(entries)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.4 * cols, 4.0 * rows), layout="constrained")
    gs = fig.add_gridspec(rows * 2, cols, height_ratios=[3, 1] * rows)
    for i, e in enumerate(entries):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[2 * r, c])
        axc = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        st = style_for(e["index"])
        b = e["bins"]
        width = (b["upper"] - b["lower"]).to_numpy()
        centers = ((b["upper"] + b["lower"]) / 2).to_numpy()
        acc = b["accuracy"].to_numpy(float)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration")
        ax.bar(centers, np.nan_to_num(acc), width=width * 0.94, color=st["color"], alpha=0.85,
               label="accuracy in bin", edgecolor="white", linewidth=0.4)
        mc = b["mean_confidence"].to_numpy(float)
        ok = np.isfinite(mc)
        ax.scatter(mc[ok], acc[ok], s=12, color=INK, zorder=3, label="mean confidence in bin")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.tick_params(labelbottom=False)
        total = int(b["count"].sum())
        ax.set_title(f"{e['label']}\nECE = {e['ece']:.4f}  (n = {total})", fontsize=9)
        axc.bar(centers, b["count"], width=width * 0.94, color=MUTED)
        if b["count"].max() > 0:
            axc.set_yscale("log")
            axc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axc.yaxis.set_minor_formatter(NullFormatter())
        axc.set_ylabel("events", fontsize=7)
        axc.set_xlabel("Model confidence (max softmax)")
        axc.grid(axis="x", visible=False)
        if i == 0:
            ax.legend(loc="upper left", fontsize=6.5)
    fig.suptitle(title)
    return save(fig, stem)


def plot_confidence_distribution(entries: Sequence[dict], stem: str | Path, title: str) -> list[Path]:
    """entries: {label, conf, correct, index}. Histogram of max-softmax confidence, correct vs incorrect."""
    apply_style()
    fig, axes = _grid(len(entries), h=2.9)
    bins = np.linspace(0, 1, 26)
    for ax, e in zip(axes, entries):
        st = style_for(e["index"])
        conf, correct = np.asarray(e["conf"]), np.asarray(e["correct"], bool)
        ax.hist(conf[correct], bins=bins, color=st["color"], alpha=0.9, label=f"correct (n={int(correct.sum())})",
                edgecolor="white", linewidth=0.4)
        ax.hist(conf[~correct], bins=bins, histtype="step", color=INK, lw=1.4, hatch="////",
                label=f"incorrect (n={int((~correct).sum())})")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Model confidence (max softmax)")
        ax.set_ylabel("Events (log scale)")
        ax.set_title(e["label"], fontsize=9)
        ax.legend(loc="upper left", fontsize=6.5)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return save(fig, stem)


def plot_sweep(entries: Sequence[dict], y_key: str, stem: str | Path, ylabel: str, title: str,
               note: str = "", xlabel: str = "Model-confidence threshold τ (keep events with max softmax ≥ τ)"
               ) -> list[Path]:
    """entries: {label, sweep (DataFrame threshold/coverage/accuracy), index}. y_key: coverage | accuracy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for e in entries:
        st = style_for(e["index"])
        s = e["sweep"]
        ax.plot(s["threshold"], s[y_key], color=st["color"], label=e["label"])
        marks = s[s["threshold"].round(6).isin([round(t, 6) for t in THRESHOLD_MARKS])]
        ax.plot(marks["threshold"], marks[y_key], ls="none", marker=st["marker"], ms=7, color=st["color"],
                markeredgecolor="white", markeredgewidth=0.8)
    for t in THRESHOLD_MARKS:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + (f"\n{note}" if note else ""))
    ax.legend(loc="best")
    fig.tight_layout()
    return save(fig, stem)


def plot_risk_coverage(entries: Sequence[dict], stem: str | Path, title: str) -> list[Path]:
    """entries: {label, coverage, risk, aurc, index}."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for e in entries:
        st = style_for(e["index"])
        cov, risk = np.asarray(e["coverage"]), np.asarray(e["risk"])
        ax.plot(cov, risk, color=st["color"], label=f"{e['label']}  (AURC = {e['aurc']:.4f})")
        if len(cov):
            pick = np.unique(np.clip(np.searchsorted(cov, [0.25, 0.5, 0.75, 1.0]), 0, len(cov) - 1))
            ax.plot(cov[pick], risk[pick], ls="none", marker=st["marker"], ms=6, color=st["color"],
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (fraction of events accepted, most confident first)")
    ax.set_ylabel("Selective risk (error rate on accepted events)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, stem)


# --------------------------------------------------------------------------- #
# O3a diagnostic extras
# --------------------------------------------------------------------------- #
def plot_confidence_by_category(groups: Sequence[tuple[str, np.ndarray]], stem: str | Path, title: str) -> list[Path]:
    """Model confidence for shared-correct / shared-incorrect / legacy / new-class events (one step line each)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = np.linspace(0, 1, 26)
    for k, (label, conf) in enumerate(groups):
        if len(conf) == 0:
            continue
        st = style_for(k)
        hist, _ = np.histogram(conf, bins=bins)
        frac = hist / max(len(conf), 1)
        ax.step(bins[:-1], frac, where="post", color=st["color"], label=f"{label} (n={len(conf)})", lw=1.8)
        ax.plot((bins[:-1] + bins[1:])[::5] / 2, frac[::5], ls="none", marker=st["marker"], color=st["color"], ms=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model confidence (max softmax over the trained classes)")
    ax.set_ylabel("Fraction of events in group")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, stem)


def plot_predicted_as(per_label: dict[str, dict[str, int]], stem: str | Path, title: str, top_k: int = 7) -> list[Path]:
    """Where events with labels outside the trained taxonomy are sent. Stacked horizontal bars of fractions."""
    apply_style()
    labels = list(per_label)
    if not labels:
        return []
    totals: dict[str, int] = {}
    for counts in per_label.values():
        for c, n in counts.items():
            totals[c] = totals.get(c, 0) + n
    top = [c for c, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top_k]]
    cols = top + (["Other trained classes"] if len(totals) > len(top) else [])
    fig, ax = plt.subplots(figsize=(8.0, 1.0 + 0.6 * len(labels)))
    left = np.zeros(len(labels))
    for k, col in enumerate(cols):
        vals = []
        for lbl in labels:
            counts, n = per_label[lbl], max(sum(per_label[lbl].values()), 1)
            v = sum(n_ for c, n_ in counts.items() if c not in top) if col == "Other trained classes" else counts.get(col, 0)
            vals.append(v / n)
        vals = np.asarray(vals)
        color = "#bdbdbd" if col == "Other trained classes" else style_for(k)["color"]
        ax.barh(labels, vals, left=left, color=color, edgecolor="white", linewidth=1.0, label=col)
        for i, v in enumerate(vals):
            if v >= 0.08:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5, color=INK)
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of events with this reference label, by predicted (trained) class")
    ax.set_yticks(range(len(labels)), [f"{l} (n={sum(per_label[l].values())})" for l in labels])
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25 - 0.05 * (4 - min(len(labels), 4))), ncol=4, fontsize=7)
    fig.tight_layout()
    return save(fig, stem)
