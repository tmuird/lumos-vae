"""Figures for the synthetic photon sweep and the embryo thinning study.

    python scripts/plot_evaluation.py

Reads results/sweep/sweep.json, results/thinning/thinning.json and
results/thinning/maps.npy, writes PNGs to docs/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots 1-3 of the reference palette, validated all-pairs for CVD
# and normal vision. Aqua is under 3:1 on white, so every series also has its
# own marker and a direct label.
METHODS = {
    "VAE": dict(color="#2a78d6", marker="o", label="VAE (amortised)"),
    "MLE": dict(color="#eb6834", marker="s", label="per-pixel MLE"),
    "global": dict(color="#1baf7a", marker="^", label="global analysis"),
}
REFERENCE = "#8a8984"  # neutral ink for raw data and the constant predictor
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
# One hue, light to dark, for lifetime maps. Starts off-white so the lowest
# values stay visible against the page.
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#d6e6f7", "#7fb0e8", "#2a78d6", "#123f75"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False,
})


def _series(ax, rows, key, x="photons", label_end=False, skip=()):
    ends = []
    for m, style in METHODS.items():
        if m in skip:
            continue
        pts = sorted((r[x], r[key]) for r in rows
                     if r["method"] == m and key in r and np.isfinite(r[key]))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], lw=2, ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=style["label"], zorder=3)
        ends.append((xs[-1], ys[-1], m))
    if label_end:
        _end_labels(ax, ends)


def _end_labels(ax, ends, gap_px=11):
    """Direct labels at the right end of each line, nudged apart vertically."""
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    placed = sorted(((to_px((x, y))[1], x, y, m) for x, y, m in ends))
    last = -np.inf
    for py, x, y, m in placed:
        py = max(py, last + gap_px)
        last = py
        dy = py - to_px((x, y))[1]
        ax.annotate(m, (x, y), xytext=(7, dy), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8)


def _reference(ax, xs, ys, text):
    ax.plot(xs, ys, color=REFERENCE, lw=1.5, ls="--", zorder=2)
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=8)


def sweep_figure(path_json, out):
    rows = json.load(open(path_json))
    fits = [r for r in rows if r["method"] in METHODS]
    raw = sorted((r["photons"], r["denoise"]) for r in rows if r["method"] == "raw data")
    photons = sorted({r["photons"] for r in fits})

    panels = [
        ("tau_amp_iqr", "Mean lifetime: spread of error", "IQR of relative error (lower is better)", "const_amp"),
        ("tau_amp_r", "Mean lifetime: tracks the truth", "correlation with truth (higher is better)", None),
        ("tau_long_r", "Long lifetime: tracks the truth", "correlation with truth", None),
        ("tau_short_iqr", "Short lifetime: spread of error", "IQR of relative error", None),
        ("alpha_mae", "Long-component amplitude fraction", "median absolute error", None),
        ("denoise", "Reconstruction against noise-free truth", "mean (μ̂ − μ)² / μ per bin (log)", "raw"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for ax, (key, title, ylabel, ref) in zip(axes.flat, panels):
        skip = ("global",) if key == "tau_long_r" else ()
        if skip:
            ax.text(0.02, 0.98, "global: one shared lifetime,\nno per-pixel value",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK_2)
        ax.set_xscale("log")
        if ref == "raw":
            ax.set_yscale("log")
        # References first, so the axis limits are final when labels are placed.
        if ref == "raw":
            _reference(ax, *zip(*raw), "raw data")
        if ref == "const_amp":
            _reference(ax, photons, [0.33] * len(photons), "constant")
        _series(ax, fits, key, label_end=True, skip=skip)
        ax.set_xticks(photons, [f"{int(p)}" for p in photons])
        ax.minorticks_off()
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("photons per pixel")
        ax.margins(x=0.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Synthetic data, 50 to 1000 photons per pixel (held-out pixels)", x=0.01,
                 ha="left", y=0.995, fontsize=11, color=INK, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def thinning_figure(path_json, out):
    rows = json.load(open(path_json))
    fits = [r for r in rows if r["method"] in METHODS]
    raw = sorted((r["photons"], r["heldout_nll"]) for r in rows
                 if r["method"] == "raw data" and "heldout_nll" in r)
    # Held-out NLL relative to the best method at each level, so the gaps are visible.
    best = {}
    for r in fits:
        if "heldout_nll" in r:
            best[r["photons"]] = min(best.get(r["photons"], np.inf), r["heldout_nll"])
    for r in fits + [r for r in rows if r["method"] == "raw data"]:
        if "heldout_nll" in r:
            r["excess_nll"] = (r["heldout_nll"] - best[r["photons"]]) * 1000

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax in axes:
        ax.set_xscale("log")
    ax = axes[0]
    _series(ax, fits, "excess_nll", label_end=True)
    ax.set_title("Predicting photons it never saw", loc="left")
    ax.set_ylabel("held-out NLL above best, millinats/photon")
    raw_ex = sorted((r["photons"], r["excess_nll"]) for r in rows
                    if r["method"] == "raw data" and "excess_nll" in r)
    ax.text(0.02, 0.97, "raw thinned histogram: " + ", ".join(f"+{v:.0f}" for _, v in raw_ex),
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK_2)
    ax = axes[1]
    _series(ax, fits, "tau_amp_r", label_end=True)
    ax.set_title("Mean lifetime vs full-count reference", loc="left")
    ax.set_ylabel("correlation with MLE on all photons")
    ax = axes[2]
    _series(ax, fits, "tau_amp_dev_iqr", label_end=True)
    ax.set_title("Mean lifetime: spread of deviation", loc="left")
    ax.set_ylabel("IQR of relative deviation")
    photons = sorted({r["photons"] for r in fits})
    for ax in axes:
        ax.set_xticks(photons, [f"{int(p)}" for p in photons])
        ax.minorticks_off()
        ax.set_xlabel("photons per pixel (after thinning)")
        ax.margins(x=0.14)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle("Zebrafish embryo (FLUTE), photons thinned from ~1300 per pixel", x=0.01,
                 ha="left", y=0.995, fontsize=11, color=INK, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def maps_figure(path_maps, out, fractions=(1.0, 0.2, 0.05)):
    maps = np.load(path_maps, allow_pickle=True).item()
    ys, xs, shape = maps["y"], maps["x"], maps["shape"]
    rows = ["VAE", "MLE", "global"]
    ref = maps["MLE_1"]["tau_amp"]
    lo, hi = np.percentile(ref, [2, 98])
    # Crop to the pixels that were kept.
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fig, axes = plt.subplots(len(rows), len(fractions), figsize=(3.3 * len(fractions), 3.0 * len(rows)))
    for i, m in enumerate(rows):
        for j, f in enumerate(fractions):
            img = np.full(shape, np.nan)
            img[ys, xs] = maps[f"{m}_{f:g}"]["tau_amp"]
            ax = axes[i, j]
            im = ax.imshow(img[y0:y1, x0:x1], cmap=SEQ, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_facecolor("#f0efec")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                photons = np.median(maps[f"counts_{f:g}"])
                ax.set_title(f"{f:.0%} of photons (~{photons:.0f}/pixel)", fontsize=9)
            if j == 0:
                ax.set_ylabel(METHODS[m]["label"], fontsize=9, color=INK)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("amplitude-weighted lifetime (ns)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle("Embryo mean lifetime as photons are removed", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def examples_figure(path_maps, out, f=0.05, n=3):
    maps = np.load(path_maps, allow_pickle=True).item()
    ex = maps[f"examples_{f:g}"]
    t = np.arange(ex["kept"].shape[-1])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), sharey=False)
    for j in range(n):
        ax = axes[j]
        kept, held = ex["kept"][j], ex["held"][j]
        # Held-out photons rescaled to the kept total, as an independent look at the shape.
        ax.plot(t, held * kept.sum() / max(held.sum(), 1), color=REFERENCE, lw=1.2,
                label="held-out photons (rescaled)", zorder=1)
        ax.bar(t, kept, width=0.9, color="#d9d8d3", label="kept photons", zorder=0)
        for m, style in METHODS.items():
            ax.plot(t, ex[m][j], color=style["color"], lw=2, label=style["label"], zorder=3)
        ax.set_title(f"pixel {ex['idx'][j]}: {int(kept.sum())} kept, {int(held.sum())} held out",
                     loc="left", fontsize=9)
        ax.set_xlabel("delay bin")
        ax.set_ylabel("counts")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    Path("docs").mkdir(exist_ok=True)
    if Path("results/sweep/sweep.json").exists():
        sweep_figure("results/sweep/sweep.json", "docs/eval_synthetic_sweep.png")
    if Path("results/thinning/thinning.json").exists():
        thinning_figure("results/thinning/thinning.json", "docs/eval_embryo_thinning.png")
    if Path("results/thinning/maps.npy").exists():
        maps_figure("results/thinning/maps.npy", "docs/eval_embryo_maps.png")
        examples_figure("results/thinning/maps.npy", "docs/eval_embryo_fits.png")


if __name__ == "__main__":
    main()
