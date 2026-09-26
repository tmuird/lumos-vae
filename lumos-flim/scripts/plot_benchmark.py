"""Figures for scripts/benchmark.py.

    python scripts/plot_benchmark.py

Reads results/benchmark/{realistic,embryo}.json and *_maps.npy, writes
docs/bench_*.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_evaluation import INK, INK_2, SEQ, SURFACE  # shared style

# The reference categorical palette in its fixed slot order (never cycled),
# validated on adjacent pairs. With eight series, end labels collide, so
# identity is carried by the legend and a distinct marker per series; the
# README tables are the table view that the low-contrast slots require.
STYLE = {
    "vae": ("#2a78d6", "o", "VAE"),
    "vae_spatial": ("#eb6834", "D", "VAE + summed 3x3"),
    "vae_stack": ("#1baf7a", "^", "VAE + stacked 3x3"),
    "eb": ("#eda100", "v", "empirical Bayes"),
    "tv": ("#e87ba4", "s", "TV-regularised"),
    "binned": ("#008300", "P", "3x3 binned"),
    "global": ("#4a3aa7", "X", "global lifetimes"),
    "mle": ("#e34948", "*", "per-pixel MLE"),
}
SHORT = {"vae": "VAE", "vae_spatial": "VAE+sum", "vae_stack": "VAE+stack", "eb": "EB", "tv": "TV", "binned": "binned",
         "global": "global", "mle": "MLE"}


def _lines(ax, rows, key, skip=()):
    for m, (c, mk, label) in STYLE.items():
        if m in skip:
            continue
        pts = sorted((r["photons"], r[key]) for r in rows if r["method"] == m and key in r
                     and np.isfinite(r[key]))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c, marker=mk, lw=2, ms=7 if mk != "*" else 10,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=3)


def _axes_common(ax, photons, title, ylabel):
    ax.set_xticks(photons, [f"{int(p)}" for p in photons])
    ax.minorticks_off()
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("photons per pixel (fitting half)")
    ax.margins(x=0.16)


def _excess_nll(rows):
    best = {}
    for r in rows:
        best[r["photons"]] = min(best.get(r["photons"], np.inf), r["heldout_nll"])
    for r in rows:
        r["excess_nll"] = 1000 * (r["heldout_nll"] - best[r["photons"]])


def realistic_figure(rows, out):
    _excess_nll(rows)
    photons = sorted({r["photons"] for r in rows})
    panels = [
        ("excess_nll", "Predicting held-out photons", "NLL above best, millinats/photon", ()),
        ("all_tau_amp_iqr", "Mean lifetime: error spread", "IQR of relative error", ()),
        ("all_tau_amp_r", "Mean lifetime: tracks truth", "correlation with truth", ()),
        ("all_tau_long_r", "Long lifetime: tracks truth", "correlation with truth", ("global",)),
        ("all_alpha_mae", "Bound (long) amplitude fraction", "median absolute error", ()),
        ("edge_tau_amp_iqr", "Mean lifetime at region edges", "IQR of relative error, edge pixels", ()),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for ax, (key, title, ylabel, skip) in zip(axes.flat, panels):
        ax.set_xscale("log")
        _lines(ax, rows, key, skip)
        _axes_common(ax, photons, title, ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Realistic synthetic tissue: cells with sharp edges, tailed IRF (held-out pixels)",
                 x=0.01, ha="left", y=0.995, fontsize=11, color=INK, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=8, bbox_to_anchor=(0.005, 0.965), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def embryo_figure(rows, out):
    _excess_nll(rows)
    photons = sorted({r["photons"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax in axes:
        ax.set_xscale("log")
    panels = [
        ("excess_nll", "Predicting held-out photons", "NLL above best, millinats/photon"),
        ("ref_r", "Mean lifetime vs full-photon MLE", "correlation"),
        ("ref_dev_iqr", "Mean lifetime: deviation spread", "IQR of relative deviation"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        _lines(ax, rows, key)
        _axes_common(ax, photons, title, ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Zebrafish embryo, photons thinned from ~1250 per pixel (held-out pixels)", x=0.01,
                 ha="left", y=0.995, fontsize=11, color=INK, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.95), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def maps_figure(maps, tag, out, title, truth=None):
    g = maps["grid"]
    y, x, shape = g["y"], g["x"], tuple(g["shape"])
    ref = truth if truth is not None else g.get("reference")
    lo, hi = np.percentile(ref, [2, 98])
    y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
    panels = [("truth" if truth is not None else "MLE, all photons", ref)]
    panels += [(STYLE[m][2], maps[f"{tag}/{m}"]["tau_amp"]) for m in STYLE if f"{tag}/{m}" in maps]
    n = len(panels)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.3 * ((n + 1) // 2), 6.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, v) in zip(axes.flat, panels):
        img = np.full(shape, np.nan)
        img[y, x] = v
        im = ax.imshow(img[y0:y1, x0:x1], cmap=SEQ, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(name, fontsize=9, color=INK)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("amplitude-weighted lifetime (ns)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path("results/benchmark")
    Path("docs").mkdir(exist_ok=True)
    if (root / "realistic.json").exists():
        realistic_figure(json.load(open(root / "realistic.json")), "docs/bench_realistic.png")
    if (root / "embryo.json").exists():
        embryo_figure(json.load(open(root / "embryo.json")), "docs/bench_embryo.png")
    maps_file = next((root / f for f in ("realistic_b_maps.npy", "realistic_maps.npy")
                      if (root / f).exists()), None)
    if maps_file is not None:
        maps = np.load(maps_file, allow_pickle=True).item()
        gt = maps["grid"]["gt"]
        import torch
        from lumos_flim.predict import derived
        truth = derived(torch.as_tensor(1 / gt["gt_tau"]), torch.as_tensor(gt["gt_fraction"]))["tau_amp"].numpy()
        tag = next((t for t in ("realistic_p250", "realistic_p100") if f"{t}/vae" in maps),
                   sorted(k.split("/")[0] for k in maps if "/" in k)[0])
        maps_figure(maps, tag, "docs/bench_realistic_maps.png",
                    f"Mean lifetime maps, realistic tissue, {tag.split('_p')[1]} photons per pixel", truth)
    if (root / "embryo_maps.npy").exists():
        maps = np.load(root / "embryo_maps.npy", allow_pickle=True).item()
        tag = "embryo_f0.05" if "embryo_f0.05/vae" in maps else sorted(k.split("/")[0] for k in maps if "/" in k)[-1]
        maps_figure(maps, tag, "docs/bench_embryo_maps.png",
                    f"Embryo mean lifetime at {tag.split('_f')[1]} of the photons")


if __name__ == "__main__":
    main()
