"""Real data at varying noise levels, scored on held-out photons.

    python scripts/thinning_study.py --image Embryo.tif --fractions 1 0.5 0.2 0.1 0.05

Each photon of a pixel is kept with probability ``f`` (binomial thinning).
Thinning a Poisson process gives exactly the Poisson data of an acquisition
``f`` times as long, so this simulates nothing. The dropped photons are an
independent measurement of the same pixel, which gives a score that needs no
ground truth: how well does each method's fitted histogram predict photons
it never saw?

Scores, on the val pixels (never seen by the VAE):
  - held-out NLL per photon, -sum h log p / sum h, where p is the method's
    normalised histogram and h the dropped photons. Lower is better. The raw
    thinned histogram (with a +0.5 pseudo-count) is the no-model reference.
  - agreement of the amplitude-weighted mean lifetime with the per-pixel MLE
    on all the photons (f = 1), the best available stand-in for the truth.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from lumos_flim.baseline import fit_pixels
from lumos_flim.data import SPLITS, assign_splits, irf_t0_guess, read_imspector_tiff, spatial_bin, write_store
from lumos_flim.predict import run_model
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule


def heldout_nll(pred, held):
    p = pred / pred.sum((1, 2), keepdims=True)
    return -(held * np.log(np.maximum(p, 1e-12))).sum((1, 2)) / np.maximum(held.sum((1, 2)), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="/home/user/phasorpy/phasorpy-data/zenodo_8046636/Embryo.tif")
    p.add_argument("--bin", type=int, default=2)
    p.add_argument("--min_counts", type=float, default=1000)
    p.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1, 0.05])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--vae_args", default="")
    p.add_argument("--out", default="results/thinning")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    counts, bin_width = read_imspector_tiff(a.image)
    counts = spatial_bin(counts, a.bin)
    T, Y, X = counts.shape
    flat = counts.reshape(T, -1).T
    keep = np.flatnonzero(flat.sum(1) >= a.min_counts)
    full = flat[keep][:, None, :].astype(np.int64)  # [P, 1, T]
    yy, xx = np.divmod(keep, X)
    split = assign_splits(len(full), 0.1, 0.1, a.seed)
    val = split == SPLITS["val"]
    name = Path(a.image).stem
    print(f"{name}: {len(full)} pixels, median {np.median(full.sum((1, 2))):.0f} photons at f=1")

    rows, maps = [], {"y": yy, "x": xx, "shape": (Y, X), "val": val}
    reference = None
    for f in sorted(a.fractions, reverse=True):
        kept = rng.binomial(full, f) if f < 1 else full.copy()
        held = full - kept
        tag = f"{name}_f{f:g}"
        store = f"data/thin_{tag}.zarr"
        write_store(store, kept, bin_width, np.zeros(len(kept)), yy, xx, [name], [(Y, X)], split,
                    attrs=dict(source=f"{a.image} thinned to {f:g}"))
        init = FlimVAE(n_bins=T, bin_width=bin_width,
                       irf_t0=irf_t0_guess(kept, bin_width), irf_sigma=0.1)

        results = {}
        res, rec = fit_pixels(kept, init, fit_irf=True)
        results["MLE"] = (res, rec)
        res, rec = fit_pixels(kept, init, fit_irf=True, global_lifetimes=True)
        results["global"] = (res, rec)
        run = f"thin_{tag}_vae"
        shutil.rmtree(f"checkpoints/{run}", ignore_errors=True)
        subprocess.run([sys.executable, "-m", "lumos_flim.train", "--data", store, "--max_epochs",
                        str(a.epochs), "--batch_size", "256", "--run_name", run, *a.vae_args.split()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vae = FlimModule.load_from_checkpoint(f"checkpoints/{run}/best.ckpt", map_location="cpu")
        res = run_model(vae, kept)
        results["VAE"] = (res, res.pop("expected"))
        if reference is None:
            reference = results["MLE"][0]["tau_amp"]  # per-pixel MLE on every photon

        raw = kept + 0.5
        for method, (res, rec) in list(results.items()) + [("raw data", (None, raw))]:
            row = dict(fraction=f, photons=float(np.median(kept[val].sum((1, 2)))), method=method)
            if f < 1:
                row["heldout_nll"] = float(np.mean(heldout_nll(rec[val], held[val])))
            if res is not None:
                d = (res["tau_amp"][val] - reference[val]) / reference[val]
                row["tau_amp_dev_median"] = float(np.median(d))
                row["tau_amp_dev_iqr"] = float(np.subtract(*np.percentile(d, [75, 25])))
                row["tau_amp_r"] = float(np.corrcoef(res["tau_amp"][val], reference[val])[0, 1]) \
                    if np.std(res["tau_amp"][val]) > 1e-9 else 0.0
                row["chi2r"] = float(np.median(res["chi2r"][val]))
                maps[f"{method}_{f:g}"] = {k: res[k] for k in ("tau_amp", "tau", "alpha")}
            rows.append(row)
            print(json.dumps(row), flush=True)
        maps[f"counts_{f:g}"] = kept[:, 0].sum(-1)
        # A few example pixels for the fit plots.
        idx = np.flatnonzero(val)[:6]
        maps[f"examples_{f:g}"] = dict(idx=idx, kept=kept[idx, 0], held=held[idx, 0],
                                       **{m: results[m][1][idx, 0] for m in results})
        json.dump(rows, open(out / "thinning.json", "w"), indent=1)
    np.save(out / "maps.npy", maps, allow_pickle=True)
    print("reference: per-pixel MLE on all photons")


if __name__ == "__main__":
    main()
