"""The VAE with a spatial (MRF) latent prior, on the benchmark's own photon splits.

    python scripts/mrf_study.py --lams 0.3 1 3

Trains the stacked-context VAE with the latent MRF prior at each strength on
the realistic synthetic data (100 photons) and the embryo (5% of photons),
reproducing exactly the photon splits scripts/benchmark.py used, so the
held-out scores are directly comparable with its tables.
"""

import argparse
import json
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, "scripts")
from benchmark import EMBRYO, _iqr, _r, heldout_nll, score_truth  # noqa: E402
from irf_tail_study import benchmark_split  # noqa: E402

from lumos_flim.baseline import PixelFitter
from lumos_flim.data import (SPLITS, assign_splits, irf_t0_guess, read_imspector_tiff, spatial_bin,
                             spatial_context)
from lumos_flim.predict import run_model
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule


def embryo_split(fraction=0.05, fractions=(0.2, 0.1, 0.05), seed=0):
    """Reproduce benchmark.py's embryo thinning and reference."""
    counts, bw = read_imspector_tiff(EMBRYO)
    counts = spatial_bin(counts, 2)
    T, Y, X = counts.shape
    flat = counts.reshape(T, -1).T
    keep = np.flatnonzero(flat.sum(1) >= 1000)
    full = flat[keep][:, None, :].astype(np.int64)
    y, x = np.divmod(keep, X)
    rng = np.random.default_rng(seed)
    model = FlimVAE(n_bins=T, bin_width=bw, irf_t0=irf_t0_guess(full, bw), irf_sigma=0.1)
    reference = PixelFitter(full, model).fit(1500, fit_irf=True).results()[0]["tau_amp"]
    for f in fractions:
        kept = rng.binomial(full, f)
        if f == fraction:
            return kept, full - kept, bw, y, x, (Y, X), reference
    raise ValueError(fraction)


def run(store, kept, held, y, x, shape, lam, epochs, tag, size=3):
    run_name = f"mrf_{tag}_{lam:g}" + (f"_k{size}" if size != 3 else "")
    shutil.rmtree(f"checkpoints/{run_name}", ignore_errors=True)
    args = [sys.executable, "-m", "lumos_flim.train", "--data", store, "--max_epochs", str(epochs),
            "--batch_size", "512", "--kl_warmup_epochs", "20", "--spatial", str(size),
            "--spatial_mode", "stack", "--run_name", run_name]
    if lam > 0:
        args += ["--spatial_prior", str(lam)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ckpt = f"checkpoints/{run_name}/{'last' if lam > 0 else 'best'}.ckpt"
    module = FlimModule.load_from_checkpoint(ckpt, map_location="cpu")
    ctx = spatial_context(kept, np.zeros(len(y)), y, x, [shape], size, "stack")
    res = run_model(module, kept, context=ctx)
    return res, res.pop("expected")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--datasets", nargs="+", default=["realistic", "embryo"])
    p.add_argument("--window", type=int, default=3, help="encoder context window (3 or 5)")
    p.add_argument("--out", default="results/benchmark/mrf.json")
    a = p.parse_args()
    rows = []
    for name in a.datasets:
        if name == "realistic":
            kept, held, bw, y, x, gt = benchmark_split(100)
            shape, reference, store = (96, 96), None, "data/bench_realistic_p100.zarr"
        else:
            kept, held, bw, y, x, shape, reference = embryo_split(0.05)
            gt, store = None, "data/bench_embryo_f0.05.zarr"
        split = assign_splits(len(kept), 0.1, 0.1, 0)
        val = split == SPLITS["val"]
        for lam in a.lams:
            res, mu = run(store, kept, held, y, x, shape, lam, a.epochs, name, a.window)
            row = dict(dataset=name, lam=lam, window=a.window, heldout_nll=heldout_nll(mu[val], held[val]))
            if gt is not None:
                row.update(score_truth(res, gt, val & (gt["gt_empty"] < 0.5)))
            else:
                d = (res["tau_amp"][val] - reference[val]) / reference[val]
                row.update(ref_dev_median=float(np.median(d)), ref_dev_iqr=_iqr(d),
                           ref_r=_r(res["tau_amp"][val], reference[val]))
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
