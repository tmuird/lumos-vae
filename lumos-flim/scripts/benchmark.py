"""Every method on the same data, scored on held-out photons and, where known, the truth.

    python scripts/benchmark.py --dataset realistic --photons 50 100 250 1000
    python scripts/benchmark.py --dataset embryo --fractions 0.2 0.1 0.05

Protocol, identical for synthetic and real data: each pixel's photons are
split at random into a fitting part and a held-out part (binomial thinning,
which for Poisson data is exactly two independent shorter acquisitions).
Every method fits the fitting part; each is scored on how well its fitted
histograms predict the held-out photons (negative log-likelihood per photon,
lower is better), which needs no ground truth. On synthetic data the fitted
parameters are also scored against the truth, separately for interior and
region-boundary pixels. On the embryo, mean lifetimes are compared with
per-pixel MLE on all the photons.

All scores are on the val pixels, which neither VAE was fitted on.

Methods: vae, vae_spatial (encoder also sees the summed 3x3 neighbourhood),
vae_stack (encoder sees the 8 neighbours individually), mle,
global, binned (3x3), tv (edge-preserving spatial penalty), eb (hierarchical
empirical Bayes). All fit the IRF from the data.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lumos_flim.baseline import PixelFitter, run_method
from lumos_flim.data import (SPLITS, assign_splits, irf_t0_guess, open_store, read_imspector_tiff,
                             spatial_bin, spatial_context, write_store)
from lumos_flim.predict import derived, run_model
from lumos_flim.synthetic import simulate_realistic
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule

METHODS = ["vae", "vae_spatial", "vae_stack", "eb", "tv", "binned", "global", "mle"]
SPATIAL_MODE = {"vae_spatial": "sum", "vae_stack": "stack"}
# Directory holding the FLUTE files (zenodo_8046636 in phasorpy-data).
FLUTE_DIR = os.environ.get("FLUTE_DIR", "/home/user/phasorpy/phasorpy-data/zenodo_8046636")
EMBRYO = os.path.join(FLUTE_DIR, "Embryo.tif")


def heldout_nll(pred, held):
    p = pred / np.maximum(pred.sum((1, 2), keepdims=True), 1e-12)
    return float((-(held * np.log(np.maximum(p, 1e-12))).sum((1, 2))).sum() / max(held.sum(), 1))


def _iqr(v):
    return float(np.subtract(*np.percentile(v, [75, 25])))


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 else 0.0


def score_truth(res, gt, mask):
    """Parameter errors against ground truth on the pixels in ``mask``."""
    tau, gtau = res["tau"][mask], gt["gt_tau"][mask]
    gamp = derived(torch.as_tensor(1 / gtau), torch.as_tensor(gt["gt_fraction"][mask]))["tau_amp"].numpy()
    amp_rel = (res["tau_amp"][mask] - gamp) / gamp
    long_rel = (tau[:, 0] - gtau[:, 0]) / gtau[:, 0]
    short_rel = (tau[:, 1] - gtau[:, 1]) / gtau[:, 1]
    return dict(
        tau_amp_bias=float(np.median(amp_rel)), tau_amp_iqr=_iqr(amp_rel), tau_amp_r=_r(res["tau_amp"][mask], gamp),
        tau_long_bias=float(np.median(long_rel)), tau_long_iqr=_iqr(long_rel), tau_long_r=_r(tau[:, 0], gtau[:, 0]),
        tau_short_iqr=_iqr(short_rel),
        alpha_mae=float(np.median(np.abs(res["alpha"][mask, 0] - gt["gt_alpha"][mask, 0]))),
    )


def train_vae(store, run, epochs, spatial_mode=None):
    shutil.rmtree(f"checkpoints/{run}", ignore_errors=True)
    args = [sys.executable, "-m", "lumos_flim.train", "--data", store, "--max_epochs", str(epochs),
            "--batch_size", "256", "--kl_warmup_epochs", "20", "--run_name", run]
    if spatial_mode:
        args += ["--spatial", "3", "--spatial_mode", spatial_mode]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return FlimModule.load_from_checkpoint(f"checkpoints/{run}/best.ckpt", map_location="cpu")


def run_all(tag, kept, held, bin_width, meta_img, split, gt, reference, methods, epochs, out_rows,
            maps):
    image, y, x, shapes = meta_img
    store = f"data/bench_{tag}.zarr"
    write_store(store, kept, bin_width, image, y, x, [tag], shapes, split, gt=gt,
                attrs=dict(source=f"benchmark {tag}"))
    ds, meta = open_store(store)
    val = split == SPLITS["val"]
    model = FlimVAE(n_bins=kept.shape[-1], bin_width=bin_width,
                    irf_t0=irf_t0_guess(kept, bin_width), irf_sigma=0.1)
    base = None
    for method in methods:
        t = time.time()
        if method.startswith("vae"):
            mode = SPATIAL_MODE.get(method)
            module = train_vae(store, f"bench_{tag}_{method}", epochs, mode)
            ctx = spatial_context(kept, image, y, x, shapes, 3, mode) if mode else None
            res = run_model(module, kept, context=ctx)
            mu = res.pop("expected")
        else:
            if method in ("binned", "tv", "eb") and base is None:
                base = PixelFitter(kept, model).fit(1500, fit_irf=True)
            res, mu = run_method(method, kept, model, ds, meta, irf_from=base)
        row = dict(tag=tag, photons=float(np.median(kept[val].sum((1, 2)))), method=method,
                   seconds=round(time.time() - t), heldout_nll=heldout_nll(mu[val], held[val]))
        if gt is not None:
            real = val & (gt["gt_empty"] < 0.5)
            row.update({f"all_{k}": v for k, v in score_truth(res, gt, real).items()})
            row.update({f"edge_{k}": v for k, v in score_truth(res, gt, real & (gt["gt_boundary"] > 0.5)).items()})
            row.update({f"interior_{k}": v for k, v in score_truth(res, gt, real & (gt["gt_boundary"] < 0.5)).items()})
        if reference is not None:
            d = (res["tau_amp"][val] - reference[val]) / reference[val]
            row.update(ref_dev_median=float(np.median(d)), ref_dev_iqr=_iqr(d),
                       ref_r=_r(res["tau_amp"][val], reference[val]))
        out_rows.append(row)
        maps[f"{tag}/{method}"] = dict(tau_amp=res["tau_amp"], tau=res["tau"], alpha=res["alpha"])
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                          if not k.startswith(("edge_", "interior_"))}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["realistic", "embryo"], required=True)
    p.add_argument("--photons", type=float, nargs="+", default=[50, 100, 250, 1000],
                   help="realistic: photons per pixel in the fitting half")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.1, 0.05],
                   help="embryo: fraction of photons kept for fitting")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--suffix", default="", help="appended to the output file names, for split runs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows, maps = [], {}
    rows_path = out / f"{a.dataset}{a.suffix}.json"

    if a.dataset == "realistic":
        for n in a.photons:
            # Simulate twice the budget and split it in half: fit on one half,
            # score on the other.
            counts, bw, y, x, gt, _ = simulate_realistic(photons=2 * n, seed=a.seed)
            kept = rng.binomial(counts, 0.5)
            held = counts - kept
            split = assign_splits(len(counts), 0.1, 0.1, a.seed)
            maps["grid"] = dict(y=y, x=x, shape=(96, 96), gt=gt)
            run_all(f"realistic_p{int(n)}", kept, held, bw, (np.zeros(len(y)), y, x, [(96, 96)]),
                    split, gt, None, a.methods, a.epochs, rows, maps)
            json.dump(rows, open(rows_path, "w"), indent=1)
    else:
        counts, bw = read_imspector_tiff(EMBRYO)
        counts = spatial_bin(counts, 2)
        T, Y, X = counts.shape
        flat = counts.reshape(T, -1).T
        keep = np.flatnonzero(flat.sum(1) >= 1000)
        full = flat[keep][:, None, :].astype(np.int64)
        y, x = np.divmod(keep, X)
        split = assign_splits(len(full), 0.1, 0.1, a.seed)
        model = FlimVAE(n_bins=T, bin_width=bw, irf_t0=irf_t0_guess(full, bw), irf_sigma=0.1)
        reference = PixelFitter(full, model).fit(1500, fit_irf=True).results()[0]["tau_amp"]
        maps["grid"] = dict(y=y, x=x, shape=(Y, X), reference=reference)
        for f in a.fractions:
            kept = rng.binomial(full, f)
            run_all(f"embryo_f{f:g}", kept, full - kept, bw, (np.zeros(len(y)), y, x, [(Y, X)]),
                    split, None, reference, a.methods, a.epochs, rows, maps)
            json.dump(rows, open(rows_path, "w"), indent=1)
    np.save(out / f"{a.dataset}{a.suffix}_maps.npy", maps, allow_pickle=True)


if __name__ == "__main__":
    main()
