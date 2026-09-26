"""Does modelling the IRF's diffusion tail remove the lifetime bias?

    python scripts/irf_tail_study.py --photons 100 250

On the realistic synthetic data (whose IRF has an exponential tail on 25% of
photons), fits the VAE and per-pixel MLE with the plain Gaussian IRF and with
the tailed IRF, on exactly the photon split scripts/benchmark.py uses, and
scores both against ground truth and held-out photons.
"""

import argparse
import json
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, "scripts")
from benchmark import heldout_nll, score_truth  # noqa: E402

from lumos_flim.baseline import PixelFitter
from lumos_flim.data import SPLITS, assign_splits, irf_t0_guess, write_store
from lumos_flim.predict import run_model
from lumos_flim.synthetic import simulate_realistic
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule


def benchmark_split(photons, seed=0, levels=(50, 100, 250, 1000)):
    """Reproduce benchmark.py's random stream up to the requested level."""
    rng = np.random.default_rng(seed)
    for n in levels:
        counts, bw, y, x, gt, _ = simulate_realistic(photons=2 * n, seed=seed)
        kept = rng.binomial(counts, 0.5)
        if n == photons:
            return kept, counts - kept, bw, y, x, gt
    raise ValueError(photons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--photons", type=int, nargs="+", default=[100, 250])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--irfs", nargs="+", default=["gaussian", "tailed"], choices=["gaussian", "tailed"],
                   help="the Gaussian fits repeat benchmark.py on the same split, so can be skipped")
    p.add_argument("--out", default="results/benchmark/irf_tail.json")
    a = p.parse_args()
    rows = []
    for n in a.photons:
        kept, held, bw, y, x, gt = benchmark_split(n)
        split = assign_splits(len(kept), 0.1, 0.1, 0)
        val = split == SPLITS["val"]
        real = val & (gt["gt_empty"] < 0.5)
        store = f"data/tail_p{n}.zarr"
        write_store(store, kept, bw, np.zeros(len(y)), y, x, ["tail"], [(96, 96)], split, gt=gt)
        for tail in [irf == "tailed" for irf in a.irfs]:
            run = f"tail_p{n}_{'tail' if tail else 'gauss'}"
            shutil.rmtree(f"checkpoints/{run}", ignore_errors=True)
            args = [sys.executable, "-m", "lumos_flim.train", "--data", store, "--max_epochs",
                    str(a.epochs), "--batch_size", "256", "--kl_warmup_epochs", "20", "--run_name", run]
            if tail:
                args.append("--irf_tail")
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            module = FlimModule.load_from_checkpoint(f"checkpoints/{run}/best.ckpt", map_location="cpu")
            res = run_model(module, kept)
            mu = res.pop("expected")
            w, q = module.model.irf_tail_params
            rows.append(dict(photons=n, method="VAE", irf="tailed" if tail else "gaussian",
                             tail_weight=None if w is None else w.item(),
                             tail_tau=None if q is None else 1 / q.item(),
                             heldout_nll=heldout_nll(mu[val], held[val]), **score_truth(res, gt, real)))
            print(json.dumps(rows[-1]), flush=True)

            model = FlimVAE(n_bins=kept.shape[-1], bin_width=bw, irf_t0=irf_t0_guess(kept, bw),
                            irf_sigma=0.1, irf_tail=tail)
            f = PixelFitter(kept, model).fit(1500, fit_irf=True)
            res, mu = f.results()
            rows.append(dict(photons=n, method="MLE", irf="tailed" if tail else "gaussian",
                             heldout_nll=heldout_nll(mu[val], held[val]), **score_truth(res, gt, real)))
            print(json.dumps(rows[-1]), flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
