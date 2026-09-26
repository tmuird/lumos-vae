"""Low-photon sweep: the VAE against the direct per-pixel MLE, with ground truth.

    python scripts/photon_sweep.py --photons 50 100 200 500 1000 --out results/sweep

For each photon budget the same synthetic maps are simulated photon by photon,
each method is fitted, and the held-out val pixels (never seen by the VAE)
are scored on
  - lifetime and amplitude-fraction errors,
  - amplitude-weighted mean lifetime error, the quantity usually reported
    at low counts,
  - denoising: mean over bins of (mu_hat - mu_true)^2 / mu_true, where mu_true
    is the noise-free histogram. The raw data scores about 1 on this, so below
    1 means the reconstruction is closer to the truth than the measurement.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from lumos_flim.baseline import fit_pixels
from lumos_flim.data import SPLITS, irf_t0_guess, open_store
from lumos_flim.physics import decay_histograms, expected_counts
from lumos_flim.predict import derived, run_model
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule


def sh(args):
    subprocess.run([sys.executable, "-m", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def truth(ds, meta):
    counts = torch.as_tensor(ds["counts"].values, dtype=torch.float32)
    tau = torch.as_tensor(ds["gt_tau"].values, dtype=torch.float32)
    decays = decay_histograms(1.0 / tau, meta["gt_irf_t0"], meta["gt_irf_sigma"],
                              counts.shape[-1], meta["bin_width_ns"])
    return expected_counts(counts.sum((1, 2)), torch.as_tensor(ds["gt_fraction"].values),
                           torch.as_tensor(ds["gt_background"].values), decays).numpy()


@torch.no_grad()
def recon_flimvae(module, counts):
    return module.model(torch.as_tensor(counts, dtype=torch.float32), sample=False)["expected"].numpy()



def score(name, res, recon, ds, mu_true, held):
    gt_tau = ds["gt_tau"].values[held]
    gt_alpha = ds["gt_alpha"].values[held]
    gt = derived(torch.as_tensor(1.0 / gt_tau), torch.as_tensor(ds["gt_fraction"].values[held]))
    tau, alpha = res["tau"][held], res["alpha"][held]
    rel = (tau - gt_tau) / gt_tau
    tau_amp_rel = (res["tau_amp"][held] - gt["tau_amp"].numpy()) / gt["tau_amp"].numpy()
    mt = mu_true[held]
    denoise = ((recon[held] - mt) ** 2 / np.maximum(mt, 1e-9)).mean((1, 2))
    return dict(
        method=name,
        tau_long_bias=float(np.median(rel[:, 0])),
        tau_long_iqr=float(np.subtract(*np.percentile(rel[:, 0], [75, 25]))),
        tau_short_bias=float(np.median(rel[:, 1])),
        tau_short_iqr=float(np.subtract(*np.percentile(rel[:, 1], [75, 25]))),
        alpha_mae=float(np.median(np.abs(alpha[:, 0] - gt_alpha[:, 0]))),
        tau_amp_bias=float(np.median(tau_amp_rel)),
        tau_amp_iqr=float(np.subtract(*np.percentile(tau_amp_rel, [75, 25]))),
        denoise=float(np.median(denoise)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--photons", type=float, nargs="+", default=[50, 100, 200, 500, 1000])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out", default="results/sweep")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = []

    for n in a.photons:
        tag = f"p{int(n)}"
        store = f"data/sweep_{tag}.zarr"
        sh(["lumos_flim.synthetic", "--out", store, "--photons", str(n)])
        ds, meta = open_store(store)
        counts = ds["counts"].values
        held = ds["split"].values == SPLITS["val"]
        mu_true = truth(ds, meta)
        raw_score = float(np.median(((counts[held] - mu_true[held]) ** 2
                                     / np.maximum(mu_true[held], 1e-9)).mean((1, 2))))

        sh(["lumos_flim.train", "--data", store, "--max_epochs", str(a.epochs),
            "--batch_size", "256", "--run_name", f"sweep_{tag}_vae"])
        vae = FlimModule.load_from_checkpoint(f"checkpoints/sweep_{tag}_vae/best.ckpt", map_location="cpu")
        rows.append(dict(photons=n, **score("VAE", run_model(vae, counts),
                                            recon_flimvae(vae, counts), ds, mu_true, held)))

        init = FlimVAE(n_bins=counts.shape[2], bin_width=meta["bin_width_ns"],
                       irf_t0=irf_t0_guess(counts, meta["bin_width_ns"]), irf_sigma=0.1)
        res, rec = fit_pixels(counts, init, fit_irf=True)
        rows.append(dict(photons=n, **score("MLE", res, rec, ds, mu_true, held)))
        rows.append(dict(photons=n, method="raw data", denoise=raw_score))

        with open(f"{a.out}/sweep.json", "w") as f:
            json.dump(rows, f, indent=1)
        for r in rows[-3:]:
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
