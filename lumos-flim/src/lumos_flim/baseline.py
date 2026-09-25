"""Independent per-pixel maximum-likelihood fits with the same physics.

    python -m lumos_flim.baseline --data data/synthetic.zarr --checkpoint checkpoints/<run>/best.ckpt

The reference point for the VAE: each pixel is fitted on its own, with the
same forward model, parametrisation, initialisation and Poisson likelihood,
but nothing is shared between pixels. The IRF (and the component spectra, for
multichannel data) are taken from a trained checkpoint or a calibration file
so the two methods differ only in how per-pixel parameters are estimated.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from lumos_flim.data import open_store
from lumos_flim.physics import decay_histograms, expected_counts
from lumos_flim.predict import derived, gt_report, summarise
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule, n_free_parameters, pearson_chi2, poisson_half_deviance


def fit_pixels(counts, model: FlimVAE, steps=1500, lr=0.05, chunk=20000):
    """Adam on every pixel at once. Pixels are independent, so the summed loss
    is separable and one optimiser is equivalent to one per pixel."""
    Fn = model.n_components
    C = counts.shape[1]
    t0, sigma = model.irf_t0.detach(), model.irf_sigma.detach()
    bases = model.bases.detach() if model.bases is not None else None
    rate_bias = model.decoder.head_rate.bias.detach()
    frac_bias = model.decoder.head_fraction.bias.detach()

    results = []
    for start in range(0, len(counts), chunk):
        x = torch.as_tensor(counts[start:start + chunk], dtype=torch.float32)
        B = len(x)
        totals = x.sum((1, 2)).clamp(min=1.0)
        rate_raw = rate_bias.expand(B, Fn).clone().requires_grad_()
        frac_logits = frac_bias.expand(B, Fn + 1).clone().requires_grad_()
        params = [rate_raw, frac_logits]
        bg_logits = None
        if C > 1:
            bg_logits = torch.zeros(B, C, requires_grad=True)
            params.append(bg_logits)
        opt = torch.optim.Adam(params, lr=lr)

        def forward():
            rates = model.rate_min + torch.cumsum(F.softplus(rate_raw), -1)
            probs = F.softmax(frac_logits, -1)
            bg_spec = F.softmax(bg_logits, -1) if bg_logits is not None else None
            decays = decay_histograms(rates, t0, sigma, model.n_bins, model.bin_width)
            mu = expected_counts(totals, probs[:, :-1], probs[:, -1], decays, bases, bg_spec)
            return rates, probs, mu

        for _ in range(steps):
            opt.zero_grad()
            _, _, mu = forward()
            poisson_half_deviance(x, mu).sum().backward()
            opt.step()
        with torch.no_grad():
            rates, probs, mu = forward()
            results.append((rates, probs, pearson_chi2(x, mu)))

    rates = torch.cat([r[0] for r in results])
    probs = torch.cat([r[1] for r in results])
    chi2 = torch.cat([r[2] for r in results])
    out = {k: v.numpy() for k, v in derived(rates, probs[:, :-1]).items()}
    out.update(fractions=probs[:, :-1].numpy(), background=probs[:, -1].numpy(),
               chi2r=(chi2 / (C * model.n_bins - n_free_parameters(Fn, C))).numpy())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", default="", help="take IRF and spectra from a trained model")
    p.add_argument("--irf", default="", help="or from a calibration JSON (single channel only)")
    p.add_argument("--irf_fit", default="fixed", choices=["fixed", "free"])
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    ds, meta = open_store(a.data)
    counts = ds["counts"].values
    if a.checkpoint:
        model = FlimModule.load_from_checkpoint(a.checkpoint, map_location="cpu").model
    elif a.irf:
        with open(a.irf) as f:
            cal = json.load(f)[a.irf_fit]
        model = FlimVAE(n_bins=counts.shape[2], bin_width=meta["bin_width_ns"],
                        n_channels=counts.shape[1], n_components=a.n_components,
                        irf_t0=cal["irf_t0"], irf_sigma=cal["irf_sigma"])
        if counts.shape[1] > 1:
            raise SystemExit("multichannel data needs --checkpoint for the component spectra")
    else:
        raise SystemExit("give --checkpoint or --irf")

    result = fit_pixels(counts, model, steps=a.steps)
    gt_report(result, ds)
    summarise(result, ds, meta, a.out or None)
    if a.out:
        np.savez_compressed(f"{a.out}.npz", image=ds["image"].values, y=ds["y"].values,
                            x=ds["x"].values, split=ds["split"].values, **result)
        print(f"wrote {a.out}.npz")


if __name__ == "__main__":
    main()
