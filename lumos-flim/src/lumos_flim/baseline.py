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
from lumos_flim.data import irf_t0_guess
from lumos_flim.device import pick_device
from lumos_flim.vae import _SIGMA_FLOOR, FlimVAE
from lumos_flim.vae_module import FlimModule, n_free_parameters, pearson_chi2, poisson_half_deviance


def fit_pixels(counts, model: FlimVAE, steps=1500, lr=0.05, chunk=20000, fit_irf=False,
               global_lifetimes=False, device="auto"):
    """Adam on every pixel at once. Pixels are independent, so the summed loss
    is separable and one optimiser is equivalent to one per pixel.

    With ``fit_irf`` the IRF centre and width are fitted as well, shared by
    every pixel, so all pixels go in one chunk. No network is involved: this
    is the direct, non-amortised fit of the physics model to the data.

    With ``global_lifetimes`` the lifetimes are shared by every pixel as well
    and only the fractions and background are per pixel: global analysis,
    the usual remedy for low counts.
    """
    Fn = model.n_components
    C = counts.shape[1]
    device = pick_device(device)
    t0 = model.irf_t0.detach().to(device).clone().requires_grad_(fit_irf)
    sigma_raw = model.irf_sigma_raw.detach().to(device).clone().requires_grad_(fit_irf)
    if fit_irf or global_lifetimes:
        chunk = len(counts)
    bases = model.bases.detach().to(device) if model.bases is not None else None
    rate_bias = model.decoder.head_rate.bias.detach().to(device)
    frac_bias = model.decoder.head_fraction.bias.detach().to(device)

    results = []
    for start in range(0, len(counts), chunk):
        x = torch.as_tensor(counts[start:start + chunk], dtype=torch.float32, device=device)
        B = len(x)
        totals = x.sum((1, 2)).clamp(min=1.0)
        rate_raw = (rate_bias[None] if global_lifetimes else rate_bias.expand(B, Fn)).clone().requires_grad_()
        frac_logits = frac_bias.expand(B, Fn + 1).clone().requires_grad_()
        params = [rate_raw, frac_logits] + ([t0, sigma_raw] if fit_irf else [])
        bg_logits = None
        if C > 1:
            bg_logits = torch.zeros(B, C, device=device, requires_grad=True)
            params.append(bg_logits)
        opt = torch.optim.Adam(params, lr=lr)

        def forward():
            rates = (model.rate_min + torch.cumsum(F.softplus(rate_raw), -1)).expand(B, Fn)
            probs = F.softmax(frac_logits, -1)
            bg_spec = F.softmax(bg_logits, -1) if bg_logits is not None else None
            sigma = F.softplus(sigma_raw) + _SIGMA_FLOOR
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
            results.append(tuple(v.cpu() for v in (rates, probs, pearson_chi2(x, mu), mu)))

    rates = torch.cat([r[0] for r in results])
    probs = torch.cat([r[1] for r in results])
    chi2 = torch.cat([r[2] for r in results])
    expected = torch.cat([r[3] for r in results])
    out = {k: v.numpy() for k, v in derived(rates, probs[:, :-1]).items()}
    out.update(fractions=probs[:, :-1].numpy(), background=probs[:, -1].numpy(),
               chi2r=(chi2 / (C * model.n_bins - n_free_parameters(Fn, C))).numpy())
    out["irf_t0"] = t0.item()
    out["irf_sigma"] = (F.softplus(sigma_raw) + _SIGMA_FLOOR).item()
    return out, expected.numpy()


def plot_fits(counts, expected, bin_width, path, n=4, seed=0):
    """Data, model and normalised residuals for a few random pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(seed).choice(len(counts), n, replace=False)
    t = (np.arange(counts.shape[-1]) + 0.5) * bin_width
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for j, i in enumerate(idx):
        h, mu = counts[i].sum(0), expected[i].sum(0)
        axes[0, j].semilogy(t, np.maximum(h, 0.5), ".", color="0.3", label="data")
        axes[0, j].semilogy(t, mu, "-", color="C0", label="fit")
        axes[0, j].set_title(f"pixel {i}, {h.sum():.0f} photons", fontsize=9)
        axes[1, j].bar(t, (h - mu) / np.sqrt(np.maximum(mu, 1e-9)), width=bin_width, color="C1")
        axes[1, j].axhline(0, color="0.5", lw=0.8)
        axes[1, j].set_xlabel("delay (ns)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("counts")
    axes[1, 0].set_ylabel("residual (sigma)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", default="", help="take IRF and spectra from a trained model")
    p.add_argument("--irf", default="", help="or from a calibration JSON (single channel only)")
    p.add_argument("--irf_fit", default="fixed", choices=["fixed", "free"])
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--fit_irf", action="store_true",
                   help="fit the IRF jointly with the pixels; with neither --checkpoint nor "
                        "--irf it starts from the rising edge of the data")
    p.add_argument("--global_lifetimes", action="store_true",
                   help="share the lifetimes across all pixels (global analysis)")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
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
    elif a.fit_irf:
        if counts.shape[1] > 1:
            raise SystemExit("multichannel data needs --checkpoint for the component spectra")
        model = FlimVAE(n_bins=counts.shape[2], bin_width=meta["bin_width_ns"],
                        n_components=a.n_components,
                        irf_t0=irf_t0_guess(counts, meta["bin_width_ns"]), irf_sigma=0.1)
    else:
        raise SystemExit("give --checkpoint, --irf or --fit_irf")

    result, expected = fit_pixels(counts, model, steps=a.steps, fit_irf=a.fit_irf,
                                  global_lifetimes=a.global_lifetimes, device=a.device)
    print(f"IRF: t0={result.pop('irf_t0'):.3f} ns sigma={result.pop('irf_sigma'):.3f} ns "
          f"({'fitted' if a.fit_irf else 'fixed'})")
    gt_report(result, ds)
    summarise(result, ds, meta, a.out or None)
    if a.out:
        np.savez_compressed(f"{a.out}.npz", image=ds["image"].values, y=ds["y"].values,
                            x=ds["x"].values, split=ds["split"].values, **result)
        plot_fits(counts, expected, meta["bin_width_ns"], f"{a.out}_fits.png")
        print(f"wrote {a.out}.npz and {a.out}_fits.png")


if __name__ == "__main__":
    main()
