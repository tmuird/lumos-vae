"""Inference for the LUMOS port: ensemble posterior sampling, and per-pixel maps.

    python -m lumos_flim.lumos.predict checkpoints/<run>/last.ckpt --data data/hmsc.zarr --out results/hmsc_lumos

The rates are unordered, as in LUMOS, so for reporting each pixel's
components are sorted by lifetime, longest first.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lumos_flim.data import open_store
from lumos_flim.lumos.vae_module import VAEModule
from lumos_flim.predict import derived, gt_report, summarise


def sample_posterior(model, sample_tensor, n_predictions=50):
    """N stochastic forward passes, returned individually as in LUMOS.

    sample_tensor is [B, C, T] std-normalised. Returns a dict of arrays with a
    leading ensemble axis: 'static' [N, B, C] counts/ns, 'rates' [N, B, F] 1/ns,
    'abundances' [N, B, F] photons per period, 'bases' [N, F, C], and
    'reconstruction' [N, B, C, T_full] in counts over the full period.
    """
    n_train = model.hparams.n_times_train
    model.eval()
    out = {k: [] for k in ("static", "rates", "abundances", "bases", "reconstruction")}
    with torch.no_grad():
        x = sample_tensor[:, :, :n_train].to(next(model.parameters()).device)
        for _ in range(n_predictions):
            _, _, _, lambdas, abundances, static, bases = model.model(x, sample=n_predictions > 1)
            recon, _ = model.model.physics_forward(lambdas, abundances, static, bases,
                                                   time_values=model.t_full)
            for k, v in zip(out, (static, lambdas, abundances, bases, recon)):
                out[k].append(v.cpu())
    return {k: torch.stack(v).numpy() for k, v in out.items()}


def to_photon_fractions(model, rates, abundances, static, bases):
    """Photon fractions of each component and of the static term, per pixel."""
    comp = abundances * bases.sum(-1)[None, :]  # photons per component
    bg = static.sum(-1) * model.model.frame_duration * model.model.n_full_timepoints
    total = comp.sum(-1) + bg
    return comp / total[:, None], bg / total


@torch.no_grad()
def run(module, counts, std, batch_size=4096):
    m = module.model.eval()
    keys = ("tau", "alpha", "fractions", "background", "tau_int", "tau_amp", "chi2r")
    res = {k: [] for k in keys}
    C, N = counts.shape[1], counts.shape[2]
    dof = C * N - (2 * m.n_fluorophores + C)
    for s in range(0, len(counts), batch_size):
        raw = torch.as_tensor(counts[s:s + batch_size], dtype=torch.float32)
        _, _, _, rates, abund, static, bases = m(raw[:, :, :module.hparams.n_times_train] / (std + 1e-8),
                                                 sample=False)
        recon, _ = m.physics_forward(rates, abund, static, bases, time_values=module.t_full)
        # Longest lifetime first, so components line up across pixels.
        order = torch.argsort(rates, dim=-1)
        rates, abund = rates.gather(1, order), abund.gather(1, order)
        frac, bg = to_photon_fractions(module, rates, abund, static, bases)
        d = derived(rates, frac)
        vals = dict(d, fractions=frac, background=bg,
                    chi2r=((raw - recon) ** 2 / recon.clamp(min=1e-6)).sum((1, 2)) / dof)
        for k in keys:
            res[k].append(vals[k].numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    module = VAEModule.load_from_checkpoint(a.checkpoint, map_location="cpu")
    ds, meta = open_store(a.data)
    result = run(module, ds["counts"].values, module.hparams.dataset_std)
    m = module.model
    print(f"IRF: t0={m.irf_t0.item():.3f} ns sigma={m.irf_sigma.item():.3f} ns | "
          f"noise alpha={module.noise_alpha.item():.4g} (Poisson: {module.hparams.noise_alpha_gt:.4g}) "
          f"beta={module.noise_beta.item():.3g} (Poisson: 0)")
    gt_report(result, ds)
    summarise(result, ds, meta, a.out or None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{a.out}.npz", image=ds["image"].values, y=ds["y"].values,
                            x=ds["x"].values, split=ds["split"].values, **result)
        print(f"wrote {a.out}.npz")


if __name__ == "__main__":
    main()
