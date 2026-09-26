"""Run a trained model over a store and write lifetime maps.

    python -m lumos_flim.predict checkpoints/<run>/best.ckpt --data data/hmsc.zarr --out results/hmsc

Writes ``<out>.npz`` with per-pixel parameters and ``<out>_<image>.png`` maps,
and prints a per-image summary. Parameters are posterior medians over
``--n_samples`` draws (default 20), with their spread saved alongside; 0
decodes the posterior mean once.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lumos_flim.data import SPLITS, open_store
from lumos_flim.device import pick_device
from lumos_flim.physics import amplitude_fractions, mean_lifetimes
from lumos_flim.vae_module import FlimModule, pearson_chi2


def derived(rates, fractions):
    """Lifetimes, amplitude fractions and mean lifetimes from rates and photon fractions."""
    tau_int, tau_amp = mean_lifetimes(fractions, rates)
    return {
        "tau": 1.0 / rates,
        "alpha": amplitude_fractions(fractions, rates),
        "tau_int": tau_int,
        "tau_amp": tau_amp,
    }


@torch.no_grad()
def run_model(module, counts, batch_size=4096, n_samples=20, seed=0, device="auto"):
    """Per-pixel parameters from the posterior.

    With ``n_samples`` > 0 (the default, as in LUMOS's sample_posterior) each
    parameter is the median over that many posterior draws, its standard
    deviation over the draws is returned as ``<name>_std``, and the
    reconstruction is the mean over draws. With 0 the posterior mean of the
    latent is decoded once. Runs on ``device`` ("auto" for CUDA, then MPS,
    then CPU); results come back as numpy on the host.
    """
    torch.manual_seed(seed)
    device = pick_device(device)
    model = module.model.to(device).eval()
    keys = ("tau", "fractions", "background", "alpha", "tau_int", "tau_amp")
    out = {k: [] for k in keys + ("chi2r",)}
    spread = {k: [] for k in keys}
    recon = []
    for start in range(0, len(counts), batch_size):
        x = torch.as_tensor(counts[start:start + batch_size], dtype=torch.float32, device=device)
        draws = [model(x, sample=True) for _ in range(n_samples)] if n_samples else [model(x, sample=False)]
        draws = [{k: v.cpu() for k, v in r.items() if v is not None} for r in draws]
        x = x.cpu()
        vals = [dict(derived(r["rates"], r["fractions"]), fractions=r["fractions"],
                     background=r["background"]) for r in draws]
        for k in keys:
            stack = torch.stack([v[k] for v in vals])
            out[k].append(stack.median(0).values.numpy())
            spread[k].append(stack.std(0).numpy() if n_samples > 1 else np.zeros(stack.shape[1:]))
        expected = torch.stack([r["expected"] for r in draws]).mean(0)
        recon.append(expected.numpy())
        out["chi2r"].append((pearson_chi2(x, expected) / module.dof).numpy())
    result = {k: np.concatenate(v) for k, v in out.items()}
    if n_samples > 1:
        result.update({f"{k}_std": np.concatenate(v) for k, v in spread.items()})
    result["expected"] = np.concatenate(recon)
    return result


def gt_report(result, ds):
    """Errors against synthetic ground truth, on the val and test pixels."""
    if "gt_tau" not in ds:
        return
    held = ds["split"].values != SPLITS["train"]
    gt_tau = ds["gt_tau"].values[held]
    order = np.argsort(-gt_tau, axis=1)
    gt_tau = np.take_along_axis(gt_tau, order, 1)
    gt_alpha = np.take_along_axis(ds["gt_alpha"].values[held], order, 1)
    tau, alpha = result["tau"][held], result["alpha"][held]
    print(f"ground truth, {held.sum()} held-out pixels (median abs error):")
    for i in range(tau.shape[1]):
        rel = np.abs(tau[:, i] - gt_tau[:, i]) / gt_tau[:, i]
        print(f"  tau{i}: rel err {np.median(rel):.3f}   alpha{i}: abs err "
              f"{np.median(np.abs(alpha[:, i] - gt_alpha[:, i])):.3f}")
    bg = np.median(np.abs(result["background"][held] - ds["gt_background"].values[held]))
    print(f"  background: abs err {bg:.4f}")


def summarise(result, ds, meta, out_prefix=None):
    image = ds["image"].values
    ys, xs = ds["y"].values, ds["x"].values
    photons = ds["counts"].values.sum((1, 2))
    F = result["tau"].shape[1]
    print(f"{'image':24s} {'pixels':>7s} " + " ".join(f"{'tau' + str(i):>7s}" for i in range(F))
          + " " + " ".join(f"{'alpha' + str(i):>7s}" for i in range(F))
          + f" {'tau_amp':>8s} {'bg':>6s} {'chi2r':>6s}")
    for i, name in enumerate(meta["image_names"]):
        m = image == i
        med = lambda v: np.median(v[m], axis=0)
        print(f"{name:24s} {m.sum():7d} " + " ".join(f"{v:7.3f}" for v in np.atleast_1d(med(result['tau'])))
              + " " + " ".join(f"{v:7.3f}" for v in np.atleast_1d(med(result['alpha'])))
              + f" {med(result['tau_amp']):8.3f} {med(result['background']):6.3f} {med(result['chi2r']):6.2f}")
        if out_prefix:
            _plot_maps(result, m, ys[m], xs[m], photons[m], meta["image_shapes"][i],
                       f"{out_prefix}_{name.replace(' ', '_')}.png", name)


def _plot_maps(result, mask, ys, xs, photons, shape, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def img(values):
        a = np.full(shape, np.nan)
        a[ys, xs] = values
        return a

    F = result["tau"].shape[1]
    panels = [("photons", img(photons), "gray")]
    panels.append(("amplitude-weighted lifetime (ns)", img(result["tau_amp"][mask]), "viridis"))
    for i in range(F):
        panels.append((f"tau{i} (ns)", img(result["tau"][mask, i]), "magma"))
    panels.append((f"alpha{F - 1} (shortest, amplitude)", img(result["alpha"][mask, F - 1]), "coolwarm"))
    panels.append(("reduced chi-square", img(result["chi2r"][mask]), "cividis"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (name, a, cmap) in zip(axes, panels):
        finite = a[np.isfinite(a)]
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    a = p.parse_args(argv)

    module = FlimModule.load_from_checkpoint(a.checkpoint, map_location="cpu")
    ds, meta = open_store(a.data)
    result = run_model(module, ds["counts"].values, n_samples=a.n_samples, device=a.device)
    module.cpu()
    m = module.model
    print(f"IRF: t0={m.irf_t0.item():.3f} ns sigma={m.irf_sigma.item():.3f} ns")
    if m.bases is not None:
        print("component spectra:\n", np.round(m.bases.detach().numpy(), 3))
    gt_report(result, ds)
    summarise(result, ds, meta, a.out or None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        result.pop("expected")
        np.savez_compressed(f"{a.out}.npz", image=ds["image"].values, y=ds["y"].values,
                            x=ds["x"].values, split=ds["split"].values, **result)
        print(f"wrote {a.out}.npz")


if __name__ == "__main__":
    main()
