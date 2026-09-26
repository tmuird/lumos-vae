"""Rescore the LUMOS port from the sweep with posterior-ensemble inference.

Most latent dimensions collapse to the prior, and the decoder is trained on
samples from them, so decoding the posterior mean puts it off its training
distribution. LUMOS's own sample_posterior averages over draws; this does the
same: per-pixel parameters are the ensemble median (components sorted by
lifetime in each draw) and the reconstruction is the ensemble mean.
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from photon_sweep import score, truth  # noqa: E402

from lumos_flim.data import SPLITS, open_store
from lumos_flim.lumos.predict import to_photon_fractions
from lumos_flim.lumos.vae_module import VAEModule
from lumos_flim.predict import derived


@torch.no_grad()
def ensemble(module, counts, n=20, seed=0):
    torch.manual_seed(seed)
    m = module.model.eval()
    x = torch.as_tensor(counts, dtype=torch.float32) / (module.hparams.dataset_std + 1e-8)
    taus, alphas, fracs, bgs, amps, recons = [], [], [], [], [], []
    for _ in range(n):
        _, _, _, rates, abund, static, bases = m(x, sample=True)
        recons.append(m.physics_forward(rates, abund, static, bases, time_values=module.t_full)[0])
        order = torch.argsort(rates, -1)
        rates, abund = rates.gather(1, order), abund.gather(1, order)
        f, bg = to_photon_fractions(module, rates, abund, static, bases)
        d = derived(rates, f)
        taus.append(d["tau"]); alphas.append(d["alpha"]); amps.append(d["tau_amp"])
        fracs.append(f); bgs.append(bg)
    med = lambda v: torch.stack(v).median(0).values.numpy()
    res = dict(tau=med(taus), alpha=med(alphas), tau_amp=med(amps), fractions=med(fracs),
               background=med(bgs))
    return res, torch.stack(recons).mean(0).numpy()


rows = json.load(open("results/sweep/sweep.json"))
rows = [r for r in rows if r["method"] != "LUMOS port (ensemble)"]
for p in sorted({r["photons"] for r in rows}):
    tag = f"p{int(p)}"
    ds, meta = open_store(f"data/sweep_{tag}.zarr")
    counts = ds["counts"].values
    held = ds["split"].values == SPLITS["val"]
    module = VAEModule.load_from_checkpoint(f"checkpoints/sweep_{tag}_lumos/last.ckpt", map_location="cpu")
    res, rec = ensemble(module, counts)
    row = dict(photons=p, **score("LUMOS port (ensemble)", res, rec, ds, truth(ds, meta), held))
    rows.append(row)
    print(json.dumps(row), flush=True)
rows.sort(key=lambda r: r["photons"])
json.dump(rows, open("results/sweep/sweep.json", "w"), indent=1)
