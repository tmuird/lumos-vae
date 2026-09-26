"""Does the latent collapse because of the objective or the optimisation?

    python scripts/kl_study.py --photons 200 1000

For each photon budget, trains the VAE with the plain beta = 1 ELBO, with KL
warm-up (ends at beta = 1, so the same objective), and with free bits (a
different objective). All are judged on the beta = 1 ELBO of the val pixels
and on accuracy against ground truth. If warm-up reaches a better ELBO than
plain training, the collapse was an optimisation failure; if only free bits
uses more of the latent, and at a worse ELBO, the collapse is the optimum.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lumos_flim.data import SPLITS, open_store
from lumos_flim.predict import derived, run_model
from lumos_flim.train import DEFAULTS, train
from lumos_flim.vae_module import FlimModule, poisson_half_deviance

CONFIGS = {
    "beta1": {},
    "warmup20": {"kl_warmup_epochs": 20},
    "freebits0.25": {"free_bits": 0.25},
    "freebits1": {"free_bits": 1.0},
}


@torch.no_grad()
def elbo_terms(module, counts):
    x = torch.as_tensor(counts, dtype=torch.float32)
    torch.manual_seed(0)
    out = module.model(x, sample=True)
    nll = poisson_half_deviance(x, out["expected"])
    kl_dims = -0.5 * (1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp())
    per_dim = kl_dims.mean(0)
    return float((nll + kl_dims.sum(1)).mean()), float(per_dim.sum()), int((per_dim > 0.1).sum())


def evaluate(module, ds, held):
    counts = ds["counts"].values[held]
    neg_elbo, kl, active = elbo_terms(module, counts)
    res = run_model(module, counts)
    gt_tau = ds["gt_tau"].values[held]
    gt_amp = derived(torch.as_tensor(1 / gt_tau), torch.as_tensor(ds["gt_fraction"].values[held]))["tau_amp"].numpy()
    iqr = lambda r: float(np.subtract(*np.percentile(r, [75, 25])))
    row = dict(neg_elbo=neg_elbo, kl_nats=kl, active_dims=active)
    for name, pred, gt in (("tau_long", res["tau"][:, 0], gt_tau[:, 0]),
                           ("tau_short", res["tau"][:, 1], gt_tau[:, 1]),
                           ("tau_amp", res["tau_amp"], gt_amp)):
        rel = (pred - gt) / gt
        row[f"{name}_bias"] = float(np.median(rel))
        row[f"{name}_iqr"] = iqr(rel)
        row[f"{name}_r"] = float(np.corrcoef(pred, gt)[0, 1])
    row["alpha_mae"] = float(np.median(np.abs(res["alpha"][:, 0] - ds["gt_alpha"].values[held][:, 0])))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--photons", type=int, nargs="+", default=[200, 1000])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out", default="results/kl_study.json")
    a = p.parse_args()
    rows = []
    for n in a.photons:
        store = f"data/sweep_p{n}.zarr"
        ds, _ = open_store(store)
        held = ds["split"].values == SPLITS["val"]
        for name, overrides in CONFIGS.items():
            cfg = dict(DEFAULTS, data=store, max_epochs=a.epochs, batch_size=256,
                       run_name=f"kl_p{n}_{name}", wandb=False, **overrides)
            ckpt = train(cfg)
            module = FlimModule.load_from_checkpoint(ckpt, map_location="cpu")
            row = dict(photons=n, config=name, **evaluate(module, ds, held))
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
