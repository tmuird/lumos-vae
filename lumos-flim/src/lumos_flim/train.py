"""Train the FLIM VAE on a pixel store.

    python -m lumos_flim.train --data data/hmsc.zarr --irf irf_hmsc.json --fix_irf

Without ``--irf`` the IRF starts at the steepest rise of the data and is
learned with everything else. Ground truth in the store, when present, is
only logged.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lumos_flim.data import FlimDataModule
from lumos_flim.vae_module import FlimModule


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--irf", default="", help="JSON written by lumos_flim.calibrate")
    p.add_argument("--irf_fit", default="fixed", choices=["fixed", "free"],
                   help="which calibration fit to take the IRF from: lifetime held at the "
                        "reference value, or lifetime free")
    p.add_argument("--fix_irf", action="store_true", help="hold the calibrated IRF fixed")
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--tau_max", type=float, default=0.0, help="longest lifetime, ns (default: one period)")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--decoder_dim", type=int, default=128)
    p.add_argument("--conv_channels", default="32,64,128")
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no_transductive", action="store_true", help="fit on the train split only")
    p.add_argument("--run_name", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    pl.seed_everything(a.seed)
    torch.set_float32_matmul_precision("medium")
    dm = FlimDataModule(a.data, batch_size=a.batch_size, transductive=not a.no_transductive)
    dm.setup()

    irf_t0, irf_sigma = dm.irf_t0_guess, 0.1
    if a.irf:
        with open(a.irf) as f:
            cal = json.load(f)[a.irf_fit]
        irf_t0, irf_sigma = cal["irf_t0"], cal["irf_sigma"]
        print(f"IRF from {a.irf}: t0={irf_t0:.3f} ns sigma={irf_sigma:.3f} ns "
              f"({'fixed' if a.fix_irf else 'initial'})")
    elif a.fix_irf:
        raise SystemExit("--fix_irf needs --irf")

    module = FlimModule(
        n_bins=dm.n_bins, bin_width=dm.bin_width, n_channels=dm.n_channels,
        n_components=a.n_components, latent_dim=a.latent_dim, hidden_dim=a.hidden_dim,
        decoder_dim=a.decoder_dim, conv_channels=a.conv_channels, tau_max=a.tau_max,
        irf_t0=irf_t0, irf_sigma=irf_sigma, fix_irf=a.fix_irf,
        learning_rate=a.learning_rate, kl_weight=a.kl_weight, max_epochs=a.max_epochs,
    )

    run_name = a.run_name or f"{Path(a.data).stem}_F{a.n_components}_{datetime.now():%Y%m%d-%H%M%S}"
    ckpt = ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor="val_nll", mode="min",
                           save_last=True, filename="best")
    trainer = pl.Trainer(
        max_epochs=a.max_epochs,
        logger=CSVLogger("logs", name=run_name),
        callbacks=[ckpt, EarlyStopping(monitor="val_nll", patience=a.patience)],
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        enable_progress_bar=sys.stderr.isatty(),
    )
    trainer.fit(module, dm)
    print(f"best checkpoint: {ckpt.best_model_path} (val_nll={ckpt.best_model_score:.4f})")


if __name__ == "__main__":
    main()
