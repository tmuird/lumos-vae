"""Train the LUMOS port on a pixel store.

    python -m lumos_flim.lumos.train --data data/hmsc.zarr

Defaults follow lumos-vae except for the time window. TCSPC records every
delay bin in parallel, so there is nothing to gain by fitting early bins and
extrapolating: the encoder sees the whole period by default
(``n_times_train=0``) with a fixed window (``t_sampling=""``). Both can still
be set as in LUMOS.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lumos_flim.lumos.callbacks import DecompositionEvalCallback
from lumos_flim.lumos.datamodule import ZarrDataModule
from lumos_flim.lumos.vae_module import VAEModule

DEFAULTS = dict(
    latent_dim=64,
    hidden_dim=128,
    decoder_dim=256,
    learning_rate=5e-4,
    kl_weight=1.0,
    n_gaussian_components=3,
    static_l1_weight=0.01,
    lambda_min=0.0,  # 0 means one per period
    t_sampling="",
    conv_channels="8,16,32,64",
    pool_spectral=1,
    pool_time=8,
    sum_loss_weight=0.0,
    irf_sigma_pinned=0.0,
    n_fluorophores=2,
    n_times_train=0,  # 0 means all bins
    transductive=True,
    max_epochs=60,
    batch_size=256,
    gradient_clip_val=1.0,
    lr_scheduler_patience=7,
    lr_schedule="cosine",
    seed=8,
    run_name="",
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=val)
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    cfg = vars(p.parse_args(argv))

    pl.seed_everything(cfg["seed"], workers=True)
    dm = ZarrDataModule(cfg["data"], batch_size=cfg["batch_size"],
                        n_times_train=cfg["n_times_train"] or None,
                        transductive=cfg["transductive"])
    dm.setup()
    model_keys = ("latent_dim", "hidden_dim", "decoder_dim", "learning_rate", "kl_weight",
                  "n_gaussian_components", "static_l1_weight", "lambda_min", "t_sampling",
                  "conv_channels", "pool_spectral", "pool_time", "sum_loss_weight",
                  "irf_sigma_pinned", "n_fluorophores", "lr_scheduler_patience", "lr_schedule")
    model = VAEModule.from_datamodule(dm, **{k: cfg[k] for k in model_keys})

    run_name = cfg["run_name"] or f"{Path(cfg['data']).stem}_lumos_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("checkpoints") / run_name
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        logger=CSVLogger("logs", name=run_name),
        callbacks=[ModelCheckpoint(dirpath=str(run_dir), filename="last", monitor=None),
                   DecompositionEvalCallback(dm)],
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg["gradient_clip_val"],
        enable_model_summary=False,
        enable_progress_bar=sys.stdout.isatty(),
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=dm)
    print(f"checkpoint: {run_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
