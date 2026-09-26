"""Train the FLIM VAE on a pixel store.

    python -m lumos_flim.train --data data/hmsc.zarr --irf calibration/irf_hmsc.json --irf_fit free

Without ``--irf`` the IRF starts at the steepest rise of the data and is
learned with everything else. Ground truth in the store, when present, is
only logged. Checkpoints are chosen on the beta = 1 ELBO of the val pixels,
whatever KL schedule was used to get there.
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
from lumos_flim.device import pick_device
from lumos_flim.vae_module import FlimModule

DEFAULTS = dict(
    # IRF
    irf="",  # JSON written by lumos_flim.calibrate
    irf_fit="fixed",  # which calibration fit: "fixed" (reference lifetime held) or "free"
    fix_irf=False,
    irf_tail=False,  # learn an exponential diffusion tail on the Gaussian IRF
    # Model
    n_components=2,
    tau_max=0.0,  # longest lifetime in ns, 0 for one period
    latent_dim=16,
    hidden_dim=128,
    decoder_dim=128,
    conv_channels="32,64,128",
    spatial=0,  # encoder also sees the k x k neighbourhood (e.g. 3); 0 for none
    spatial_mode="sum",  # "sum": one summed context histogram; "stack": each neighbour separately
    # Latent MRF prior between neighbouring pixels (0 for the independent
    # prior). Trains on contiguous tiles; best used with --spatial 3
    # --spatial_mode stack so the encoder can see the neighbours it is tied to.
    spatial_prior=0.0,
    tile=16,
    # Objective. beta = 1 is the ELBO; warm-up ramps up to it and free bits
    # departs from it, so both are off by default.
    kl_weight=1.0,
    kl_warmup_epochs=0,
    free_bits=0.0,
    # Optimisation
    learning_rate=1e-3,
    batch_size=512,
    max_epochs=100,
    lr_schedule="cosine",  # "cosine" runs to max_epochs; "none" allows early stopping
    early_stopping_patience=20,  # in validation checks, ignored under cosine
    val_check_steps=0,  # validate about every N optimiser steps, 0 for every epoch
    gradient_clip_val=1.0,
    transductive=True,
    # Hardware. "auto" picks CUDA, then Apple MPS, then CPU. With preload the
    # whole store is copied to the GPU once instead of batch by batch.
    accelerator="auto",
    preload=True,
    seed=0,
    run_name="",
    resume="",  # existing run directory to continue from its last.ckpt
)


def make_run_name(cfg) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg["run_name"]:
        return cfg["run_name"]
    parts = [Path(cfg["data"]).stem, f"F{cfg['n_components']}", f"z{cfg['latent_dim']}"]
    if cfg["irf_tail"]:
        parts.append("tail")
    if cfg["spatial_prior"]:
        parts.append(f"mrf{cfg['spatial_prior']:g}")
    if cfg["spatial"]:
        parts.append(f"sp{cfg['spatial']}{cfg['spatial_mode'][0]}")
    if cfg["kl_warmup_epochs"]:
        parts.append(f"warm{cfg['kl_warmup_epochs']}")
    if cfg["free_bits"]:
        parts.append(f"fb{cfg['free_bits']:g}")
    if cfg["kl_weight"] != 1.0:
        parts.append(f"kl{cfg['kl_weight']:g}")
    parts += [f"s{cfg['seed']}", ts]
    return "_".join(parts)


def train(cfg):
    pl.seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    device = pick_device(cfg["accelerator"])
    preload = device if cfg["preload"] and device.type != "cpu" else None
    print(f"training on {device}")
    dm = FlimDataModule(cfg["data"], batch_size=cfg["batch_size"], transductive=cfg["transductive"],
                        preload_device=preload, spatial=cfg["spatial"],
                        spatial_mode=cfg["spatial_mode"], tiles=cfg["spatial_prior"] > 0,
                        tile=cfg["tile"])
    dm.setup()

    irf_t0, irf_sigma = dm.irf_t0_guess, 0.1
    if cfg["irf"]:
        with open(cfg["irf"]) as f:
            cal = json.load(f)[cfg["irf_fit"]]
        irf_t0, irf_sigma = cal["irf_t0"], cal["irf_sigma"]
        print(f"IRF from {cfg['irf']}: t0={irf_t0:.3f} ns sigma={irf_sigma:.3f} ns "
              f"({'fixed' if cfg['fix_irf'] else 'initial'})")
    elif cfg["fix_irf"]:
        raise SystemExit("--fix_irf needs --irf")

    run_name = make_run_name(cfg)
    run_dir = Path("checkpoints") / run_name
    ckpt_path = None
    if cfg["resume"]:
        run_dir = Path("checkpoints") / cfg["resume"]
        ckpt_path = run_dir / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No last.ckpt in {run_dir}")
        run_name = cfg["resume"]
        print(f"Resuming from {ckpt_path}")

    module = FlimModule(
        n_bins=dm.n_bins, bin_width=dm.bin_width, n_channels=dm.n_channels,
        n_components=cfg["n_components"], latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"], decoder_dim=cfg["decoder_dim"],
        conv_channels=cfg["conv_channels"], tau_max=cfg["tau_max"],
        irf_t0=irf_t0, irf_sigma=irf_sigma, fix_irf=cfg["fix_irf"],
        learning_rate=cfg["learning_rate"], kl_weight=cfg["kl_weight"],
        kl_warmup_epochs=cfg["kl_warmup_epochs"], free_bits=cfg["free_bits"],
        max_epochs=cfg["max_epochs"], lr_schedule=cfg["lr_schedule"], spatial=cfg["spatial"],
        spatial_mode=cfg["spatial_mode"], irf_tail=cfg["irf_tail"],
        spatial_prior=cfg["spatial_prior"],
    )

    # Validate about every val_check_steps optimiser steps.
    batches = max(1, len(dm.train_dataloader()))
    val_every = max(1, round(cfg["val_check_steps"] / batches)) if cfg["val_check_steps"] else 1

    callbacks = [
        ModelCheckpoint(dirpath=str(run_dir), filename="best", monitor="val_neg_elbo",
                        mode="min", save_top_k=1),
        ModelCheckpoint(dirpath=str(run_dir), filename="last", monitor=None,
                        every_n_epochs=val_every, enable_version_counter=False),
    ]
    # Cutting a cosine schedule short leaves the model wherever the rate had
    # got to, so early stopping only runs without it.
    if cfg["lr_schedule"] != "cosine" and cfg["early_stopping_patience"] > 0:
        callbacks.append(EarlyStopping(monitor="val_neg_elbo",
                                       patience=cfg["early_stopping_patience"], mode="min"))

    logger = CSVLogger("logs", name=run_name)
    if cfg.get("wandb"):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="LUMOS-FLIM", name=run_name, log_model=False)

    trainer = pl.Trainer(
        accelerator=device.type,
        devices=1,
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=val_every,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg["gradient_clip_val"],
        enable_model_summary=False,
        log_every_n_steps=20,
        # A progress bar redirected to a file repaints into it and bloats logs.
        enable_progress_bar=sys.stdout.isatty(),
    )
    trainer.fit(module, dm, ckpt_path=ckpt_path)
    best = callbacks[0]
    print(f"best checkpoint: {best.best_model_path} (val_neg_elbo={float(best.best_model_score):.4f})")
    if cfg["spatial_prior"] > 0:
        # The val ELBO is computed per pixel, without the neighbour coupling,
        # so it is not the training objective; with the cosine schedule the
        # final weights are the ones to use.
        last = str(Path(callbacks[1].dirpath) / "last.ckpt")
        print(f"spatial prior: using {last}")
        return last
    return best.best_model_path


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                   help="log to Weights and Biases instead of CSV")
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=val)
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    return p


def main(argv=None):
    cfg = vars(_build_parser().parse_args(argv))
    train(cfg)


if __name__ == "__main__":
    main()
