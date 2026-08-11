"""Train the LUMOS VAE from a processed Zarr store.

    e.g python -m lumos.train --data path/to/dataset.zarr --learning_rate 5e-4

Data loading is Zarr-only and strictly unsupervised: the model is fit on the
observed ``time_series`` and validated on its ability to extrapolate later
frames. Any ground truth in the store is used only for logging.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lumos.callbacks import DecompositionEvalCallback
from lumos.datamodule import ZarrDataModule
from lumos.vae_module import VAEModule

torch.set_float32_matmul_precision("medium")

# Model and trainer defaults. The only data input is a processed Zarr store.
DEFAULTS = dict(
    # Model
    basis_mode="mog",
    latent_dim=128,
    hidden_dim=512,
    physics_model="factored",
    learning_rate=5e-4,
    kl_weight=1.0,
    n_gaussian_components=3,
    polynomial_degree=3,
    raman_l1_weight=0.01,
    raman_lr_multiplier=3.0,
    mog_lr_multiplier=3.0,
    noise_lr_multiplier=10.0,
    lambda_min=0.001,
    raman_mode="pseudo_voigt",
    n_raman_peaks=300,
    fwhm_G_init=5.0,
    fwhm_G_trainable=True,
    n_fluorophores=3,
    n_model_components=16,
    # Data
    n_times_train=16,
    semi_supervised=False,
    # Trainer
    max_epochs=15000,
    batch_size=8,
    gradient_clip_val=1.0,
    num_workers=4,
    early_stopping_patience=200,
    lr_scheduler_patience=20,
    steps_per_epoch=200,
    val_check_steps=500,
    seed=8,
)


def _read_store_config(zarr_path: str, cfg: dict):
    """Build the config object from_datamodule expects, from the store attrs.

    Frame duration and the injected-noise parameters are read from the Zarr
    ``.attrs`` where available, falling back to config defaults.
    """
    import zarr

    attrs = dict(zarr.open(str(zarr_path), mode="r").attrs)

    def pick(keys, default):
        for k in keys:
            if attrs.get(k) is not None:
                return attrs[k]
        return default

    return SimpleNamespace(
        physics_model=cfg["physics_model"],
        bleaching_interval=float(pick(["frame_duration_s", "frame_duration"], 0.1)),
        noise_type=str(
            pick(["injected_noise_type", "noise_type", "synthesis_noise_type"],
                 "poisson_gaussian")
        ),
        gaussian_noise_scale=float(
            pick(["injected_gaussian_noise_scale", "gaussian_noise_scale",
                  "synthesis_gaussian_noise_scale"], 0.0)
        ),
        poisson_noise_scale=float(
            pick(["injected_poisson_noise_scale", "poisson_noise_scale",
                  "synthesis_poisson_noise_scale"], 1.0)
        ),
        seed=cfg["seed"],
    )


def _preload_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def build_datamodule(cfg: dict) -> ZarrDataModule:
    store_config = _read_store_config(cfg["data"], cfg)
    return ZarrDataModule(
        zarr_path=cfg["data"],
        config=store_config,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        n_times_train=cfg["n_times_train"],
        normalize=True,
        preload_device=_preload_device(),
        semi_supervised=cfg["semi_supervised"],
    )


def build_model(cfg: dict, dm: ZarrDataModule) -> VAEModule:
    model_kwargs = dict(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        kl_weight=cfg["kl_weight"],
        basis_mode=cfg["basis_mode"],
        lambda_min=cfg["lambda_min"],
        raman_l1_weight=cfg["raman_l1_weight"],
        raman_lr_multiplier=cfg["raman_lr_multiplier"],
        mog_lr_multiplier=cfg["mog_lr_multiplier"],
        noise_lr_multiplier=cfg["noise_lr_multiplier"],
        raman_mode=cfg["raman_mode"],
        n_raman_peaks=cfg["n_raman_peaks"],
        fwhm_G_init=cfg["fwhm_G_init"],
        fwhm_G_trainable=cfg["fwhm_G_trainable"],
        lr_scheduler_patience=cfg["lr_scheduler_patience"],
        polynomial_degree=cfg["polynomial_degree"],
        semi_supervised=cfg["semi_supervised"],
    )
    n_model = cfg["n_model_components"] or cfg["n_fluorophores"]
    if cfg["basis_mode"] == "mog":
        model_kwargs["n_gaussian_components"] = cfg["n_gaussian_components"]
        model_kwargs["n_fluorophores"] = n_model
    elif cfg["basis_mode"] == "polynomial":
        model_kwargs["n_fluorophores"] = n_model
    else:
        raise ValueError(
            f"basis_mode='{cfg['basis_mode']}' needs an external fluorophore "
            f"dictionary. Use 'mog' or 'polynomial'."
        )
    return VAEModule.from_datamodule(dm, **model_kwargs)


def make_run_name(cfg: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(cfg["data"]).stem
    return (
        f"{stem}_{cfg['basis_mode']}_t{cfg['n_times_train']}"
        f"_z{cfg['latent_dim']}_h{cfg['hidden_dim']}_lr{cfg['learning_rate']}_{ts}"
    )


def train(cfg: dict):
    pl.seed_everything(cfg["seed"], workers=True)

    dm = build_datamodule(cfg)
    dm.setup()
    print(f"Datamodule ready: {len(dm.full_ds)} train / {len(dm.val_ds)} val samples")

    # Validate every ~val_check_steps optimiser steps.
    steps_per_epoch = cfg["steps_per_epoch"]
    batches = min(len(dm.train_dataloader()), steps_per_epoch or 10**9)
    val_every = max(1, round(cfg["val_check_steps"] / max(1, batches)))

    model = build_model(cfg, dm)
    run_name = make_run_name(cfg)
    run_dir = Path("checkpoints") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename="{epoch:04d}-{val_recon_loss:.4f}",
            save_last=True,
            save_top_k=3,
            monitor="val_recon_loss",
            mode="min",
        ),
        DecompositionEvalCallback(dm, n_samples=4),
    ]
    if cfg["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_recon_loss",
                patience=cfg["early_stopping_patience"],
                mode="min",
            )
        )

    logger = CSVLogger(save_dir="logs", name=run_name)
    if cfg.get("wandb"):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="LUMOS", name=run_name, log_model=False)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg["max_epochs"],
        limit_train_batches=cfg["steps_per_epoch"],
        check_val_every_n_epoch=val_every,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg["gradient_clip_val"],
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=2,
    )
    trainer.fit(model, datamodule=dm)


def _resolve_run_dir(run_name=None, checkpoint_dir="checkpoints"):
    """Resolve a run directory by name, or pick the most recently created run."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory at {checkpoint_dir}")
    if run_name:
        return ckpt_dir / run_name

    def _ctime(d):
        s = d.stat()
        return getattr(s, "st_birthtime", s.st_ctime)

    run_dir = max((d for d in ckpt_dir.iterdir() if d.is_dir()), key=_ctime)
    print(f"Using latest run: {run_dir.name}")
    return run_dir


def load_checkpoint(run_name=None, checkpoint_dir="checkpoints", filename=None):
    """Load a trained VAEModule. Uses last.ckpt unless a filename is given."""
    run_dir = _resolve_run_dir(run_name, checkpoint_dir)
    if filename:
        checkpoint_path = run_dir / filename
    else:
        last = run_dir / "last.ckpt"
        if last.exists():
            checkpoint_path = last
        else:
            ckpts = sorted(run_dir.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {run_dir}")
            checkpoint_path = ckpts[-1]
    print(f"Loading: {checkpoint_path}")
    model = VAEModule.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu", strict=False
    )
    model.eval()
    return model, run_dir.name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LUMOS VAE from a Zarr store")
    parser.add_argument(
        "--data",
        "--dataset_path",
        dest="data",
        required=True,
        help="Path to a processed Zarr store",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    optional_int = {"n_model_components", "steps_per_epoch"}
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        elif key in optional_int:
            parser.add_argument(f"--{key}", type=int, default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser


def main():
    # Ignore unrecognised flags so stray arguments do not stop a run.
    args, unknown = _build_parser().parse_known_args()
    if unknown:
        print(f"Ignoring unrecognised arguments: {' '.join(unknown)}")
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    cfg["data"] = args.data
    cfg["wandb"] = args.wandb
    train(cfg)


if __name__ == "__main__":
    main()
