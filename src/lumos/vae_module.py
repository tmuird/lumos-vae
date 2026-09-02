from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from lumos.vae import VAE



_RETIRED = {
    "sum_channel": False,
    "raman_rank": 0,
    "diff_loss_weight": 0.0,
    "integrate_target": 0,
    "temporal_pool": "mean",
    "log1p": False,
    "center_input": False,
    "encoder_norm": "none",
    "ordered_peaks": False,
    "basis_mode": "mog",
    "decay_mode": "per_sample",
    "physics_model": "factored",
    "dictionary_bases": None,
    "raman_mode": None,
}


def _reject_retired(kwargs):
    stale = {k: kwargs[k] for k, ok in _RETIRED.items()
             if k in kwargs and kwargs[k] != ok}
    if stale:
        raise ValueError(
            f"Checkpoint was trained with removed options {stale}. The current "
            f"code cannot reproduce it; load it with the pre-cleanup revision "
            f"or retrain."
        )


class VAEModule(pl.LightningModule):

    def __init__(
        self,
        n_wavenumbers: int = 1024,
        n_times_train: int = 1,
        n_full_timepoints: Optional[int] = None,
        time_values: Optional[torch.Tensor] = None,
        wavenumbers: Optional[torch.Tensor] = None,
        dataset_std: float = 1.0,
        dataset_mean: float = 0.0,
        pool_spectral: int = 16,
        pool_time: int = 1,
        sum_loss_weight: float = 0.0,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        decoder_dim: int = 256,
        learning_rate: float = 1e-3,
        kl_weight: float = 1,
        raman_l1_weight: float = 0.0,
        noise_alpha_gt: float = 0.0,
        noise_beta_gt: float = 0.0,
        t_sampling: str = "",
        lr_scheduler_patience: int = 50,
        lr_schedule: str = "plateau",
        **model_kwargs,
    ):
        super().__init__()
        _reject_retired(model_kwargs)
        self.save_hyperparameters(ignore=["time_values", "wavenumbers"])

        self.log_alpha = nn.Parameter(torch.tensor(-4.0))
        self.log_beta = nn.Parameter(torch.tensor(-6.0))

        spec = (t_sampling or "").strip()
        self._t_choices = (
            [int(v) for v in spec.split(",")] if spec and spec != "uniform" else None
        )
        self._mog_means_grad_norm: Optional[float] = None

        if n_full_timepoints is None:
            n_full_timepoints = n_times_train

        if time_values is not None:
            self.register_buffer(
                "t_full",
                torch.as_tensor(time_values, dtype=torch.float32),
            )
        else:
            self.register_buffer("t_full", torch.zeros(n_full_timepoints))

        self.model = VAE(
            time_values=time_values,
            wavenumbers=wavenumbers,
            n_wavenumbers=n_wavenumbers,
            n_times_train=n_times_train,
            n_full_timepoints=n_full_timepoints,
            dataset_std=dataset_std,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            pool_spectral=pool_spectral,
            pool_time=pool_time,
            decoder_dim=decoder_dim,
            **model_kwargs,
        )

    @property
    def noise_alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha) + 1e-6

    @property
    def noise_beta(self) -> torch.Tensor:
        return F.softplus(self.log_beta) + 1e-6

    @classmethod
    def from_datamodule(cls, datamodule, **kwargs):
        ds_attr = "full_ds" if hasattr(datamodule, "full_ds") else "train_ds"
        if getattr(datamodule, ds_attr) is None:
            datamodule.setup()
        n_times_train = datamodule.n_times_train
        backing_ds = getattr(datamodule, ds_attr)
        ds = backing_ds if not hasattr(backing_ds, "dataset") else backing_ds.dataset
        std = float(ds.std) if getattr(ds, "normalize", False) else 1.0

        # Full time axis for the model. The training data may be cropped to
        # n_times_train, but reconstructing the extrapolation window needs the
        # full axis, which the datamodule exposes directly.
        full_time = getattr(datamodule, "time_values", None)
        if full_time is not None:
            time_vals = torch.as_tensor(full_time, dtype=torch.float32)
        else:
            time_vals = torch.tensor(ds.time_values).float()
        wavs = (
            torch.tensor(ds.wavenumbers).float()
            if getattr(ds, "wavenumbers", None) is not None
            else None
        )
        if wavs is not None and wavs.ndim > 1:
            wavs = wavs.mean(dim=0)
        n_w = wavs.shape[0] if wavs is not None else 1024
        n_t_total = time_vals.shape[0]


        frame_duration = getattr(datamodule.config, "bleaching_interval", 0.1)

        noise_type = getattr(datamodule.config, "noise_type", "gaussian")
        gauss_scale = getattr(datamodule.config, "gaussian_noise_scale", 0.0)
        poisson_scale = getattr(datamodule.config, "poisson_noise_scale", 1.0)
        _std = max(std, 1e-8)

        noise_alpha_gt = (
            (1.0 / (poisson_scale * _std))
            if noise_type in ("poisson", "poisson_gaussian")
            else 0.0
        )
        print(f"gaussian scale  {gauss_scale:.3g}  poisson scale  {poisson_scale:.3g}")

        noise_beta_gt = (
            ((gauss_scale / _std) ** 2)
            if noise_type in ("gaussian", "poisson_gaussian")
            else 0.0
        )
        print(
            f"frame_duration={frame_duration}  n_times_train={n_times_train}  "
            f"n_times_total={n_t_total}  n_wavenumbers={n_w}  "
            f"noise_type={noise_type}  noise_alpha_gt={noise_alpha_gt:.3g}  noise_beta_gt={noise_beta_gt:.3g}"
        )

        model = cls(
            time_values=time_vals,
            wavenumbers=wavs,
            n_wavenumbers=n_w,
            n_times_train=n_times_train,
            n_full_timepoints=n_t_total,
            dataset_std=std,
            frame_duration=frame_duration,
            noise_alpha_gt=noise_alpha_gt,
            noise_beta_gt=noise_beta_gt,
            **kwargs,
        )

        return model

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):

        finite = all(
            torch.isfinite(p.grad).all()
            for p in self.parameters() if p.grad is not None
        )
        if not finite:
            self._skipped_steps = getattr(self, "_skipped_steps", 0) + 1
            if self._skipped_steps in (1, 10, 100, 1000):
                print(
                    f"non-finite gradient at step {self.global_step}, skipping "
                    f"({self._skipped_steps} so far)"
                )
            optimizer.zero_grad(set_to_none=True)
            return

        clip_val = gradient_clip_val if gradient_clip_val is not None else 1.0
        self.clip_gradients(
            optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm"
        )

    def on_fit_start(self):
        if hasattr(self.model, "mog_means"):

            def _hook(grad):
                self._mog_means_grad_norm = grad.norm().item()
                return grad

            self.model.mog_means.register_hook(_hook)

    def configure_optimizers(self):
        # One learning rate for everything
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        if self.hparams.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * 1e-3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.learning_rate * 1e-3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_recon_loss",
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }

    def training_step(self, batch, batch_idx):
        if len(batch) >= 3:
            x_full, _, lengths, *_ = batch
        else:
            x_full, _ = batch[:2]
            lengths = torch.full(
                (x_full.shape[0],), x_full.shape[-1], device=x_full.device
            )

        B, W, T_max = x_full.shape

        # Batches are homogeneous in frame count
        limit = min(self.hparams.n_times_train, int(lengths.min()))
        if self._t_choices is not None:
            usable = [t for t in self._t_choices if t <= limit] or [limit]
            time_slice = usable[int(torch.randint(len(usable), (1,)).item())]
        elif self.hparams.t_sampling == "uniform":
            time_slice = int(torch.randint(1, limit + 1, (1,)).item())
        else:
            time_slice = limit

        # Encoder always sees ONLY the first n_times_train frames.
        x_input = x_full[:, :, :time_slice]

        x_recon, mu, logvar, lambdas, abundances, raman, bases = (
            self.model(x_input)
        )

        # Report which tensor degraded first, so a NaN identifies its own source
        # rather than needing to be reproduced.
        tensors = {
            "mu": mu, "logvar": logvar, "lambdas": lambdas,
            "abundances": abundances, "raman": raman, "x_recon": x_recon,
        }
        bad = [k for k, v in tensors.items() if not torch.isfinite(v).all()]
        if bad:
            summary = "  ".join(
                f"{k}: absmax={v.abs().max().item():.3e}"
                for k, v in tensors.items()
                if torch.isfinite(v).all()
            )
            raise RuntimeError(
                f"Non-finite values in {bad} at epoch={self.current_epoch} "
                f"batch={batch_idx}. Finite tensors: {summary}"
            )

        x_target = x_input
        x_recon_norm = x_recon[: x_target.shape[0], :, :time_slice]

          # Per-element means so losses are independent of batch/spectrum/time size.
        n_elements = x_input.shape[0] * x_input.shape[1] * x_input.shape[2]
        mse = F.mse_loss(x_recon_norm, x_target, reduction="none").sum() / n_elements

        noise_var = (
            self.noise_alpha * x_recon_norm.clamp(min=0.0) + self.noise_beta
        ).clamp(min=1e-6)

        per_element_recon = (
            0.5 * (x_target - x_recon_norm).pow(2) / noise_var + 0.5 * noise_var.log()
        )
        recon_loss = per_element_recon.sum() / n_elements

        if self.hparams.kl_weight > 0:
            kld_loss = -0.5 * torch.sum(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            ) / n_elements
            loss = recon_loss + self.hparams.kl_weight * kld_loss
            self.log("kld_loss", kld_loss, prog_bar=True)
        else:
            loss = recon_loss

        if self.hparams.raman_l1_weight > 0:
            raman_scale = max(self.hparams.dataset_std, 1e-6)
            rl1_penalty = raman.mean() / raman_scale
            loss = loss + self.hparams.raman_l1_weight * rl1_penalty
            self.log("raman_l1_penalty", rl1_penalty, prog_bar=False)

        if self.hparams.sum_loss_weight > 0 and time_slice > 1:

            pred_sum = x_recon_norm.sum(-1)
            targ_sum = x_target.sum(-1)
            var_sum = (self.noise_alpha * pred_sum.clamp(min=0.0)
                       + time_slice * self.noise_beta).clamp(min=1e-6)
            sum_loss = (0.5 * (targ_sum - pred_sum).pow(2) / var_sum
                        + 0.5 * var_sum.log()).mean()
            loss = loss + self.hparams.sum_loss_weight * sum_loss
            self.log("sum_loss", sum_loss, prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", mse, prog_bar=False)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("noise_alpha", self.noise_alpha.detach(), prog_bar=True)
        self.log("noise_beta", self.noise_beta.detach(), prog_bar=True)
        self.log("noise_alpha_gt", self.hparams.noise_alpha_gt, prog_bar=False)
        self.log("noise_beta_gt", self.hparams.noise_beta_gt, prog_bar=False)

        with torch.no_grad():
            _std = max(self.hparams.dataset_std, 1e-6)
            _T = getattr(self.model, "frame_duration", 0.1)
            if self.hparams.kl_weight > 0:
                kl_frac = (self.hparams.kl_weight * kld_loss.detach()) / (
                    loss.detach().abs() + 1e-8
                )
                self.log("kl_frac", kl_frac, prog_bar=True)
            self.log("lambda_mean", lambdas.mean(), prog_bar=True)
            self.log("abundance_mean_norm", abundances.mean() / _std, prog_bar=True)
            self.log("raman_per_frame_norm", raman.mean() * _T / _std, prog_bar=True)

            self.log("recon_mse", mse, prog_bar=False)

            if hasattr(self.model, "log_fwhm_L"):
                # Must match the forward pass, including the pin.
                f_L = F.softplus(self.model.log_fwhm_L) + self.model.width_floor
                f_G = (self.model.log_fwhm_G.new_tensor(self.model.fwhm_G_pinned)
                       if self.model.fwhm_G_pinned
                       else F.softplus(self.model.log_fwhm_G) + self.model.width_floor)
                self.log("fwhm_G", f_G.detach(), prog_bar=True)
                self.log("fwhm_L_mean", f_L.mean().detach(), prog_bar=True)
                self.log("fwhm_L_min", f_L.min().detach(), prog_bar=False)
                self.log("fwhm_L_widest", f_L.max().detach(), prog_bar=False)

            if hasattr(self.model, "mog_means"):
                scales = self.model.mog_log_scales.exp()
                self.log("mog_scale_mean", scales.mean(), prog_bar=False)
                self.log("mog_scale_min", scales.min(), prog_bar=False)
                means = self.model.mog_means
                self.log("mog_means_spread", means.std(), prog_bar=True)
                if self._mog_means_grad_norm is not None:
                    self.log(
                        "mog_means_grad_norm", self._mog_means_grad_norm, prog_bar=True
                    )
            self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        time_slice = self.hparams.n_times_train

        if len(batch) >= 3:
            x_full, _, lengths, *_ = batch
        else:
            x_full, _ = batch[:2]
            lengths = torch.full(
                (x_full.shape[0],), x_full.shape[-1], device=x_full.device
            )
        B, W, T_max = x_full.shape
        # Encoder always sees only the first n_times_train frames.
        x_input = x_full[:, :, :time_slice]

        x_recon, mu, logvar, lambdas, abundances, raman, bases = (
            self.model(x_input)
        )
        # raman_1, raman_2 = raman[: raman.shape[0] // 2], raman[raman.shape[0] // 2 :]

        n_valid_full = min(T_max, self.t_full.shape[0])
        x_target = x_input
        x_recon_norm = x_recon[: x_target.shape[0]]  # already [B, W, time_slice]

        n_elements = x_input.shape[0] * x_input.shape[1] * x_input.shape[2]
        mse = F.mse_loss(x_recon_norm, x_target, reduction="none").sum() / n_elements

        noise_var_v = (
            self.noise_alpha * x_recon_norm.clamp(min=0.0) + self.noise_beta
        ).clamp(min=1e-6)

        per_element_recon = (
            0.5 * (x_target - x_recon_norm).pow(2) / noise_var_v
            + 0.5 * noise_var_v.log()
        )
        recon_loss = per_element_recon.sum() / n_elements
        if self.hparams.kl_weight > 0:
            kld_loss = -0.5 * torch.sum(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            ) / n_elements
            loss = recon_loss + self.hparams.kl_weight * kld_loss
        else:
            loss = recon_loss

        # Extrapolation loss (monitoring only - not added to val_loss).
        # Reuse x_recon_norm if the full reconstruction was already computed,
        # otherwise run physics_forward over the full window now.
        if time_slice < n_valid_full:
            x_recon_phys_full, _ = self.model.physics_forward(
                lambdas,
                abundances,
                raman,
                self.model.bases,
                time_values=self.t_full[:n_valid_full],
            )
            x_recon_full = x_recon_phys_full / self.model.dataset_std

            x_target_extrap = x_full[: x_target.shape[0], :, time_slice:n_valid_full]
            x_recon_extrap = x_recon_full[
                : x_target.shape[0], :, time_slice:n_valid_full
            ]

            t_idx_extrap = torch.arange(
                time_slice, n_valid_full, device=x_full.device
            ).view(1, 1, -1)
            valid_mask_extrap = (
                (t_idx_extrap < lengths.view(-1, 1, 1)).float().expand(-1, W, -1)
            )

            if valid_mask_extrap.sum() > 0:
                extrap_loss = (
                    F.mse_loss(x_recon_extrap, x_target_extrap, reduction="none")
                    * valid_mask_extrap
                ).sum() / valid_mask_extrap.sum()
                self.log("val_extrap_loss", extrap_loss, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_mse", mse, prog_bar=False)
        return loss
