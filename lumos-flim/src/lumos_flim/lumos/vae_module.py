from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from lumos_flim.lumos.vae import VAE


class VAEModule(pl.LightningModule):
    """Loss and training protocol as in lumos-vae.

    The likelihood is the learned heteroscedastic Gaussian of LUMOS,
    ``var = alpha * mu + beta`` in normalised units. For photon counting the
    true values are known (alpha = 1/std, beta = 0) and are logged as
    ``noise_alpha_gt`` / ``noise_beta_gt`` so the learned noise can be checked.
    """

    def __init__(
        self,
        n_wavenumbers: int = 1,
        n_times_train: int = 1,
        n_full_timepoints: Optional[int] = None,
        time_values: Optional[torch.Tensor] = None,
        dataset_std: float = 1.0,
        pool_spectral: int = 1,
        pool_time: int = 1,
        sum_loss_weight: float = 0.0,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        decoder_dim: int = 256,
        learning_rate: float = 1e-3,
        kl_weight: float = 1,
        static_l1_weight: float = 0.0,
        noise_alpha_gt: float = 0.0,
        noise_beta_gt: float = 0.0,
        t_sampling: str = "",
        lr_scheduler_patience: int = 50,
        lr_schedule: str = "plateau",
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["time_values"])

        self.log_alpha = nn.Parameter(torch.tensor(-4.0))
        self.log_beta = nn.Parameter(torch.tensor(-6.0))

        spec = (t_sampling or "").strip()
        self._t_choices = (
            [int(v) for v in spec.split(",")] if spec and spec != "uniform" else None
        )

        if n_full_timepoints is None:
            n_full_timepoints = n_times_train
        if time_values is not None:
            self.register_buffer("t_full", torch.as_tensor(time_values, dtype=torch.float32))
        else:
            self.register_buffer("t_full", torch.zeros(n_full_timepoints))

        self.model = VAE(
            time_values=time_values,
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
        if datamodule.full_ds is None:
            datamodule.setup()
        ds = datamodule.full_ds
        std = float(ds.std) if ds.normalize else 1.0
        time_vals = torch.as_tensor(datamodule.time_values, dtype=torch.float32)
        # Photon counts are Poisson, so in std-normalised units var = mu / std.
        noise_alpha_gt = 1.0 / max(std, 1e-8)
        print(f"bin width={datamodule.bin_width:.4f} ns  n_times_train={datamodule.n_times_train}  "
              f"n_bins={len(time_vals)}  channels={datamodule.n_channels}  "
              f"std={std:.3g}  noise_alpha_gt={noise_alpha_gt:.3g}")
        kwargs.setdefault("irf_t0", datamodule.irf_t0_guess)
        return cls(
            time_values=time_vals,
            n_wavenumbers=datamodule.n_channels,
            n_times_train=datamodule.n_times_train,
            n_full_timepoints=len(time_vals),
            dataset_std=std,
            frame_duration=datamodule.bin_width,
            noise_alpha_gt=noise_alpha_gt,
            noise_beta_gt=0.0,
            **kwargs,
        )

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        finite = all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None)
        if not finite:
            self._skipped_steps = getattr(self, "_skipped_steps", 0) + 1
            if self._skipped_steps in (1, 10, 100, 1000):
                print(f"non-finite gradient at step {self.global_step}, skipping "
                      f"({self._skipped_steps} so far)")
            optimizer.zero_grad(set_to_none=True)
            return
        clip_val = gradient_clip_val if gradient_clip_val is not None else 1.0
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=1e-4)
        if self.hparams.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * 1e-3)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.learning_rate * 1e-3)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_recon_loss",
                                 "interval": "epoch",
                                 "frequency": self.trainer.check_val_every_n_epoch}}

    def _recon_loss(self, x_target, x_recon, n_elements):
        noise_var = (self.noise_alpha * x_recon.clamp(min=0.0) + self.noise_beta).clamp(min=1e-6)
        per_element = 0.5 * (x_target - x_recon).pow(2) / noise_var + 0.5 * noise_var.log()
        return per_element.sum() / n_elements

    def _kld(self, mu, logvar, n_elements):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n_elements

    def training_step(self, batch, batch_idx):
        x_full, _, lengths = batch
        limit = min(self.hparams.n_times_train, int(lengths.min()))
        if self._t_choices is not None:
            usable = [t for t in self._t_choices if t <= limit] or [limit]
            time_slice = usable[int(torch.randint(len(usable), (1,)).item())]
        elif self.hparams.t_sampling == "uniform":
            time_slice = int(torch.randint(1, limit + 1, (1,)).item())
        else:
            time_slice = limit

        # Encoder sees only the first time_slice bins.
        x_input = x_full[:, :, :time_slice]
        x_recon, mu, logvar, lambdas, abundances, static, bases = self.model(x_input)

        tensors = {"mu": mu, "logvar": logvar, "lambdas": lambdas,
                   "abundances": abundances, "static": static, "x_recon": x_recon}
        bad = [k for k, v in tensors.items() if not torch.isfinite(v).all()]
        if bad:
            raise RuntimeError(f"Non-finite values in {bad} at epoch={self.current_epoch} "
                               f"batch={batch_idx}")

        n_elements = x_input.numel()
        mse = F.mse_loss(x_recon, x_input, reduction="sum") / n_elements
        recon_loss = self._recon_loss(x_input, x_recon, n_elements)
        loss = recon_loss
        if self.hparams.kl_weight > 0:
            kld_loss = self._kld(mu, logvar, n_elements)
            loss = loss + self.hparams.kl_weight * kld_loss
            self.log("kld_loss", kld_loss, prog_bar=True)

        if self.hparams.static_l1_weight > 0:
            penalty = static.mean() / max(self.hparams.dataset_std, 1e-6)
            loss = loss + self.hparams.static_l1_weight * penalty
            self.log("static_l1_penalty", penalty)

        if self.hparams.sum_loss_weight > 0 and time_slice > 1:
            pred_sum, targ_sum = x_recon.sum(-1), x_input.sum(-1)
            var_sum = (self.noise_alpha * pred_sum.clamp(min=0.0)
                       + time_slice * self.noise_beta).clamp(min=1e-6)
            sum_loss = (0.5 * (targ_sum - pred_sum).pow(2) / var_sum + 0.5 * var_sum.log()).mean()
            loss = loss + self.hparams.sum_loss_weight * sum_loss
            self.log("sum_loss", sum_loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", mse)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("noise_alpha", self.noise_alpha.detach(), prog_bar=True)
        self.log("noise_beta", self.noise_beta.detach())
        self.log("noise_alpha_gt", self.hparams.noise_alpha_gt)
        with torch.no_grad():
            self.log("lambda_mean", lambdas.mean())
            self.log("tau_median", (1.0 / lambdas).median())
            self.log("abundance_mean_norm", abundances.mean() / self.hparams.dataset_std)
            self.log("static_per_bin_norm",
                     static.mean() * self.model.frame_duration / self.hparams.dataset_std)
            self.log("irf_t0", self.model.irf_t0.detach(), prog_bar=True)
            self.log("irf_sigma", self.model.irf_sigma.detach(), prog_bar=True)
            self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        time_slice = self.hparams.n_times_train
        x_full, _, lengths = batch
        W = x_full.shape[1]
        x_input = x_full[:, :, :time_slice]
        x_recon, mu, logvar, lambdas, abundances, static, bases = self.model(x_input)

        n_elements = x_input.numel()
        mse = F.mse_loss(x_recon, x_input, reduction="sum") / n_elements
        recon_loss = self._recon_loss(x_input, x_recon, n_elements)
        loss = recon_loss
        if self.hparams.kl_weight > 0:
            loss = loss + self.hparams.kl_weight * self._kld(mu, logvar, n_elements)

        # Extrapolation to the bins the encoder never saw (monitoring only).
        n_valid_full = min(x_full.shape[-1], self.t_full.shape[0])
        if time_slice < n_valid_full:
            x_full_recon, _ = self.model.physics_forward(
                lambdas, abundances, static, bases, time_values=self.t_full[:n_valid_full])
            x_full_recon = x_full_recon / self.model.dataset_std
            target = x_full[:, :, time_slice:n_valid_full]
            pred = x_full_recon[:, :, time_slice:n_valid_full]
            t_idx = torch.arange(time_slice, n_valid_full, device=x_full.device).view(1, 1, -1)
            mask = (t_idx < lengths.view(-1, 1, 1)).float().expand(-1, W, -1)
            if mask.sum() > 0:
                extrap = (F.mse_loss(pred, target, reduction="none") * mask).sum() / mask.sum()
                self.log("val_extrap_loss", extrap, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_mse", mse)
        return loss
