from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from lumos.vae import VAE


class VAEModule(pl.LightningModule):

    def __init__(
        self,
        physics_model: str = "pointsample",
        n_wavenumbers: int = 1024,
        n_times_train: int = 1,
        n_full_timepoints: Optional[int] = None,
        time_values: Optional[torch.Tensor] = None,
        wavenumbers: Optional[torch.Tensor] = None,
        dataset_std: float = 1.0,
        dataset_mean: float = 0.0,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        kl_weight: float = 1,
        raman_l1_weight: float = 0.0,
        noise_alpha_gt: float = 0.0,
        noise_beta_gt: float = 0.0,
        raman_lr_multiplier: float = 1.0,
        mog_lr_multiplier: float = 3.0,
        noise_lr_multiplier: float = 10.0,
        lr_scheduler_patience: int = 50,
        semi_supervised: bool = False,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["time_values", "wavenumbers"])

        self.log_alpha = nn.Parameter(torch.tensor(-4.0))
        self.log_beta = nn.Parameter(torch.tensor(-6.0))
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
            physics_model=physics_model,
            dataset_std=dataset_std,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
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

        dictionary_bases = None
        if hasattr(ds, "dictionary_bases") and ds.dictionary_bases is not None:
            dictionary_bases = torch.as_tensor(ds.dictionary_bases, dtype=torch.float32)

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
            physics_model=datamodule.config.physics_model,
            dataset_std=std,
            dictionary_bases=dictionary_bases,
            frame_duration=frame_duration,
            noise_alpha_gt=noise_alpha_gt,
            noise_beta_gt=noise_beta_gt,
            **kwargs,
        )

        return model

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
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
        dec = self.model.decoder
        if self.model.raman_mode == "pseudo_voigt":
            raman_params = set(dec.head_raman_amplitudes.parameters())
            global_shape_params = {
                self.model.peak_positions_raw,
                self.model.log_fwhm_L,
                self.model.log_fwhm_G,
            }
        else:
            raman_params = set(dec.head_raman.parameters()) | set(
                dec.trunk_raman.parameters()
            )
            global_shape_params = set()
        raman_param_ids = {id(p) for p in raman_params | global_shape_params}

        mog_mean_params = set()
        mog_scale_params = set()
        if hasattr(self.model, "mog_means"):
            mog_mean_params = {self.model.mog_means, self.model.mog_logits}
            mog_scale_params = {self.model.mog_log_scales}

        mog_param_ids = {id(p) for p in mog_mean_params | mog_scale_params}

        base_params = [
            p
            for p in self.model.parameters()
            if id(p) not in raman_param_ids and id(p) not in mog_param_ids
        ]
        base_params += [p for p in global_shape_params if p.requires_grad]

        param_groups = [
            {"params": base_params},
            {
                "params": [self.log_alpha, self.log_beta],
                "lr": self.hparams.learning_rate * self.hparams.noise_lr_multiplier,
            },
            {
                "params": list(raman_params),
                "lr": self.hparams.learning_rate * self.hparams.raman_lr_multiplier,
            },
        ]

        if mog_mean_params:
            param_groups.append(
                {
                    "params": list(mog_mean_params),
                    "lr": self.hparams.learning_rate
                    * self.hparams.mog_lr_multiplier
                    * 3.0,
                }
            )
        if mog_scale_params:
            param_groups.append(
                {
                    "params": list(mog_scale_params),
                    "lr": self.hparams.learning_rate * self.hparams.mog_lr_multiplier,
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
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
        time_slice = self.hparams.n_times_train

        # Safely unpack based on the dataset output
        if len(batch) == 4:
            x_full, _, lengths, is_test = batch
        elif len(batch) == 3:
            x_full, _, lengths = batch
            is_test = None
        else:
            x_full, _ = batch[:2]
            lengths = torch.full(
                (x_full.shape[0],), x_full.shape[-1], device=x_full.device
            )
            is_test = None

        B, W, T_max = x_full.shape

        # Encoder always sees ONLY the first n_times_train frames.
        x_input = x_full[:, :, :time_slice]

        x_recon, mu, logvar, lambdas, abundances, raman, bases, c_fluo_counts = (
            self.model(x_input)
        )

        if torch.isnan(raman).any() or torch.isnan(x_recon).any():
            raise RuntimeError(
                f"NaN detected in model outputs at epoch={self.current_epoch} "
                f"batch={batch_idx}. Check input data and encoder."
            )

        # # validity and semi-supervised masks
        # if self.hparams.semi_supervised and T_max > time_slice:
        #     T_target = T_max
        #     x_recon_phys, _ = self.model.physics_forward(
        #         lambdas,
        #         abundances,
        #         raman + c_fluo_counts,
        #         self.model.bases,
        #         time_values=self.t_full[:T_target],
        #     )
        #     x_recon_norm = x_recon_phys / self.model.dataset_std
        #     x_target = x_full
        # else:
        T_target = time_slice
        x_target = x_input
        x_recon_norm = x_recon[: x_target.shape[0], :, :time_slice]

        # 1. Base Mask: True if frame < true sample length (ignores zero-padding)
        t_idx = torch.arange(T_target, device=x_full.device).view(1, 1, -1)
        lengths_view = lengths.view(-1, 1, 1)
        # final_mask = t_idx < lengths_view

        # # 2. Semi-Supervised Mask: Test samples ignore frames beyond `time_slice`
        # if self.hparams.semi_supervised and is_test is not None:
        #     test_mask = is_test.view(-1, 1, 1)
        #     extrap_window = t_idx >= time_slice
        #     # Force test samples to False during the extrapolation window
        #     final_mask = final_mask & ~(test_mask & extrap_window)

        # final_mask = final_mask.float().expand(-1, W, -1)
        # mask_sum = final_mask.sum().clamp(min=1.0)
        # print(f"Final Mask Shape:  {final_mask.shape}")

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
                f_L = F.softplus(self.model.log_fwhm_L) + 5.0
                f_G = F.softplus(self.model.log_fwhm_G) + 1.0
                self.log("fwhm_G", f_G.detach(), prog_bar=True)
                self.log("fwhm_L_mean", f_L.mean().detach(), prog_bar=True)
                self.log("fwhm_L_min", f_L.min().detach(), prog_bar=False)
                self.log("fwhm_L_max", f_L.max().detach(), prog_bar=False)

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
            opt = self.optimizers()
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=False)
            self.log("lr_raman", opt.param_groups[1]["lr"], prog_bar=False)
            if len(opt.param_groups) > 2:
                self.log("lr_mog", opt.param_groups[2]["lr"], prog_bar=False)

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

        x_recon, mu, logvar, lambdas, abundances, raman, bases, c_fluo_counts = (
            self.model(x_input)
        )
        # raman_1, raman_2 = raman[: raman.shape[0] // 2], raman[raman.shape[0] // 2 :]

        # In semi-supervised mode val samples are all non-test, so they get
        # the full time series as reconstruction target (same as training_step
        # does for non-test samples). This keeps val_recon_loss comparable to
        # the training loss and gives the LR scheduler a meaningful signal.
        n_valid_full = min(T_max, self.t_full.shape[0])
        if self.hparams.semi_supervised and n_valid_full > time_slice:
            x_recon_phys_full, _ = self.model.physics_forward(
                lambdas,
                abundances,
                raman + c_fluo_counts,
                self.model.bases,
                time_values=self.t_full[:n_valid_full],
            )
            x_recon_norm = x_recon_phys_full / self.model.dataset_std
            x_target = x_full[:, :, :n_valid_full]
            T_target = n_valid_full
        else:

            x_target = x_input
            x_recon_norm = x_recon[: x_target.shape[0]]  # already [B, W, time_slice]
            T_target = time_slice

        # Mask to ignore zero-padding (lengths may exceed T_target on padded samples)
        t_idx = torch.arange(T_target, device=x_full.device).view(1, 1, -1)
        # valid_mask = (t_idx < lengths.view(-1, 1, 1)).float().expand(-1, W, -1)
        # mask_sum = valid_mask.sum().clamp(min=1.0)

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
            if not (self.hparams.semi_supervised):
                # Full reconstruction not yet computed; run it now for monitoring.
                x_recon_phys_full, _ = self.model.physics_forward(
                    lambdas,
                    abundances,
                    raman + c_fluo_counts,
                    self.model.bases,
                    time_values=self.t_full[:n_valid_full],
                )
                x_recon_full = x_recon_phys_full / self.model.dataset_std
            else:
                x_recon_full = x_recon_norm  # already covers [0, n_valid_full)

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
