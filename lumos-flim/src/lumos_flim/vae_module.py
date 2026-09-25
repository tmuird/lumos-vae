import pytorch_lightning as pl
import torch

from lumos_flim.vae import FlimVAE


def poisson_half_deviance(x, mu):
    """sum x log(x / mu) per pixel, over channels and bins. [B]

    The model's counts sum to the observed total, so this is the multinomial
    negative log-likelihood less its value at the saturated model: an exact
    likelihood for photon counting, zero for a perfect fit, and about half the
    degrees of freedom for a correct model with Poisson noise.
    """
    mu = mu.clamp(min=1e-10)
    terms = torch.where(x > 0, x * (torch.log(x.clamp(min=1e-10)) - torch.log(mu)), 0.0)
    return terms.sum(dim=(1, 2))


def pearson_chi2(x, mu):
    """Pearson chi-square per pixel. [B]"""
    mu = mu.clamp(min=1e-10)
    return ((x - mu) ** 2 / mu).sum(dim=(1, 2))


def n_free_parameters(n_components: int, n_channels: int) -> int:
    """Per-pixel parameters: rates, fractions (F + 1 summing to one), and the
    background spectrum when there is more than one channel."""
    return 2 * n_components + (n_channels - 1 if n_channels > 1 else 0)


def gt_errors(out, batch):
    """Lifetime and fraction errors against ground truth, both sorted longest first."""
    tau = (1.0 / out["rates"]).detach()
    gt_tau = batch["gt_tau"]
    gt_tau, order = gt_tau.sort(dim=-1, descending=True)
    gt_frac = batch["gt_fraction"].gather(-1, order)
    rel_tau = ((tau - gt_tau).abs() / gt_tau).median(dim=0).values
    frac_err = (out["fractions"].detach() - gt_frac).abs().mean(dim=0)
    bg_err = (out["background"].detach() - batch["gt_background"]).abs().mean()
    return rel_tau, frac_err, bg_err


class FlimModule(pl.LightningModule):
    def __init__(
        self,
        n_bins: int,
        bin_width: float,
        n_channels: int = 1,
        n_components: int = 2,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        decoder_dim: int = 128,
        conv_channels: str = "32,64,128",
        pool_time: int = 7,
        tau_max: float = 0.0,
        irf_t0: float = 1.0,
        irf_sigma: float = 0.1,
        fix_irf: bool = False,
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlimVAE(
            n_bins=n_bins, bin_width=bin_width, n_channels=n_channels,
            n_components=n_components, latent_dim=latent_dim, hidden_dim=hidden_dim,
            decoder_dim=decoder_dim, conv_channels=conv_channels, pool_time=pool_time,
            tau_max=tau_max, irf_t0=irf_t0, irf_sigma=irf_sigma, fix_irf=fix_irf,
        )
        self.dof = n_channels * n_bins - n_free_parameters(n_components, n_channels)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _step(self, batch, stage):
        x = batch["x"]
        out = self.model(x)
        recon = poisson_half_deviance(x, out["expected"])
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        loss = (recon + self.hparams.kl_weight * kl).mean()

        on_step = stage == "train"
        log = dict(on_step=on_step, on_epoch=True, batch_size=x.shape[0])
        self.log(f"{stage}_loss", loss, prog_bar=True, **log)
        self.log(f"{stage}_nll", recon.mean(), prog_bar=stage == "val", **log)
        self.log(f"{stage}_kl", kl.mean(), **log)
        with torch.no_grad():
            chi2r = pearson_chi2(x, out["expected"]) / self.dof
            self.log(f"{stage}_chi2r", chi2r.mean(), prog_bar=True, **log)
            tau = 1.0 / out["rates"]
            for i in range(tau.shape[1]):
                self.log(f"{stage}_tau{i}_median", tau[:, i].median(), **log)
                self.log(f"{stage}_frac{i}_mean", out["fractions"][:, i].mean(), **log)
            self.log(f"{stage}_background_mean", out["background"].mean(), **log)
            if stage == "val":
                self.log("irf_t0", self.model.irf_t0.detach(), prog_bar=True)
                self.log("irf_sigma", self.model.irf_sigma.detach(), prog_bar=True)
            if "gt_tau" in batch:
                rel_tau, frac_err, bg_err = gt_errors(out, batch)
                for i in range(rel_tau.shape[0]):
                    self.log(f"{stage}_gt_tau{i}_rel_err", rel_tau[i], **log)
                    self.log(f"{stage}_gt_frac{i}_abs_err", frac_err[i], **log)
                self.log(f"{stage}_gt_background_abs_err", bg_err, **log)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.hparams.max_epochs, 1),
            eta_min=self.hparams.learning_rate * 1e-2,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
