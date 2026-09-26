"""LUMOS VAE with the photobleaching physics swapped for TCSPC.

Kept from lumos-vae: the 2D CNN encoder over [channel, time] with positional
encodings, the parameter decoder with separate trunks, per-sample softplus
rates, softplus abundances scaled by the dataset std, global mixture-of-
Gaussian bases, a global Gaussian instrument response with an optional pin.

Changed: the time axis is TCSPC delay bins, the wavenumber axis is detection
channels, the static Raman spectrum is a static background (flat in delay, so
one amplitude per channel instead of Voigt peaks), and ``exp(-lambda t)`` is
replaced by the exact periodic IRF-convolved bin probability.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from lumos_flim.physics import decay_histograms

_WIDTH_EPS = 1e-4
_SIGMA_FLOOR = 0.005  # ns


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        decoder_dim: int = 256,
        n_fluorophores: int = 2,
        n_wavenumbers: int = 1,  # detection channels; name kept from LUMOS
        n_times_train: int = 56,
        n_full_timepoints: Optional[int] = None,
        time_values: Optional[torch.Tensor] = None,
        frame_duration: float = 0.2229,  # TCSPC bin width in ns
        dataset_std: float = 1.0,
        n_gaussian_components: int = 3,
        lambda_min: float = 0.0,
        pool_spectral: int = 1,
        pool_time: int = 1,
        irf_t0: float = 1.0,
        irf_sigma: float = 0.1,
        irf_sigma_pinned: float = 0.0,  # known IRF width in ns, as fwhm_G in LUMOS
        conv_channels: str = "8,16,32,64",
        **kwargs,
    ):
        super().__init__()
        self.frame_duration = frame_duration
        self.dataset_std = dataset_std
        self.n_wavenumbers = n_wavenumbers
        self.n_times_train = n_times_train
        self.n_full_timepoints = n_full_timepoints or n_times_train
        self.period = self.n_full_timepoints * frame_duration
        # A lifetime longer than the period is flat and cannot be told from
        # the static term, so by default rates stop at one per period.
        self.lambda_min = lambda_min if lambda_min > 0 else 1.0 / self.period

        if time_values is not None:
            t_train = torch.as_tensor(time_values, dtype=torch.float32)[:n_times_train]
        else:
            t_train = torch.arange(n_times_train, dtype=torch.float32) * frame_duration
        self.register_buffer("t", t_train)

        # Channel positional encoding in [-1, 1], zero for a single channel.
        wn_norm = (torch.linspace(-1.0, 1.0, n_wavenumbers) if n_wavenumbers > 1
                   else torch.zeros(1))
        self.register_buffer("wn_norm", wn_norm)
        t_norm = 2.0 * (self.t - self.t[0]) / (self.t[-1] - self.t[0] + 1e-8) - 1.0
        self.register_buffer("t_norm_full", t_norm)

        self.n_fluorophores = n_fluorophores

        # Instrument response. The width can be pinned when it is known, as
        # the Raman instrument blur can in LUMOS.
        self.irf_t0 = nn.Parameter(torch.tensor(float(irf_t0)))
        self.irf_sigma_pinned = float(irf_sigma_pinned)
        s0 = self.irf_sigma_pinned or irf_sigma
        self.log_irf_sigma = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(s0 - _SIGMA_FLOOR, 1e-6)))),
            requires_grad=not irf_sigma_pinned,
        )

        n_k = n_gaussian_components
        centres = torch.linspace(-1.0, 1.0, n_fluorophores)
        share = 2.0 / n_fluorophores
        offsets = torch.zeros(1) if n_k == 1 else torch.linspace(-share / 2, share / 2, n_k)
        self.mog_means = nn.Parameter(centres[:, None] + offsets[None, :])
        self.mog_log_scales = nn.Parameter(torch.zeros(n_fluorophores, n_k))
        self.mog_logits = nn.Parameter(torch.zeros(n_fluorophores, n_k))

        if isinstance(conv_channels, str):
            conv_channels = tuple(int(v) for v in conv_channels.split(","))
        self.encoder = VAEEncoder(
            n_wavenumbers, hidden_dim, latent_dim, conv_channels=conv_channels,
            n_times_train=n_times_train, pool_spectral=pool_spectral, pool_time=pool_time,
        )
        self.decoder = ParameterDecoder(latent_dim, decoder_dim, n_fluorophores, n_wavenumbers)

        # Rates are only observable between one bin and the full period.
        fastest = 1.0 / frame_duration
        slowest = 1.0 / self.period
        targets = torch.logspace(math.log10(slowest), math.log10(fastest), n_fluorophores)
        self.decoder.head_lambda.bias.data.copy_(
            torch.log(torch.expm1((targets - self.lambda_min).clamp(min=_WIDTH_EPS)))
        )

    @property
    def irf_sigma(self):
        if self.irf_sigma_pinned:
            return self.log_irf_sigma.new_tensor(self.irf_sigma_pinned)
        return F.softplus(self.log_irf_sigma) + _SIGMA_FLOOR

    @property
    def bases(self) -> torch.Tensor:
        """Global MoG spectra over channels, L2-normalised. [F, C]"""
        weights = F.softmax(self.mog_logits, dim=-1)
        scales = self.mog_log_scales.exp() + _WIDTH_EPS
        diff = self.wn_norm[None, None, :] - self.mog_means[:, :, None]
        bases = (weights[:, :, None] * torch.exp(-0.5 * (diff / scales[:, :, None]) ** 2)).sum(1)
        return bases / bases.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def forward(self, x, sample=None, scale=1):
        # x: [B, C, T] std-normalised counts
        T = x.shape[-1]
        if T > self.n_times_train:
            raise ValueError(f"Input has T={T} bins but the model was trained on "
                             f"n_times_train={self.n_times_train}.")
        t_use = self.t[:T]
        mu, logvar = self.encoder(x, self.wn_norm, self.t_norm_full[:T])
        if sample is None:
            sample = self.training
        z = self.reparameterize(mu, logvar, scale) if sample else mu

        lambdas_raw, static_out, abundances_raw = self.decoder(z)
        lambdas = F.softplus(lambdas_raw) + self.lambda_min

        # Static background as counts per ns per channel, like raman_counts in
        # LUMOS (counts per second); the physics multiplies by the bin width.
        static_amp = F.softplus(static_out)
        self._static_amplitudes = static_amp
        static_counts = static_amp * self.dataset_std / self.frame_duration

        bases = self.bases
        # Effective amplitudes: photons per period from each component.
        abundances = F.softplus(abundances_raw) * self.dataset_std

        x_recon_phys, _ = self.physics_forward(lambdas, abundances, static_counts, bases,
                                               time_values=t_use)
        x_recon = x_recon_phys / self.dataset_std
        return x_recon, mu, logvar, lambdas, abundances, static_counts, bases

    def physics_forward(self, lambdas, abundances, static, bases=None, time_values=None):
        """Expected counts [B, C, T] at the bins starting at ``time_values``.

        The decay is periodic, so it is always evaluated over the full period
        and then read off at the requested bins.
        """
        if bases is None:
            bases = self.bases
        if time_values is None:
            time_values = self.t
        decays = decay_histograms(lambdas, self.irf_t0, self.irf_sigma,
                                  self.n_full_timepoints, self.frame_duration)  # [B, F, N]
        idx = torch.round(time_values / self.frame_duration).long().clamp(0, self.n_full_timepoints - 1)
        decays = decays[..., idx]
        fluorescence = torch.einsum("bf,fc,bft->bct", abundances, bases, decays)
        return fluorescence + static[:, :, None] * self.frame_duration, bases

    def reparameterize(self, mu, logvar, scale=1.0):
        std = torch.exp(0.5 * logvar) * scale
        return mu + torch.randn_like(std) * std


class VAEEncoder(nn.Module):
    """Joint 2D CNN encoder for [C, T] histograms, as in LUMOS."""

    def __init__(self, n_wavenumbers, hidden_dim, latent_dim,
                 conv_channels=(8, 16, 32, 64), n_times_train=56,
                 pool_spectral=1, pool_time=1):
        super().__init__()
        self.conv_channels = tuple(conv_channels)
        self.pool_spectral = pool_spectral
        self.pool_time = pool_time

        kernels = [(7, 3), (7, 3), (5, 3), (5, 3)]
        paddings = [(3, 1), (3, 1), (2, 1), (2, 1)]
        layers, in_channels = [], 3
        for out_channels, kernel, padding in zip(self.conv_channels, kernels, paddings):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                 stride=(2, 1), padding=padding),
                       nn.LeakyReLU(0.2)]
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)

        flat_dim = self.conv_channels[-1] * pool_spectral * pool_time
        self.fc_net = nn.Sequential(nn.Linear(flat_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, wn_norm, t_norm):
        B, W, T = x.shape
        x_2d = F.relu(x).unsqueeze(1)
        wn_pe = wn_norm.view(1, 1, W, 1).expand(B, 1, W, T)
        t_pe = t_norm.view(1, 1, 1, T).expand(B, 1, W, T)
        h = self.conv_net(torch.cat([x_2d, wn_pe, t_pe], dim=1))
        h = F.adaptive_avg_pool2d(h, output_size=(self.pool_spectral, self.pool_time))
        h = self.fc_net(h.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-20.0, 10.0)


class ParameterDecoder(nn.Module):
    def __init__(self, latent_dim, decoder_dim, n_fluorophores, n_wavenumbers):
        super().__init__()
        self.trunk_shared = nn.Sequential(nn.Linear(latent_dim, decoder_dim), nn.LeakyReLU(0.2))
        self.trunk_lambda = nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.LeakyReLU(0.2))
        self.trunk_abundance = nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.LeakyReLU(0.2))
        self.trunk_static = nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.LeakyReLU(0.2))

        self.head_lambda = nn.Linear(decoder_dim, n_fluorophores)
        self.head_abundance = nn.Linear(decoder_dim, n_fluorophores)
        # One amplitude per channel; the static term has no shape in delay.
        self.head_static = nn.Linear(decoder_dim, n_wavenumbers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        h = self.trunk_shared(z)
        return (self.head_lambda(self.trunk_lambda(h)),
                self.head_static(self.trunk_static(h)),
                self.head_abundance(self.trunk_abundance(h)))
