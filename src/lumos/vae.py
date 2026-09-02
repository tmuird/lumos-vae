import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


from lumos.physics import reconstruct_time_series_factored_torch


_WIDTH_EPS = 1e-4


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        decoder_dim: int = 256,
        n_fluorophores: int = 3,
        n_wavenumbers: int = 1024,
        n_times_train: int = 20,
        n_full_timepoints: Optional[int] = None,
        time_values: Optional[torch.Tensor] = None,
        wavenumbers: Optional[torch.Tensor] = None,
        frame_duration: float = 0.1,
        dataset_std: float = 1.0,
        n_gaussian_components: int = 5,
        lambda_min: float = 0.001,
        n_raman_peaks: int = 50,
        pool_spectral: int = 16,
        pool_time: int = 1,
        fwhm_G: float = 0.0, # Measured instrument response in cm^-1, when the optics are known.
        conv_channels: str = "64,128,256,512",
        **kwargs,
    ):
        super().__init__()



        self.frame_duration = frame_duration
        self.lambda_min = lambda_min

        self.dataset_std = dataset_std
        self.n_wavenumbers = n_wavenumbers
        self.n_times_train = n_times_train

        # n_full_timepoints: number of frames in the full dataset - used only for
        # validation extrapolation
        if n_full_timepoints is None:
            n_full_timepoints = n_times_train
        self.n_full_timepoints = n_full_timepoints

        # self.t - training window time axis, shape [n_times_train].
        # Stores only the frames the encoder is trained on.
        if time_values is not None:
            t_train = torch.as_tensor(time_values, dtype=torch.float32)[:n_times_train]
        else:
            t_train = torch.zeros(
                n_times_train
            )  # placeholder; overwritten by checkpoint
        self.register_buffer("t", t_train)  # [n_times_train]

        if wavenumbers is not None:
            if wavenumbers.ndim > 1:
                wavenumbers = wavenumbers.squeeze()
            self.register_buffer(
                "wavenumbers", torch.as_tensor(wavenumbers, dtype=torch.float32)
            )
        else:
            self.register_buffer("wavenumbers", torch.zeros(n_wavenumbers))

        wn_for_norm = (
            wavenumbers
            if wavenumbers is not None
            else torch.linspace(400, 1800, n_wavenumbers)
        )
        wn_min, wn_max = wn_for_norm.min(), wn_for_norm.max()
        wn_norm = 2.0 * (wn_for_norm - wn_min) / (wn_max - wn_min + 1e-8) - 1.0
        self.register_buffer("wn_norm", wn_norm)

        # t_norm: positional encoding for the training window.
        # lower frame counts receive a crop
        t_end = self.t[-1]
        t_norm = 2.0 * (self.t - self.t[0]) / (t_end - self.t[0] + 1e-8) - 1.0
        self.register_buffer("t_norm_full", t_norm)  # [n_times_train], spans [-1, +1]

        self.n_fluorophores = n_fluorophores

        # Widths below are derived from the wavenumber axis, not chosen.
        if wavenumbers is not None:
            wn_t = torch.as_tensor(wavenumbers, dtype=torch.float32)
            span = float(wn_t.max() - wn_t.min())
            # Widest pixel, so a width at the floor is resolvable everywhere.
            steps = torch.diff(wn_t).abs()
            spacing = float(steps.max()) if steps.numel() else span
        else:
            # Loading a checkpoint: these only seed values the state dict
            # then overwrites.
            span, spacing = float(n_wavenumbers), 1.0

        self.n_raman_peaks = n_raman_peaks

        def inv_softplus(val):
            return math.log(math.expm1(max(val, _WIDTH_EPS)))

        # Peaks tile the window evenly, spanning it fully.
        self.peak_positions_raw = nn.Parameter(torch.linspace(0.0, 1.0, n_raman_peaks))

        # Start each peak as wide as the spacing between peaks, so the initial
        # basis covers the window once without piling up.
        self.log_fwhm_L = nn.Parameter(
            torch.full((n_raman_peaks,), inv_softplus(span / n_raman_peaks - spacing))
        )

        # The instrument response cannot be narrower than the sampling interval,

        self.fwhm_G_pinned = float(fwhm_G) if fwhm_G else 0.0
        g_init = self.fwhm_G_pinned or 4.0 * spacing
        self.log_fwhm_G = nn.Parameter(
            torch.tensor(inv_softplus(g_init - spacing)),
            requires_grad=not fwhm_G,
        )

        n_k = n_gaussian_components
        # Fluorophore centres tile the normalised window, and the components
        # within one fluorophore tile its own share of that window.
        centres = torch.linspace(-1.0, 1.0, n_fluorophores)  # [F]
        share = 2.0 / n_fluorophores
        offsets = (
            torch.zeros(1)
            if n_k == 1
            else torch.linspace(-share / 2, share / 2, n_k)
        )
        self.mog_means = nn.Parameter(centres[:, None] + offsets[None, :])
        # Start at the window half width. Starting narrow strands them.
        self.mog_log_scales = nn.Parameter(torch.zeros(n_fluorophores, n_k))
        self.mog_logits = nn.Parameter(torch.zeros(n_fluorophores, n_k))

        if isinstance(conv_channels, str):
            conv_channels = tuple(int(v) for v in conv_channels.split(","))
        self.encoder = VAEEncoder(
            self.n_wavenumbers, hidden_dim, latent_dim, conv_channels=conv_channels,
            n_times_train=n_times_train,
            pool_spectral=pool_spectral, pool_time=pool_time,
        )

        self.decoder = ParameterDecoder(
            latent_dim,
            decoder_dim,
            self.n_fluorophores,
            self.n_wavenumbers,
            n_raman_peaks=n_raman_peaks,
        )

        # Rates are only observable between one frame and the full window.
        t_span = float(self.t[-1] - self.t[0]) if self.t.numel() > 1 else frame_duration
        fastest = 1.0 / max(frame_duration, 1e-6)
        slowest = 1.0 / max(t_span, frame_duration)
        targets = torch.logspace(
            math.log10(slowest), math.log10(fastest), self.n_fluorophores
        )
        self.decoder.head_lambda.bias.data.copy_(
            torch.log(torch.expm1((targets - self.lambda_min).clamp(min=_WIDTH_EPS)))
        )

        print(f"[VAE init] n_raman_peaks={n_raman_peaks}")
        print(
            f"[VAE init] head_abundance: weight_norm={self.decoder.head_abundance.weight.norm():.4g}"
            f"  bias={self.decoder.head_abundance.bias.data.tolist()}  (softplus -> {torch.nn.functional.softplus(self.decoder.head_abundance.bias).data.tolist()})"
        )

        # Register a pre-hook to resize buffers before state_dict is applied.
        self._register_load_state_dict_pre_hook(self._resize_buffers_hook)

    def _resize_buffers_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Official PyTorch hook to resize buffers to match checkpoint shape."""
        for name in ["t", "wavenumbers", "wn_norm", "t_norm_full"]:
            key = prefix + name
            if key in state_dict:
                checkpoint_value = state_dict[key]
                current_value = getattr(self, name, None)
                if (
                    current_value is not None
                    and current_value.shape != checkpoint_value.shape
                ):
                    self.register_buffer(name, torch.zeros_like(checkpoint_value))

    @property
    def _mog_bases(self) -> torch.Tensor:
        """Evaluate global MoG bases at the registered wavenumber grid. [F, W]"""
        weights = F.softmax(self.mog_logits, dim=-1)  # [F, K], sums to 1
        scales = self.mog_log_scales.exp() + _WIDTH_EPS  # [F, K]

        diff = self.wn_norm[None, None, :] - self.mog_means[:, :, None]  # [F, K, W]
        gaussians = torch.exp(-0.5 * (diff / scales[:, :, None]).pow(2))
        bases = (weights[:, :, None] * gaussians).sum(dim=1)  # [F, W]

        # L2 normalise: unit-energy bases give equal weight to narrow and broad shapes.
        norms = bases.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return bases / norms

    @property
    def bases(self) -> torch.Tensor:
        """Current fluorophore basis spectra, L2-normalised. [F, W]"""
        return self._mog_bases

    def _spacing(self, reduce):
        """Channel width by the given reduction, one if the axis is not set yet.

        Reconstructing from a checkpoint leaves the wavenumber buffer zeroed
        until the state dict lands, so anything derived from it has to tolerate
        a degenerate axis rather than dividing by zero.
        """
        steps = torch.diff(self.wavenumbers).abs()
        if steps.numel() == 0:
            return self.wavenumbers.new_tensor(1.0)
        value = reduce(steps)
        return value if float(value) > 0 else self.wavenumbers.new_tensor(1.0)

    @property
    def width_floor(self):
        """Widest channel. A profile narrower than this is not resolvable.

        Derived from the wavenumber buffer rather than stored, so it is always
        consistent with the axis a checkpoint was trained on.
        """
        return self._spacing(torch.Tensor.max)

    @property
    def mean_spacing(self):
        """Mean channel width, the dispersion the blur kernel is defined at."""
        return self._spacing(torch.Tensor.mean)

    @property
    def peak_positions(self):
        """Peak positions in [0, 1].

        """
        return self.peak_positions_raw.clamp(0.0, 1.0)

    def _get_voigt_raman(self, amplitudes):
        """Sum of Lorentzians, blurred once by the instrument response.
        """
        wn = self.wavenumbers  # [W], actual cm^-1
        x0 = wn.min() + (wn.max() - wn.min()) * self.peak_positions  # [N_peaks]
        fwhm_L = F.softplus(self.log_fwhm_L) + self.width_floor
        gamma = fwhm_L / 2.0  # HWHM [N_peaks]

        delta = x0[:, None] - wn[None, :]  # [N_peaks, W]
        g = gamma[:, None]
        lorentz = g / (delta**2 + g**2)  # area-normalised up to a factor of pi
        return self._instrument_blur(amplitudes @ lorentz)  # [B, W]

    def _instrument_blur(self, spectrum):
        """Blur [B, W] by one Gaussian, the instrument response.
        """
        step = self.mean_spacing
        if getattr(self, "fwhm_G_pinned", 0.0):
            fwhm_G = self.log_fwhm_G.new_tensor(self.fwhm_G_pinned)
        else:
            fwhm_G = F.softplus(self.log_fwhm_G) + self.width_floor
        sigma = fwhm_G / (2.0 * math.sqrt(2.0 * math.log(2.0))) / step # turn fwhm into standard deviation
        radius = int(max(1, math.ceil(4.0 * float(sigma.detach()))))
        offsets = torch.arange(
            -radius, radius + 1, device=spectrum.device, dtype=spectrum.dtype
        )
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()

        padded = F.pad(spectrum[:, None, :], (radius, radius), mode="reflect")
        return F.conv1d(padded, kernel[None, None, :]).squeeze(1)


    def forward(self, x, sample=None):
        # x: [B, W, T] - std-normalised signal (may be slightly negative after dark subtraction)

        # Derive t_use from actual input length.
        # T must be <= n_times_train. t_pe is derived from the longest series available

        T = x.shape[-1]
        if T > self.n_times_train:
            raise ValueError(
                f"Input has T={T} time frames but model was trained on "
                f"n_times_train={self.n_times_train}. Crop the input to "
                f"<= {self.n_times_train} frames before encoding."
            )
        t_use = self.t[:T]
        t_norm = self.t_norm_full[:T]

        mu, logvar = self.encoder(x, self.wn_norm, t_norm)
        # Sample the posterior during training; use the mean for a
        # deterministic estimate at inference unless sampling is requested.
        if sample is None:
            sample = self.training
        z = self.reparameterize(mu, logvar) if sample else mu

        lambdas_raw, raman_out, abundances_raw = self.decoder(z)

        lambdas = F.softplus(lambdas_raw) + self.lambda_min

        amplitudes = F.softplus(raman_out)
        self._raman_amplitudes = amplitudes  # exposed for the training step L1
        raman_spectrum = self._get_voigt_raman(amplitudes)  # [B, W]

        # Scale Raman to physical counts/sec.
        raman_counts = (raman_spectrum * self.dataset_std) / self.frame_duration


        bases_normalised = self.bases  # [F, W]

        # Abundances are learned end-to-end from the decoder; positivity enforced by softplus.
        # Scaled by dataset_std so the network doesn't have to output huge values directly.
        abundances = F.softplus(abundances_raw) * self.dataset_std

        x_recon_phys, bases = self.physics_forward(
            lambdas, abundances, raman_counts, bases_normalised,
            time_values=t_use,
        )

        x_recon = (x_recon_phys) / self.dataset_std
        # Exposed for dashboard visualization
        return (
            x_recon,
            mu,
            logvar,
            lambdas,
            abundances,
            raman_counts,
            bases_normalised
        )

    def physics_forward(
        self,
        lambdas,
        abundances,
        raman,
        bases_normalised=None,
        time_values=None,
    ):
        if bases_normalised is None:
            bases_normalised = self.bases
        if time_values is None:
            time_values = self.t

        x_recon = reconstruct_time_series_factored_torch(
            raman=raman,
            bases=bases_normalised,
            effective_amplitudes=abundances,
            decay_rates=lambdas,
            time_values=time_values,
            frame_duration=self.frame_duration,
        )

        if x_recon.shape[-1] != time_values.shape[0]:
            x_recon = x_recon.transpose(-1, -2)

        return x_recon, bases_normalised

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class VAEEncoder(nn.Module):
    """Joint 2D CNN encoder for [W, T] spectrograms.
    """

    def __init__(self, n_wavenumbers, hidden_dim, latent_dim,
                 conv_channels=(64, 128, 256, 512), n_times_train=16,
                 pool_spectral=16, pool_time=1):
        super().__init__()
        self.conv_channels = tuple(conv_channels)
        self.n_times_flat = n_times_train
        # The pooled grid is a budget: (16, 1) spends every slot on wavenumber
        self.pool_spectral = pool_spectral
        self.pool_time = pool_time

        kernels = [(7, 3), (7, 3), (5, 3), (5, 3)]
        paddings = [(3, 1), (3, 1), (2, 1), (2, 1)]
        layers, in_channels = [], 3
        for out_channels, kernel, padding in zip(self.conv_channels, kernels, paddings):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                          stride=(2, 1), padding=padding)
            )
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)

        flat_dim = self.conv_channels[-1] * self.pool_spectral * self.pool_time

        self.fc_net = nn.Sequential(nn.Linear(flat_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, wn_norm, t_norm):
        """
        x       : [B, W, T]  std-normalised signal
        wn_norm : [W]        wavenumber positional encoding in [-1, 1]
        t_norm  : [T]        time positional encoding in [-1, 1]
        """
        B, W, T = x.shape

        x_2d = F.relu(x).unsqueeze(1)  # [B, 1, W, T]
        wn_pe = wn_norm.view(1, 1, W, 1).expand(B, 1, W, T)  # [B, 1, W, T]
        t_pe = t_norm.view(1, 1, 1, T).expand(B, 1, W, T)  # [B, 1, W, T]
        x_in = torch.cat([x_2d, wn_pe, t_pe], dim=1)

        # Here we are reshaping the data, and PE arrays to be the same so that they are then concatenated along the channel dimension.
        # This means that every B,W,T point will have 3 values, its intensity (C0), its wavenumber (C1), and its time (C2).

        # The convolutional stack preserves T, but divides W by 16 (if W=1024, W becomes 64)
        h = self.conv_net(x_in)  # [B, 128, W/16, T]

        # Adaptive pooling:
        h = F.adaptive_avg_pool2d(
            h, output_size=(self.pool_spectral, self.pool_time))

        h = h.flatten(1)
        h = self.fc_net(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar.clamp(-20.0, 10.0) # clamped or it runs away


class ParameterDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        decoder_dim,
        n_fluorophores,
        n_wavenumbers,
        n_raman_peaks: int = 150,
    ):
        super().__init__()
        self.n_wavenumbers = n_wavenumbers

        self.trunk_shared = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim),
            nn.LeakyReLU(0.2),
        )
        self.trunk_lambda = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.LeakyReLU(0.2),
        )
        self.trunk_abundance = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.LeakyReLU(0.2),
        )
        self.trunk_raman = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.LeakyReLU(0.2),
        )

        self.head_lambda = nn.Linear(decoder_dim, n_fluorophores)
        self.head_abundance = nn.Linear(decoder_dim, n_fluorophores)

        # One positive amplitude per Voigt component.
        self.head_raman_amplitudes = nn.Linear(decoder_dim, n_raman_peaks)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



    def forward(self, z):
        h = self.trunk_shared(z)
        h_lam = self.trunk_lambda(h)
        h_abd = self.trunk_abundance(h)
        h_ram = self.trunk_raman(h)

        lambdas = self.head_lambda(h_lam)
        abundances_raw = self.head_abundance(h_abd)

        raman_out = self.head_raman_amplitudes(h_ram)
        return lambdas, raman_out, abundances_raw
