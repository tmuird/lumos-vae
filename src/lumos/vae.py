import math
from enum import Enum
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


from lumos.physics import (
    evaluate_polynomial_bases_torch,
    reconstruct_time_series_factored_torch,
    reconstruct_time_series_integrated_torch,
    reconstruct_time_series_torch,
)


class BasisMode(str, Enum):
    POLYNOMIAL = "polynomial"
    MOG = "mog"
    DICTIONARY = "dictionary"


_FWHM_L_MIN = 2.0  # narrowest physically realisable biological Raman Lorentzian
_FWHM_G_MIN = 1.0  # below any real CCD spectrometer IRF


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
        physics_model: str = "factored",
        dataset_std: float = 1.0,
        basis_mode: str = "mog",
        polynomial_degree: int = 3,
        n_gaussian_components: int = 5,
        dictionary_bases: Optional[torch.Tensor] = None,
        dynamic_bases: bool = False,
        lambda_min: float = 0.001,
        decay_mode: str = "per_sample",
        raman_mode: str = "conv",
        n_raman_peaks: int = 50,
        fwhm_G_init: float = 1.0,
        fwhm_G_trainable: bool = True,
        conv_channels: str = "64,128,256,512",
        **kwargs,
    ):
        super().__init__()

        # Normalise separator so "pseudo-voigt" and "pseudo_voigt" are equivalent.
        raman_mode = raman_mode.replace("-", "_")

        self.physics_model = physics_model
        self.basis_mode = BasisMode(basis_mode)
        self.raman_mode = raman_mode
        self.frame_duration = frame_duration
        self.lambda_min = lambda_min

        self.dataset_std = dataset_std
        self.n_wavenumbers = n_wavenumbers
        self.n_times_train = n_times_train
        self.poly_degree = polynomial_degree

        # n_full_timepoints: number of frames in the full dataset - used only for
        # validation extrapolation bookkeeping. Never drives model buffers.
        if n_full_timepoints is None:
            n_full_timepoints = n_times_train
        self.n_full_timepoints = n_full_timepoints

        # self.t - training window time axis, shape [n_times_train].
        # Stores only the frames the encoder is trained on. Physics reconstruction
        # at inference uses caller-supplied time_values, not this buffer.
        # When loading from a checkpoint, time_values=None here; load_state_dict
        # restores the saved buffer immediately after __init__ completes.
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
        # Normalised to [-1, +1] over [t[0], t[n_times_train-1]].
        # Encoder always receives t_norm[:T] for T <= n_times_train - a crop of this
        # buffer - so any shorter inference window stays in-distribution.
        # Predicting with T > n_times_train is not supported: PE would extrapolate
        # beyond +1 (OOD). The encoder should always see <= n_times_train frames;
        # longer reconstructions are handled by physics_forward with explicit time_values.
        t_end = self.t[-1]
        t_norm = 2.0 * (self.t - self.t[0]) / (t_end - self.t[0] + 1e-8) - 1.0
        self.register_buffer("t_norm_full", t_norm)  # [n_times_train], spans [-1, +1]

        self.n_fluorophores = n_fluorophores

        # -- Pseudo-Voigt global Raman parameters ------------------------------
        if raman_mode == "pseudo_voigt":
            self.n_raman_peaks = n_raman_peaks

            # Peak positions (Logit uniform spacing)
            init_peaks = torch.linspace(0.05, 0.95, n_raman_peaks)
            self.peak_positions_raw = nn.Parameter(init_peaks)

            # --- Secure Inverse Softplus Helper ---
            def inv_softplus(val):
                val = max(val, 1e-6)
                return math.log(math.exp(val) - 1.0)

            # Lorentzian FWHM initialisation (target 8.0 cm^-1)
            target_fwhm_L = 8.0
            safe_val_L = max(target_fwhm_L - _FWHM_L_MIN, 1e-4)
            self.log_fwhm_L = nn.Parameter(
                torch.full((n_raman_peaks,), inv_softplus(safe_val_L))
            )

            # Gaussian FWHM initialisation (hardware IRF)
            safe_val_G = max(float(fwhm_G_init) - _FWHM_G_MIN, 1e-4)
            self.log_fwhm_G = nn.Parameter(
                torch.tensor(inv_softplus(safe_val_G)),
                requires_grad=fwhm_G_trainable,
            )

        if self.basis_mode == BasisMode.DICTIONARY:
            if dictionary_bases is not None:
                if not isinstance(dictionary_bases, torch.Tensor):
                    dictionary_bases = torch.as_tensor(
                        dictionary_bases, dtype=torch.float32
                    )
                self.dictionary_bases = nn.Parameter(
                    dictionary_bases, requires_grad=dynamic_bases
                )
                self.n_fluorophores = dictionary_bases.shape[0]
            else:
                raise ValueError(
                    "basis_mode 'dictionary' requires 'dictionary_bases' argument."
                )
        elif self.basis_mode == BasisMode.MOG:
            n_k = n_gaussian_components
            # Initialise means spread across [-0.8, 0.8] in wn_norm space.
            # Each fluorophore gets its own centre; within a fluorophore the K
            # components are offset by +/-half_spacing around that centre.
            fluor_centers = torch.linspace(-0.8, 0.8, n_fluorophores)  # [F]
            if n_k == 1:
                means_init = fluor_centers.unsqueeze(1)  # [F, 1]
            else:
                half_spacing = 0.8 / n_fluorophores
                offsets = torch.linspace(-half_spacing, half_spacing, n_k)  # [K]
                means_init = fluor_centers[:, None] + offsets[None, :]  # [F, K]
            self.mog_means = nn.Parameter(means_init)

            # Log scales in wn_norm space (range [-1, +1] = full spectral window).
            init_scale = 1.0
            scales_init = torch.full((n_fluorophores, n_k), init_scale)
            if n_k > 1:
                scales_init[:, 0] = 2.0  # component 0: flat background anchor
            self.mog_log_scales = nn.Parameter(torch.log(scales_init))
            # Mixture logits: bias component 0 so it starts with higher softmax weight.
            logits_init = torch.zeros(n_fluorophores, n_k)
            if n_k > 1:
                logits_init[:, 0] = 1.0
            self.mog_logits = nn.Parameter(logits_init)
        elif self.basis_mode == BasisMode.POLYNOMIAL:
            # Initialise each polynomial as a Gaussian bump at an evenly-spaced
            # position in [-0.8, 0.8] wn_norm space
            fluor_centers = torch.linspace(-0.8, 0.8, n_fluorophores)
            sigma = max(0.8 / n_fluorophores, 0.15)
            coeffs = torch.zeros(n_fluorophores, polynomial_degree + 1)
            for f in range(n_fluorophores):
                mu = fluor_centers[f].item()
                if polynomial_degree >= 0:
                    coeffs[f, 0] = -0.5 * mu**2 / sigma**2
                if polynomial_degree >= 1:
                    coeffs[f, 1] = mu / sigma**2
                if polynomial_degree >= 2:
                    coeffs[f, 2] = -0.5 / sigma**2
                # degree >= 3: higher terms stay zero (small random noise)
            coeffs += torch.randn_like(coeffs) * 0.01
            self.log_poly_coeffs = nn.Parameter(coeffs)

        self.decay_mode = decay_mode
        if decay_mode in ("global", "global_scaled"):
            # One characteristic rate per fluorophore, shared by every sample.
            self.log_lambda_global = nn.Parameter(
                torch.linspace(-2.0, 3.0, self.n_fluorophores)
            )

        if isinstance(conv_channels, str):
            conv_channels = tuple(int(v) for v in conv_channels.split(","))
        self.encoder = VAEEncoder(
            self.n_wavenumbers, hidden_dim, latent_dim, conv_channels=conv_channels,
        )

        self.decoder = ParameterDecoder(
            latent_dim,
            decoder_dim,
            self.n_fluorophores,
            self.n_wavenumbers,
            raman_mode=raman_mode,
            n_raman_peaks=n_raman_peaks,
            poly_degree=self.poly_degree,
        )

        print(
            f"[VAE init] raman_mode={raman_mode}"
            + (
                f"  n_raman_peaks={n_raman_peaks}"
                if raman_mode == "pseudo_voigt"
                else ""
            )
        )
        print(
            f"[VAE init] head_abundance: weight_norm={self.decoder.head_abundance.weight.norm():.4g}"
            f"  bias={self.decoder.head_abundance.bias.data.tolist()}  (softplus -> {torch.nn.functional.softplus(self.decoder.head_abundance.bias).data.tolist()})"
        )

        # Register a pre-hook to resize buffers before state_dict is applied.
        self._register_load_state_dict_pre_hook(self._resize_buffers_hook)

    @torch.no_grad()
    def get_individual_peak_profiles(self, x):
        """
        Extracts the individual TCH Pseudo-Voigt profiles and parameters for a single sample.
        Assumes x is a single sample [1, W, T] or takes the first sample of a batch.
        """
        # 1. Run the encoder and decoder to get the sample-specific amplitudes
        T = x.shape[-1]
        t_norm = self.t_norm_full[:T]

        # Take just the first sample if a batch is passed
        if x.dim() == 3:
            x_single = x[0:1]
        else:
            x_single = x

        mu, _ = self.encoder(x_single, self.wn_norm, t_norm)
        # Use mu directly (no noise) for clean extraction
        res_dec = self.decoder(mu)
        raman_out = res_dec[1]
        amplitudes = F.softplus(raman_out)  # [1, N_peaks]

        # 2. Extract the global physical parameters
        wn = self.wavenumbers
        wn_min, wn_max = wn.min(), wn.max()
        x0 = wn_min + (wn_max - wn_min) * self.peak_positions
        f_L = F.softplus(self.log_fwhm_L) + _FWHM_L_MIN
        f_G = F.softplus(self.log_fwhm_G) + _FWHM_G_MIN

        # 3. Compute TCH variables
        f5 = (
            f_L**5
            + 2.69269 * f_L**4 * f_G
            + 2.42843 * f_L**3 * f_G**2
            + 4.47163 * f_L**2 * f_G**3
            + 0.07842 * f_L * f_G**4
            + f_G**5
        )
        f_V = f5**0.2
        ratio = f_L / f_V
        eta = (1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3).clamp(0, 1)

        # 4. Compute the individual profiles [N_peaks, W]
        delta = x0[:, None] - wn[None, :]
        hwhm = f_V[:, None] / 2.0

        # Normalised peak=1 shapes
        L_shape = hwhm**2 / (delta**2 + hwhm**2)
        G_shape = torch.exp(-4.0 * math.log(2.0) * delta**2 / f_V[:, None] ** 2)

        # Mixed and amplitude-scaled shapes
        amps = amplitudes.squeeze(0)[:, None]  # [N_peaks, 1]

        L_component = amps * eta[:, None] * L_shape
        G_component = amps * (1.0 - eta[:, None]) * G_shape
        pV_component = L_component + G_component

        return {
            "wavenumbers": wn.cpu().numpy(),
            "amplitudes": amplitudes.squeeze(0).cpu().numpy(),
            "centers": x0.cpu().numpy(),
            "fwhm_L": f_L.cpu().numpy(),
            "fwhm_G": f_G.item(),  # Scalar
            "fwhm_V": f_V.cpu().numpy(),
            "eta": eta.cpu().numpy(),
            "L_profiles": L_component.cpu().numpy(),  # [N_peaks, W]
            "G_profiles": G_component.cpu().numpy(),  # [N_peaks, W]
            "pV_profiles": pV_component.cpu().numpy(),  # [N_peaks, W]
        }

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
        scales = self.mog_log_scales.exp() + 0.05  # [F, K]

        diff = self.wn_norm[None, None, :] - self.mog_means[:, :, None]  # [F, K, W]
        gaussians = torch.exp(-0.5 * (diff / scales[:, :, None]).pow(2))
        bases = (weights[:, :, None] * gaussians).sum(dim=1)  # [F, W]

        # L2 normalise: unit-energy bases give equal weight to narrow and broad shapes.
        norms = bases.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return bases / norms

    @property
    def _polynomial_bases(self) -> torch.Tensor:
        """Evaluate polynomial bases at the registered wavenumber grid. [F, W]"""
        bases = evaluate_polynomial_bases_torch(self.log_poly_coeffs, self.wn_norm)
        peaks = bases.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return bases / peaks

    @property
    def bases(self) -> torch.Tensor:
        """Current fluorophore basis spectra, L2-normalised. [F, W]"""
        if self.basis_mode == BasisMode.DICTIONARY:
            b = self.dictionary_bases
            peaks = b.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            return b / peaks
        elif self.basis_mode == BasisMode.MOG:
            return self._mog_bases
        elif self.basis_mode == BasisMode.POLYNOMIAL:
            return self._polynomial_bases
        else:
            raise NotImplementedError(f"basis_mode {self.basis_mode!r} not implemented")

    @property
    def peak_positions(self):
        """Peak positions in [0, 1], clamped in the forward pass only.

        The backward pass sees the unclamped parameter, so every peak trains at
        the same rate wherever it sits. Without the clamp peaks drift outside the
        spectral range, where their Lorentzian tails act as a broad baseline and
        compete with the fluorescence bases.
        """
        raw = self.peak_positions_raw
        return raw + (raw.clamp(0.0, 1.0) - raw).detach()

    def _get_pseudo_voigt_raman(self, amplitudes):
        """
        Construct a Raman spectrum as a weighted sum of N pseudo-Voigt peaks.

        Uses the Thompson-Cox-Hastings (1987) approximation:
            f_V^5 = f_L^5 + 2.69269*f_L^4*f_G + 2.42843*f_L^3*f_G^2
                  + 4.47163*f_L^2*f_G^3 + 0.07842*f_L*f_G^4 + f_G^5
            eta = 1.36603(f_L/f_V) - 0.47719(f_L/f_V)^2 + 0.11116(f_L/f_V)^3
            pV = eta*L(f_V) + (1-eta)*G(f_V)

        All widths are FWHM in cm^-1 (actual wavenumber units, not normalised).

        Parameters
        ----------
        amplitudes : [B, N_peaks]  positive peak amplitudes (softplus already applied)

        Returns
        -------
        spectrum : [B, W]  Raman spectrum in the same scale as the conv-stack output
        """
        wn = self.wavenumbers  # [W], actual cm^-1
        wn_min = wn.min()
        wn_max = wn.max()

        # Peak positions constrained to [wn_min, wn_max] via sigmoid
        x0 = wn_min + (wn_max - wn_min) * self.peak_positions  # [N_peaks]

        # Lorentzian FWHM per peak (global, shared across all samples).
        f_L = F.softplus(self.log_fwhm_L) + _FWHM_L_MIN  # [N_peaks]

        # Gaussian FWHM: global instrument response function, constant per session.
        # Scalar - same for all samples and all peaks.
        f_G = F.softplus(self.log_fwhm_G) + _FWHM_G_MIN  # scalar

        # -- TCH pseudo-Voigt --------------------------------------------------
        f_L_b = f_L[None, :]  # [1, N_peaks]  broadcast over batch
        # f_G: scalar - broadcasts over N_peaks; f5/f_V/eta are [1, N_peaks]
        f5 = (
            f_L_b**5
            + 2.69269 * f_L_b**4 * f_G
            + 2.42843 * f_L_b**3 * f_G**2
            + 4.47163 * f_L_b**2 * f_G**3
            + 0.07842 * f_L_b * f_G**4
            + f_G**5
        )  # [1, N_peaks]
        f_V = f5**0.2  # [1, N_peaks] total FWHM
        ratio = f_L_b / f_V  # [1, N_peaks]
        eta = (1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3).clamp(
            0.0, 1.0
        )  # [1, N_peaks]

        # -- Evaluate profiles at every wavenumber -----------------------------
        # Profiles are global (shared across batch) - only amplitudes are per-sample.
        delta = x0[None, :, None] - wn[None, None, :]  # [1, N_peaks, W]
        hwhm = f_V[:, :, None] / 2.0  # [1, N_peaks, 1]

        L_pv = hwhm**2 / (delta**2 + hwhm**2)  # [1, N_peaks, W] peak=1
        G_pv = torch.exp(
            -4.0 * math.log(2.0) * delta**2 / f_V[:, :, None] ** 2
        )  # [1, N_peaks, W] peak=1
        pV = eta[:, :, None] * L_pv + (1.0 - eta[:, :, None]) * G_pv  # [1, N_peaks, W]

        # amplitudes [B, N_peaks] x pV [N_peaks, W] -> [B, W]
        return amplitudes @ pV.squeeze(0)  # [B, W]


    def forward(self, x, sample=None):
        # x: [B, W, T] - std-normalised signal (may be slightly negative after dark subtraction)

        # Derive t_use from actual input length.
        # T must be <= n_times_train: the PE (t_norm_full) only covers the training
        # window and values beyond it are OOD. For longer reconstructions, call
        # physics_forward directly with explicit time_values.
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

        if self.decay_mode == "global":
            lambdas = (
                F.softplus(self.log_lambda_global).unsqueeze(0).expand(z.shape[0], -1)
                + self.lambda_min
            )
        elif self.decay_mode == "global_scaled":
            # Per-fluorophore characteristic rate, modulated by one shared factor
            # per spectrum (local intensity and oxygen scale every rate together).
            # Bounded to [0.1x, 10x]; the measured spread across spectra is ~16x.
            log_scale = torch.tanh(lambdas_raw.mean(dim=1, keepdim=True)) * math.log(10.0)
            lambdas = (
                F.softplus(self.log_lambda_global).unsqueeze(0) * torch.exp(log_scale)
                + self.lambda_min
            )
        else:
            lambdas = F.softplus(lambdas_raw) + self.lambda_min

        if self.raman_mode == "pseudo_voigt":
            # raman_out: [B, N_peaks] raw amplitudes
            amplitudes = F.softplus(raman_out)  # [B, N_peaks], positive
            self._raman_amplitudes = amplitudes  # exposed for training step L1
            raman_spectrum = self._get_pseudo_voigt_raman(amplitudes)  # [B, W]
        else:
            self._raman_amplitudes = None
            raman_spectrum = raman_out  # [B, W], softplus already applied

        # Scale Raman to physical counts/sec.
        raman_counts = (raman_spectrum * self.dataset_std) / self.frame_duration


        bases_normalised = self.bases  # [F, W]

        # Abundances are learned end-to-end from the decoder; positivity enforced by softplus.
        # Scaled by dataset_std so the network doesn't have to output huge values directly.
        abundances = F.softplus(abundances_raw) * self.dataset_std

        x_recon_phys, bases = self.physics_forward(
            lambdas,
            abundances,
            raman_counts,
            bases_normalised,
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

        if self.physics_model == "factored":
            x_recon = reconstruct_time_series_factored_torch(
                raman=raman,
                bases=bases_normalised,
                effective_amplitudes=abundances,
                decay_rates=lambdas,
                time_values=time_values,
                frame_duration=self.frame_duration,
            )
        elif self.physics_model == "pointsample":
            x_recon = reconstruct_time_series_torch(
                raman=raman,
                bases=bases_normalised,
                abundances=abundances,
                decay_rates=lambdas,
                time_values=time_values,
                frame_duration=self.frame_duration,
            )
        elif self.physics_model == "integrated":
            x_recon = reconstruct_time_series_integrated_torch(
                raman=raman,
                bases=bases_normalised,
                abundances=abundances,
                decay_rates=lambdas,
                time_values=time_values,
                frame_duration=self.frame_duration,
            )
        else:
            raise ValueError(
                f"Unknown physics_model: {self.physics_model!r}. Expected 'factored', 'pointsample', or 'integrated'."
            )

        if x_recon.shape[-1] != time_values.shape[0]:
            x_recon = x_recon.transpose(-1, -2)

        return x_recon, bases_normalised

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class VAEEncoder(nn.Module):
    """Joint 2D CNN encoder for [W, T] spectrograms.

    Input: 3 channels - log1p-compressed signal, wavenumber PE, time PE.
    Kernels span both W and T so the network can learn joint spectro-temporal
    features.

    Spectral axis: stride=2 on all 4 layers -> W/16 resolution.
    Temporal axis: stride=1, k=3 throughout -> T preserved.

    Pooling: Adaptive Average Pooling. Reduces any temporal length T to 1,
    extracting global temporal features without time-warping. Forces the
    spectral axis to a fixed size of 16 to maintain the flat_dim of 2048.
    """

    def __init__(self, n_wavenumbers, hidden_dim, latent_dim,
                 conv_channels=(64, 128, 256, 512)):
        super().__init__()
        self.conv_channels = tuple(conv_channels)

        # Four layers, stride 2 on the spectral axis and 1 on time, so W is
        # reduced 16-fold and T is preserved. Widths are configurable because
        # this stack holds around 90% of the model's parameters, while the
        # latent, hidden and decoder sizes together hold about 1%.
        kernels = [(7, 3), (7, 3), (5, 3), (5, 3)]
        paddings = [(3, 1), (3, 1), (2, 1), (2, 1)]
        layers, in_channels = [], 3
        for out_channels, kernel, padding in zip(self.conv_channels, kernels, paddings):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                          stride=(2, 1), padding=padding),
                nn.LeakyReLU(0.2),
            ]
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)

        flat_dim = self.conv_channels[-1] * 16

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

        # Compressed so the encoder is not dominated by the bright, slowly
        # varying fluorescence background.
        x_enc = torch.log1p(F.relu(x))

        x_2d = x_enc.unsqueeze(1)  # [B, 1, W, T]
        wn_pe = wn_norm.view(1, 1, W, 1).expand(B, 1, W, T)  # [B, 1, W, T]
        t_pe = t_norm.view(1, 1, 1, T).expand(B, 1, W, T)  # [B, 1, W, T]
        x_in = torch.cat([x_2d, wn_pe, t_pe], dim=1)  # [B, 3, W, T]

        # Here we are reshaping the data, and PE arrays to be the same so that they are then concatenated along the channel dimension.
        # This means that every B,W,T point will have 3 values, its intensity (C0), its wavenumber (C1), and its time (C2).

        # The convolutional stack preserves T, but divides W by 16 (if W=1024, W becomes 64)
        h = self.conv_net(x_in)  # [B, 128, W/16, T]

        # Adaptive pooling:
        # Forces the spatial dimension to exactly 16 bins.
        # Averages the entire temporal dimension T down to exactly 1 bin.
        # This makes the output strictly length-invariant.
        # Collapses any number of frames to one bin, so the encoder accepts any
        # window length. Adding a standard deviation here was measured to make no
        # difference once runs were trained to convergence.
        h = F.adaptive_avg_pool2d(h, output_size=(16, 1))

        h = h.flatten(1)  # [B, 2048]
        h = self.fc_net(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ParameterDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        decoder_dim,
        n_fluorophores,
        n_wavenumbers,
        raman_mode: str = "conv",
        n_raman_peaks: int = 150,
        poly_degree: int = 3,
    ):
        super().__init__()
        self.n_wavenumbers = n_wavenumbers
        self.raman_mode = raman_mode
        self.poly_degree = poly_degree

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

        if raman_mode == "pseudo_voigt":
            # Amplitude head: one positive scalar per peak [B, N_peaks].
            self.head_raman_amplitudes = nn.Linear(decoder_dim, n_raman_peaks)
        else:
            n_basis = min(64, n_wavenumbers)
            self.head_raman = nn.Sequential(
                nn.Linear(decoder_dim, n_basis),
                nn.LeakyReLU(0.2),
                nn.Linear(n_basis, n_wavenumbers),
            )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if raman_mode == "pseudo_voigt":
            nn.init.constant_(self.head_raman_amplitudes.bias, -2.0)

        # Initialise the lambda head with spread values to cover different decay scales
        self.head_lambda.bias.data.copy_(torch.linspace(-2.0, 3.0, n_fluorophores))

    def forward(self, z):
        h = self.trunk_shared(z)
        h_lam = self.trunk_lambda(h)
        h_abd = self.trunk_abundance(h)
        h_ram = self.trunk_raman(h)

        lambdas = self.head_lambda(h_lam)
        abundances_raw = self.head_abundance(h_abd)

        if self.raman_mode == "pseudo_voigt":
            raman_out = self.head_raman_amplitudes(h_ram)
        else:
            raman_out = F.softplus(self.head_raman(h_ram))  # [B, W], non-negative

        return lambdas, raman_out, abundances_raw
