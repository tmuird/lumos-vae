import math

import torch
from torch import nn
from torch.nn import functional as F

from lumos_flim.physics import decay_histograms, expected_counts

# Lower limit on the fitted IRF width in ns. Anything narrower is a delta at
# this bin width, so the floor only keeps the parameter away from zero.
_SIGMA_FLOOR = 0.005


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(max(v, 1e-6)))


class FlimVAE(nn.Module):
    """Encoder to per-pixel physical parameters, decoded by the TCSPC model.

    Per pixel: ``n_components`` lifetimes, their photon fractions and a
    background fraction. Global: the instrument response (Gaussian, centre
    ``t0`` and width ``sigma``) and, with more than one detection channel, the
    emission spectrum of each component. This mirrors LUMOS, where rates and
    abundances are per sample and the fluorophore bases are shared.
    """

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
        spatial: int = 0,  # number of context histograms per pixel, 0 for none
        irf_tail: bool = False,
    ):
        super().__init__()
        self.spatial = spatial
        self.n_bins = n_bins
        self.bin_width = float(bin_width)
        self.period = n_bins * self.bin_width
        self.n_channels = n_channels
        self.n_components = n_components

        # A lifetime much longer than the period is a flat line and cannot be
        # told apart from background, so the slowest rate is bounded below.
        self.rate_min = 1.0 / (tau_max if tau_max > 0 else self.period)

        self.irf_t0 = nn.Parameter(torch.tensor(float(irf_t0)), requires_grad=not fix_irf)
        self.irf_sigma_raw = nn.Parameter(
            torch.tensor(_inv_softplus(irf_sigma - _SIGMA_FLOOR)), requires_grad=not fix_irf
        )

        # Optional exponential diffusion tail on the IRF: weight w (sigmoid)
        # and rate q (softplus), shared by every pixel, learned with the rest.
        # Starts at 10% of photons with a 0.3 ns tail.
        self.irf_tail = irf_tail
        if irf_tail:
            self.irf_tail_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)),
                                               requires_grad=not fix_irf)
            self.irf_tail_rate_raw = nn.Parameter(torch.tensor(_inv_softplus(1 / 0.3)),
                                                  requires_grad=not fix_irf)

        if n_channels > 1:
            # Components start with distinct, broad spectra so they are not
            # symmetric at initialisation.
            pos = torch.linspace(-1.0, 1.0, n_channels)
            centres = torch.linspace(-0.6, 0.6, n_components)
            self.basis_logits = nn.Parameter(-2.0 * (pos[None, :] - centres[:, None]) ** 2)
        else:
            self.basis_logits = None

        if isinstance(conv_channels, str):
            conv_channels = tuple(int(v) for v in conv_channels.split(","))
        self.encoder = Encoder(n_channels, n_bins, hidden_dim, latent_dim, conv_channels, pool_time,
                               spatial=spatial)
        self.decoder = Decoder(latent_dim, decoder_dim, n_components, n_channels)

        # Rates start log-spaced across what the window can resolve: from a
        # third of the period down to two bins. Rates are built as a cumulative
        # sum, so the bias holds the increments.
        targets = torch.logspace(
            math.log10(3.0 / self.period), math.log10(0.5 / self.bin_width), n_components
        )
        steps = torch.diff(targets, prepend=torch.tensor([self.rate_min]))
        self.decoder.head_rate.bias.data.copy_(
            torch.tensor([_inv_softplus(float(s)) for s in steps.clamp(min=1e-3)])
        )
        # Background starts small but not negligible.
        self.decoder.head_fraction.bias.data[-1] = -3.0

    @property
    def irf_sigma(self) -> torch.Tensor:
        return F.softplus(self.irf_sigma_raw) + _SIGMA_FLOOR

    @property
    def irf_tail_params(self):
        """(weight, rate) of the IRF tail, or (None, None) without one."""
        if not getattr(self, "irf_tail", False):
            return None, None
        return torch.sigmoid(self.irf_tail_logit), F.softplus(self.irf_tail_rate_raw) + 0.1

    @property
    def bases(self):
        """Emission spectrum of each component, [F, C], or None for one channel."""
        if self.basis_logits is None:
            return None
        return F.softmax(self.basis_logits, dim=-1)

    def decode(self, z, totals):
        rate_raw, frac_logits, bg_logits = self.decoder(z)
        # Ascending rates, so component 0 always has the longest lifetime and
        # the labels cannot swap between pixels.
        rates = self.rate_min + torch.cumsum(F.softplus(rate_raw), dim=-1)
        probs = F.softmax(frac_logits, dim=-1)
        fractions, background = probs[:, :-1], probs[:, -1]
        bg_spectrum = F.softmax(bg_logits, dim=-1) if bg_logits is not None else None

        w, q = self.irf_tail_params
        decays = decay_histograms(rates, self.irf_t0, self.irf_sigma, self.n_bins, self.bin_width,
                                  tail_weight=w, tail_rate=q)
        expected = expected_counts(totals, fractions, background, decays, self.bases, bg_spectrum)
        return {
            "expected": expected,
            "rates": rates,
            "fractions": fractions,
            "background": background,
            "background_spectrum": bg_spectrum,
        }

    def forward(self, x, sample=None, scale=1.0, context=None):
        # x: [B, C, N] raw photon counts. context: [B, M, C, N], the
        # neighbourhood histograms, when the encoder is spatial.
        totals = x.sum(dim=(1, 2)).clamp(min=1.0)
        if self.spatial and context is None:
            raise ValueError("spatial model needs the neighbourhood context")
        mu, logvar = self.encoder(x, totals, context if self.spatial else None)
        if sample is None:
            sample = self.training
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) * scale if sample else mu
        out = self.decode(z, totals)
        out.update(mu=mu, logvar=logvar, totals=totals)
        return out


class Encoder(nn.Module):
    """1D CNN over the delay axis, with detection channels as input channels.

    The delay axis is periodic, so padding is circular and position is given
    as the phase of each bin (cos, sin), the same basis the phasor method uses.
    Counts are divided by the pixel total, which varies by orders of magnitude
    across an image, and the log total is passed alongside so the encoder still
    knows how noisy the pixel is.

    With ``spatial`` = M the M context histograms (the summed neighbourhood,
    or each neighbour separately) come in as further input channels, each
    normalised the same way, with their log totals. They only inform the
    encoder: the likelihood is still that of the centre pixel alone. Empty
    neighbours (outside the image or dropped as too dim) arrive as zeros with
    log total 0, which the encoder can learn to ignore.
    """

    def __init__(self, n_channels, n_bins, hidden_dim, latent_dim, conv_channels, pool_time,
                 spatial=False):
        super().__init__()
        phase = 2 * math.pi * torch.arange(n_bins) / n_bins
        self.register_buffer("phase", torch.stack([phase.cos(), phase.sin()]))  # [2, N]
        self.n_bins = n_bins
        self.pool_time = pool_time
        self.spatial = spatial

        layers, c_in = [], n_channels * (1 + spatial) + 2
        for i, c_out in enumerate(conv_channels):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2, padding_mode="circular",
                          stride=1 if i == 0 else 2),
                nn.LeakyReLU(0.2),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        n_scalars = 1 + spatial
        self.fc = nn.Sequential(nn.Linear(c_in * pool_time + n_scalars, hidden_dim), nn.LeakyReLU(0.2))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    @staticmethod
    def _shape(x):
        """Histogram scaled to mean one, and its log total."""
        C, N = x.shape[1:]
        total = x.sum(dim=(1, 2)).clamp(min=1.0)
        return x * (C * N) / total[:, None, None], total.log().unsqueeze(1)

    def forward(self, x, totals, context=None):
        B = x.shape[0]
        shape, log_total = self._shape(x)
        channels, scalars = [shape], [log_total]
        if self.spatial:
            M = context.shape[1]
            ctx_shape, ctx_log_total = self._shape(context.flatten(0, 1))
            channels.append(ctx_shape.view(B, M * ctx_shape.shape[1], -1))
            scalars.append(ctx_log_total.view(B, M))
        pe = self.phase.unsqueeze(0).expand(B, -1, -1)
        h = self.conv(torch.cat(channels + [pe], dim=1))
        h = F.adaptive_avg_pool1d(h, self.pool_time).flatten(1)
        h = self.fc(torch.cat([h] + scalars, dim=1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-20.0, 10.0)


class Decoder(nn.Module):
    def __init__(self, latent_dim, decoder_dim, n_components, n_channels):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim), nn.LeakyReLU(0.2),
            nn.Linear(decoder_dim, decoder_dim), nn.LeakyReLU(0.2),
        )
        self.head_rate = nn.Linear(decoder_dim, n_components)
        # Last logit is the background.
        self.head_fraction = nn.Linear(decoder_dim, n_components + 1)
        self.head_bg_spectrum = nn.Linear(decoder_dim, n_channels) if n_channels > 1 else None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
        # Heads start near their bias so the physics sees sensible values at step 0.
        for head in (self.head_rate, self.head_fraction, self.head_bg_spectrum):
            if head is not None:
                head.weight.data.mul_(0.1)

    def forward(self, z):
        h = self.trunk(z)
        bg = self.head_bg_spectrum(h) if self.head_bg_spectrum is not None else None
        return self.head_rate(h), self.head_fraction(h), bg
