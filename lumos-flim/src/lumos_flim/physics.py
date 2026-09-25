"""Analytical TCSPC forward model.

A pulsed laser excites the sample every ``P`` ns and each detected photon is
time-stamped relative to the pulse, then binned into ``N`` bins of width
``D = P / N``. For a pixel the expected histogram is

    h_n = n_tot * [ f_bg / N + sum_i f_i * p_i(n) ]

where ``p_i(n)`` is the probability that a photon from component ``i`` lands in
bin ``n``. That photon was emitted a time ``Exp(k_i)`` after the pulse, which
itself arrived at ``t0 + N(0, sigma^2)`` (the instrument response). Because the
laser is periodic, photons from earlier pulses whose decay has not finished
wrap into the current window, which is why the counts before the rising edge
match the tail.

This is the same structure as LUMOS: a constant term (there the static Raman
spectrum, here the uncorrelated background) plus exponentials with per-sample
rates, each integrated over a detector bin. The decoder emits photon fractions
rather than pre-exponential amplitudes, for the same reason LUMOS emits
effective amplitudes: the fraction is what the histogram actually constrains,
and it keeps amplitude and rate from trading off against each other.

``p_i(n)`` is evaluated exactly. For one pulse the arrival-time CDF is the
exponentially modified Gaussian

    F(t) = Phi(u) - exp(-k (t - t0) + k^2 sigma^2 / 2) * Phi(u - k sigma),
    u = (t - t0) / sigma

and the periodic histogram is the sum of ``F(b + jP) - F(a + jP)`` over pulses
``j``. Pulses ``j`` in {-1, 0, 1} are summed explicitly; earlier ones add a
geometric series in ``exp(-k P)``. Everything is done in log space so very
short lifetimes do not overflow.

Times are in ns and rates in 1/ns throughout.
"""

import math

import torch
from torch.special import log_ndtr, ndtr


def bin_edges(n_bins: int, bin_width: float, device=None, dtype=torch.float32):
    """The ``n_bins + 1`` bin edges covering one laser period."""
    return torch.arange(n_bins + 1, device=device, dtype=dtype) * bin_width


def _exgauss_tail(t, rate, t0, sigma):
    """exp(-k (t - t0) + k^2 sigma^2 / 2) * Phi(u - k sigma), without overflow.

    Written directly, the two exponents cancel catastrophically once k sigma is
    large. Where k sigma > u the identity

        exp(-k sigma u + (k sigma)^2 / 2) Phi(u - k sigma)
            = exp(-u^2 / 2) erfcx((k sigma - u) / sqrt 2) / 2

    avoids that; elsewhere Phi is near one and the direct form is safe. Both
    branches are clamped into their own domain so the unused one stays finite
    and does not poison the gradient.
    """
    ks = rate * sigma
    u = (t - t0) / sigma
    w = (ks - u) / math.sqrt(2.0)
    scaled = 0.5 * torch.exp(-0.5 * u**2) * torch.special.erfcx(w.clamp(min=0.0))
    u_c = torch.maximum(u, ks)
    direct = torch.exp(ks * (0.5 * ks - u_c) + log_ndtr(u_c - ks))
    return torch.where(w > 0, scaled, direct)


def _periodic_cdf(t, rate, t0, sigma, period):
    """Cumulative arrival probability up to ``t`` summed over all past pulses.

    Only differences of this function are meaningful. Pulses more than one
    period back contribute ``Phi = 1`` (a constant, dropped) and a decay term
    that forms a geometric series.
    """
    total = 0.0
    for j in (-1, 0, 1):
        tj = t + j * period
        total = total + ndtr((tj - t0) / sigma) - _exgauss_tail(tj, rate, t0, sigma)

    # Pulses j >= 2. Phi(u_j - k sigma) grows with j, so evaluating it at j = 2
    # is exact wherever this term is not already negligible.
    geom = 1.0 / -torch.expm1(-rate * period)
    return total - _exgauss_tail(t + 2 * period, rate, t0, sigma) * geom


def decay_histograms(
    rates: torch.Tensor,  # [..., F]
    t0: torch.Tensor,  # scalar or broadcastable to [..., F]
    sigma: torch.Tensor,  # scalar or broadcastable to [..., F]
    n_bins: int,
    bin_width: float,
) -> torch.Tensor:
    """Probability that a photon from each component lands in each bin.

    Returns [..., F, n_bins]. Each row sums to one up to truncation of the
    Gaussian, which is negligible while ``sigma`` is well below the period.
    """
    period = n_bins * bin_width
    edges = bin_edges(n_bins, bin_width, device=rates.device, dtype=rates.dtype)
    k = rates.unsqueeze(-1)  # [..., F, 1]
    t0 = torch.as_tensor(t0, dtype=rates.dtype, device=rates.device)
    sigma = torch.as_tensor(sigma, dtype=rates.dtype, device=rates.device)
    if t0.dim() > 0:
        t0 = t0.unsqueeze(-1)
    if sigma.dim() > 0:
        sigma = sigma.unsqueeze(-1)
    cdf = _periodic_cdf(edges, k, t0, sigma, period)  # [..., F, N + 1]
    return (cdf[..., 1:] - cdf[..., :-1]).clamp(min=0.0)


def irf_histogram(t0, sigma, n_bins: int, bin_width: float) -> torch.Tensor:
    """The instrument response itself, wrapped onto one period. [n_bins]"""
    t0 = torch.as_tensor(t0, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    period = n_bins * bin_width
    edges = bin_edges(n_bins, bin_width)
    cdf = sum(ndtr((edges + j * period - t0) / sigma) for j in (-1, 0, 1))
    return cdf[1:] - cdf[:-1]


def expected_counts(
    totals: torch.Tensor,  # [B]
    fractions: torch.Tensor,  # [B, F] photon fraction of each decaying component
    background: torch.Tensor,  # [B] photon fraction of the constant term
    decays: torch.Tensor,  # [B, F, N] from decay_histograms
    bases: torch.Tensor = None,  # [F, C] emission spectra, rows sum to one
    background_spectrum: torch.Tensor = None,  # [B, C], rows sum to one
) -> torch.Tensor:
    """Expected photon counts, [B, C, N]. Sums to ``totals`` over (C, N).

    With a single detection channel ``bases`` and ``background_spectrum`` are
    both identically one and can be left out.
    """
    B, F, N = decays.shape
    if bases is None:
        fluorescence = torch.einsum("bf,bfn->bn", fractions, decays).unsqueeze(1)
    else:
        fluorescence = torch.einsum("bf,fc,bfn->bcn", fractions, bases, decays)
    if background_spectrum is None:
        flat = (background / N)[:, None, None]
    else:
        flat = (background[:, None] * background_spectrum / N)[:, :, None]
    return totals[:, None, None] * (fluorescence + flat)


def amplitude_fractions(fractions, rates):
    """Photon (intensity) fractions to pre-exponential amplitude fractions.

    A component with amplitude ``a`` and rate ``k`` emits ``a / k`` photons per
    pulse, so ``a`` is proportional to ``f * k``. These are the fractions
    usually quoted for free and bound NADH.
    """
    a = fractions * rates
    return a / a.sum(-1, keepdim=True).clamp(min=1e-12)


def mean_lifetimes(fractions, rates):
    """Intensity- and amplitude-weighted mean lifetimes over decaying components."""
    tau = 1.0 / rates
    f = fractions / fractions.sum(-1, keepdim=True).clamp(min=1e-12)
    alpha = amplitude_fractions(fractions, rates)
    return (f * tau).sum(-1), (alpha * tau).sum(-1)


def phasor(counts, harmonic: int = 1):
    """First-harmonic phasor (g, s) of histograms along the last axis.

    Uncalibrated: the instrument response rotates and shrinks it. Used only as
    a model-free cross-check on the fitted lifetimes.
    """
    n = counts.shape[-1]
    phase = 2 * math.pi * harmonic * torch.arange(n, dtype=counts.dtype) / n
    total = counts.sum(-1).clamp(min=1e-12)
    g = (counts * torch.cos(phase)).sum(-1) / total
    s = (counts * torch.sin(phase)).sum(-1) / total
    return g, s


def lifetime_phasor(tau, n_bins: int, bin_width: float, harmonic: int = 1):
    """Phasor of a pure single exponential, the reference for calibration."""
    omega = 2 * math.pi * harmonic / (n_bins * bin_width)
    wt = omega * tau
    return 1.0 / (1.0 + wt**2), wt / (1.0 + wt**2)
