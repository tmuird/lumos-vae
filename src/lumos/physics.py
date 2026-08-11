"""Analytical photobleaching forward model.

The observation model for a CCD integrating over each frame of duration ``T`` is

    S_n(nu) = S(nu)*T + sum_i w_i * B_i(nu) * exp(-lambda_i * t_n) * (1 - exp(-lambda_i*T)) / lambda_i

where S(nu) is the static Raman spectrum, B_i the fluorophore basis spectra,
w_i the abundances and lambda_i the per-sample decay rates. Three reconstruction
variants are provided; the decoder in ``vae.py`` selects one via ``physics_model``.
"""

from typing import Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    TORCH_AVAILABLE = False


def interpolate_bases(
    bases: np.ndarray,
    source_wn: np.ndarray,
    target_wn: np.ndarray,
    method: str = "pchip",
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Interpolate fluorophore bases from one wavenumber axis onto another.

    Handles unsorted source axes. Negative overshoot is clipped to zero.
    ``method`` is one of "pchip" (monotone cubic, default), "spline" (exact
    cubic) or "linear".
    """
    axes_match = len(source_wn) == len(target_wn) and np.allclose(source_wn, target_wn)
    if axes_match:
        result = bases.copy()
    else:
        try:
            from scipy.interpolate import PchipInterpolator, UnivariateSpline

            _have_scipy = True
        except ImportError:
            _have_scipy = False

        sort_idx = np.argsort(source_wn)
        source_wn_sorted = source_wn[sort_idx]
        bases_sorted = bases[:, sort_idx]

        result = np.zeros((bases.shape[0], len(target_wn)))
        for i in range(bases.shape[0]):
            if method == "pchip" and _have_scipy:
                interp = PchipInterpolator(
                    source_wn_sorted, bases_sorted[i], extrapolate=True
                )
                result[i] = interp(target_wn)
            elif method == "spline" and _have_scipy:
                spline = UnivariateSpline(source_wn_sorted, bases_sorted[i], k=3, s=0)
                result[i] = spline(target_wn)
            else:
                result[i] = np.interp(
                    target_wn, source_wn_sorted, bases_sorted[i], left=0.0, right=0.0
                )

    if smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter1d

        wn_spacing = float(np.mean(np.diff(np.sort(target_wn))))
        sigma_px = smooth_sigma / wn_spacing
        result = gaussian_filter1d(result, sigma=sigma_px, axis=1)

    return np.maximum(result, 0.0)


def effective_to_physical_abundance(
    effective_amplitudes: np.ndarray,
    decay_rates: np.ndarray,
    frame_duration: float,
) -> np.ndarray:
    """Convert factored-model effective amplitudes to physical abundances.

    w = a_tilde * lambda / (1 - exp(-lambda*T)). Well behaved as lambda -> 0.
    """
    return (
        effective_amplitudes
        * decay_rates
        / (1.0 - np.exp(-decay_rates * frame_duration) + 1e-8)
    )


if TORCH_AVAILABLE:

    def evaluate_polynomial_bases_torch(
        log_poly_coeffs: "torch.Tensor",
        wn_normalised: "torch.Tensor",
    ) -> "torch.Tensor":
        """Evaluate log-polynomial fluorophore bases, then exponentiate.

        ``log_poly_coeffs`` has shape (n_fluorophores, degree+1) in ascending
        power order; ``wn_normalised`` is the wavenumber axis in [-1, 1].
        """
        degree = log_poly_coeffs.shape[1] - 1
        vandermonde = torch.vander(wn_normalised, N=degree + 1, increasing=True)
        if log_poly_coeffs.dtype != vandermonde.dtype:
            vandermonde = vandermonde.to(log_poly_coeffs.dtype)
        log_intensity_values = torch.matmul(log_poly_coeffs, vandermonde.T)
        return torch.exp(log_intensity_values)

    def reconstruct_time_series_torch(
        raman: "torch.Tensor",  # [B, W]
        bases: "torch.Tensor",  # [F, W] or [B, F, W]
        abundances: "torch.Tensor",  # [B, F]
        decay_rates: "torch.Tensor",  # [B, F]
        time_values: "torch.Tensor",  # [T]
        frame_duration: float = 0.1,
    ) -> "torch.Tensor":
        """Point-sampled decay: fluorescence(t) = sum_i w_i B_i exp(-lambda_i t).

        Raman is treated as a rate and multiplied by ``frame_duration`` to give
        counts per frame. Returns [B, W, T].
        """
        lam = decay_rates.unsqueeze(1)  # [B, 1, F]
        t = time_values.view(1, -1, 1)  # [1, T, 1]
        decay_matrix = torch.exp(-lam * t)  # [B, T, F]

        w = abundances.unsqueeze(2)  # [B, F, 1]
        weighted_bases = w * bases  # [B, F, W]

        fluorescence = torch.matmul(decay_matrix, weighted_bases)  # [B, T, W]
        raman_integrated = raman.unsqueeze(1) * frame_duration  # [B, 1, W]
        total_signal = fluorescence + raman_integrated
        return total_signal.transpose(1, 2)  # [B, W, T]

    def reconstruct_time_series_integrated_torch(
        raman: "torch.Tensor",  # [B, W]
        bases: "torch.Tensor",  # [F, W]
        abundances: "torch.Tensor",  # [B, F]
        decay_rates: "torch.Tensor",  # [B, F]
        time_values: "torch.Tensor",  # [T] frame start times
        frame_duration: float = 0.1,
    ) -> "torch.Tensor":
        """CCD-integrated decay over each frame [t_n, t_n + T].

            measured(n) = S*T + sum_i w_i B_i * (1/lambda_i) * [exp(-lambda_i t_n) - exp(-lambda_i (t_n+T))]

        This is the physically correct model: fast components contribute near
        zero because they bleach within the frame. Returns [B, W, T].
        """
        lam = decay_rates.unsqueeze(1)  # [B, 1, F]
        t_start = time_values.view(1, -1, 1)  # [1, T, 1]
        t_end = t_start + frame_duration
        decay_matrix = (torch.exp(-lam * t_start) - torch.exp(-lam * t_end)) / (
            lam + 1e-8
        )  # [B, T, F]

        w = abundances.unsqueeze(2)  # [B, F, 1]
        B = bases.unsqueeze(0)  # [1, F, W]
        weighted_bases = w * B  # [B, F, W]

        fluorescence = torch.matmul(decay_matrix, weighted_bases)  # [B, T, W]
        raman_integrated = raman.unsqueeze(1) * frame_duration
        total_signal = fluorescence + raman_integrated
        return total_signal.transpose(1, 2)  # [B, W, T]

    def reconstruct_time_series_factored_torch(
        raman: "torch.Tensor",  # [B, W]
        bases: "torch.Tensor",  # [F, W] or [B, F, W]
        effective_amplitudes: "torch.Tensor",  # [B, F] - a_tilde values
        decay_rates: "torch.Tensor",  # [B, F]
        time_values: "torch.Tensor",  # [T]
        frame_duration: float = 0.1,
    ) -> "torch.Tensor":
        """Factored form that decouples observable amplitude from decay rate.

            S_n(nu) = S(nu)*T + sum_i a_tilde_i * B_i(nu) * exp(-lambda_i t_n)

        with a_tilde_i = w_i * (1 - exp(-lambda_i T)) / lambda_i. Mathematically
        equivalent to the integrated model but the decoder outputs a_tilde
        directly, which avoids the amplitude-rate identifiability problem.
        Returns [B, W, T].
        """
        lam = decay_rates.unsqueeze(1)  # [B, 1, F]
        t = time_values.view(1, -1, 1)  # [1, T, 1]
        decay_matrix = torch.exp(-lam * t)  # [B, T, F]

        a = effective_amplitudes.unsqueeze(2)  # [B, F, 1]
        B = bases if bases.dim() == 3 else bases.unsqueeze(0)  # [B or 1, F, W]
        weighted_bases = a * B  # [B, F, W]

        fluorescence = torch.matmul(decay_matrix, weighted_bases)  # [B, T, W]
        raman_integrated = raman.unsqueeze(1) * frame_duration
        total_signal = fluorescence + raman_integrated
        return total_signal.transpose(1, 2)  # [B, W, T]
