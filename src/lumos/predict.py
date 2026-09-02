"""
Prediction utilities for the VAE model.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from lumos.types import DecompositionResult, SpectralData
from lumos.physics import effective_to_physical_abundance


def predict(
    model,
    early_data,
    dataset=None,
    n_early=20,
    n_predictions=1,
    denormalise=True,
    stochastic=False,
) -> Tuple[DecompositionResult, np.ndarray]:
    """
    Thin wrapper around sample_posterior that returns a single averaged DecompositionResult.

    For a combined ensemble + decomposition in one call use sample_posterior directly
    with return_decomposition=True - that runs only N (not 2N) forward passes.

    Args:
        model:         Trained VAEModule.
        early_data:    [1, W, T] normalised tensor.
        n_early:       How many early timepoints the encoder sees.
        n_predictions: Stochastic draws to average (1 = deterministic eval mode).
        stochastic:    Ignored when n_predictions > 1 (always stochastic then).
                       When n_predictions == 1, False uses eval/deterministic mode.
        denormalise:   Kept for API compatibility; outputs are always physical units.
    """
    n_pred = n_predictions if (n_predictions > 1 or stochastic) else 1
    ens, decomp = sample_posterior(
        model,
        early_data[:, :, :n_early] if early_data.shape[-1] > n_early else early_data,
        n_predictions=n_pred,
        return_decomposition=True,
    )
    # Second return value: MMSE reconstruction averaged over ensemble [T_out, W]
    mean_recon = ens["reconstruction"].mean(axis=0)  # [T_out, W]
    return decomp, mean_recon


def sample_posterior(
    model,
    sample_tensor: torch.Tensor,
    n_predictions: int = 50,
    t_reconstruct: Optional[np.ndarray] = None,
    return_decomposition: bool = False,
) -> dict:
    """
    Run N stochastic forward passes and return individual (non-averaged) outputs.

    The encoder always sees the first n_times_train timepoints (matching training).
    The reconstruction is computed at ``t_reconstruct`` via physics_forward so it
    covers any requested time window regardless of training mode.

    This is the primary inference function.  ``predict`` is a thin
    wrapper that calls this with ``return_decomposition=True`` and returns only the
    averaged ``DecompositionResult``.

    Args:
        model:               Trained VAEModule.
        sample_tensor:       [1, W, T] normalised input tensor.
        n_predictions:       Number of stochastic samples.  Use 1 for a deterministic
                             point estimate (model is put in eval mode).
        t_reconstruct:       [T_out] time axis (seconds) for the reconstruction.
                             Defaults to model.model.t (the full training time axis).
        return_decomposition: If True, also return a DecompositionResult built from
                             the ensemble-mean parameters alongside the sample dict.
                             Use this to replace separate predict calls.

    Returns:
        ensemble : dict with keys
            'raman'              [N, W]        - Raman rate (counts/sec)
            'rates'              [N, F]        - decay rates (s^-1)
            'abundances'         [N, F]        - physical abundances
            'bases'              [N, F, W]     - normalised fluorophore spectra
            'reconstruction'     [N, T_out, W] - physical counts/frame at t_reconstruct
            't_reconstruct'      [T_out]       - the time axis used
        decomposition : DecompositionResult (only when return_decomposition=True)
            Built from ensemble-mean parameters; suitable for visualise_decomposition.
    """
    n_train = model.hparams.n_times_train

    frame_dur = getattr(model.model, "frame_duration", 0.1)

    # Time axis for reconstruction - independent of n_times_train.
    # Defaults to the training window (model.t) if not provided.
    # Pass an explicit t_reconstruct array to reconstruct beyond the training window.
    if t_reconstruct is None:
        t_reconstruct = model.model.t.cpu().numpy()  # [n_times_train]
    t_tensor = torch.tensor(t_reconstruct, dtype=torch.float32)

    model.eval()

    ramans, rates_list, abundances_list, bases_list, recons = [], [], [], [], []

    with torch.no_grad():
        x_input = sample_tensor[:, :, :n_train]
        for _ in range(n_predictions):
            _, _, _, lambdas, abundances, raman, bases = model.model(
                x_input, sample=n_predictions > 1
            )

            lambdas_np = lambdas.squeeze(0).cpu().numpy()
            abundances_np = abundances.squeeze(0).cpu().numpy()

            # The factored model outputs effective amplitudes.
            abundances_np = effective_to_physical_abundance(
                abundances_np, lambdas_np, frame_dur
            )

            ramans.append(raman.squeeze(0).cpu().numpy())
            rates_list.append(lambdas_np)
            abundances_list.append(abundances_np)
            bases_list.append(bases.cpu().numpy())

            # Re-run physics at the requested time axis.
            # Re-run physics at the requested time axis.
            x_full, _ = model.model.physics_forward(
                lambdas,
                abundances,
                raman,
                bases,
                time_values=t_tensor.to(lambdas.device),
            )  # [1, W, T_out], ADU (fluorescence + static*dt)

            recons.append((x_full.squeeze(0).T).cpu().numpy())  # [T_out, W]

    model.eval()

    ramans_arr = np.stack(ramans)  # [N, W]  counts/sec
    abundances_arr = np.stack(abundances_list)  # [N, F]
    bases_arr = np.stack(bases_list)  # [N, F, W]
    rates_arr = np.stack(rates_list)  # [N, F]
    recons_arr = np.stack(recons)  # [N, T_out, W]  ADU counts/frame

    ensemble = {
        "raman": ramans_arr,
        "rates": rates_arr,
        "abundances": abundances_arr,
        "bases": bases_arr,
        "abundance_times_basis": abundances_arr[:, :, None] * bases_arr,  # [N, F, W]
        "reconstruction": recons_arr,
        "t_reconstruct": t_reconstruct,
    }

    if not return_decomposition:
        return ensemble

    # Build a DecompositionResult from ensemble-mean parameters.
    # Using mean parameters (not mean reconstructions) keeps DecompositionResult
    # self-consistent - its .reconstruction(t) method always re-derives from params.

    wn = model.model.wavenumbers.cpu().numpy()
    mean_raman = ramans_arr.mean(axis=0)
    # [W] counts/sec - Raman peaks only
    mean_rates = rates_arr.mean(axis=0)  # [F]
    mean_abunds = abundances_arr.mean(axis=0)  # [F]
    mean_bases = bases_arr.mean(axis=0)  # [F, W]

    decomposition = DecompositionResult(
        raman=SpectralData(mean_raman, wavenumbers=wn),
        fluorophore_spectra=SpectralData(mean_bases, wavenumbers=wn),
        abundances=mean_abunds,
        rates=mean_rates,
        frame_duration=frame_dur,
    )
    return ensemble, decomposition
