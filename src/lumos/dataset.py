import torch
from torch.utils.data import Dataset
import numpy as np


class BleachingDataset(Dataset):
    def __init__(
        self,
        intensities,
        labels=None,
        time_values=None,
        wavenumbers=None,
        normalize=True,
        initial_bases=None,
        n_times_train=None,
        dictionary_bases=None,
        lengths=None,
        device=None,
    ):
        """
        Args:
            intensities (np.array): Shape [N_Samples, Time, Wavenumbers]
            labels (list/array): Optional labels (e.g. 'GFP', 'RFP')
            normalize (bool): Divide by the standard deviation. No mean is
                subtracted, since the physics output is non-negative.
            n_times_train (int): Number of training timepoints (compute stats only from these)
            dictionary_bases (np.ndarray): Optional [D, W] dictionary for dictionary mode
        """
        self.labels = labels
        self.n_times_train = n_times_train
        self.time_values = time_values
        self.wavenumbers = wavenumbers
        self.fluorophore_bases_gt = initial_bases
        self.dictionary_bases = dictionary_bases
        self.normalize = normalize
        # Per-spot valid frame counts (None for synthetic / fixed-length data).
        # Frames beyond lengths[i] are zero-padding and should not be visualised.
        self.lengths = lengths

        # Detect truly lazy (dask/zarr-backed) xarray DataArrays.
        # Numpy-backed xarrays are loaded eagerly - per-sample isel() in __getitem__
        # has high Python overhead and kills DataLoader throughput.
        import xarray as xr

        _is_xarray = isinstance(intensities, xr.DataArray)
        _is_dask = _is_xarray and hasattr(intensities.data, "__dask_graph__")
        self._lazy = _is_dask

        if self._lazy:
            self._lazy_data = intensities  # truly zarr/dask-backed, not loaded into RAM
            n_pilot = min(500, len(intensities))
            pilot = torch.from_numpy(
                intensities.isel(sample=slice(n_pilot)).values
            ).float()
            stats_data = pilot[:, :n_times_train, :] if n_times_train else pilot
        else:
            arr = intensities.values if _is_xarray else intensities
            self.data = torch.from_numpy(np.asarray(arr)).float()
            if device is not None:
                self.data = self.data.to(device)
                print(
                    f"Dataset preloaded to {device} ({self.data.nbytes / 1e9:.2f} GB)"
                )
            stats_data = self.data[:, :n_times_train, :] if n_times_train else self.data

        if self.normalize:
            if n_times_train is None:
                print(
                    "Warning: n_times_train is None, computing normalisation stats from entire dataset"
                )
            self.std = stats_data.std()

    def __len__(self):
        if self._lazy:
            return len(self._lazy_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self._lazy:
            sample = torch.from_numpy(
                self._lazy_data.isel(sample=int(idx)).values
            ).float()
        else:
            sample = self.data[idx]

        if self.normalize:
            sample = sample / (self.std + 1e-8)
        # Transpose for the VAE: data arrives as [Time, Wavenumbers], model expects [W, T]
        sample = sample.transpose(0, 1)  # [W, T_full]

        label = str(self.labels[idx]) if self.labels is not None else 0

        # Extact the true unpadded length for this sample
        valid_length = (
            int(self.lengths[idx]) if self.lengths is not None else sample.shape[-1]
        )
        valid_length_tensor = torch.tensor(valid_length, dtype=torch.long)

        return sample, label, valid_length_tensor

    def get_full(self, idx) -> torch.Tensor:
        """Return the complete valid time series for sample idx as [W, valid_T].

        Never used during training - intended for evaluation, visualisation,
        and the extrap loss callback which need the full unpadded series.
        """
        if self._lazy:
            sample = torch.from_numpy(
                self._lazy_data.isel(sample=int(idx)).values
            ).float()
        else:
            sample = self.data[idx]

        if self.normalize:
            sample = sample / (self.std + 1e-8)
        sample = sample.transpose(0, 1)  # [W, T_max]

        valid_T = (
            int(self.lengths[idx]) if self.lengths is not None else sample.shape[-1]
        )
        return sample[:, :valid_T]  # [W, valid_T]

    def __getitems__(self, indices):
        """Batch-load for lazy data: one xarray read instead of N individual isel() calls."""
        if self._lazy:
            batch = torch.from_numpy(
                self._lazy_data.isel(sample=list(indices)).values
            ).float()  # [B, T, W]
        else:
            batch = self.data[list(indices)]  # [B, T, W]

        if self.normalize:
            batch = batch / (self.std + 1e-8)
        batch = batch.transpose(1, 2)  # [B, W, T_max]

        labels = (
            [str(self.labels[i]) for i in indices]
            if self.labels is not None
            else [0] * len(indices)
        )

        # Extract true unpadded lengths for the entire batch
        valid_lengths = [
            (
                torch.tensor(int(self.lengths[i]), dtype=torch.long)
                if self.lengths is not None
                else torch.tensor(batch.shape[-1], dtype=torch.long)
            )
            for i in indices
        ]

        return list(zip(batch, labels, valid_lengths))

