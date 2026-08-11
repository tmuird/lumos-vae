"""Data loading from a processed Zarr store.

The store holds the observed time series plus a train/val/test split. Training is
strictly unsupervised: only ``time_series`` is used to fit the model. Ground-truth
variables, when present, are read solely for logging and never enter any loss.
"""

from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lumos.dataset import BleachingDataset

# Ground-truth variables loaded for logging only. Absent in real-data stores.
_GT_VARS = (
    "gt_raman",
    "gt_raman_sum",
    "gt_late_mean",
    "decay_rates_gt",
    "abundances_gt",
    "fluorophore_bases_gt",
    "intensity_clean",
)


class ZarrDataModule(pl.LightningDataModule):
    """Load a processed Zarr store and expose train/val/test dataloaders.

    Training uses the train and test splits (the model is transductive and
    unsupervised); the val split is held out purely to monitor ``val_recon_loss``
    for the LR scheduler and early stopping. Normalisation std is computed from
    the train split only, over the first ``n_times_train`` frames, to avoid
    leakage from the extrapolation region.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store.
    config : object
        Model config, passed through for callbacks that inspect the datamodule.
    n_times_train : int or None
        Number of early frames the encoder sees; also the window used for the
        normalisation std.
    """

    def __init__(
        self,
        zarr_path: str,
        config,
        batch_size: int = 32,
        num_workers: int = 0,
        n_times_train: Optional[int] = None,
        normalize: bool = True,
        preload_device=None,
        semi_supervised: bool = False,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.config = config
        self.batch_size = batch_size
        self.num_workers = 0 if preload_device is not None else num_workers
        self.n_times_train = n_times_train
        self.normalize = normalize
        self.preload_device = preload_device
        self.semi_supervised = semi_supervised

        self.full_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_dataset = None
        self.dims = None
        # Ground-truth arrays for logging only, keyed by variable name. None when
        # the store carries no ground truth (real data).
        self.gt = None

    def setup(self, stage: Optional[str] = None):
        if self.full_ds is not None:
            return

        import xarray as xr

        ds = xr.open_zarr(self.zarr_path)

        # [N, W, T] in the store -> [N, T, W] for BleachingDataset.
        intensities = ds["time_series"].transpose("sample", "time", "wavenumber")
        labels = ds["labels"].values
        lengths = ds["lengths"].values.astype(int)
        split_arr = ds["split"].values
        wavenumbers = ds.coords["wavenumber"].values
        time_values = ds.coords["time"].values

        train_idx = np.where(split_arr == "train")[0]
        val_idx = np.where(split_arr == "val")[0]
        test_idx = np.where(split_arr == "test")[0]

        min_valid = int(lengths.min())
        if self.n_times_train is not None and self.n_times_train > min_valid:
            raise ValueError(
                f"n_times_train ({self.n_times_train}) must be <= shortest spot "
                f"({min_valid} valid frames)"
            )

        mode_str = (
            "semi-supervised" if self.semi_supervised else "unsupervised (transductive)"
        )
        print(
            f"ZarrDataModule: {len(split_arr)} total samples | mode={mode_str} | "
            f"n_times_train={self.n_times_train} | min_valid_frames={min_valid}"
        )

        def _make_ds(idx, is_test_mask=None):
            return BleachingDataset(
                intensities=intensities.isel(sample=idx.tolist()),
                labels=labels[idx],
                time_values=time_values,
                wavenumbers=wavenumbers,
                lengths=lengths[idx],
                n_times_train=self.n_times_train,
                normalize=self.normalize,
                device=self.preload_device,
                is_test_mask=is_test_mask,
            )

        # Std from train split only, to avoid leakage from val/test.
        ref_std = _make_ds(train_idx).std

        # Train on train+test; val is genuinely held out. In semi-supervised mode
        # test samples are restricted to n_times_train frames in the loss.
        train_test_idx = np.concatenate([train_idx, test_idx])
        if self.semi_supervised:
            is_test_mask = np.zeros(len(train_test_idx), dtype=bool)
            is_test_mask[len(train_idx):] = True
            self.full_ds = _make_ds(train_test_idx, is_test_mask=is_test_mask)
        else:
            self.full_ds = _make_ds(train_test_idx)
        self.full_ds.std = ref_std

        self.val_ds = _make_ds(val_idx)
        self.test_ds = _make_ds(test_idx)
        if self.normalize:
            self.val_ds.std = ref_std
            self.test_ds.std = ref_std

        self.dims = self.full_ds[0][0].shape
        self.gt = self._load_ground_truth(ds, val_idx)

        if stage in ("predict", None):
            self.predict_dataset = self.full_ds

    def _load_ground_truth(self, ds, val_idx) -> Optional[dict]:
        """Read ground-truth arrays for the val split, for logging only.

        Returns None when the store has no ground truth. These arrays are never
        used in training.
        """
        present = [v for v in _GT_VARS if v in ds]
        if not present:
            return None
        gt = {"val_idx": val_idx, "wavenumber": ds.coords["wavenumber"].values}
        for name in present:
            arr = ds[name]
            if "sample" in arr.dims:
                arr = arr.isel(sample=val_idx.tolist())
            gt[name] = arr.values
        return gt

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.full_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> Any:
        return self._loader(self.predict_dataset, shuffle=False)
