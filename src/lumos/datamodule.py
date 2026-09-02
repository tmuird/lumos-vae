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
from lumos.sampler import LengthGroupedBatchSampler

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

    Fits on the train and test splits by default and monitors ``val_recon_loss``
    on val. Fitting is unsupervised, so the test spectra can be used without
    their targets, which a supervised baseline cannot do; it matches the
    deployment case where the spectra to be corrected are already in hand.
    ``transductive=False`` fits on train alone, for a strictly held-out estimate.

    The normalisation std comes from the same spectra the model fits on, over the
    first ``n_times_train`` frames, so it follows the pool automatically.

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
        transductive: bool = True,
        lazy: bool = False,
        n_train: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers if lazy else 0
        self.n_times_train = n_times_train
        self.normalize = normalize
        self.preload_device = None if lazy else preload_device
        self.transductive = transductive
        self.lazy = lazy
        # Cap the training pool, to measure how performance scales with the
        # number of bleaching series. 0 uses everything.
        self.n_train = n_train
        self.seed = seed
        # Fixed fluorophore emission spectra, when the dyes are known in advance.

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
        if self.n_train and self.n_train < len(train_idx):
            rng = np.random.default_rng(self.seed)
            train_idx = np.sort(rng.permutation(train_idx)[:self.n_train])
        val_idx = np.where(split_arr == "val")[0]
        test_idx = np.where(split_arr == "test")[0]

        min_valid = int(lengths.min())
        if self.n_times_train is not None and self.n_times_train > min_valid:
            # Not an error: spots shorter than the window form their own batch
            # group and train at whatever length they have.
            print(
                f"  n_times_train={self.n_times_train} exceeds the shortest spot "
                f"({min_valid} frames); short spots will batch separately"
            )

        print(
            f"ZarrDataModule: {len(split_arr)} total samples | "
            f"n_times_train={self.n_times_train} | min_valid_frames={min_valid}"
        )

        # Full axes, exposed for model construction. The train data is cropped to
        # n_times_train, but the model still needs the full time axis to
        # reconstruct the extrapolation window during validation.
        self.time_values = time_values
        self.wavenumbers = wavenumbers

        def _make_ds(idx, crop_T=None):
            inten = intensities.isel(sample=idx.tolist())
            tvals = time_values
            lens = lengths[idx]
            if crop_T is not None:
                inten = inten.isel(time=slice(0, crop_T))
                tvals = time_values[:crop_T]
                lens = np.minimum(lens, crop_T)
            return BleachingDataset(
                intensities=inten,
                labels=labels[idx],
                time_values=tvals,
                wavenumbers=wavenumbers,
                lengths=lens,
                n_times_train=self.n_times_train,
                normalize=self.normalize,
                device=self.preload_device,
                lazy=self.lazy,
            )

        # Transductive fitting leaks no labels but reports on seen spectra.
        fit_idx = np.concatenate([train_idx, test_idx]) if self.transductive else train_idx
        print(
            f"  fitting on {len(fit_idx)} spectra "
            f"({'train+test, transductive' if self.transductive else 'train only'}), "
            f"{len(val_idx)} val, {len(test_idx)} test"
        )
        self.full_ds = _make_ds(fit_idx, crop_T=self.n_times_train)

        # Std over exactly the spectra the model trains on, so it follows the
        # split automatically if the training set changes.
        ref_std = self.full_ds.std

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
        """Batches are homogeneous in frame count, so each has one window length."""
        sampler = LengthGroupedBatchSampler(
            self.full_ds.lengths, self.batch_size, shuffle=True
        )
        return DataLoader(
            self.full_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> Any:
        return self._loader(self.predict_dataset, shuffle=False)
