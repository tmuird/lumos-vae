"""Dataset and datamodule in the LUMOS style, reading the pixel store.

As in lumos-vae: global std-only normalisation computed over the fitting set
and the first ``n_times_train`` bins, transductive fitting by default, and
batches of (sample [C, T], label, length). All TCSPC histograms have the same
length, so the length-grouped sampler of LUMOS is not needed.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from lumos_flim.data import GT_VARS, SPLITS, irf_t0_guess, open_store


class TCSPCDataset(Dataset):
    def __init__(self, counts, labels, n_times_train=None, normalize=True, std=None):
        self.data = torch.as_tensor(counts, dtype=torch.float32)  # [N, C, T]
        self.labels = labels
        self.normalize = normalize
        self.lengths = np.full(len(counts), counts.shape[-1])
        if std is None:
            stats = self.data[..., :n_times_train] if n_times_train else self.data
            std = float(stats.std())
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.normalize:
            sample = sample / (self.std + 1e-8)
        return sample, int(self.labels[idx]), torch.tensor(int(self.lengths[idx]))


class ZarrDataModule(pl.LightningDataModule):
    def __init__(self, zarr_path, batch_size: int = 256, n_times_train: Optional[int] = None,
                 normalize: bool = True, transductive: bool = True, num_workers: int = 0):
        super().__init__()
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.n_times_train = n_times_train
        self.normalize = normalize
        self.transductive = transductive
        self.num_workers = num_workers
        self.full_ds = None
        self.gt = None

    def setup(self, stage=None):
        if self.full_ds is not None:
            return
        ds, meta = open_store(self.zarr_path)
        counts = ds["counts"].values
        split = ds["split"].values
        labels = ds["image"].values
        self.meta = meta
        self.n_channels = counts.shape[1]
        self.bin_width = float(meta["bin_width_ns"])
        self.time_values = ds.coords["time"].values
        if self.n_times_train is None:
            self.n_times_train = counts.shape[-1]

        fit = split != SPLITS["val"] if self.transductive else split == SPLITS["train"]
        val_idx = np.flatnonzero(split == SPLITS["val"])
        self.full_ds = TCSPCDataset(counts[fit], labels[fit], self.n_times_train, self.normalize)
        self.val_ds = TCSPCDataset(counts[val_idx], labels[val_idx], normalize=self.normalize,
                                   std=self.full_ds.std)
        self.irf_t0_guess = irf_t0_guess(counts[fit], self.bin_width)

        present = [k for k in GT_VARS + ("gt_alpha",) if k in ds]
        if present:
            self.gt = {k: ds[k].values[val_idx] for k in present}
        print(f"ZarrDataModule: {fit.sum()} fit, {len(val_idx)} val | n_times_train="
              f"{self.n_times_train} of {counts.shape[-1]} bins | std={self.full_ds.std:.3g}")

    def train_dataloader(self):
        return DataLoader(self.full_ds, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
