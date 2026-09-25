"""Reading TCSPC images and the processed Zarr store.

Store layout, one row per pixel (after any spatial binning):

    counts      [sample, channel, time]  photon counts
    split       [sample]                 0 train, 1 val, 2 test
    image       [sample]                 index into attrs["image_names"]
    y, x        [sample]                 pixel position in that image
    gt_tau, gt_fraction [sample, component], gt_background [sample]
                                         synthetic data only, never used in a loss

Attributes: ``bin_width_ns``, ``period_ns``, ``image_names``, ``image_shapes``.
"""

import json
from typing import Optional
from xml.etree import ElementTree

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = {"train": 0, "val": 1, "test": 2}
GT_VARS = ("gt_tau", "gt_fraction", "gt_background")


def read_imspector_tiff(path):
    """TCSPC histogram from an ImSpector OME-TIFF, as in the FLUTE dataset.

    Returns (counts [T, Y, X], bin_width_ns). The histogram spans exactly one
    laser period, which the file stores as the extent of its time axis.
    """
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data, axes = np.asarray(series.asarray()), series.axes
        omexml = tif.pages.first.tags.valueof(270, "")
    ns = {
        "": "http://www.openmicroscopy.org/Schemas/OME/2008-02",
        "ca": "http://www.openmicroscopy.org/Schemas/CA/2008-02",
    }
    root = ElementTree.fromstring(omexml)
    pixels = root.find(".//Image/Pixels", ns)
    labels = root.find(".//Image/ca:CustomAttributes/AxesLabels", ns)
    if pixels is None or labels is None or not axes.endswith("YX") or len(axes) < 3:
        raise ValueError(f"{path} is not an ImSpector FLIM TIFF")

    # ImSpector stores the delay axis as whichever OME axis precedes Y and X,
    # and labels it "TCSPC T" in its own AxesLabels element.
    if not labels.attrib.get("FirstAxis", "").endswith("TCSPC T"):
        raise ValueError(f"{path} has no TCSPC delay axis")
    ax = axes[-3]
    attrib = "TimeIncrement" if ax == "T" else f"PhysicalSize{ax}"
    bin_width = float(pixels.attrib[attrib])
    data = data.reshape(data.shape[-3:]) if data.ndim > 3 else data
    return data, bin_width


def spatial_bin(counts, factor: int):
    """Sum ``factor x factor`` blocks of a [T, Y, X] stack."""
    if factor <= 1:
        return counts
    T, Y, X = counts.shape
    Y2, X2 = Y // factor, X // factor
    c = counts[:, : Y2 * factor, : X2 * factor].astype(np.int64)
    return c.reshape(T, Y2, factor, X2, factor).sum(axis=(2, 4))


def assign_splits(n: int, val_frac: float, test_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    split = np.zeros(n, dtype=np.int8)
    perm = rng.permutation(n)
    n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
    split[perm[:n_val]] = SPLITS["val"]
    split[perm[n_val:n_val + n_test]] = SPLITS["test"]
    return split


def write_store(path, counts, bin_width, image, y, x, image_names, image_shapes,
                split, gt: Optional[dict] = None, attrs: Optional[dict] = None):
    import xarray as xr

    n, c, t = counts.shape
    data = {
        "counts": (("sample", "channel", "time"), counts.astype(np.float32)),
        "split": (("sample",), split.astype(np.int8)),
        "image": (("sample",), image.astype(np.int32)),
        "y": (("sample",), y.astype(np.int32)),
        "x": (("sample",), x.astype(np.int32)),
    }
    for key, value in (gt or {}).items():
        dims = ("sample", "component") if value.ndim == 2 else ("sample",)
        data[key] = (dims, value.astype(np.float32))
    ds = xr.Dataset(data, coords={"time": np.arange(t) * bin_width, "channel": np.arange(c)})
    ds.attrs.update(
        bin_width_ns=float(bin_width),
        period_ns=float(bin_width * t),
        image_names=json.dumps(list(image_names)),
        image_shapes=json.dumps([list(map(int, s)) for s in image_shapes]),
        **(attrs or {}),
    )
    ds.to_zarr(path, mode="w")
    return ds


def open_store(path):
    import xarray as xr

    ds = xr.open_zarr(path)
    meta = dict(ds.attrs)
    meta["image_names"] = json.loads(meta["image_names"])
    meta["image_shapes"] = json.loads(meta["image_shapes"])
    return ds, meta


def irf_t0_guess(counts, bin_width: float) -> float:
    """Pulse arrival guessed from the steepest rise of the summed histogram."""
    h = np.asarray(counts).reshape(-1, counts.shape[-1]).sum(0)
    rise = np.diff(np.concatenate([h[-1:], h]))  # periodic
    return float(np.argmax(rise)) * bin_width


class PixelDataset(Dataset):
    def __init__(self, counts, gt: Optional[dict] = None):
        self.counts = torch.as_tensor(counts, dtype=torch.float32)
        self.gt = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in (gt or {}).items()}

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, i):
        item = {"x": self.counts[i], "index": i}
        for k, v in self.gt.items():
            item[k] = v[i]
        return item


class FlimDataModule(pl.LightningDataModule):
    """Loads the whole store into memory; pixel stores are small.

    Fitting is unsupervised, so by default the model fits on train and test
    together (``transductive``) and is monitored on val, as in LUMOS.
    """

    def __init__(self, path, batch_size: int = 512, transductive: bool = True,
                 num_workers: int = 0):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.transductive = transductive
        self.num_workers = num_workers
        self.train_ds = None

    def setup(self, stage=None):
        if self.train_ds is not None:
            return
        ds, meta = open_store(self.path)
        counts = ds["counts"].values
        split = ds["split"].values
        gt = {k: ds[k].values for k in GT_VARS if k in ds}
        self.meta = meta
        self.n_channels, self.n_bins = counts.shape[1], counts.shape[2]
        self.bin_width = float(meta["bin_width_ns"])

        fit = split != SPLITS["val"] if self.transductive else split == SPLITS["train"]
        val = split == SPLITS["val"]
        self.train_ds = PixelDataset(counts[fit], {k: v[fit] for k, v in gt.items()})
        self.val_ds = PixelDataset(counts[val], {k: v[val] for k, v in gt.items()})
        self.irf_t0_guess = irf_t0_guess(counts[fit], self.bin_width)
        print(f"FlimDataModule: {len(self.train_ds)} fit, {len(self.val_ds)} val pixels, "
              f"{self.n_channels} channel(s) x {self.n_bins} bins of {self.bin_width:.4f} ns, "
              f"median {np.median(counts[fit].sum((1, 2))):.0f} photons/pixel")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=4 * self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
