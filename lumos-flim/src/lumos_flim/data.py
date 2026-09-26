"""Reading TCSPC images and the processed Zarr store.

Store layout, one row per pixel (after any spatial binning):

    counts      [sample, channel, time]  photon counts
    split       [sample]                 0 train, 1 val, 2 test
    image       [sample]                 index into attrs["image_names"]
    y, x        [sample]                 pixel position in that image
    gt_tau, gt_fraction [sample, component], gt_background [sample]
                                         synthetic data only, never used in a loss
    gt_boundary [sample]                 realistic synthetic data: pixel on a region edge

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


def _dense(counts, image, y, x, shapes):
    """Scatter pixel rows back onto their image grids, zeros where no pixel."""
    grids = [np.zeros((h, w) + counts.shape[1:], dtype=np.float64) for h, w in shapes]
    for i, g in enumerate(grids):
        m = image == i
        g[y[m], x[m]] = counts[m]
    return grids


def neighbourhood_sum(counts, image, y, x, shapes, k: int = 3, include_centre: bool = False):
    """Sum of each pixel's k x k neighbourhood, [N, C, T].

    Pixels absent from the store (dropped as too dim) count as zero. With
    ``include_centre`` this is the usual binned histogram; without it, the
    context the spatial encoder sees alongside the pixel itself.
    """
    r = k // 2
    out = np.zeros(counts.shape, dtype=np.float64)
    for i, g in enumerate(_dense(counts, image, y, x, shapes)):
        # Box sum via a 2D cumulative sum over a zero-padded grid.
        pad = np.pad(g, ((r + 1, r), (r + 1, r)) + ((0, 0),) * (g.ndim - 2))
        cs = pad.cumsum(0).cumsum(1)
        box = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
        m = image == i
        out[m] = box[y[m], x[m]]
    if not include_centre:
        out -= counts
    return out


def neighbour_stack(counts, image, y, x, shapes, k: int = 3):
    """Each pixel's k x k neighbours as separate histograms, [N, k*k - 1, C, T].

    Unlike ``neighbourhood_sum`` this keeps which neighbour is which, so an
    encoder can tell a neighbour across an edge from one that agrees.
    Neighbours outside the image or absent from the store are zeros.
    """
    r = k // 2
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]
    out = np.zeros((len(counts), len(offsets)) + counts.shape[1:], dtype=np.float32)
    for i, g in enumerate(_dense(counts, image, y, x, shapes)):
        m = image == i
        pad = np.pad(g, ((r, r), (r, r)) + ((0, 0),) * (g.ndim - 2))
        for j, (dy, dx) in enumerate(offsets):
            out[m, j] = pad[y[m] + r + dy, x[m] + r + dx]
    return out


def spatial_context(counts, image, y, x, shapes, k: int, mode: str = "sum"):
    """Encoder context, [N, M, C, T]: the summed neighbourhood (M = 1) or the
    neighbours stacked individually (M = k*k - 1)."""
    if mode == "stack":
        return neighbour_stack(counts, image, y, x, shapes, k)
    return neighbourhood_sum(counts, image, y, x, shapes, k)[:, None].astype(np.float32)


def context_slots(k: int, mode: str) -> int:
    return k * k - 1 if mode == "stack" else 1


def neighbour_pairs(image, y, x):
    """Index pairs (i, j) of 4-connected neighbours present in the store."""
    key = {(int(im), int(a), int(b)): n for n, (im, a, b) in enumerate(zip(image, y, x))}
    pairs = []
    for n, (im, a, b) in enumerate(zip(image, y, x)):
        for da, db in ((1, 0), (0, 1)):
            j = key.get((int(im), int(a) + da, int(b) + db))
            if j is not None:
                pairs.append((n, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def irf_t0_guess(counts, bin_width: float) -> float:
    """Pulse arrival guessed from the steepest rise of the summed histogram."""
    h = np.asarray(counts).reshape(-1, counts.shape[-1]).sum(0)
    rise = np.diff(np.concatenate([h[-1:], h]))  # periodic
    return float(np.argmax(rise)) * bin_width


class PixelDataset(Dataset):
    """Pixel histograms, optionally held on the training device.

    Batches are gathered with one indexing call (``__getitems__``) rather than
    per pixel, which matters once the tensors live on a GPU.
    """

    def __init__(self, counts, gt: Optional[dict] = None, device=None, context=None):
        self.counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
        self.context = (torch.as_tensor(context, dtype=torch.float32, device=device)
                        if context is not None else None)
        self.gt = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                   for k, v in (gt or {}).items()}

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, i):
        return self.__getitems__([i])

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, device=self.counts.device)
        batch = {"x": self.counts[idx], "index": idx}
        if self.context is not None:
            batch["context"] = self.context[idx]
        for k, v in self.gt.items():
            batch[k] = v[idx]
        return batch


def _identity(batch):
    return batch


class FlimDataModule(pl.LightningDataModule):
    """Loads the whole store into memory; pixel stores are small.

    Fitting is unsupervised, so by default the model fits on train and test
    together (``transductive``) and is monitored on val, as in LUMOS. With
    ``preload_device`` the arrays are copied to that device once, so no batch
    crosses from host memory during training.
    """

    def __init__(self, path, batch_size: int = 512, transductive: bool = True,
                 num_workers: int = 0, preload_device=None, spatial: int = 0,
                 spatial_mode: str = "sum"):
        super().__init__()
        # Neighbourhood size for the spatial encoder's context, 0 for none,
        # and whether the neighbours are summed or kept separate.
        self.spatial = spatial
        self.spatial_mode = spatial_mode
        self.path = path
        self.batch_size = batch_size
        self.transductive = transductive
        self.preload_device = preload_device
        # Worker processes cannot share tensors that already sit on a GPU.
        self.num_workers = 0 if preload_device is not None else num_workers
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
        dev = self.preload_device
        ctx = None
        if self.spatial:
            ctx = spatial_context(counts, ds["image"].values, ds["y"].values, ds["x"].values,
                                  meta["image_shapes"], self.spatial, self.spatial_mode)
        pick = lambda m: None if ctx is None else ctx[m]  # noqa: E731
        self.train_ds = PixelDataset(counts[fit], {k: v[fit] for k, v in gt.items()}, dev, pick(fit))
        self.val_ds = PixelDataset(counts[val], {k: v[val] for k, v in gt.items()}, dev, pick(val))
        self.irf_t0_guess = irf_t0_guess(counts[fit], self.bin_width)
        print(f"FlimDataModule: {len(self.train_ds)} fit, {len(self.val_ds)} val pixels, "
              f"{self.n_channels} channel(s) x {self.n_bins} bins of {self.bin_width:.4f} ns, "
              f"median {np.median(counts[fit].sum((1, 2))):.0f} photons/pixel"
              + (f", preloaded to {dev}" if dev is not None else ""))

    def _loader(self, ds, batch_size, shuffle, drop_last):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.num_workers, collate_fn=_identity)

    def train_dataloader(self):
        return self._loader(self.train_ds, self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, 4 * self.batch_size, shuffle=False, drop_last=False)
