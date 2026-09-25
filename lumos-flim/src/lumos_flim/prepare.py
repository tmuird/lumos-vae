"""Turn ImSpector FLIM TIFFs into a pixel store for training.

    python -m lumos_flim.prepare "hMSC control.tif" hMSC_rotenone.tif \\
        --out data/hmsc.zarr --bin 2 --min_counts 300

Each image is binned spatially, pixels below ``--min_counts`` photons (mostly
empty background) are dropped, and the rest are split at random into
train/val/test. All images must share the same time axis.
"""

import argparse
from pathlib import Path

import numpy as np

from lumos_flim.data import assign_splits, read_imspector_tiff, spatial_bin, write_store


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--bin", type=int, default=2, help="spatial binning factor")
    p.add_argument("--min_counts", type=float, default=300)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    rows, names, shapes, bin_width = [], [], [], None
    for i, path in enumerate(a.images):
        counts, bw = read_imspector_tiff(path)
        if bin_width is not None and not np.isclose(bw, bin_width):
            raise ValueError(f"{path} has bin width {bw}, expected {bin_width}")
        bin_width = bw
        counts = spatial_bin(counts, a.bin)
        T, Y, X = counts.shape
        flat = counts.reshape(T, -1).T  # [Y*X, T]
        keep = np.flatnonzero(flat.sum(1) >= a.min_counts)
        yy, xx = np.divmod(keep, X)
        rows.append((flat[keep], np.full(keep.size, i), yy, xx))
        names.append(Path(path).stem)
        shapes.append((Y, X))
        print(f"{path}: {T} bins x {Y}x{X} after binning, kept {keep.size} pixels, "
              f"median {np.median(flat[keep].sum(1)):.0f} photons")

    counts = np.concatenate([r[0] for r in rows])[:, None, :]  # one channel
    image = np.concatenate([r[1] for r in rows])
    yy = np.concatenate([r[2] for r in rows])
    xx = np.concatenate([r[3] for r in rows])
    split = assign_splits(len(counts), a.val_frac, a.test_frac, a.seed)
    write_store(a.out, counts, bin_width, image, yy, xx, names, shapes, split,
                attrs=dict(source="imspector", spatial_bin=a.bin, min_counts=a.min_counts))
    print(f"wrote {a.out}: {counts.shape}, bin width {bin_width:.4f} ns, "
          f"period {bin_width * counts.shape[-1]:.3f} ns")


if __name__ == "__main__":
    main()
