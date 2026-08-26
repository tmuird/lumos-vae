"""Build a store that mixes photobleaching series with single-frame spectra.

This simulates the deployment case the model is aimed at: a set of cells
measured as a full bleaching series, plus cells for which only one frame exists,
as flow cytometry would give. The single-frame cells carry no targets, so a
supervised method cannot use them at all; LUMOS can, because fitting needs no
labels.

Only ``lengths`` changes. The spectra themselves are untouched, so the frames
beyond the first are present but never read for the single-frame cells: the
batch sampler places them in their own group and training uses T=1 there.
"""

import numpy as np
import xarray as xr

SOURCE = "/home/tom/Developments/Raman/oracle/data/processed/glasgow/glasgow_16_trimmed_smoothed.zarr"
OUT = "/home/tom/Developments/Raman/oracle/data/processed/glasgow/glasgow_mixed_1_16.zarr"
N_FRAMES = 16


def main():
    ds = xr.open_zarr(SOURCE).isel(time=slice(0, N_FRAMES)).load()
    split = ds["split"].values

    # Train keeps the full window; test is reduced to a single frame, which is
    # what a flow measurement of those cells would have given.
    lengths = np.where(split == "test", 1, N_FRAMES).astype(ds["lengths"].dtype)
    ds["lengths"] = ("sample", lengths)
    ds.attrs["note"] = (
        f"{SOURCE.split('/')[-1]} cropped to {N_FRAMES} frames, with the test "
        "split marked as single-frame to simulate flow acquisition"
    )

    for var in ds.data_vars:
        ds[var].encoding.clear()
    for coord in ds.coords:
        ds[coord].encoding.clear()
    ds.to_zarr(OUT, mode="w")

    counts = {int(n): int((lengths == n).sum()) for n in np.unique(lengths)}
    print(f"wrote {OUT}")
    print(f"  {ds.sizes['sample']} spectra, {ds.sizes['time']} frames stored")
    print(f"  lengths: {counts}")
    splits = {k: int((split == k).sum()) for k in ("train", "val", "test")}
    print(f"  splits:  {splits}")


if __name__ == "__main__":
    main()
