"""Build a store that mixes photobleaching series with single-frame spectra.

This simulates the deployment case the model is aimed at: some cells measured
as a full bleaching series, others for which only one frame exists, as flow
cytometry would give. The single-frame cells carry no targets, so a supervised
method cannot use them at all; LUMOS can, because fitting needs no labels.

Only ``lengths`` changes. The spectra themselves are untouched, so the frames
beyond the first are present but never read for the single-frame cells: the
batch sampler places them in their own group and training uses T=1 there.

Two layouts:

  transductive  the test split becomes single-frame. The flow spectra the model
                trains on are the ones it is scored on, which is legitimate for
                an unsupervised method but does not measure generalisation.

  inductive     part of the train split becomes single-frame and the test split
                is held out entirely. The model still learns what a lone frame
                looks like, but never sees a spectrum it is scored on.

``--flow_fraction`` sets how much of the train split is reduced, since the
realistic case has far more flow frames than bleaching series. Whether that
imbalance then hurts is a question for ``--length_balance``: batches are drawn
per frame-count group in proportion to group size unless told otherwise.
"""

import argparse
import collections

import numpy as np
import xarray as xr

ROOT = "/home/tom/Developments/Raman/oracle/data/processed/glasgow"
SOURCE = f"{ROOT}/glasgow_16_trimmed_smoothed.zarr"
N_FRAMES = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("transductive", "inductive"),
                    default="inductive")
    ap.add_argument("--flow_fraction", type=float, default=0.5,
                    help="Share of the train split reduced to one frame")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = xr.open_zarr(args.source).isel(time=slice(0, N_FRAMES)).load()
    split = np.array([str(v) for v in ds["split"].values])
    lengths = np.full(len(split), N_FRAMES, dtype=ds["lengths"].dtype)

    if args.mode == "transductive":
        lengths[split == "test"] = 1
        note = "test split marked single-frame"
    else:
        train_idx = np.where(split == "train")[0]
        n_flow = int(round(args.flow_fraction * len(train_idx)))
        rng = np.random.default_rng(args.seed)
        lengths[rng.permutation(train_idx)[:n_flow]] = 1
        note = f"{n_flow} of {len(train_idx)} train spectra marked single-frame"

    ds["lengths"] = ("sample", lengths)
    ds.attrs["note"] = (
        f"{args.source.split('/')[-1]} cropped to {N_FRAMES} frames; {note} "
        "to simulate flow acquisition")

    out = args.out or (
        f"{ROOT}/glasgow_mixed_{args.mode}"
        + (f"_{args.flow_fraction:g}".replace(".", "p")
           if args.mode == "inductive" else "") + ".zarr")
    for var in ds.data_vars:
        ds[var].encoding.clear()
    for coord in ds.coords:
        ds[coord].encoding.clear()
    ds.to_zarr(out, mode="w")

    print(f"wrote {out}")
    for name in ("train", "val", "test"):
        m = split == name
        counts = dict(collections.Counter(lengths[m].tolist()))
        print(f"  {name:<6} {int(m.sum()):>4} spectra, lengths {counts}")


if __name__ == "__main__":
    main()
