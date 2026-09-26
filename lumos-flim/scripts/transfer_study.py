"""Does a trained VAE carry over to a new image? The practical case for amortisation.

    python scripts/transfer_study.py --fraction 0.2

Both hMSC images (control and rotenone, same instrument) are binned 2x2,
pixels with at least 500 photons kept, and thinned to ``fraction`` of their
photons; the dropped photons are held out. A VAE trained on control is applied
to rotenone without retraining and compared with a VAE trained on rotenone
itself and with TV, binned and per-pixel MLE fitted on rotenone. All are
scored on rotenone's held-out photons; wall-clock time is recorded for what a
user would actually wait for on a new image (inference only for the
transferred VAE, the full fit for everything else).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from benchmark import heldout_nll  # noqa: E402

from lumos_flim.baseline import PixelFitter, run_method
from lumos_flim.data import (assign_splits, irf_t0_guess, open_store, read_imspector_tiff,
                             spatial_bin, spatial_context, write_store)
from lumos_flim.predict import run_model
from lumos_flim.vae import FlimVAE
from lumos_flim.vae_module import FlimModule

HMSC = os.environ.get("FLUTE_DIR", "/home/user/phasorpy/phasorpy-data/zenodo_8046636") + "/"


def load(name, fraction, rng, min_counts=500):
    counts, bw = read_imspector_tiff(HMSC + name)
    counts = spatial_bin(counts, 2)
    T, Y, X = counts.shape
    flat = counts.reshape(T, -1).T
    keep = np.flatnonzero(flat.sum(1) >= min_counts)
    full = flat[keep][:, None, :].astype(np.int64)
    kept = rng.binomial(full, fraction)
    y, x = np.divmod(keep, X)
    return kept, full - kept, bw, y, x, (Y, X)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--vae_args", default="--kl_warmup_epochs 20",
                   help="extra lumos_flim.train flags for both VAEs")
    p.add_argument("--out", default="results/benchmark/transfer.json")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    data = {}
    for key, name in (("control", "hMSC control.tif"), ("rotenone", "hMSC_rotenone.tif")):
        kept, held, bw, y, x, shape = load(name, a.fraction, rng)
        split = assign_splits(len(kept), 0.1, 0.1, 0)
        store = f"data/transfer_{key}.zarr"
        write_store(store, kept, bw, np.zeros(len(y)), y, x, [key], [shape], split)
        data[key] = dict(kept=kept, held=held, bw=bw, y=y, x=x, shape=shape, store=store)
        print(f"{key}: {len(kept)} pixels, median {np.median(kept.sum((1, 2))):.0f} photons kept")

    rows = []
    rot = data["rotenone"]
    k = [f for f in a.vae_args.split()]
    spatial = "--spatial" in k
    mode = k[k.index("--spatial_mode") + 1] if "--spatial_mode" in k else "sum"
    size = int(k[k.index("--spatial") + 1]) if spatial else 0

    def context(d):
        return spatial_context(d["kept"], np.zeros(len(d["y"])), d["y"], d["x"], [d["shape"]],
                               size, mode) if spatial else None

    def train(key):
        run = f"transfer_{key}"
        shutil.rmtree(f"checkpoints/{run}", ignore_errors=True)
        t = time.time()
        subprocess.run([sys.executable, "-m", "lumos_flim.train", "--data", data[key]["store"],
                        "--max_epochs", str(a.epochs), "--batch_size", "512", "--run_name", run,
                        *k], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ckpt = f"checkpoints/{run}/{'last' if '--spatial_prior' in k else 'best'}.ckpt"
        return FlimModule.load_from_checkpoint(ckpt, map_location="cpu"), time.time() - t

    def record(method, res, mu, seconds, note=""):
        rows.append(dict(method=method, seconds=round(seconds, 1), note=note,
                         heldout_nll=heldout_nll(mu, rot["held"]),
                         tau_amp_median=float(np.median(res["tau_amp"]))))
        print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    vae_control, train_s = train("control")
    t = time.time()
    res = run_model(vae_control, rot["kept"], context=context(rot))
    record("VAE trained on control", res, res.pop("expected"), time.time() - t,
           f"inference only; training on control took {train_s:.0f} s")

    vae_rot, train_s = train("rotenone")
    t = time.time()
    res = run_model(vae_rot, rot["kept"], context=context(rot))
    record("VAE trained on rotenone", res, res.pop("expected"), train_s + time.time() - t)

    ds, meta = open_store(rot["store"])
    model = FlimVAE(n_bins=rot["kept"].shape[-1], bin_width=rot["bw"],
                    irf_t0=irf_t0_guess(rot["kept"], rot["bw"]), irf_sigma=0.1)
    t = time.time()
    base = PixelFitter(rot["kept"], model).fit(1500, fit_irf=True)
    res, mu = base.results()
    mle_s = time.time() - t
    record("per-pixel MLE", res, mu, mle_s)
    for method in ("binned", "tv"):
        t = time.time()
        res, mu = run_method(method, rot["kept"], model, ds, meta, irf_from=base)
        record(method, res, mu, mle_s + time.time() - t, "includes the IRF fit")


if __name__ == "__main__":
    main()
