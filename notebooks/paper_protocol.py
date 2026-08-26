"""LUMOS against the supervised MLP of Peng et al. (ICBSP 2025), on fish.

Both methods predict the same object: the bleached reference spectrum of a cell,
background included. Their MLP is trained on paired (raw, bleached) examples and
sees one frame. LUMOS never sees a bleached reference at all: it fits a decay
model to the first few frames and extrapolates to the reference time.

Scored two ways. Their protocol is cosine and Pearson on raw spectra with the
dual threshold at 0.95, which is the measure their headline quotes. Peak
correlation strips a smooth background from both sides first, which is the
measure that actually separates methods here: a constant predictor already
clears 71% of their dual threshold.

Their percentages are quoted over all measurements, including the 80% fitted on.
Everything below is held-out only unless stated.
"""

import numpy as np
import torch
import xarray as xr
from pybaselines import Baseline, polynomial
from scipy.signal import savgol_filter
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from lumos.metrics import scale_invariant_psnr
from lumos.train import load_checkpoint

ZARR = "/home/tom/Developments/Raman/oracle/data/processed/fish/fish_6frame.zarr"
CKPT_DIR = "/home/tom/Developments/lumos/src/lumos/checkpoints"
SEED = 0

# label -> (run directory, frames fed to the encoder). Add rows to ablate.
LUMOS_RUNS = {
    "LUMOS 3 frames, inductive": (
        "fish_6frame_mog_t3_z64_h128_lr0.0005_20260825_160257", 3),
    "LUMOS 6 frames, inductive": (
        "fish_6frame_mog_t6_z64_h128_lr0.0005_20260825_160303", 6),
    "LUMOS 3 frames, transductive": (
        "fish_6frame_mog_t3_z64_h128_lr0.0005_20260825_161453", 3),
    "LUMOS 6 frames, transductive": (
        "fish_6frame_mog_t6_z64_h128_lr0.0005_20260825_161459", 6),
}


def cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def pearson(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return cosine(a, b)


def sam(a, b):
    """Spectral angle in degrees."""
    return np.degrees(np.arccos(np.clip(cosine(a, b), -1, 1)))


def raman_head(run, series, n_frames):
    """LUMOS's own Raman estimate: no baseline algorithm involved."""
    model, _ = load_checkpoint(run, CKPT_DIR)
    vae = model.model
    std = float(vae.dataset_std)
    with torch.no_grad():
        x = torch.from_numpy(series[:, :n_frames, :]).float().transpose(1, 2) / std
        _, _, _, _, _, raman, _ = vae(x, sample=False)
    return raman.cpu().numpy()


def strip_background(spectra, fitter):
    return np.stack([s - fitter.arpls(s.astype(float), lam=1e5)[0] for s in spectra])


def predict_mlp(raw, ref, train_idx):
    """Their model: 512/256 ReLU, Adam at 1e-3, early stopping.

    Each spectrum is normalised by its own area and log-compressed first. This
    store spans 352x in intensity, which per-channel standardisation alone
    cannot cope with; without it the same network scores 0.26 instead of 0.82.
    """
    forward = lambda x: np.log1p(np.clip(x, 0, None) / x.sum(1, keepdims=True) * 1e4)
    xs, ys = StandardScaler(), StandardScaler()
    mlp = MLPRegressor(
        hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
        learning_rate_init=1e-3, max_iter=1000, early_stopping=True,
        random_state=SEED,
    )
    mlp.fit(xs.fit_transform(forward(raw[train_idx])),
            ys.fit_transform(forward(ref[train_idx])))
    pred = ys.inverse_transform(mlp.predict(xs.transform(forward(raw))))
    return np.expm1(pred) / 1e4 * raw.sum(1, keepdims=True)


def predict_lumos(run, series, time_values, n_frames):
    """Reconstruct the final frame from the first n_frames."""
    model, _ = load_checkpoint(run, CKPT_DIR)
    vae = model.model
    std = float(vae.dataset_std)
    with torch.no_grad():
        x = torch.from_numpy(series[:, :n_frames, :]).float().transpose(1, 2) / std
        _, _, _, lam, ab, raman, bases = vae(x, sample=False)
        recon, _ = vae.physics_forward(
            lam, ab, raman, bases, time_values=torch.from_numpy(time_values).float()
        )
        recon = recon.cpu().numpy()
    return recon[:, :, -1] if recon.shape[-1] == len(time_values) else recon[:, -1, :]


def main():
    ds = xr.open_zarr(ZARR)
    X = ds["time_series"].transpose("sample", "time", "wavenumber").values
    t_vals = ds.coords["time"].values
    split = ds["split"].values
    raw, ref = X[:, 0, :], X[:, -1, :]
    # Same fitting pool as the model: LUMOS is trained on the train split, so
    # giving the MLP train+val would hand it more cells than we used.
    train_idx = np.where(split == "train")[0]
    test_idx = np.where(split == "test")[0]

    fitter = Baseline(x_data=np.arange(X.shape[2]))
    ref_peaks = strip_background(ref[test_idx], fitter)

    methods = {
        # Kept as a floor: it ignores its input entirely, so anything scoring
        # near it has not demonstrated anything.
        "constant: mean reference": np.repeat(
            ref[train_idx].mean(0, keepdims=True), len(X), axis=0
        ),
        "MLP (Peng et al.), 1 frame": predict_mlp(raw, ref, train_idx),
    }
    for label, (run, n_frames) in LUMOS_RUNS.items():
        methods[label] = predict_lumos(run, X, t_vals, n_frames)

    print(f"{len(X)} cells, {len(train_idx)} fitted on, {len(test_idx)} held out")
    print("target: bleached reference (final frame)\n")
    header = f"{'':<30}{'cosine':>8}{'pearson':>9}{'both>.95':>10}{'r(peaks)':>10}"
    print(header)
    print("-" * len(header))

    for name, pred in methods.items():
        p = pred[test_idx]
        c, r = cosine(p, ref[test_idx]), pearson(p, ref[test_idx])
        peaks = pearson(strip_background(p, fitter), ref_peaks).mean()
        print(f"{name:<30}{c.mean():>8.4f}{r.mean():>9.4f}"
              f"{((c > 0.95) & (r > 0.95)).mean():>10.1%}{peaks:>10.4f}")

    # Second table: background-free Raman. Classical correction cannot predict a
    # spectrum that still has a background, so it cannot appear above. Stripping
    # the background from every prediction and from the target puts all methods,
    # including LUMOS's own Raman head, on one footing. Metrics are
    # scale-invariant because the Raman head is in counts per second while the
    # rest are counts per frame.
    raman_methods = {
        "airPLS on raw": strip_background(raw[test_idx], fitter),
        "Savitzky-Golay + airPLS": strip_background(
            savgol_filter(raw[test_idx], 11, 3, axis=-1), fitter
        ),
        "imodpoly on raw": np.stack([
            s - polynomial.imodpoly(s.astype(float), poly_order=5)[0]
            for s in raw[test_idx]
        ]),
        "MLP output, background stripped": strip_background(
            methods["MLP (Peng et al.), 1 frame"][test_idx], fitter
        ),
    }
    for label, (run, n_frames) in LUMOS_RUNS.items():
        raman_methods[f"{label}: Raman head"] = raman_head(run, X, n_frames)[test_idx]

    print("\n\ntarget: bleached reference with background stripped")
    header = f"{'':<34}{'Pearson':>9}{'SAM':>8}{'SI-PSNR':>10}"
    print(header)
    print("-" * len(header))
    for name, pred in raman_methods.items():
        print(f"{name:<34}{pearson(pred, ref_peaks).mean():>9.4f}"
              f"{sam(pred, ref_peaks).mean():>8.2f}"
              f"{np.mean([scale_invariant_psnr(ref_peaks[i], pred[i]) for i in range(len(pred))]):>10.2f}")

    print("\nMLP is supervised on paired bleached references; LUMOS sees none.")


if __name__ == "__main__":
    main()
