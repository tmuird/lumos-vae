"""Does shrinking the encoder cost anything beyond reconstruction?

Reconstruction quality is the easy thing to preserve. The properties that matter
downstream are whether the recovered Raman still separates cell types without
labels, and whether the model still behaves at shorter input windows.

ARI and AMI are chance-corrected, so 0 is a random assignment. The raw-input rows
bound nothing here: clustering a representation can beat clustering the raw data,
because the decomposition removes fluorescence and intensity variation that
dominate Euclidean distance. That is the point of the model, not a paradox.
"""

import numpy as np
import torch
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from lumos.metrics import pearson_r, scale_invariant_psnr
from lumos.train import load_checkpoint

ZARR = "/home/tom/Developments/Raman/oracle/data/processed/glasgow/glasgow_16_trimmed_smoothed.zarr"
CKPT_DIR = "/home/tom/Developments/lumos/src/lumos/checkpoints"
SEED = 0

RUNS = {
    "64,128,256,512": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260816_162145",
    "32,64,128,256": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260825_211022",
    "16,32,64,128": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260825_211027",
    "8,16,32,64": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260825_212328",
    "4,8,16,32": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260825_212333",
}


def main():
    ds = xr.open_zarr(ZARR)
    split = ds["split"].values
    fd = float(ds.attrs.get("frame_duration_s", 0.1))
    wn = ds.coords["wavenumber"].values
    mask = (wn >= 400) & (wn <= 1800)

    X_all = ds["time_series"].transpose("sample", "time", "wavenumber").values
    labels_all = np.array([str(v) for v in ds["labels"].values])
    test = np.where(split == "test")[0]
    X_test, gt_test = X_all[test], ds["gt_raman_smoothed"].values[test] / fd
    k = len(set(labels_all))

    # Clustering uses every spectrum, since it needs no target.
    raw_ari = adjusted_rand_score(
        labels_all,
        KMeans(k, n_init=10, random_state=SEED).fit_predict(
            StandardScaler().fit_transform(X_all[:, 0, :])
        ),
    )
    print(f"{len(X_all)} spectra, {k} classes; raw first frame ARI {raw_ari:.3f}\n")

    header = (f"{'conv channels':<18}{'params':>9}{'Pearson':>9}{'SI-PSNR':>9}"
              f"{'ARI':>7}{'AMI':>7}{'r@T=8':>8}{'r@T=1':>8}")
    print(header)
    print("-" * len(header))

    for tag, run in RUNS.items():
        model, _ = load_checkpoint(run, CKPT_DIR)
        vae = model.model
        std = float(vae.dataset_std)
        n_params = sum(p.numel() for p in vae.parameters())

        with torch.no_grad():
            x = torch.from_numpy(X_test[:, :16, :]).float().transpose(1, 2) / std
            _, _, _, _, _, raman, _ = vae(x, sample=False)
            short = {
                t: vae(x[:, :, :t], sample=False)[5].cpu().numpy()[:, mask]
                for t in (8, 1)
            }
        pred, ref = raman.cpu().numpy()[:, mask], gt_test[:, mask]
        r = np.mean([pearson_r(ref[i], pred[i]) for i in range(len(ref))])
        si = np.mean([scale_invariant_psnr(ref[i], pred[i]) for i in range(len(ref))])
        r_short = {
            t: np.mean([pearson_r(ref[i], v[i]) for i in range(len(ref))])
            for t, v in short.items()
        }

        with torch.no_grad():
            xa = torch.from_numpy(X_all[:, :16, :]).float().transpose(1, 2) / std
            ram_all = torch.cat([
                vae(xa[i:i + 32], sample=False)[5] for i in range(0, len(xa), 32)
            ]).cpu().numpy()
        Z = StandardScaler().fit_transform(
            ram_all / (np.linalg.norm(ram_all, axis=1, keepdims=True) + 1e-12)
        )
        clusters = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
        print(f"{tag:<18}{n_params / 1e6:>8.2f}M{r:>9.3f}{si:>9.2f}"
              f"{adjusted_rand_score(labels_all, clusters):>7.3f}"
              f"{adjusted_mutual_info_score(labels_all, clusters):>7.3f}"
              f"{r_short[8]:>8.3f}{r_short[1]:>8.3f}")


if __name__ == "__main__":
    main()
