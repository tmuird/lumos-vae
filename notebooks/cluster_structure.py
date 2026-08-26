"""Do the learned parameters carry class structure a downstream user could use?

Scored three ways, because each answers a different question.

  ARI / AMI of k-means   purely unsupervised: cluster the representation without
                         looking at labels, then ask how well the clusters line
                         up with the true classes. Both are chance-corrected, so
                         0 means no better than random assignment.
  LDA accuracy           supervised ceiling: how separable are the classes if
                         you are allowed to fit a boundary. Cross-validated,
                         because in-sample LDA on this many channels separates
                         random labels at 0.90.
  permuted-label null    the same pipeline with labels shuffled, which is the
                         floor any real result has to clear.

The raw input rows matter: no representation derived from a spectrum can carry
more class information than the spectrum itself, so they bound everything below.
"""

import numpy as np
import torch
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lumos.train import load_checkpoint

ZARR = "/home/tom/Developments/Raman/oracle/data/processed/glasgow/glasgow_16_trimmed_smoothed.zarr"
CKPT_DIR = "/home/tom/Developments/lumos/src/lumos/checkpoints"
RUNS = {
    "fixed T": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260816_162145",
    "variable T": "glasgow_16_trimmed_smoothed_mog_t16_z64_h128_lr0.0005_20260816_162150",
}
SEED = 0


def representations(run, X):
    """Per-sample latent, abundances, rates, fluorescence and recovered Raman."""
    model, _ = load_checkpoint(run, CKPT_DIR)
    vae = model.model
    n_times, std = vae.n_times_train, float(vae.dataset_std)
    rates, abundances, raman, mu = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            x = torch.from_numpy(X[i:i + 16, :n_times, :]).float().transpose(1, 2) / std
            _, m, _, lam, ab, ram, _ = vae(x, sample=False)
            rates.append(lam.numpy())
            abundances.append(ab.numpy())
            raman.append(ram.numpy())
            mu.append(m.numpy())
    rates, abundances, raman, mu = (
        np.concatenate(v) for v in (rates, abundances, raman, mu)
    )
    bases = vae.bases.detach().numpy()
    fluorescence = (abundances[:, :, None] * bases[None]).sum(1)

    def unit(v):
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    return {
        "latent mu": mu,
        "abundances": abundances / (abundances.sum(1, keepdims=True) + 1e-12),
        "log decay rates": np.log(rates),
        "fluorescence t=0": unit(fluorescence),
        "recovered Raman": unit(raman),
    }


def score(Z, labels, k, rng):
    """Unsupervised agreement, supervised accuracy, and a permuted-label null."""
    Z = StandardScaler().fit_transform(Z)
    clusters = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    lda = make_pipeline(LinearDiscriminantAnalysis())
    acc = cross_val_score(lda, Z, labels, cv=cv).mean()
    null = np.mean([
        cross_val_score(lda, Z, rng.permutation(labels), cv=cv).mean()
        for _ in range(3)
    ])
    return (adjusted_rand_score(labels, clusters),
            adjusted_mutual_info_score(labels, clusters),
            acc, null)


def main():
    rng = np.random.default_rng(SEED)
    ds = xr.open_zarr(ZARR)
    X = ds["time_series"].transpose("sample", "time", "wavenumber").values
    labels = np.array([str(v) for v in ds["labels"].values])
    k = len(set(labels))
    majority = max(np.mean(labels == c) for c in set(labels))
    print(f"{len(X)} spectra, {k} classes: {sorted(set(labels))}")
    print(f"majority-class baseline {majority:.3f}\n")

    header = (f"{'':<34}{'ARI':>8}{'AMI':>8}{'LDA':>8}{'LDA null':>10}")
    print(header)
    print("-" * len(header))

    print("raw input (bounds everything below)")
    for name, Z in [("  first frame", X[:, 0, :]),
                    ("  all 16 frames", X[:, :16, :].reshape(len(X), -1))]:
        ari, ami, acc, null = score(Z, labels, k, rng)
        print(f"{name:<34}{ari:>8.3f}{ami:>8.3f}{acc:>8.3f}{null:>10.3f}")

    for tag, run in RUNS.items():
        print(f"\n{tag}")
        for name, Z in representations(run, X).items():
            ari, ami, acc, null = score(Z, labels, k, rng)
            print(f"  {name:<32}{ari:>8.3f}{ami:>8.3f}{acc:>8.3f}{null:>10.3f}")


if __name__ == "__main__":
    main()
