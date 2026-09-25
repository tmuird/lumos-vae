"""Synthetic TCSPC images with known lifetimes, simulated photon by photon.

Photons are drawn as pulse jitter + exponential delay, wrapped onto the laser
period, and then histogrammed. This does not reuse the analytical model, so a
good fit is evidence that the model is right rather than that it agrees with
itself.

    python -m lumos_flim.synthetic --out data/synthetic.zarr
"""

import argparse

import numpy as np

from lumos_flim.data import assign_splits, write_store


def smooth_field(rng, shape, n_blobs=6, lo=0.0, hi=1.0):
    """Random smooth map scaled to [lo, hi]."""
    yy, xx = np.mgrid[: shape[0], : shape[1]] / max(shape)
    field = np.zeros(shape)
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, 1, 2)
        w = rng.uniform(0.08, 0.3)
        field += rng.uniform(-1, 1) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * w**2))
    field = (field - field.min()) / (np.ptp(field) + 1e-12)
    return lo + (hi - lo) * field


def simulate(
    size=96,
    n_bins=56,
    period=12.483,
    photons=1000,
    tau_long=(2.0, 3.5),
    tau_short=(0.3, 0.6),
    alpha_long=(0.15, 0.6),
    background=(0.01, 0.08),
    irf_t0=1.0,
    irf_sigma=0.12,
    n_channels=1,
    seed=0,
):
    rng = np.random.default_rng(seed)
    shape = (size, size)
    n_pix = size * size

    tau = np.stack([
        smooth_field(rng, shape, lo=tau_long[0], hi=tau_long[1]).ravel(),
        smooth_field(rng, shape, lo=tau_short[0], hi=tau_short[1]).ravel(),
    ], axis=1)  # [P, 2], longest first
    alpha = smooth_field(rng, shape, lo=alpha_long[0], hi=alpha_long[1]).ravel()
    amp = np.stack([alpha, 1 - alpha], axis=1)
    photon_frac = amp * tau / (amp * tau).sum(1, keepdims=True)
    bg = smooth_field(rng, shape, lo=background[0], hi=background[1]).ravel()
    probs = np.concatenate([photon_frac * (1 - bg[:, None]), bg[:, None]], axis=1)

    intensity = smooth_field(rng, shape, lo=0.2, hi=1.8).ravel()
    totals = rng.poisson(photons * intensity)
    per_source = rng.multinomial(totals, probs)  # [P, 3]

    bin_width = period / n_bins
    if n_channels > 1:
        pos = np.linspace(0, 1, n_channels)
        bases = np.stack([np.exp(-0.5 * ((pos - c) / 0.18) ** 2) for c in (0.35, 0.65)])
        bases /= bases.sum(1, keepdims=True)
        bg_spec = np.full(n_channels, 1.0 / n_channels)
    else:
        bases, bg_spec = np.ones((2, 1)), np.ones(1)

    counts = np.zeros(n_pix * n_channels * n_bins, dtype=np.int64)
    for src in range(3):
        pix = np.repeat(np.arange(n_pix), per_source[:, src])
        if src < 2:
            t = rng.normal(irf_t0, irf_sigma, pix.size) + rng.exponential(tau[pix, src])
            spec = bases[src]
        else:
            t = rng.uniform(0, period, pix.size)
            spec = bg_spec
        b = np.floor(np.mod(t, period) / bin_width).astype(np.int64).clip(0, n_bins - 1)
        ch = rng.choice(n_channels, size=pix.size, p=spec)
        np.add.at(counts, (pix * n_channels + ch) * n_bins + b, 1)
    counts = counts.reshape(n_pix, n_channels, n_bins)

    yy, xx = np.divmod(np.arange(n_pix), size)
    gt = {
        "gt_tau": tau,
        "gt_fraction": probs[:, :2],
        "gt_background": bg,
        "gt_alpha": amp,
    }
    return counts, bin_width, yy, xx, gt, bases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--photons", type=float, default=1000, help="median photons per pixel")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--irf_t0", type=float, default=1.0)
    p.add_argument("--irf_sigma", type=float, default=0.12)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    counts, bin_width, yy, xx, gt, bases = simulate(
        size=a.size, photons=a.photons, n_channels=a.channels,
        irf_t0=a.irf_t0, irf_sigma=a.irf_sigma, seed=a.seed,
    )
    split = assign_splits(len(counts), a.val_frac, a.test_frac, a.seed)
    write_store(
        a.out, counts, bin_width, np.zeros(len(counts)), yy, xx,
        ["synthetic"], [(a.size, a.size)], split, gt=gt,
        attrs=dict(source="synthetic", gt_irf_t0=a.irf_t0, gt_irf_sigma=a.irf_sigma,
                   gt_bases=bases.tolist()),
    )
    print(f"wrote {a.out}: {counts.shape}, median {np.median(counts.sum((1, 2))):.0f} photons/pixel")


if __name__ == "__main__":
    main()
