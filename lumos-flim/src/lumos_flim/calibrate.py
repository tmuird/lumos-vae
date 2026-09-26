"""Fit the instrument response to a reference dye of known lifetime.

    python -m lumos_flim.calibrate Fluorescein_hMSC.tif --tau 4.2 --out irf_hmsc.json

The reference is summed over all pixels into one high-count histogram and
fitted as a single exponential plus background with a Gaussian IRF. Two fits
are reported: one with the lifetime fixed at ``--tau`` (whose IRF is written
out) and one with the lifetime free. If the free fit does not land near the
known lifetime, or the reduced chi-square is far above one, the Gaussian IRF
is a poor description of the instrument and the lifetimes fitted with it will
be biased accordingly.

Runs on the CPU in float64: it is one histogram, fitted in seconds, and MPS
has no float64 support.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from lumos_flim.data import irf_t0_guess, open_store, read_imspector_tiff
from lumos_flim.physics import decay_histograms


def fit_reference(hist, bin_width, tau=None, t0_init=None, steps=300):
    """Maximum-likelihood fit of one exponential, background and a Gaussian IRF."""
    h = torch.as_tensor(hist, dtype=torch.float64)
    n = h.shape[-1]
    total = h.sum()
    if t0_init is None:
        t0_init = irf_t0_guess(hist, bin_width)

    best = None
    for dt in (-1.0, -0.5, 0.0, 0.5):  # a few starts along the rising edge
        t0 = torch.tensor(t0_init + dt * bin_width, dtype=torch.float64, requires_grad=True)
        s_raw = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
        log_tau = torch.tensor(np.log(tau or 2.0), dtype=torch.float64, requires_grad=tau is None)
        bg_logit = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
        params = [t0, s_raw, bg_logit] + ([log_tau] if tau is None else [])
        opt = torch.optim.LBFGS(params, max_iter=steps, line_search_fn="strong_wolfe")

        def model():
            sigma = F.softplus(s_raw) + 1e-3
            p = decay_histograms((-log_tau).exp().view(1), t0, sigma, n, bin_width)[0]
            bg = torch.sigmoid(bg_logit)
            return total * ((1 - bg) * p + bg / n)

        def closure():
            opt.zero_grad()
            mu = model().clamp(min=1e-12)
            loss = (mu - h * mu.log()).sum() / total
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            mu = model()
            nll = float((mu - h * mu.clamp(min=1e-12).log()).sum())
            dof = n - len(params)
            result = dict(
                irf_t0=float(t0), irf_sigma=float(F.softplus(s_raw) + 1e-3),
                tau=float(log_tau.exp()), background=float(torch.sigmoid(bg_logit)),
                chi2r=float(((h - mu) ** 2 / mu.clamp(min=1e-12)).sum() / dof),
                photons=float(total),
            )
        if best is None or nll < best[0]:
            best = (nll, result, mu.numpy())
    return best[1], best[2]


def load_hist(path, min_counts=0.0):
    if str(path).rstrip("/").endswith(".zarr"):
        ds, meta = open_store(path)
        counts = ds["counts"].values.sum(1)  # [sample, time]
        bin_width = meta["bin_width_ns"]
    else:
        counts, bin_width = read_imspector_tiff(path)
        counts = counts.reshape(counts.shape[0], -1).T
    keep = counts.sum(1) >= min_counts
    return counts[keep].sum(0).astype(np.float64), float(bin_width)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", help="ImSpector TIFF or pixel store of the reference dye")
    p.add_argument("--tau", type=float, required=True, help="known reference lifetime in ns")
    p.add_argument("--out", default="")
    p.add_argument("--min_counts", type=float, default=0.0)
    a = p.parse_args(argv)

    hist, bin_width = load_hist(a.reference, a.min_counts)
    fixed, _ = fit_reference(hist, bin_width, tau=a.tau)
    free, _ = fit_reference(hist, bin_width, tau=None)
    print(f"{hist.sum():.3g} photons, {len(hist)} bins of {bin_width:.4f} ns")
    for name, r in (("fixed tau", fixed), ("free tau ", free)):
        print(f"  {name}: tau={r['tau']:.3f} ns  t0={r['irf_t0']:.3f} ns  "
              f"sigma={r['irf_sigma']:.3f} ns  background={r['background']:.4f}  "
              f"chi2r={r['chi2r']:.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"reference": str(a.reference), "reference_tau": a.tau,
                       "bin_width_ns": bin_width, "fixed": fixed, "free": free}, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
