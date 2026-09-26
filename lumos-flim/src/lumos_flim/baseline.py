"""Non-amortised fits of the same physics model: the baselines for the VAE.

    python -m lumos_flim.baseline --data data/hmsc.zarr --fit_irf --method mle

Every method fits the same forward model, parametrisation and Poisson
likelihood as the VAE; they differ only in how the per-pixel parameters are
tied together:

  mle       every pixel on its own
  global    one set of lifetimes for the whole image, fractions per pixel
  binned    each pixel fitted on the sum of its k x k neighbourhood
  tv        per-pixel fits plus an edge-preserving total-variation penalty
            between neighbours, its strength chosen on held-out photons
  eb        hierarchical empirical Bayes: a full-covariance Gaussian
            population prior over the per-pixel parameters, estimated from
            the image by EM with Laplace-approximated pixel posteriors

The per-pixel parameters are the same unconstrained vector the VAE decoder
emits: rate increments (softplus, cumulative sum, so lifetimes stay ordered),
photon-fraction logits including the background, and background-spectrum
logits for multichannel data.
"""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from lumos_flim.data import irf_t0_guess, neighbour_pairs, neighbourhood_sum, open_store
from lumos_flim.device import pick_device
from lumos_flim.physics import decay_histograms, expected_counts
from lumos_flim.predict import derived, gt_report, summarise
from lumos_flim.vae import _SIGMA_FLOOR, FlimVAE
from lumos_flim.vae_module import FlimModule, n_free_parameters, pearson_chi2, poisson_half_deviance


class PixelFitter:
    """Holds the counts, the per-pixel parameter matrix and the shared IRF.

    ``theta`` is [N, P] (or [1, F] rows shared for global lifetimes). The loss
    is separable over pixels apart from the shared IRF and any penalty, so
    one Adam optimiser on the whole matrix is one optimiser per pixel.
    """

    def __init__(self, counts, model: FlimVAE, device="auto", global_lifetimes=False):
        self.device = pick_device(device)
        self.model = model
        self.Fn = model.n_components
        self.C = counts.shape[1]
        self.global_lifetimes = global_lifetimes
        d = self.device
        self.x = torch.as_tensor(counts, dtype=torch.float32, device=d)
        self.totals = self.x.sum((1, 2)).clamp(min=1.0)
        N = len(self.x)
        self.t0 = model.irf_t0.detach().to(d).clone()
        self.sigma_raw = model.irf_sigma_raw.detach().to(d).clone()
        self.tail = getattr(model, "irf_tail", False)
        if self.tail:
            self.tail_logit = model.irf_tail_logit.detach().to(d).clone()
            self.tail_rate_raw = model.irf_tail_rate_raw.detach().to(d).clone()
        self.bases = model.bases.detach().to(d) if model.bases is not None else None
        rate_bias = model.decoder.head_rate.bias.detach().to(d)
        frac_bias = model.decoder.head_fraction.bias.detach().to(d)
        self.rates_raw = (rate_bias[None] if global_lifetimes else rate_bias.expand(N, -1)).clone()
        rest = [frac_bias.expand(N, -1)]
        if self.C > 1:
            rest.append(torch.zeros(N, self.C, device=d))
        self.rest = torch.cat(rest, 1).clone()

    # Parameters -------------------------------------------------------------
    @property
    def theta(self):
        """Per-pixel parameter matrix [N, P]."""
        if self.global_lifetimes:
            return self.rest
        return torch.cat([self.rates_raw, self.rest], 1)

    def set_theta(self, theta):
        if self.global_lifetimes:
            self.rest = theta.detach().clone()
        else:
            self.rates_raw = theta[:, :self.Fn].detach().clone()
            self.rest = theta[:, self.Fn:].detach().clone()

    def _split(self, theta):
        if self.global_lifetimes:
            rate_raw, rest = self.rates_raw, theta
        else:
            rate_raw, rest = theta[:, :self.Fn], theta[:, self.Fn:]
        frac_logits = rest[:, :self.Fn + 1]
        bg_logits = rest[:, self.Fn + 1:] if self.C > 1 else None
        return rate_raw, frac_logits, bg_logits

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw) + _SIGMA_FLOOR

    def forward(self, theta=None, x=None):
        """Rates, photon probabilities and expected counts for ``theta``."""
        theta = self.theta if theta is None else theta
        totals = self.totals if x is None else x.sum((1, 2)).clamp(min=1.0)
        rate_raw, frac_logits, bg_logits = self._split(theta)
        N = len(frac_logits)
        rates = (self.model.rate_min + torch.cumsum(F.softplus(rate_raw), -1)).expand(N, self.Fn)
        probs = F.softmax(frac_logits, -1)
        bg_spec = F.softmax(bg_logits, -1) if bg_logits is not None else None
        w = torch.sigmoid(self.tail_logit) if self.tail else None
        q = F.softplus(self.tail_rate_raw) + 0.1 if self.tail else None
        decays = decay_histograms(rates, self.t0, self.sigma, self.model.n_bins, self.model.bin_width,
                                  tail_weight=w, tail_rate=q)
        mu = expected_counts(totals, probs[:, :-1], probs[:, -1], decays, self.bases, bg_spec)
        return rates, probs, mu

    # Optimisation ------------------------------------------------------------
    def fit(self, steps=1500, lr=0.05, fit_irf=False, penalty=None):
        """Maximise the likelihood (plus ``-penalty(theta)``) with Adam."""
        rates_raw = self.rates_raw.requires_grad_()
        rest = self.rest.requires_grad_()
        params = [rates_raw, rest]
        if fit_irf:
            irf = [self.t0, self.sigma_raw] + ([self.tail_logit, self.tail_rate_raw] if self.tail else [])
            for p in irf:
                p.requires_grad_()
            params += irf
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            theta = self.theta
            _, _, mu = self.forward(theta)
            loss = poisson_half_deviance(self.x, mu).sum()
            if penalty is not None:
                loss = loss + penalty(theta)
            loss.backward()
            opt.step()
        for p in (self.rates_raw, self.rest, self.t0, self.sigma_raw) + (
                (self.tail_logit, self.tail_rate_raw) if self.tail else ()):
            p.requires_grad_(False)
        return self

    def results(self):
        """Per-pixel results as numpy, and the expected counts."""
        with torch.no_grad():
            rates, probs, mu = self.forward()
            chi2 = pearson_chi2(self.x, mu)
        rates, probs, mu, chi2 = (v.cpu() for v in (rates, probs, mu, chi2))
        out = {k: v.numpy() for k, v in derived(rates, probs[:, :-1]).items()}
        dof = self.C * self.model.n_bins - n_free_parameters(self.Fn, self.C)
        out.update(fractions=probs[:, :-1].numpy(), background=probs[:, -1].numpy(),
                   chi2r=(chi2 / dof).numpy(), irf_t0=self.t0.item(), irf_sigma=self.sigma.item())
        return out, mu.numpy()

    def copy_irf_from(self, other):
        self.t0 = other.t0.detach().clone()
        self.sigma_raw = other.sigma_raw.detach().clone()
        if self.tail and other.tail:
            self.tail_logit = other.tail_logit.detach().clone()
            self.tail_rate_raw = other.tail_rate_raw.detach().clone()
        return self


def _heldout_nll(mu, held):
    """Held-out photons' negative log-likelihood under the fitted shapes."""
    p = mu / mu.sum((1, 2), keepdim=True)
    return float(-(held * torch.log(p.clamp(min=1e-12))).sum() / held.sum().clamp(min=1))


def fit_total_variation(counts, model, pairs, irf_from, lams=(1.0, 3.0, 10.0, 30.0, 100.0),
                        steps=1500, select_steps=600, keep=0.8, delta=1e-2, seed=0,
                        device="auto"):
    """Per-pixel fits with a smoothed total-variation penalty between neighbours.

    penalty = lam * sum over neighbour pairs of sqrt(|theta_i - theta_j|^2 + delta^2)

    TV is used rather than a quadratic penalty because it keeps sharp edges
    between regions. The strength is chosen without ground truth: each pixel's
    photons are split at random (``keep`` / 1 - ``keep``), each candidate is
    fitted on the first part and scored on how well it predicts the second,
    and the winner is refitted on everything with its strength scaled up by
    1 / ``keep`` (the likelihood grows with the photon count, the penalty
    does not). The IRF is taken from ``irf_from``.
    """
    rng = np.random.default_rng(seed)
    part = rng.binomial(counts.astype(np.int64), keep).astype(np.float32)
    held = torch.as_tensor(counts - part, dtype=torch.float32)
    pairs_t = torch.as_tensor(pairs)

    def penalty_for(lam, fitter):
        i, j = pairs_t[:, 0].to(fitter.device), pairs_t[:, 1].to(fitter.device)

        def penalty(theta):
            d = theta[i] - theta[j]
            return lam * torch.sqrt((d ** 2).sum(-1) + delta ** 2).sum()
        return penalty

    scores = {}
    for lam in lams:
        f = PixelFitter(part, model, device).copy_irf_from(irf_from)
        f.fit(select_steps, penalty=penalty_for(lam, f))
        with torch.no_grad():
            scores[lam] = _heldout_nll(f.forward()[2].cpu(), held)
    best = min(scores, key=scores.get)
    f = PixelFitter(counts, model, device).copy_irf_from(irf_from)
    f.fit(steps, penalty=penalty_for(best / keep, f))
    out, mu = f.results()
    out["tv_lambda"] = best / keep
    out["tv_scores"] = scores
    return out, mu


def _laplace_covariance(fitter, penalty, eps=5e-3):
    """Per-pixel posterior covariance from the Hessian of the MAP objective.

    The objective is separable over pixels, so perturbing column j of every
    pixel at once and differencing the gradients gives column j of all the
    per-pixel Hessians together: 2P gradient evaluations in total.
    """
    theta = fitter.theta.detach()
    N, P = theta.shape

    def grad(th):
        th = th.clone().requires_grad_()
        _, _, mu = fitter.forward(th)
        loss = poisson_half_deviance(fitter.x, mu).sum() + penalty(th)
        return torch.autograd.grad(loss, th)[0]

    H = torch.zeros(N, P, P, device=theta.device)
    for j in range(P):
        e = torch.zeros(P, device=theta.device)
        e[j] = eps
        H[:, :, j] = (grad(theta + e) - grad(theta - e)) / (2 * eps)
    H = 0.5 * (H + H.transpose(1, 2))
    # Posterior curvature must be positive definite; clamp any direction the
    # finite differences left flat or negative.
    evals, evecs = torch.linalg.eigh(H.cpu())
    evals = evals.clamp(min=1e-4)
    return (evecs @ torch.diag_embed(1.0 / evals) @ evecs.transpose(1, 2)).to(theta.device)


def fit_empirical_bayes(counts, model, irf_from, rounds=8, steps=300, device="auto",
                        init_theta=None):
    """Hierarchical model with a Gaussian population prior, fitted by EM.

    theta_p ~ N(m, S) for every pixel, with m and S estimated from the image:

      E-step  MAP of each theta_p under the current prior, with a Laplace
              approximation of its posterior covariance Sigma_p;
      M-step  m = mean(theta_p), S = mean((theta_p - m)(theta_p - m)^T + Sigma_p).

    This learns a population prior, as the VAE does, but without a network
    and with a Gaussian in parameter space in place of the decoder's learned
    map from a Gaussian latent. The IRF is taken from ``irf_from``.
    """
    f = PixelFitter(counts, model, device).copy_irf_from(irf_from)
    if init_theta is not None:
        f.set_theta(init_theta)
    theta = f.theta.detach()
    P = theta.shape[1]
    ridge = 1e-3 * torch.eye(P, device=theta.device)
    m = theta.mean(0)
    S = torch.cov(theta.T) + ridge
    for _ in range(rounds):
        S_inv = torch.linalg.inv(S)

        def penalty(th, m=m, S_inv=S_inv):
            d = th - m
            return 0.5 * ((d @ S_inv) * d).sum()

        f.fit(steps, penalty=penalty)
        theta = f.theta.detach()
        cov = _laplace_covariance(f, penalty)
        m = theta.mean(0)
        d = theta - m
        S = (d.T @ d) / len(d) + cov.mean(0) + ridge
    out, mu = f.results()
    out["eb_prior_mean"] = m.cpu().numpy()
    out["eb_prior_cov"] = S.cpu().numpy()
    return out, mu


def fit_binned(counts, model, image, y, x, shapes, irf_from, k=3, steps=1500, device="auto"):
    """Fit each pixel on the sum of its k x k neighbourhood, the usual FLIM remedy.

    Parameters are assigned to the centre pixel; its expected histogram is the
    binned fit's shape scaled to the centre pixel's own total.
    """
    binned = neighbourhood_sum(counts, image, y, x, shapes, k, include_centre=True).astype(np.float32)
    f = PixelFitter(binned, model, device).copy_irf_from(irf_from).fit(steps)
    out, mu = f.results()
    shape = mu / np.maximum(mu.sum((1, 2), keepdims=True), 1e-12)
    return out, shape * counts.sum((1, 2), keepdims=True)


def fit_pixels(counts, model: FlimVAE, steps=1500, lr=0.05, fit_irf=False,
               global_lifetimes=False, device="auto"):
    """Per-pixel MLE (or global analysis) with the IRF optionally fitted.

    Kept as the simple entry point used by the scripts. Returns
    (results, expected counts).
    """
    f = PixelFitter(counts, model, device, global_lifetimes=global_lifetimes)
    f.fit(steps, lr, fit_irf=fit_irf)
    return f.results()


def plot_fits(counts, expected, bin_width, path, n=4, seed=0):
    """Data, model and normalised residuals for a few random pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(seed).choice(len(counts), n, replace=False)
    t = (np.arange(counts.shape[-1]) + 0.5) * bin_width
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for j, i in enumerate(idx):
        h, mu = counts[i].sum(0), expected[i].sum(0)
        axes[0, j].semilogy(t, np.maximum(h, 0.5), ".", color="0.3", label="data")
        axes[0, j].semilogy(t, mu, "-", color="C0", label="fit")
        axes[0, j].set_title(f"pixel {i}, {h.sum():.0f} photons", fontsize=9)
        axes[1, j].bar(t, (h - mu) / np.sqrt(np.maximum(mu, 1e-9)), width=bin_width, color="C1")
        axes[1, j].axhline(0, color="0.5", lw=0.8)
        axes[1, j].set_xlabel("delay (ns)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("counts")
    axes[1, 0].set_ylabel("residual (sigma)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_method(method, counts, model, ds, meta, irf_from=None, device="auto", steps=1500):
    """One baseline by name. ``irf_from`` is a fitted PixelFitter whose IRF the
    structured methods reuse; mle and global fit their own."""
    if method == "mle":
        return fit_pixels(counts, model, steps, fit_irf=True, device=device)
    if method == "global":
        return fit_pixels(counts, model, steps, fit_irf=True, global_lifetimes=True, device=device)
    if irf_from is None:
        irf_from = PixelFitter(counts, model, device).fit(steps, fit_irf=True)
    img, y, x = ds["image"].values, ds["y"].values, ds["x"].values
    if method == "binned":
        return fit_binned(counts, model, img, y, x, meta["image_shapes"], irf_from, device=device)
    if method == "tv":
        return fit_total_variation(counts, model, neighbour_pairs(img, y, x), irf_from, device=device)
    if method == "eb":
        return fit_empirical_bayes(counts, model, irf_from, device=device, init_theta=irf_from.theta)
    raise ValueError(f"unknown method {method}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--method", default="mle", choices=["mle", "global", "binned", "tv", "eb"])
    p.add_argument("--checkpoint", default="", help="take IRF and spectra from a trained model")
    p.add_argument("--irf", default="", help="or from a calibration JSON (single channel only)")
    p.add_argument("--irf_fit", default="fixed", choices=["fixed", "free"])
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--fit_irf", action="store_true",
                   help="fit the IRF from the data, starting from its rising edge")
    p.add_argument("--irf_tail", action="store_true", help="give the IRF a fitted exponential tail")
    p.add_argument("--global_lifetimes", action="store_true", help="same as --method global")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    if a.global_lifetimes:
        a.method = "global"

    ds, meta = open_store(a.data)
    counts = ds["counts"].values
    if a.checkpoint:
        model = FlimModule.load_from_checkpoint(a.checkpoint, map_location="cpu").model
    elif counts.shape[1] > 1:
        raise SystemExit("multichannel data needs --checkpoint for the component spectra")
    elif a.irf:
        with open(a.irf) as f:
            cal = json.load(f)[a.irf_fit]
        model = FlimVAE(n_bins=counts.shape[2], bin_width=meta["bin_width_ns"],
                        n_components=a.n_components, irf_t0=cal["irf_t0"], irf_sigma=cal["irf_sigma"])
    elif a.fit_irf:
        model = FlimVAE(n_bins=counts.shape[2], bin_width=meta["bin_width_ns"],
                        n_components=a.n_components, irf_tail=a.irf_tail,
                        irf_t0=irf_t0_guess(counts, meta["bin_width_ns"]), irf_sigma=0.1)
    else:
        raise SystemExit("give --checkpoint, --irf or --fit_irf")

    if a.method == "mle" and not a.fit_irf:
        result, expected = PixelFitter(counts, model, a.device).fit(a.steps).results()
    else:
        result, expected = run_method(a.method, counts, model, ds, meta, device=a.device, steps=a.steps)
    print(f"IRF: t0={result.pop('irf_t0'):.3f} ns sigma={result.pop('irf_sigma'):.3f} ns")
    extras = {k: result.pop(k) for k in list(result) if k.startswith(("tv_", "eb_"))}
    if "tv_lambda" in extras:
        print(f"TV strength {extras['tv_lambda']:.3g} (held-out scores {extras['tv_scores']})")
    gt_report(result, ds)
    summarise(result, ds, meta, a.out or None)
    if a.out:
        np.savez_compressed(f"{a.out}.npz", image=ds["image"].values, y=ds["y"].values,
                            x=ds["x"].values, split=ds["split"].values, **result)
        plot_fits(counts, expected, meta["bin_width_ns"], f"{a.out}_fits.png")
        print(f"wrote {a.out}.npz and {a.out}_fits.png")


if __name__ == "__main__":
    main()
