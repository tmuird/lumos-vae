# LUMOS-FLIM

The LUMOS physics-informed VAE, adapted from photobleaching Raman to
time-correlated single photon counting (TCSPC) fluorescence lifetime imaging.
As in LUMOS, the network does not reconstruct the signal. It predicts the
parameters of an analytical forward model (per-pixel lifetimes, photon
fractions and a constant background), and the model produces the histogram.
Training is unsupervised and uses the exact Poisson likelihood of photon
counting.

A full walk-through of the physics, every component and how each differs from
lumos-vae is in [docs/how_it_works.md](docs/how_it_works.md).

## Why FLIM

Of the candidates (FLIM, NMR T2 relaxometry, DOSY), FLIM is the closest match
to LUMOS and the easiest to get real data for:

| LUMOS (photobleaching) | FLIM (TCSPC) |
|---|---|
| static Raman spectrum, constant in time | uncorrelated background (dark counts, ambient light), constant in delay |
| fluorophores bleaching at per-sample rates | fluorophores decaying at per-pixel lifetimes |
| CCD integrates each frame: `(1 - e^(-λT))/λ` | TCSPC integrates each delay bin, the same factor |
| decoder emits effective amplitudes | decoder emits photon fractions |
| instrument blur, a learnable Gaussian | instrument response (IRF), a learnable Gaussian |
| global fluorophore bases `B_i(ν)` | global emission spectra per detection channel (spectral FLIM) |

It also adds physics LUMOS did not need: excitation is periodic, so decay left
over from earlier laser pulses wraps into the window. That shows up in real
data as counts before the rising edge that match the tail. The noise model is
exact rather than learned, because every count is a photon.

NMR T2 would also work, but a CPMG decay has no second axis (the spectral
dimension that makes LUMOS a matrix problem). Clean multi-echo data also means
stimulated-echo corrections (EPG), and public data for it is harder to get
hold of.

## The physics

For pixel `p` the expected count in delay bin `n` (width `D`, period `P = N D`) is

    h_n = n_p * [ f_bg / N + sum_i f_i * p_i(n) ]

- `n_p` is the pixel's photon total. The model is conditioned on it, so the
  likelihood is multinomial and the overall scale is not a free parameter.
- `f_i` and `f_bg` are the photon fractions of each lifetime component and of
  the background. They sum to one.
- `p_i(n)` is the probability that a photon of lifetime `1/k_i` lands in bin
  `n`. The photon arrives `t0 + N(0, σ²) + Exp(k_i)` after its pulse, summed
  over all earlier pulses and integrated over the bin. With a Gaussian IRF this
  has a closed form (a periodic, bin-integrated exponentially modified
  Gaussian), evaluated exactly and stably in `physics.py`. It is not
  discretised and convolved on a grid.

Lifetimes are built as a cumulative sum of rates, so component 0 always has
the longest lifetime and labels cannot swap between pixels. The slowest rate
is bounded below by one period, because a much longer lifetime is flat and
indistinguishable from background (the same degeneracy LUMOS has between slow
bleaching and the static Raman). Photon fractions convert to the amplitude
fractions usually quoted (free and bound NADH, say) via `a_i ∝ f_i k_i`.

The loss is `sum_n h_n log(h_n / μ_n)` per pixel plus the KL term. That is the
exact negative log-likelihood less its saturated value, and half the Poisson
deviance. A reduced chi-square near 1 therefore means the physics and the noise
model both fit, which is a stronger check than LUMOS could make with its learned
noise.

## Data

Real data comes from the FLUTE dataset (Gottlieb et al. 2023, Zenodo
[8046636](https://doi.org/10.5281/zenodo.8046636), CC BY 4.0), mirrored in
[phasorpy-data](https://github.com/phasorpy/phasorpy-data/tree/main/zenodo_8046636).
It has NADH autofluorescence of human mesenchymal stem cells, control and
rotenone-treated, plus a zebrafish embryo, each with a fluorescein reference.
It is ImSpector OME-TIFF, 56 bins over one 12.48 ns period (80.11 MHz).

```bash
pip install -e ".[test]"
git clone --depth 1 https://github.com/phasorpy/phasorpy-data   # ~2 GB, only zenodo_8046636 is needed
D=phasorpy-data/zenodo_8046636

# pixels -> store (2x2 binning, drop pixels under 500 photons)
python -m lumos_flim.prepare "$D/hMSC control.tif" "$D/hMSC_rotenone.tif" --out data/hmsc.zarr --bin 2 --min_counts 500

# IRF from the fluorescein reference
python -m lumos_flim.calibrate "$D/Fluorescein_hMSC.tif" --tau 4.2 --out calibration/irf_hmsc.json

python -m lumos_flim.train --data data/hmsc.zarr --irf calibration/irf_hmsc.json --irf_fit free
python -m lumos_flim.predict checkpoints/<run>/best.ckpt --data data/hmsc.zarr --out results/hmsc --n_samples 10
```

Training runs on PyTorch Lightning and picks the hardware itself: CUDA if
present, then Apple MPS, then CPU (`--accelerator auto`; pass `cuda`, `mps` or
`cpu` to choose). On a GPU the whole pixel store is copied to the device once
(`--preload`, on by default) and batches are gathered with a single indexing
call. `predict.py` and `baseline.py` take `--device` in the same way. The
normal CDF, its log and the scaled complementary error function are built
from elementary operations (Numerical Recipes' erfc fit, relative error below
1.2e-7), because MPS has no kernels for `torch.special.erfcx` or `log_ndtr`.
`tests/test_physics.py::test_runs_on_device` checks the physics and a training
step on every accelerator present against the CPU result. Calibration stays on
the CPU in float64. The GPU paths were written for CUDA and MPS but only the
CPU path has been run so far; run the tests on the target machine first.

Synthetic data with ground truth is simulated photon by photon. The generator
does not use the analytical model, so a fit is not the model agreeing with
itself:

```bash
python -m lumos_flim.synthetic --out data/synthetic.zarr --photons 1000
python -m lumos_flim.synthetic --out data/synthetic_spec.zarr --channels 8   # spectral FLIM
```

`baseline.py` fits every pixel independently by maximum likelihood with the
same physics, parametrisation and IRF, so the difference from the VAE is only
in how the per-pixel parameters are estimated.

## Results so far

These come from short CPU runs (60 epochs, default settings, one seed). They
show the approach works. They are not tuned numbers.

**Physics check.** The analytical histograms match Monte-Carlo photon
simulations (reduced chi-square about 1) for lifetimes from 1 ps to 30 ns,
including an IRF that straddles the end of the window. See
`tests/test_physics.py`.

**Direct fit, no network.** Before any amortisation, `baseline.py --fit_irf`
fits the physics model straight to the data. Per-pixel lifetimes and fractions
and one shared IRF are optimised jointly by maximum likelihood, starting from
the rising edge of the data:

```bash
python -m lumos_flim.baseline --data data/hmsc.zarr --fit_irf --out results/hmsc_direct
```

On synthetic data this recovers the IRF from scratch (t0 = 1.006 ns and
σ = 0.122 ns, against the true 1.0 and 0.12), with reduced chi-square 0.97.
On hMSC it reaches reduced chi-square 0.99, and its IRF (0.427 / 0.109 ns)
agrees with the one the VAE learned (0.423 / 0.107 ns). The residuals show no
structure, so the forward model alone describes the data before any network
is involved. Its lifetimes are in the hMSC table below as "MLE". Random pixels,
data against fit, with normalised residuals:

![direct fits, synthetic](docs/synthetic_direct_fits.png)
![direct fits, hMSC](docs/hmsc_direct_fits.png)

**Synthetic ground truth.** The VAE learns the IRF from scratch to
t0 = 1.00 ns and σ = 0.120 ns (the true values) with reduced chi-square 1.0.
Accuracy against the per-pixel fit and global analysis is in the evaluation
below. With 8 spectral channels (`--channels 8`), the two emission spectra
are recovered with the right shapes and lifetimes to about 8%.

**Fluorescein references.** A free single-exponential fit gives 3.99 ns for
the embryo reference (literature 4.0-4.2 ns). The hMSC reference gives
3.59 ns. Its raw tail slope is about 3.8 ns before the wrap-around correction,
so the dye there genuinely decays faster than 4.2 ns. Its residuals also
alternate by ±50σ bin to bin, which looks like timing nonlinearity in the
electronics. Holding τ at 4.2 ns for that reference therefore gives the wrong
IRF, which is why the hMSC run uses `--irf_fit free` and lets the model
refine the IRF.

**hMSC NADH, 2x2 binned, pixels over 500 photons** (medians per image):

| | τ bound (ns) | τ free (ns) | α free | amplitude-weighted τ (ns) | reduced χ² |
|---|---|---|---|---|---|
| control, VAE | 3.17 | 0.51 | 0.79 | 1.07 | 1.04 |
| control, MLE | 3.12 | 0.48 | 0.79 | 1.08 | 0.99 |
| rotenone, VAE | 2.90 | 0.46 | 0.81 | 0.93 | 1.04 |
| rotenone, MLE | 2.80 | 0.43 | 0.82 | 0.89 | 0.99 |

The lifetimes sit in the usual NADH ranges (free about 0.4 ns, bound 2-3.4
ns). Rotenone shortens the mean lifetime and raises the free fraction. The raw
phasor shows the same shift with no model at all: phase 43.3° to 40.4°,
modulation 0.58 to 0.61. VAE numbers are posterior medians over 20 draws;
decoding the posterior mean instead put the bound lifetime about 10% high
(see Inference below).

![hMSC control](docs/hmsc_control.png)
![hMSC rotenone](docs/hmsc_rotenone.png)

Caveat on the maps: the VAE's per-pixel spread of α is narrow (0.77-0.84
in the rotenone image, 2nd to 98th percentile; the per-pixel fit spans
0.67-0.91, including its own noise). Some of that is the prior pulling
pixels towards the population, so it is not evidence of uniform metabolism.

## Evaluation

Three methods, each fitted end to end with the IRF learned from the data (no
calibration file):

- **VAE**: the amortised model, β = 1 ELBO with a 20-epoch KL warm-up,
  per-pixel parameters as posterior medians over 20 draws.
- **Per-pixel MLE**: every pixel fitted independently by maximum likelihood
  (`baseline.py --fit_irf`).
- **Global analysis**: one set of lifetimes shared by every pixel, fractions
  and background per pixel (`baseline.py --fit_irf --global_lifetimes`). This
  is the usual remedy for low counts in FLIM.

One seed, 60 epochs, CPU. Scripts: `scripts/photon_sweep.py`,
`scripts/thinning_study.py`, `scripts/kl_study.py`, and
`scripts/plot_evaluation.py` for the figures.

### Inference: decode samples, not the mean

Decoding the posterior mean of the latent put the long lifetime 22% high and
tripled the α error (0.149 against 0.046 at 200 photons, same checkpoint).
The decoder is nonlinear and the posterior is wide, so the decoded mean is
not the posterior's view of the parameters. Taking the median of the
parameters over posterior draws, as lumos-vae's `sample_posterior` does,
removes the bias. It is now the default in `predict.py`. Every VAE number in
earlier versions of this README that showed a 7-25% lifetime bias came from
mean decoding.

### KL: β = 1 is right

The model uses only one or two of its 16 latent dimensions. To test whether
that is the β = 1 optimum or an optimisation failure, the same data was fitted
with the plain ELBO, with KL warm-up (which ends at β = 1, so the objective is
unchanged), and with free bits (which changes the objective by exempting the
first few nats per dimension from the KL):

| photons | objective | −ELBO (val) | KL (nats) | active dims | τ long bias / IQR | τ_amp IQR | α MAE |
|---|---|---|---|---|---|---|---|
| 200 | beta1 | 29.80 | 0.86 | 1 | -0.004 / 0.085 | 0.219 | 0.046 |
| 200 | warmup20 | 29.77 | 0.95 | 2 | -0.005 / 0.082 | 0.205 | 0.046 |
| 200 | freebits0.25 | 31.74 | 3.70 | 16 | -0.020 / 0.138 | 0.218 | 0.056 |
| 200 | freebits1 | 37.98 | 9.69 | 16 | +0.076 / 0.160 | 0.241 | 0.065 |
| 1000 | beta1 | 31.30 | 2.06 | 2 | -0.008 / 0.064 | 0.110 | 0.026 |
| 1000 | warmup20 | 31.21 | 2.06 | 2 | -0.002 / 0.066 | 0.109 | 0.026 |
| 1000 | freebits0.25 | 33.31 | 4.76 | 16 | -0.006 / 0.077 | 0.115 | 0.029 |
| 1000 | freebits1 | 40.54 | 11.12 | 16 | +0.016 / 0.097 | 0.129 | 0.039 |

Warm-up gives a slightly better ELBO than plain training with the same
accuracy. Free bits uses all 16 dimensions and is worse on every count. So
the low latent usage is the correct β = 1 optimum: the data does not support
more per-pixel information, and forcing it in adds noise. With the exact
Poisson likelihood there is no reason to move β away from 1.

### Synthetic data, 50 to 1000 photons per pixel

Held-out pixels, scored against ground truth. Relative errors on the
amplitude-weighted mean lifetime τ_amp, r is the correlation with the truth,
α MAE is the error in the long component's amplitude fraction, and the
reconstruction error is the mean over bins of (μ̂ − μ)² / μ against the
noise-free histogram (raw counts score about 1).

![synthetic sweep](docs/eval_synthetic_sweep.png)

| photons | method | τ_amp bias / IQR | τ_amp r | τ long r | τ short IQR | α MAE | reconstruction |
|---|---|---|---|---|---|---|---|
| 50 | VAE | +0.01 / 0.30 | 0.57 | 0.58 | 0.24 | 0.058 | 0.0074 |
| 50 | MLE | -0.27 / 0.59 | 0.37 | 0.17 | 1.16 | 0.187 | 0.0375 |
| 50 | global | -0.05 / 0.43 | 0.53 | — | 0.23 | 0.121 | 0.0228 |
| 50 | raw data | | | | | | 0.93 |
| 100 | VAE | +0.00 / 0.24 | 0.70 | 0.69 | 0.22 | 0.050 | 0.0102 |
| 100 | MLE | -0.20 / 0.45 | 0.53 | 0.20 | 0.85 | 0.136 | 0.0427 |
| 100 | global | -0.02 / 0.31 | 0.66 | — | 0.24 | 0.090 | 0.0264 |
| 100 | raw data | | | | | | 0.94 |
| 200 | VAE | -0.01 / 0.21 | 0.79 | 0.78 | 0.20 | 0.044 | 0.0152 |
| 200 | MLE | -0.16 / 0.36 | 0.62 | 0.33 | 0.71 | 0.091 | 0.0485 |
| 200 | global | -0.01 / 0.22 | 0.78 | — | 0.25 | 0.067 | 0.0307 |
| 200 | raw data | | | | | | 0.96 |
| 500 | VAE | -0.00 / 0.14 | 0.89 | 0.85 | 0.17 | 0.034 | 0.0202 |
| 500 | MLE | -0.08 / 0.23 | 0.76 | 0.45 | 0.51 | 0.057 | 0.0535 |
| 500 | global | +0.01 / 0.16 | 0.88 | — | 0.27 | 0.050 | 0.0376 |
| 500 | raw data | | | | | | 0.96 |
| 1000 | VAE | -0.00 / 0.11 | 0.93 | 0.88 | 0.16 | 0.025 | 0.0242 |
| 1000 | MLE | -0.04 / 0.16 | 0.86 | 0.58 | 0.39 | 0.042 | 0.0549 |
| 1000 | global | +0.01 / 0.13 | 0.92 | — | 0.27 | 0.045 | 0.0506 |
| 1000 | raw data | | | | | | 0.97 |

- The VAE is best or joint best on every measure at every photon count. The
  clearest gains are in α (0.058 against 0.121 for global analysis and 0.187
  for per-pixel MLE at 50 photons) and in the long lifetime, which only it can
  resolve per pixel at low counts (r = 0.58 at 50 photons against 0.17).
  Global analysis's short-lifetime IQR only looks competitive because it
  gives every pixel the same value, so its IQR is just the spread of the
  truth.
- Its mean lifetime is essentially unbiased at every level. Per-pixel MLE
  underestimates it by 27% at 50 photons.
- Global analysis is close to the VAE on mean lifetime from 200 photons up.
  Here the true lifetimes vary between pixels, which is exactly what global
  analysis assumes away, so this dataset favours the VAE on that point.
- Reconstructions from all three methods are 18 to 130 times closer to the
  truth than the raw counts. The VAE's are 2 to 5 times closer again.

### Real data at varying noise levels: zebrafish embryo

The embryo image from FLUTE (not used anywhere else here), 2x2 binned,
pixels with at least 1000 photons (9094 pixels, median 1256). Noise levels
are made by keeping each photon with probability f. Binomial thinning of a
Poisson process is exactly the data of an acquisition f times as long, so
nothing is simulated. The photons removed are an independent measurement of
the same pixel, which gives a score that needs no ground truth: the negative
log-likelihood of the held-out photons under each method's fitted histogram.
Mean lifetimes are compared against per-pixel MLE on all photons. That
reference shares both photons and estimator with the thinned MLE, so it
flatters MLE; the held-out-photon score is the unbiased comparison.

![embryo thinning](docs/eval_embryo_thinning.png)

| photons | method | held-out NLL (nats/photon) | τ_amp deviation median / IQR | τ_amp r |
|---|---|---|---|---|
| 630 | VAE | 2.8493 | +0.007 / 0.109 | 0.68 |
| 630 | global | 2.8504 | +0.007 / 0.116 | 0.51 |
| 630 | MLE | 2.8519 | -0.015 / 0.089 | 0.78 |
| 630 | raw data | 2.8851 | | |
| 252 | VAE | 2.8503 | +0.005 / 0.125 | 0.48 |
| 252 | global | 2.8537 | +0.008 / 0.134 | 0.38 |
| 252 | MLE | 2.8570 | -0.049 / 0.181 | 0.56 |
| 252 | raw data | 2.9221 | | |
| 126 | VAE | 2.8500 | +0.010 / 0.127 | 0.34 |
| 126 | global | 2.8573 | +0.005 / 0.163 | 0.30 |
| 126 | MLE | 2.8637 | -0.081 / 0.266 | 0.37 |
| 126 | raw data | 2.9696 | | |
| 65 | VAE | 2.8527 | +0.072 / 0.156 | 0.25 |
| 65 | global | 2.8677 | -0.024 / 0.212 | 0.21 |
| 65 | MLE | 2.8797 | -0.134 / 0.381 | 0.26 |
| 65 | raw data | 3.0504 | | |

![embryo maps](docs/eval_embryo_maps.png)

- The VAE predicts unseen photons best at every noise level, and the margin
  grows as photons drop: 15 millinats per photon ahead of global analysis and
  27 ahead of per-pixel MLE at 65 photons. The raw thinned histogram is 36 to
  198 millinats per photon behind.
- Its mean lifetime has the smallest spread against the reference at 252
  photons and below.
- It keeps the tissue structure down to 65 photons per pixel, where per-pixel
  MLE has dissolved into speckle and drifted 13% short.
- But at 65 photons the VAE's map shifts about 7% long against the reference.
  It did not do this on synthetic data. Real tissue has a lifetime
  distribution the smooth synthetic maps do not, and at very low counts the
  model leans on what it learned from the population. The shift is visible in
  the right-hand column of the maps and should be kept in mind below about 100
  photons per pixel.

Example fits at 5% of the photons, with the held-out photons (rescaled) as an
independent check on the shape:

![embryo fits](docs/eval_embryo_fits.png)

## Benchmark against stronger baselines

`scripts/benchmark.py` runs eight methods on the same data, each fitted end to
end with the IRF learned from the data:

- the VAE, alone and with neighbourhood context (summed, or each neighbour
  separately)
- hierarchical empirical Bayes (EB): a Gaussian population prior fitted by
  EM, the non-neural counterpart of what the VAE learns
- TV-regularised MLE, with an edge-preserving spatial penalty whose strength
  is chosen on held-out photons
- 3x3 binned MLE
- global lifetimes
- per-pixel MLE

Every pixel's photons are split at random: all methods fit one half and are
scored on how well they predict the other. For Poisson data the halves are
exactly two independent acquisitions, so this needs no ground truth. Scores
are on val pixels, which the VAEs were not fitted on. One seed, 60 epochs,
CPU.

### Realistic synthetic tissue

`synthetic.simulate_realistic` generates cell-like regions with sharp edges,
their own lifetimes and fractions, empty regions and strongly varying
brightness. Its IRF has an exponential diffusion tail on 25% of photons,
which no method models by default. Ground truth is scored on non-empty
pixels, with edge pixels separately.

![realistic benchmark](docs/bench_realistic.png)

| photons | method | held-out NLL above best (mnats/photon) | τ_amp bias / IQR / r | τ long r | α MAE | τ_amp IQR, edge / interior |
|---|---|---|---|---|---|---|
| 45 | VAE | 5.2 | +0.15 / 0.30 / 0.85 | 0.70 | 0.072 | 0.30 / 0.30 |
| 45 | VAE + summed 3x3 | 4.7 | +0.15 / 0.25 / 0.87 | 0.70 | 0.068 | 0.22 / 0.26 |
| 45 | VAE + stacked 3x3 | 5.1 | +0.15 / 0.27 / 0.86 | 0.70 | 0.068 | 0.22 / 0.28 |
| 45 | empirical Bayes | 12.5 | -0.10 / 0.35 / 0.67 | 0.42 | 0.111 | 0.34 / 0.35 |
| 45 | TV-regularised | 0.0 | -0.03 / 0.09 / 0.98 | 0.92 | 0.038 | 0.14 / 0.08 |
| 45 | 3x3 binned | 3.7 | -0.07 / 0.26 / 0.81 | 0.67 | 0.064 | 0.29 / 0.25 |
| 45 | global lifetimes | 14.6 | +0.15 / 0.32 / 0.82 | — | 0.122 | 0.35 / 0.32 |
| 45 | per-pixel MLE | 24.5 | -0.12 / 0.57 / 0.61 | 0.31 | 0.148 | 0.55 / 0.57 |
| 91 | VAE | 3.4 | +0.18 / 0.22 / 0.92 | 0.78 | 0.067 | 0.22 / 0.22 |
| 91 | VAE + summed 3x3 | 3.5 | +0.19 / 0.22 / 0.92 | 0.79 | 0.064 | 0.26 / 0.22 |
| 91 | VAE + stacked 3x3 | 3.4 | +0.19 / 0.21 / 0.92 | 0.79 | 0.058 | 0.19 / 0.21 |
| 91 | empirical Bayes | 7.3 | -0.00 / 0.28 / 0.80 | 0.52 | 0.090 | 0.30 / 0.28 |
| 91 | TV-regularised | 0.0 | +0.06 / 0.09 / 0.99 | 0.95 | 0.031 | 0.12 / 0.08 |
| 91 | 3x3 binned | 2.1 | +0.04 / 0.16 / 0.91 | 0.73 | 0.051 | 0.26 / 0.15 |
| 91 | global lifetimes | 7.8 | +0.18 / 0.26 / 0.90 | — | 0.117 | 0.30 / 0.25 |
| 91 | per-pixel MLE | 12.9 | -0.00 / 0.40 / 0.73 | 0.39 | 0.108 | 0.40 / 0.40 |
| 227 | VAE | 2.1 | +0.17 / 0.17 / 0.95 | 0.89 | 0.050 | 0.17 / 0.17 |
| 227 | VAE + summed 3x3 | 2.1 | +0.15 / 0.18 / 0.95 | 0.89 | 0.046 | 0.17 / 0.18 |
| 227 | VAE + stacked 3x3 | 2.1 | +0.17 / 0.18 / 0.95 | 0.89 | 0.051 | 0.15 / 0.18 |
| 227 | empirical Bayes | 3.7 | +0.08 / 0.23 / 0.87 | 0.62 | 0.055 | 0.25 / 0.22 |
| 227 | TV-regularised | 0.0 | +0.10 / 0.07 / 0.99 | 0.97 | 0.031 | 0.08 / 0.07 |
| 227 | 3x3 binned | 1.5 | +0.10 / 0.11 / 0.95 | 0.84 | 0.042 | 0.25 / 0.10 |
| 227 | global lifetimes | 4.2 | +0.17 / 0.20 / 0.93 | — | 0.101 | 0.19 / 0.20 |
| 227 | per-pixel MLE | 5.4 | +0.08 / 0.25 / 0.83 | 0.47 | 0.078 | 0.26 / 0.25 |
| 898 | VAE | 0.8 | +0.16 / 0.11 / 0.98 | 0.95 | 0.034 | 0.13 / 0.11 |
| 898 | VAE + summed 3x3 | 0.7 | +0.16 / 0.10 / 0.98 | 0.95 | 0.037 | 0.11 / 0.10 |
| 898 | VAE + stacked 3x3 | 0.7 | +0.16 / 0.12 / 0.98 | 0.95 | 0.035 | 0.12 / 0.11 |
| 898 | empirical Bayes | 1.2 | +0.12 / 0.13 / 0.97 | 0.56 | 0.054 | 0.15 / 0.12 |
| 898 | TV-regularised | 0.0 | +0.14 / 0.07 / 0.99 | 0.98 | 0.032 | 0.07 / 0.07 |
| 898 | 3x3 binned | 1.3 | +0.14 / 0.06 / 0.97 | 0.92 | 0.037 | 0.24 / 0.05 |
| 898 | global lifetimes | 2.3 | +0.17 / 0.15 / 0.97 | — | 0.101 | 0.18 / 0.15 |
| 898 | per-pixel MLE | 1.4 | +0.13 / 0.14 / 0.95 | 0.74 | 0.054 | 0.17 / 0.14 |

![realistic maps](docs/bench_realistic_maps.png)

- **TV wins everything here, by a wide margin**: best held-out
  likelihood, mean-lifetime IQR 0.07 to 0.09 against 0.10 to 0.30 for the
  VAEs, and correlation with truth 0.98 or more at every level. But this
  generator is piecewise constant, which is exactly TV's assumption, so it
  flatters TV as much as the smooth generator flattered the VAE.
- **On held-out photons the VAEs come third below 900 photons**, behind TV
  and binning, and second at 900. They are well ahead of EB, global analysis
  and per-pixel MLE. They keep sharp edges
  without any spatial prior (edge and interior errors are similar), where
  binning smears them.
- **Neighbourhood context barely helps.** Summed or stacked, the spatial VAEs
  are within noise of the plain VAE on every measure. An encoder that sees
  its neighbours but is trained per pixel does not learn to pool them
  effectively.
- **Empirical Bayes is worse than the VAE** on held-out photons, correlation,
  long lifetime and α at every level, though its mean-lifetime bias is
  smaller from 91 photons up. A Gaussian prior in parameter space is too
  crude for a population that is a mixture of regions; a nonlinear decoder
  from a Gaussian latent represents that better.
- **Every method's mean lifetime is biased high at 227 and 898 photons**
  (+8 to +17%). That comes from the unmodelled IRF tail (next section). Per-pixel
  MLE looked unbiased at 45 to 91 photons only because its low-count bias
  (about −20%, seen on the smooth data) cancelled it.

### Real tissue: zebrafish embryo

The embryo image (9094 pixels of at least 1000 photons after 2x2 binning),
thinned to 20%, 10% and 5% of its photons. Mean lifetimes are compared with
per-pixel MLE on all the photons. That reference is itself noisy and
low-count biased, and it flatters likelihood-maximising estimators; the
held-out likelihood is the unbiased comparison.

![embryo benchmark](docs/bench_embryo.png)

| photons | method | held-out NLL above best (mnats/photon) | τ_amp vs full-photon MLE: median dev / IQR / r |
|---|---|---|---|
| 64 | VAE | 0.2 | +0.039 / 0.143 / 0.26 |
| 64 | VAE + summed 3x3 | 0.0 | +0.040 / 0.146 / 0.23 |
| 64 | VAE + stacked 3x3 | 0.7 | +0.043 / 0.158 / 0.26 |
| 64 | empirical Bayes | 10.2 | -0.145 / 0.214 / 0.25 |
| 64 | TV-regularised | 0.1 | -0.127 / 0.100 / 0.42 |
| 64 | 3x3 binned | 3.7 | -0.128 / 0.171 / 0.32 |
| 64 | global lifetimes | 13.8 | -0.015 / 0.197 / 0.25 |
| 64 | per-pixel MLE | 25.9 | -0.121 / 0.384 / 0.26 |
| 129 | VAE | 0.0 | +0.006 / 0.133 / 0.35 |
| 129 | VAE + summed 3x3 | 0.0 | +0.007 / 0.132 / 0.36 |
| 129 | VAE + stacked 3x3 | 0.0 | +0.005 / 0.132 / 0.34 |
| 129 | empirical Bayes | 6.2 | -0.101 / 0.160 / 0.35 |
| 129 | TV-regularised | 0.1 | -0.077 / 0.101 / 0.51 |
| 129 | 3x3 binned | 1.7 | -0.081 / 0.132 / 0.40 |
| 129 | global lifetimes | 6.9 | -0.003 / 0.152 / 0.32 |
| 129 | per-pixel MLE | 13.7 | -0.098 / 0.280 / 0.37 |
| 254 | VAE | 0.2 | -0.001 / 0.120 / 0.47 |
| 254 | VAE + summed 3x3 | 0.2 | +0.003 / 0.122 / 0.45 |
| 254 | VAE + stacked 3x3 | 0.2 | +0.004 / 0.124 / 0.42 |
| 254 | empirical Bayes | 3.5 | -0.048 / 0.141 / 0.50 |
| 254 | TV-regularised | 0.0 | -0.044 / 0.103 / 0.53 |
| 254 | 3x3 binned | 0.8 | -0.044 / 0.121 / 0.49 |
| 254 | global lifetimes | 3.0 | +0.005 / 0.137 / 0.38 |
| 254 | per-pixel MLE | 6.3 | -0.047 / 0.188 / 0.53 |

![embryo maps](docs/bench_embryo_maps.png)

- **On held-out photons the best VAE and TV tie** at every level (within
  0.2 millinats per photon), ahead of binning, EB, global analysis and per-pixel
  MLE (26 millinats behind at 64 photons). TV's large synthetic lead does not
  survive real tissue.
- **The VAE's mean lifetime stays close to the full-photon reference** (+4%
  at 64 photons, under 1% above that). TV, binning, EB and MLE all drift
  short by 8 to 15% as photons drop, the usual low-count bias of maximising
  the likelihood per pixel.
- **TV tracks pixel-to-pixel variation better** (correlation 0.42 against
  0.26 at 64 photons, IQR 0.10 against 0.14). The VAE is more accurate on
  average but pulls individual pixels towards the population; TV keeps local
  contrast but shifts the whole map. The maps show both: TV's is smooth and
  uniformly light, the VAE's has the right tone and texture.

### Modelling the IRF tail

The realistic data's lifetime bias comes from its IRF tail, which a Gaussian
IRF absorbs into the fluorescence: the short lifetime reads about 40% long.
`--irf_tail` adds an exponential tail with a learned weight and rate
(closed form, `docs/how_it_works.md` section 2.1b). At 100 photons, on the
same split:

| photons | method | IRF | held-out NLL (nats/photon) | τ_amp bias / IQR / r | τ short IQR | α MAE | fitted tail |
|---|---|---|---|---|---|---|---|
| 100 | MLE | gaussian | 3.2805 | -0.001 / 0.398 / 0.73 | 0.873 | 0.108 | — |
| 100 | MLE | tailed | 3.2810 | -0.290 / 0.365 / 0.65 | 0.596 | 0.103 | — |
| 100 | VAE | gaussian | 3.2710 | +0.184 / 0.223 / 0.92 | 0.461 | 0.067 | — |
| 100 | VAE | tailed | 3.2709 | +0.100 / 0.210 / 0.92 | 0.322 | 0.056 | 12%, 0.29 ns |

- **For the VAE the tail model nearly halves the mean-lifetime bias** (+18%
  to +10%), cuts the short-lifetime spread by a third and lowers the α error.
  It recovers a tail of 12% at 0.29 ns against the true 25% at 0.25 ns. A
  tail on the IRF and a slightly longer short lifetime look much alike, so
  the tail is only partly identifiable from the data.
- **For per-pixel MLE the tail model exposes its low-count bias** (−29%),
  confirming that its apparent accuracy with the Gaussian IRF was two errors
  cancelling.
- **Held-out likelihood is unchanged either way**, so photons alone cannot
  tell the two IRF models apart at this count. A reference measurement of
  the IRF would settle it.

### Verdict

Against strong baselines the VAE is a sound, competitive method but not a
dominant one:

- **On real tissue** it ties the best spatial method (TV) on predicting
  unseen photons, and gives the least biased mean lifetimes at low counts.
- **It beats per-pixel MLE, global analysis and a hierarchical-Bayes
  population prior** by clear margins everywhere.
- **When neighbouring pixels share parameters**, an explicit spatial prior
  like TV recovers maps the VAE cannot, and showing the encoder its
  neighbours does not close that gap.

The obvious next step is to combine the two: amortised inference plus a
spatial term in the objective, for example TV on the decoded parameters of
neighbouring pixels in a batch of patches.

## Diagnosis: what limits the VAE, and a spatial prior

**It is fully trained.** Validation ELBO is flat over the last 20 to 30
epochs of every run. At 64 photons the training ELBO sits 0.3 nats per pixel
below validation: mild overfitting on a single small image, not
undertraining.

**Inference is not the limit.** The amortisation gap was measured by
optimising each pixel's posterior directly, starting from the encoder's
guess. That improves the ELBO by only 0.03 nats per pixel on the embryo and
0.06 on the realistic data. The encoder already finds nearly the best
posterior the model allows, so more epochs or per-pixel refinement cannot
help.

**The formulation was the limit.** Each pixel had its own independent
N(0, I) prior. Under that model neighbouring pixels are independent, so the
exact posterior of a pixel ignores its neighbours. That is why giving the
encoder its neighbours changed nothing: it correctly learned to ignore them.
TV wins on structured tissue because its objective ties neighbours
together.

**The fix: a latent Markov random field prior** (`--spatial_prior λ`):

```
−log p(Z) = ½ Σ_i |z_i|² + λ Σ_(i~j) sqrt(|z_i − z_j|² + δ²) + const
```

This is TV's edge-preserving coupling, applied to the latents. Its
expectation under the posterior is estimated from the same reparameterised
samples as the likelihood. For fixed λ the normaliser is a constant, so the
objective is still an ELBO. Training draws contiguous tiles of pixels
(`data.TileDataset`), so neighbour pairs share a batch. With the encoder
seeing its neighbours (`--spatial 5 --spatial_mode stack`), the model now has
both a reason and the information to pool them. `scripts/mrf_study.py` runs
it on the benchmark's own photon splits.

Realistic synthetic tissue, 100 photons:

| window | λ | held-out NLL (nats/photon) | τ_amp bias / IQR / r | τ long r | α MAE |
|---|---|---|---|---|---|
| 3x3 | 0 | 3.2712 | +0.18 / 0.245 / 0.916 | 0.79 | 0.065 |
| 3x3 | 0.3 | 3.2711 | +0.19 / 0.230 / 0.918 | 0.77 | 0.062 |
| 3x3 | 1 | 3.2706 | +0.20 / 0.216 / 0.925 | 0.79 | 0.059 |
| 3x3 | 3 | 3.2695 | +0.19 / 0.197 / 0.944 | 0.84 | 0.052 |
| 3x3 | 10 | 3.2685 | +0.20 / 0.174 / 0.960 | 0.88 | 0.049 |
| 3x3 | 30 | 3.2692 | +0.20 / 0.167 / 0.955 | 0.86 | 0.052 |
| 5x5 | 3 | 3.2694 | +0.19 / 0.184 / 0.947 | 0.86 | 0.051 |
| 5x5 | 10 | 3.2684 | +0.18 / 0.163 / 0.960 | 0.91 | 0.042 |
| TV-regularised (benchmark) | | 3.2676 | +0.06 / 0.085 / 0.987 | 0.95 | 0.031 |
| 3x3 binned (benchmark) | | 3.2698 | +0.04 / 0.163 / 0.913 | 0.73 | 0.051 |

Embryo, 5% of photons (64 per pixel), 3x3 context:

| λ | held-out NLL | τ_amp vs full-photon MLE: dev / IQR / r |
|---|---|---|
| 0 | 2.8446 | +0.030 / 0.137 / 0.20 |
| 0.3 | 2.8448 | +0.052 / 0.142 / 0.18 |
| 1 | 2.8448 | +0.047 / 0.145 / 0.12 |
| 3 | 2.8448 | +0.055 / 0.142 / 0.21 |

- **On structured tissue the prior helps steadily.** With a 5x5 window and
  λ = 10 the VAE passes binning on held-out photons. It cuts the gap to TV
  from 3.6 to 0.8 millinats per photon. Mean-lifetime spread falls by a
  third, long-lifetime correlation rises from 0.79 to 0.91, and α error
  from 0.065 to 0.042. TV is still ahead; a 5x5 window is still local,
  while TV pools across whole regions.
- **On the embryo it neither helps nor hurts.** The real tissue at 64
  photons does not have the piecewise-constant structure the prior
  rewards, consistent with TV's own lead vanishing there.
- λ has to be chosen per dataset, on held-out photons as for TV. It stays off
  by default.

## Practical advantages over TV and binning

The case for amortisation is applying a trained model to new data without
refitting. `scripts/transfer_study.py` trains on the hMSC control image and
applies the model to the rotenone image (same instrument). Both are thinned
to 20% of their photons (about 165 per pixel), and all methods are scored on
rotenone's held-out photons:

| method | held-out NLL (nats/photon) | wall-clock on CPU | median τ_amp (ns) |
|---|---|---|---|
| VAE trained on control | 3.3022 | 16 s (inference only; training on control took 143 s) | 0.988 |
| VAE trained on rotenone | 3.3006 | 186 s | 0.934 |
| per-pixel MLE | 3.3080 | 436 s | 0.846 |
| binned | 3.3022 | 805 s (includes the IRF fit) | 0.829 |
| tv | 3.3008 | 1564 s (includes the IRF fit) | 0.829 |

- **Speed.** The transferred VAE processes a new image in 16 s on CPU, with no
  tuning parameter. TV takes 26 minutes, because its strength has to be
  selected by refitting, and binning takes 13. Fitted on the new image
  itself, the VAE ties TV on held-out photons (3.3006 against 3.3008) at an
  eighth of its time, training included.
- **Accuracy of the transfer.** It predicts as well as binning, 1.6 millinats
  per photon behind a VAE trained on the image itself.
- **The catch: the learned population prior travels with the model.** The
  transferred VAE puts rotenone's median mean lifetime at 0.99 ns, against
  0.93 ns from a VAE trained on rotenone. It partly pulls the new condition
  towards the old one. To compare conditions, train one model on all the
  images together; do not train on one and apply it to another.
- **Where the methods disagree.** On rotenone, TV and binning give 0.83 ns and
  the VAE 0.93 ns. There is no ground truth. On synthetic data and the embryo
  the likelihood-maximising methods read 8 to 15% short at these photon
  counts and the VAE did not, which favours the VAE's value, but it is not
  proof.

Other practical points:
- The VAE needs no smoothing strength (TV) or bin size (binning) chosen per
  image.
- It keeps edges at full resolution where binning blurs them.
- It gives a per-pixel posterior spread, not yet checked for calibration.

## Running on a GPU or Mac

```bash
git clone -b claude/sleepy-ride-t6pomq https://github.com/tmuird/lumos-vae && cd lumos-vae/lumos-flim
pip install -e ".[test]"
pytest tests                         # includes test_runs_on_device on CUDA / MPS if present
git clone --depth 1 https://github.com/phasorpy/phasorpy-data ../phasorpy-data   # FLUTE data

# training picks CUDA, then MPS, then CPU; --accelerator cuda|mps|cpu to force
python -m lumos_flim.train --data data/hmsc.zarr --kl_warmup_epochs 20
# with the spatial prior and a tailed IRF, for structured tissue
python -m lumos_flim.train --data data/hmsc.zarr --kl_warmup_epochs 20 \
    --spatial 5 --spatial_mode stack --spatial_prior 10 --irf_tail
```

The benchmark scripts read the FLUTE files from `$FLUTE_DIR`:

```bash
export FLUTE_DIR=../phasorpy-data/zenodo_8046636
python scripts/benchmark.py --dataset embryo
python scripts/transfer_study.py
```
On a GPU the VAEs train in seconds, and the slow parts become TV and empirical
Bayes, which also run on the device through `PixelFitter`.

## A literal port of lumos-vae (removed)

An earlier variant copied lumos-vae with only the physics changed: global
std normalisation, unordered rates and the learned Gaussian noise model
`var = α·μ + β`. It did worst of the three methods at every photon level
(mean lifetime biased +30 to +50%, short lifetime off by +100 to +215%) and
has been removed. What it showed:

- The KL term of lumos-vae is sound. Dividing both reconstruction and KL by
  the element count keeps their ratio equal to the true ELBO, which is the
  same weighting this VAE uses per pixel.
- The learned noise model exists to find the right likelihood scale. For
  photon counting that scale is known exactly (in raw counts the variance
  equals the mean), so the Poisson likelihood is the model with the scale
  already fixed. In the port, α started 30 times too low and reached only
  0.03 of the correct 0.61 in 60 epochs, so the model fitted shot noise in
  bins holding 0 to 2 photons, where a Gaussian is a poor stand-in anyway.
- Its latent collapsed to one active dimension out of 64. Decoding the
  posterior mean then gave reconstructions four times worse than decoding
  samples, because the decoder had only seen samples. The same
  mean-versus-sample gap is worth checking in lumos-vae, whose `predict`
  decodes the mean when `n_predictions=1`.

## Layout

| Module | Role |
|---|---|
| `physics.py` | Exact periodic TCSPC forward model, amplitude/lifetime conversions, phasor |
| `vae.py` | Encoder (circular 1D CNN over delay), decoder, learnable IRF and spectra |
| `vae_module.py` | Lightning module: Poisson deviance + KL, chi-square and ground-truth logging |
| `data.py` | ImSpector TIFF reader, Zarr store, datamodule |
| `prepare.py` | TIFF images to pixel store |
| `synthetic.py` | Photon-level simulator with ground truth |
| `calibrate.py` | IRF from a reference dye of known lifetime |
| `train.py`, `predict.py` | Training and inference, maps and per-image summaries |
| `baseline.py` | Non-amortised fits of the same physics: per-pixel MLE, global, binned, TV, empirical Bayes |
| `scripts/photon_sweep.py` | Synthetic sweep against per-pixel MLE and global analysis |
| `scripts/thinning_study.py` | Real data at varying noise levels, scored on held-out photons |
| `scripts/kl_study.py` | β = 1 ELBO against KL warm-up and free bits |
| `scripts/plot_evaluation.py` | Evaluation figures |
| `scripts/benchmark.py`, `scripts/plot_benchmark.py`, `scripts/summarise_benchmark.py` | Eight-method benchmark, figures and tables |
| `scripts/irf_tail_study.py` | Gaussian against tailed IRF |
| `scripts/mrf_study.py` | Latent MRF prior strength and window |
| `scripts/transfer_study.py` | Train on one image, apply to another |

Store layout: `counts [sample, channel, time]`, `split`, `image`, `y`, `x`,
and `gt_*` for synthetic data. Attributes: `bin_width_ns`, `period_ns`,
`image_names`, `image_shapes`.

## Limitations

- The IRF is Gaussian. Real detector IRFs have tails, and at 10^8 photons the
  reference fits show it (reduced chi-square far above 1). A measured IRF could
  be added as a discrete circular convolution, but none ships with FLUTE.
- One IRF for the whole image. On scanning systems `t0` can drift across the
  field. A per-pixel shift would be a small extension.
- No pile-up or dead-time correction. That is fine at the count rates in
  FLUTE, but not at high rates.
- Pixels are independent apart from sharing the encoder. Spatial binning is
  done up front, not learned.
