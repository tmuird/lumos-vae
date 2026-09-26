# How LUMOS-FLIM works

This document explains the whole system: the measurement, the physics, every
class and function, and why each choice was made. It assumes familiarity with
lumos-vae (photobleaching Raman) and says at every step what is the same and
what differs.

## Summary

**The measurement.** A pulsed laser excites the sample every 12.48 ns. Each
detected photon is time-stamped against the last pulse and dropped into one of
56 delay bins. Every pixel therefore ends up with a histogram of photon
arrival times. A fluorophore with lifetime τ makes that histogram decay as
exp(−t/τ). Mixtures give sums of exponentials. A constant background (dark
counts, stray light) adds a flat floor. The instrument blurs everything with
its response function (IRF).

**The physics we exploit.** For a Gaussian IRF, periodic excitation and
binned detection, the expected histogram has an exact closed form.
`physics.py` evaluates it. Every photon is a discrete event, so the noise is
exactly Poisson, and the likelihood is known exactly, with no free noise
parameters.

**The model.** A VAE whose decoder is that closed form:

```
histogram --(normalise, CNN encoder)--> latent z --(MLP decoder)-->
    lifetimes, photon fractions, background fraction
    --(exact TCSPC physics, shared IRF)--> expected histogram
    --(Poisson likelihood against the data)--> loss (+ KL)
```

The network never draws a histogram. It only proposes physical parameters,
and the physics draws the histogram. The IRF (and, for spectral data, the
emission spectra) are global parameters learned alongside the network.

**Why amortise.** Fitting each pixel alone is noisy at low photon counts.
Training one encoder over every pixel of an image learns what parameter
combinations are plausible (a population prior) and uses that to steady each
pixel's estimate. The spatial variant also lets the encoder see the
neighbouring pixels.

**What is the same as LUMOS.**
- A VAE with an analytical physics decoder.
- Per-sample rates and amplitudes, with global shared components (IRF,
  spectra).
- A constant term plus decaying exponentials, each integrated over a detector
  interval.
- Decoding observable amplitudes rather than raw ones, to break the
  amplitude–rate trade-off.
- The ELBO at β = 1 with correctly scaled terms.
- Posterior sampling for inference.
- The Lightning training loop.

**What differs.**
- The data axis: delay bins within one laser period instead of frames over
  seconds.
- Periodicity: decay wraps around from earlier pulses.
- The instrument response is in time, not wavenumber.
- The noise model is exact Poisson instead of learned heteroscedastic
  Gaussian.
- Normalisation is per pixel instead of by one global standard deviation.
- Lifetimes are ordered.
- No extrapolation: every delay bin is recorded at once.
- The static term is a nuisance here, not the quantity of interest.

---

## 1. FLIM and TCSPC from first principles

### 1.1 What is measured

Fluorescence lifetime imaging (FLIM) records, for every pixel, how long
molecules stay excited before emitting. After excitation, an excited
population decays by first-order kinetics. The probability that a given
molecule is still excited at time t is exp(−kt), with k = 1/τ. The emission
rate is proportional to the excited population, so the photon arrival
density after an instantaneous pulse is k·exp(−kt).

Time-correlated single photon counting (TCSPC) measures this one photon at a
time. The laser fires at 80.11 MHz (period P = 12.483 ns). Detection is
kept sparse, well under one photon per pulse, so each detected photon's delay
relative to its pulse is unbiased. Its delay is placed into one of N = 56 bins
of width D = P/N = 0.2229 ns. Over a scan, each pixel accumulates a histogram
h_n, n = 0…55.

### 1.2 Why lifetimes matter: the NADH example

NADH is autofluorescent. Free NADH has a short lifetime (~0.4 ns);
protein-bound NADH, whose fluorescence is less quenched, is longer (~2–3.4
ns). A pixel containing both decays as a sum of two exponentials, weighted by
how much of each is present. The bound fraction tracks metabolic state:
glycolysis versus oxidative phosphorylation. Rotenone blocks complex I,
pushing cells towards free NADH. That is the effect the hMSC data shows.

### 1.3 Periodic excitation and wrap-around

At 12.5 ns between pulses, a 3 ns component has not fully decayed by the next
pulse (exp(−12.5/3) ≈ 1.5%). Photons from earlier pulses therefore land in
the current window. In the data this shows as counts *before* the rising edge
that match the tail at the end of the window. The model has to sum over all
past pulses. Ignoring it biases long lifetimes.

**LUMOS contrast.** Photobleaching happens once. There is no periodic
re-excitation and nothing wraps.

### 1.4 The instrument response

The laser pulse has finite width, the detector and electronics add timing
jitter, and single-photon detectors add a slow diffusion tail. All of this
blurs the arrival time. We model the IRF as a Gaussian with centre t0 and
width σ, both learned. The tail is ignored: that is a deliberate
misspecification, and the realistic benchmark tests it.

**LUMOS contrast.** LUMOS blurs the Raman peaks with a Gaussian in
wavenumber (`fwhm_G`), learned or pinned. Same idea, different axis: here
the blur is in time and applies to every component, fluorescence included.

### 1.5 Background

Dark counts and ambient light arrive uniformly in time, so they add a flat
floor. It is the constant-in-time term, the counterpart of the static Raman
spectrum in LUMOS. There the constant term is the signal of interest; here it
is a nuisance to be separated from the lifetimes.

### 1.6 Noise: exactly Poisson

Each photon is an independent, discrete arrival. The count in a bin is
therefore Poisson with mean equal to the expected count, with no read noise
or gain. Conditioned on the pixel's total photon count, the histogram is
multinomial. This is why the likelihood needs no learned noise parameters.

**LUMOS contrast.** A CCD adds read noise and gain, so LUMOS learns
var = α·μ + β. The Poisson likelihood is the special case where the scale is
known: in raw counts α = 1 and β = 0. The purpose of LUMOS's noise model, to
put the reconstruction on the correct scale against the KL, is met exactly
without fitting anything.

---

## 2. The forward model (`physics.py`)

### 2.1 Derivation

One pulse at time 0. A photon from a component with rate k arrives at
T = G + E, where G ~ N(t0, σ²) is the IRF jitter and E ~ Exp(k) the
emission delay. The distribution of T is the exponentially modified
Gaussian, with CDF

```
F(t) = Φ(u) − exp(−k(t − t0) + k²σ²/2) · Φ(u − kσ),     u = (t − t0)/σ
```

Periodic excitation: pulses at …, −2P, −P, 0, P. The probability that a
photon lands in bin [a, b) of the current window is

```
p(n) = Σ_j [ F(b + jP) − F(a + jP) ]
```

summed over past pulses j ≥ 0 and, for safety, the next pulse j = −1, in
case the IRF straddles the window start. This sums exactly to one over the
window: the telescoping sum covers the whole real line.

**Evaluating the infinite sum.** Pulses j ∈ {−1, 0, 1} are summed
explicitly. For j ≥ 2 the Gaussian part Φ(u_j) is 1 to machine precision
(u_j ≥ P/σ ≈ 100), so it cancels in the difference F(b) − F(a). The decay
part forms a geometric series in exp(−kP), summed in closed form as
1/(1 − exp(−kP)).

### 2.2 Numerics

**Cancellation.** Written directly, exp(−k(t−t0) + k²σ²/2)·Φ(u − kσ)
cancels catastrophically for short lifetimes (kσ large): a huge positive
exponent times a tiny Φ. `_exgauss_tail` uses the identity

```
exp(−kσu + (kσ)²/2) Φ(u − kσ) = ½ exp(−u²/2) erfcx((kσ − u)/√2)
```

wherever kσ > u, which has no cancellation. Elsewhere Φ is near one and the
direct form, in log space via `log_ndtr`, is safe. Both branches are clamped
into their own domains, so the branch `torch.where` discards is still finite
and cannot poison the gradient with NaNs. This was found by a test:
τ = 10 fs failed in float32 before the fix.

**Devices.** Apple's MPS has no kernels for `torch.special.erfcx` or
`log_ndtr`. On MPS the code uses portable versions built from elementary
operations, from the Numerical Recipes Chebyshev fit to erfc (relative error
below 1.2×10⁻⁷, float32 precision). Everywhere else it uses the native
kernels, which are three times faster. The normal CDF is taken through
`log_ndtr` in the lower tail, because the native `ndtr` underflows to zero
below about −8.4.

**Validation.** `tests/test_physics.py` compares the analytical histogram with
Monte-Carlo photon simulation for lifetimes from 1 ps to 30 ns, including an
IRF straddling the window edge. The reduced chi-square is about 1 in every
case. Row sums are one and gradients are finite over seven decades of rate.

### 2.3 The functions

| Function | Does | Why |
|---|---|---|
| `bin_edges(n, D)` | the N+1 edges of one period | bins span exactly one period, as the file metadata says |
| `_exgauss_tail(t, k, t0, σ)` | the decay part of the ex-Gaussian CDF, stably | see 2.2 |
| `_periodic_cdf(t, k, t0, σ, P)` | cumulative arrival probability summed over pulses | only differences are used, so constants drop out |
| `decay_histograms(rates, t0, σ, N, D)` | [..., F, N] bin probabilities per component | the core physics; exact, differentiable in rates, t0 and σ |
| `irf_histogram` | the IRF alone, wrapped | test that the zero-lifetime limit equals the IRF |
| `expected_counts(totals, fractions, background, decays, bases, bg_spectrum)` | [B, C, N] expected counts | assembles the mixture; sums to the photon total by construction |
| `amplitude_fractions(f, k)` | photon fractions to amplitude fractions | a component with amplitude a emits a/k photons, so a ∝ f·k; amplitude fractions are what NADH papers quote |
| `mean_lifetimes(f, k)` | intensity- and amplitude-weighted τ | τ_amp is the standard robust summary at low counts |
| `phasor`, `lifetime_phasor` | first-harmonic phasor | model-free cross-check (the rotenone shift) |

### 2.4 The assembled model

```
h_n ~ Poisson( n_tot · [ f_bg / N  +  Σ_i f_i · p_i(n) ] )
```

For spectral FLIM with C channels, component i emits with a global spectrum
B_i(c) summing to one over c. The background has its own per-pixel spectrum
β(c), so the model is n_tot · [f_bg β(c)/N + Σ_i f_i B_i(c) p_i(n)].

**LUMOS contrast, term by term.** LUMOS's factored model is
S_n(ν) = S(ν)·T + Σ_i ã_i B_i(ν) e^{−λ_i t_n}.
- **Constant term:** S(ν)·T there; f_bg/N here.
- **Decay:** e^{−λ t_n} at frame times there; here the exact bin-integrated,
  IRF-blurred, periodically summed p_i(n).
- **Integration factor:** LUMOS's (1 − e^{−λT})/λ (fast bleaching components
  contribute little within a frame) has an exact analogue here. p_i(n) is
  integrated over the bin, and the periodic factor 1/(1 − e^{−kP}) is its
  wrap-around counterpart.
- **Amplitudes:** ã_i (effective amplitudes) there; photon fractions f_i
  here. Both are what the data measures directly.
- **Bases:** B_i global in both.

---

## 3. Parametrisation and identifiability

These choices decide whether the problem is well posed. They sit between the
network and the physics.

**Photon fractions, conditioned on the total.** The decoder outputs fractions
(softmax over F components plus background), and the expected histogram is
scaled to the pixel's observed total. For Poisson data the maximum-likelihood
total is the observed one, so conditioning on it loses nothing. It turns the
likelihood into a multinomial over shapes and removes the overall scale from
the network's job. This is the same principle as LUMOS's effective amplitudes
ã: decode what the data measures, not a physical amplitude that trades off
against the rate. Here the trade-off would be amplitude a against rate k,
because a component emits a/k photons.

**Ordered lifetimes.** Rates are `rate_min + cumsum(softplus(raw))`, so
k_1 < k_2 < … and component 0 always has the longest lifetime. Without this,
component labels can swap between pixels: pixel A's "component 1" could be
pixel B's "component 2". The encoder would then have to learn a
discontinuous map, and averaged maps would mix labels. LUMOS leaves rates
unordered, because its components are also tied to distinct emission spectra
B_i, which identify them.

**A floor on the rate.** `rate_min = 1/P` by default (τ ≤ one period). A
lifetime much longer than the period gives a nearly flat histogram, which is
indistinguishable from background. This is the FLIM version of LUMOS's
degeneracy between very slow bleaching and the static Raman term, and it is
handled the same way, with `lambda_min` there and `tau_max` here.

**Initial rates.** Log-spaced from P/3 (a third of the window) down to 2D
(two bins), the range the window can resolve. It mirrors LUMOS initialising
between 1/span and 1/frame.

---

## 4. The network (`vae.py`)

### 4.1 `FlimVAE.__init__`

- `n_bins`, `bin_width`, `period`: the time axis, from the store.
- `rate_min`: section 3.
- `irf_t0`, `irf_sigma_raw`: the global IRF. σ = softplus(raw) + 5 ps, so it
  is positive and never exactly zero, which would make the physics a
  step function with undefined gradients. With `fix_irf` both are frozen,
  from a calibration file. Same as LUMOS's `fwhm_G` pin.
- `basis_logits` (spectral only): the global emission spectra, softmax over
  channels. They start as broad, distinct bumps so the components are not
  symmetric at initialisation; symmetric components would receive identical
  gradients and never separate. LUMOS uses a mixture of Gaussians (MoG) over
  wavenumber for the same role. With a handful of detection channels a free
  softmax is enough; LUMOS needs a smooth parametric family across 1024
  wavenumbers.
- Encoder, decoder and head initialisation: 4.3–4.4.

### 4.2 `decode` and `forward`

`forward(x, sample, scale, context)`:
1. `totals` = photons per pixel.
2. The encoder gives μ and log σ² of q(z | x) (plus the context, if spatial).
3. z is sampled by reparameterisation during training, and set to the mean
   otherwise, unless `sample=True`.
4. `decode` maps z to rates (ordered), fractions and background (softmax),
   and the background spectrum, then calls the physics.

It returns a dict (expected, rates, fractions, background, mu, logvar,
totals). LUMOS returns a tuple; a dict avoids positional mistakes as outputs
are added.

### 4.3 `Encoder`

**Input normalisation.** Each histogram is divided by its own total and
multiplied by C·N, giving a mean-one shape. The log of the total is fed in
separately, after the convolutions. Photon totals span orders of magnitude
across an image; dividing by one global standard deviation, as LUMOS does,
would make dim and bright pixels look like different curves. The shape
carries the lifetime information and the log total tells the encoder how
noisy the shape is. LUMOS's photobleaching series have comparable scales, so
a global scale works there.

**Position encoding.** Two extra input channels, cos and sin of 2πn/N: the
phase of each bin within the period. This is the phasor basis, so the first
layer can compute phasor-like projections directly. It also tells the
network where the pulse sits. LUMOS uses linear positional encodings in
wavenumber and time, since its axes are not periodic.

**Circular padding.** The delay axis wraps: bin 55 is followed by bin 0 of
the next period. Circular padding makes the convolution respect that; zero
padding would invent an edge.

**Architecture.** 1D convolutions over delay (kernel 5; 32, 64, 128
channels; stride 1, then 2, 2), LeakyReLU, adaptive average pooling to 7
positions, flatten, concatenate the log total(s), one hidden layer, then
linear heads for μ and log σ² (clamped to [−20, 10] against runaway, as in
LUMOS). Detection channels enter as input channels. LUMOS uses a 2D
convolution over [wavenumber, time] because it has 1024 spectral points; here
spectral FLIM has a handful of channels, so treating them as input channels
is simpler and loses nothing.

**Spatial context** (`spatial=True`). The summed histogram of the k×k
neighbourhood (excluding the centre) is added as further input channels,
normalised the same way, with its own log total. Only the encoder sees it:
the likelihood is still that of the centre pixel alone, so the objective is
still a valid ELBO for each pixel (any q(z | ·) gives a lower bound). Where
neighbours agree, the encoder can borrow their photons; at an edge, it can
learn to ignore them. Spatial binning, by contrast, always averages.

### 4.4 `Decoder`

A shared two-layer MLP trunk and three linear heads: rate increments (F),
fraction logits (F + 1, the last being the background), and background
spectrum logits (C, spectral only). The heads start with weights scaled by
0.1, so at step 0 every pixel decodes to roughly the bias values: sensible
lifetimes and a small background. The physics then sees valid inputs from the
first step. LUMOS has separate trunks per head (λ, abundance, Raman); with
five outputs here, one trunk is enough.

---

## 5. The objective (`vae_module.py`)

### 5.1 Likelihood: `poisson_half_deviance`

Per pixel, Σ x log(x/μ). Because μ sums to the observed total, this equals
the multinomial negative log-likelihood minus its value at the saturated
model μ = x. It is exact, zero for a perfect fit, and about half the degrees
of freedom for a correct model with Poisson noise. The constant (the
saturated log-likelihood) does not affect gradients but makes the number
interpretable.

`pearson_chi2` / dof (the reduced chi-square) is logged as a physics check.
Near 1 means the model and the noise model both fit.

### 5.2 KL and β

KL(q(z|x) ‖ N(0, I)) summed over latent dimensions, per pixel. The loss per
pixel is NLL + β·KL, averaged over the batch: the ELBO at β = 1. LUMOS
divides both reconstruction and KL by the element count. That keeps the same
ratio, so the two are identical up to a constant factor on the gradient.

**Warm-up** (`kl_warmup_epochs`) ramps β linearly from 0 to 1 and then holds
it at 1. The final objective is unchanged; early training can place pixels in
the latent before the KL pulls them together. It gave a marginally better
ELBO.

**Free bits** (`free_bits`) exempts the first λ nats per latent dimension from
the KL. It is not the ELBO, so it is only a diagnostic. It made every metric
worse. The low latent usage (one to two dimensions carry information) is the
correct β = 1 optimum, not an optimisation failure.

**Validation** always reports the β = 1 negative ELBO (`val_neg_elbo`),
whatever schedule trained the model, and checkpoints are selected on it.

### 5.3 Logging and safety

- Per-stage loss, NLL, KL, ELBO and active latent dimensions (KL > 0.1 nats).
- Reduced chi-square, median lifetimes and mean fractions.
- The IRF.
- Ground-truth errors when the store has them. They are logged only and
  never enter the loss, as in LUMOS.

`configure_gradient_clipping` skips any step whose gradients are non-finite,
from LUMOS, then clips the norm to 1. AdamW at 1e-3 with a cosine schedule
(to 1% of the rate) over the full run. There is no early stopping under
cosine, following LUMOS's reasoning that stopping early leaves the model at
whatever learning rate it had reached.

---

## 6. Data (`data.py`)

- `read_imspector_tiff`: reads the FLUTE OME-TIFFs. The delay axis is stored
  as whichever OME axis precedes Y and X (here Z), labelled "TCSPC T" in
  ImSpector's AxesLabels. The bin width is that axis's physical size. The
  histogram spans exactly one period (80.11 MHz).
- `spatial_bin`: k×k block sums, applied before storing, to trade resolution
  for photons.
- `assign_splits`: random pixel-level train / val / test.
- `write_store` / `open_store`: the Zarr layout (counts [sample, channel,
  time], split, image, y, x, optional `gt_*`, attrs with bin width and image
  shapes). Unlike LUMOS's layout there are no `lengths`: every histogram has
  all 56 bins.
- `irf_t0_guess`: the steepest rise of the summed histogram, the IRF
  starting point when no calibration is given.
- `neighbourhood_sum`: k×k box sums by 2D cumulative sums over each image
  grid, with absent pixels as zero. It serves both the spatial encoder's
  context and the binned baseline. Checked against brute force.
- `neighbour_pairs`: the 4-connected neighbour pairs, for the TV baseline.
- `PixelDataset`: tensors optionally placed on the GPU up front.
  `__getitems__` gathers a whole batch with one indexing call; per-pixel
  indexing would be slow once the data sits on a GPU. LUMOS does the same.
- `FlimDataModule`: fits on train + test by default (transductive, as in
  LUMOS: fitting is unsupervised, so using the test pixels' histograms is
  legitimate) and validates on val. It adds the neighbourhood context when
  `spatial` is set.

---

## 7. Training (`train.py`)

A `DEFAULTS` dict becomes command-line flags, as in LUMOS. `pick_device`
chooses CUDA, then MPS, then CPU; Lightning gets the matching accelerator and
the store is preloaded to the GPU. Two checkpoints are kept: `best.ckpt` on
the val ELBO and `last.ckpt`. Validation runs every N optimiser steps when
`val_check_steps` is set. `--resume` continues from `last.ckpt` with the
optimiser and schedule restored. `--wandb` switches logging to Weights &
Biases.

---

## 8. Inference (`predict.py`)

`run_model` draws 20 samples from q(z | x) per pixel, decodes each, and takes
the per-pixel median of every parameter. The spread over draws is its
uncertainty, and the mean of the draws' histograms is the reconstruction.

**Why not decode the posterior mean.** The decoder is nonlinear and the
posterior is wide in the dimensions the data barely constrains. Decoding the
mean latent is not the same as the posterior's view of the parameters. On
the same checkpoint it put the long lifetime 22% high and tripled the
fraction error. The median over draws removed the bias. LUMOS's
`sample_posterior` makes sampling its primary inference path for the same
reason.

`summarise` prints per-image medians and draws maps (photons, τ_amp, each τ,
the short component's amplitude fraction, reduced chi-square).

---

## 9. Calibration (`calibrate.py`)

A reference dye of known lifetime (fluorescein, 4.2 ns) is summed into one
high-count histogram. It is fitted as a single exponential plus background
with a Gaussian IRF, once with τ fixed and once free. The fixed fit's IRF can
initialise or pin the model's. The free fit is the honesty check: the embryo
reference gives 3.99 ns, but the hMSC reference gives 3.59 ns and shows
bin-to-bin ripple, so its fixed fit gives the wrong IRF, and for hMSC the
IRF is learned instead. Calibration runs on the CPU in float64 (MPS has no
float64, and it is one histogram).

---

## 10. Synthetic data (`synthetic.py`)

Both generators simulate **photon by photon**: pulse jitter plus exponential
delay, wrapped onto the period, then histogrammed. They never call the
analytical model, so agreement is evidence that the model is right, not that
it agrees with itself.

- `simulate`: smooth random maps of two lifetimes, amplitude fraction,
  background and intensity. This is the easy case, and it favours methods
  that smooth spatially.
- `simulate_realistic`:
  - Voronoi "cells", each with its own lifetimes and bound fraction, gentle
    variation inside and sharp edges between.
  - Empty background regions and strongly varying brightness.
  - NADH-like lifetimes.
  - An IRF with an exponential diffusion tail on a quarter of the photons.
    No method models the tail, so all are misspecified the way real data
    is.
  - It records which pixels lie on region boundaries, so edges can be scored
    separately.

---

## 11. Baselines (`baseline.py`)

All baselines share the VAE's physics, likelihood and parametrisation. They
differ only in how pixels are tied together. `PixelFitter` holds the counts,
the per-pixel parameter matrix θ (rate increments, fraction logits,
background-spectrum logits) and the shared IRF. The loss is separable over
pixels apart from the IRF and any penalty, so one Adam optimiser over the
whole matrix is one optimiser per pixel.

- **Per-pixel MLE**: each pixel alone, with the IRF fitted jointly. It shows
  what the data alone can say.
- **Global analysis**: one set of lifetimes for the image, with fractions per
  pixel. This is the standard low-count remedy, and it is right when
  lifetimes truly are shared.
- **Binned MLE**: each pixel fitted on its 3×3 sum. This is the other
  standard remedy: nine times the photons, at the cost of blurring edges.
- **Total variation (TV)**: per-pixel fits plus
  λ·Σ_neighbours sqrt(|θ_i − θ_j|² + δ²). TV is used rather than a quadratic
  penalty because it favours piecewise-constant maps and so keeps edges. The
  strength λ is chosen without ground truth: each pixel's photons are split
  80/20 at random, each λ is fitted on the 80% and scored on predicting the
  20%, and the winner is refitted on everything with λ scaled by 1/0.8 (the
  likelihood grows with the photon count; the penalty does not).
- **Hierarchical empirical Bayes (EB)**: θ_p ~ N(m, S) for every pixel, with
  m and S (full covariance) estimated from the image by EM. The E-step finds
  each pixel's MAP under the current prior, plus a Laplace approximation of
  its posterior covariance. The Hessian comes from finite differences of the
  gradient, all pixels at once, which works because the objective is
  separable. The M-step sets m = mean θ_p and
  S = mean[(θ_p − m)(θ_p − m)ᵀ + Σ_p]. This is the fair non-neural competitor:
  like the VAE, it learns a population prior from the image, but a Gaussian
  one in parameter space rather than the decoder's learned map from a
  Gaussian latent.

---

## 12. Evaluation protocol

**Held-out photons.** Each pixel's photons are split at random (binomial
thinning). For Poisson data the two parts are exactly two independent
acquisitions of the same pixel. Every method fits the first part, and is
scored by the negative log-likelihood per photon of the second part under its
fitted shape. This needs no ground truth, works on real data, and rewards
exactly what matters: predicting what the sample would emit. It is the FLIM
counterpart of LUMOS validating by extrapolation to unseen frames. Here the
unseen data is photons, not time points.

**Ground truth** (synthetic only), on held-out non-empty pixels, split into
interior and boundary pixels:
- τ_amp bias, IQR and correlation
- long-lifetime bias, IQR and correlation
- short-lifetime IQR
- amplitude fraction absolute error

**Real-data reference.** Per-pixel MLE on all photons of the embryo. It is
noisy and shares photons with the thinned MLE, so it flatters MLE; the
held-out score is the unbiased one.

---

## 13. Part-by-part comparison with lumos-vae

| Part | lumos-vae (Raman) | lumos-flim | Same? |
|---|---|---|---|
| Data | [N, W, T]: 1024 wavenumbers × frames over seconds | [N, C, T]: detection channels × 56 delay bins in one 12.5 ns period | analogous |
| What varies in time | photobleaching, once | fluorescence decay, every pulse, periodic | differs |
| Constant term | static Raman spectrum: the target | background: a nuisance | same role in the model, opposite importance |
| Decaying terms | fluorophores, per-sample rates and abundances | lifetime components, per-pixel rates and fractions | same |
| Global components | MoG fluorophore bases, instrument blur | emission spectra (softmax), IRF (t0, σ) | same idea |
| Interval integration | (1 − e^{−λT})/λ per frame | exact bin integration + periodic sum | same principle, exact here |
| Amplitude parametrisation | effective amplitudes ã | photon fractions, conditioned on the total | same principle |
| Rates | softplus + λ_min, unordered | cumulative softplus + rate_min, ordered | differs (no spectra to anchor labels in single-channel FLIM) |
| Raman / static head | Voigt peaks blurred by the instrument | one fraction (and a spectrum per pixel if spectral) | differs (a background has no line shape) |
| Normalisation | one global std | per-pixel shape plus log total | differs (photon counts span orders of magnitude) |
| Encoder | 2D CNN over [W, T], linear positional encodings | 1D circular CNN over delay, phase encodings, optional neighbourhood context | differs to fit the axis |
| Decoder | separate trunks per parameter group | one shared trunk | minor |
| Noise | learned var = α·μ + β (Gaussian) | exact Poisson / multinomial | same purpose, exact here |
| KL | per element, divided by element count | per pixel, averaged over batch | same ratio: identical ELBO |
| β | 1 | 1, optional warm-up | same |
| Training window | early frames; validate by extrapolating | all bins; validate on held-out pixels and photons | differs: TCSPC records every bin at once |
| Inference | ensemble posterior sampling | posterior median over 20 draws, the default | same |
| Loop | Lightning, AdamW, cosine, non-finite skip, transductive | same | same |
| Hardware | CUDA / MPS preload | CUDA / MPS / CPU, preload, portable special functions | same, plus MPS kernels |

---

## 14. Known limitations

- **Gaussian IRF.** Real IRFs have tails. The realistic benchmark measures
  what this costs. A measured IRF, or an exponentially modified Gaussian IRF
  (also closed form), would be the fix.
- **One IRF per image.** Scanning systems can drift in t0 across the field.
- **No pile-up or dead-time correction.** Fine at FLUTE's count rates.
- **Two components by default.** Real samples may need three.
- **Inference bias at very low counts on real tissue** (+7% τ_amp at 65
  photons on the embryo): the population prior leans towards the typical
  pixel.
