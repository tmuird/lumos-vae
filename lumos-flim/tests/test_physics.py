import numpy as np
import pytest
import torch

from lumos_flim.physics import (
    amplitude_fractions,
    erfcx_portable as erfcx,
    log_ndtr_portable as log_ndtr,
    ndtr_portable as ndtr,
    decay_histograms,
    expected_counts,
    irf_histogram,
)
from lumos_flim.vae_module import FlimModule, poisson_half_deviance

N_BINS, PERIOD = 56, 12.483
BIN = PERIOD / N_BINS


def monte_carlo(tau, t0, sigma, n, rng):
    t = rng.normal(t0, sigma, n) + rng.exponential(tau, n)
    return np.histogram(np.mod(t, PERIOD), bins=N_BINS, range=(0, PERIOD))[0]


@pytest.mark.parametrize("tau,t0,sigma", [
    (4.2, 1.1, 0.12),   # long, wraps round the period
    (0.4, 0.3, 0.10),   # short
    (0.05, 12.3, 0.15), # IRF straddles the end of the window
    (30.0, 5.0, 0.2),   # much longer than the period
])
def test_matches_photon_simulation(tau, t0, sigma):
    rng = np.random.default_rng(0)
    n = 2_000_000
    h = monte_carlo(tau, t0, sigma, n, rng)
    p = decay_histograms(torch.tensor([1 / tau], dtype=torch.float64),
                         torch.tensor(t0, dtype=torch.float64),
                         torch.tensor(sigma, dtype=torch.float64), N_BINS, BIN)[0].numpy()
    mu = n * p
    ok = mu > 5
    chi2 = ((h[ok] - mu[ok]) ** 2 / mu[ok]).sum()
    # chi2 with ~56 dof: 1.6x the dof is far out in the tail for a correct model
    assert chi2 / ok.sum() < 1.6
    assert h[~ok].sum() < 20


def test_rows_sum_to_one_and_gradients_finite():
    rates = torch.logspace(-3, 4, 40, dtype=torch.float32).requires_grad_()
    t0 = torch.tensor(1.0, requires_grad=True)
    sigma = torch.tensor(0.1, requires_grad=True)
    p = decay_histograms(rates, t0, sigma, N_BINS, BIN)
    assert torch.allclose(p.sum(-1), torch.ones(40), atol=1e-4)
    (p * torch.randn_like(p)).sum().backward()
    for g in (rates.grad, t0.grad, sigma.grad):
        assert torch.isfinite(g).all()


def test_zero_lifetime_limit_is_the_irf():
    p = decay_histograms(torch.tensor([1e5]), 2.0, 0.15, N_BINS, BIN)[0]
    assert torch.allclose(p, irf_histogram(2.0, 0.15, N_BINS, BIN), atol=1e-4)


def test_expected_counts_conserve_photons():
    B, Fn, C = 5, 2, 3
    decays = decay_histograms(torch.rand(B, Fn) + 0.2, 1.0, 0.1, N_BINS, BIN)
    fr = torch.softmax(torch.randn(B, Fn + 1), -1)
    totals = torch.rand(B) * 1000
    mu = expected_counts(totals, fr[:, :-1], fr[:, -1], decays,
                         torch.softmax(torch.randn(Fn, C), -1), torch.softmax(torch.randn(B, C), -1))
    assert mu.shape == (B, C, N_BINS)
    assert torch.allclose(mu.sum((1, 2)), totals, rtol=1e-4)


def test_amplitude_fractions():
    # Equal amplitudes: photons split in proportion to lifetime.
    rates = torch.tensor([[1 / 3.0, 1 / 0.5]])
    photon = torch.tensor([[3.0, 0.5]]) / 3.5
    assert torch.allclose(amplitude_fractions(photon, rates), torch.tensor([[0.5, 0.5]]))


@pytest.mark.parametrize("channels", [1, 4])
def test_model_step(channels):
    module = FlimModule(n_bins=N_BINS, bin_width=BIN, n_channels=channels, latent_dim=4,
                        hidden_dim=16, decoder_dim=16, conv_channels="8,8")
    x = torch.poisson(torch.full((6, channels, N_BINS), 20.0))
    out = module.model(x)
    assert out["expected"].shape == x.shape
    assert torch.allclose(out["expected"].sum((1, 2)), x.sum((1, 2)), rtol=1e-4)
    tau = 1 / out["rates"]
    assert (tau[:, 0] >= tau[:, 1]).all()  # ordered, longest first
    loss = poisson_half_deviance(x, out["expected"]).mean()
    loss.backward()
    assert torch.isfinite(module.model.irf_t0.grad)



def test_special_functions_match_torch():
    # The portable versions used on MPS, checked against torch.special.
    x = torch.linspace(-37, 37, 20001, dtype=torch.float64)
    ref_log = torch.special.log_ndtr(x)
    assert (log_ndtr(x) - ref_log).abs().max() < 2e-7
    assert ((ndtr(x) - ref_log.exp()).abs() / ref_log.exp()).max() < 2e-7
    xp = torch.linspace(0, 1e4, 20001, dtype=torch.float64)
    assert ((erfcx(xp) - torch.special.erfcx(xp)).abs() / torch.special.erfcx(xp)).max() < 2e-7


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_runs_on_device(device):
    """Physics and a full training step on each available accelerator, matching CPU."""
    rates = torch.logspace(-1, 2, 12)
    ref = decay_histograms(rates, 1.0, 0.1, N_BINS, BIN)
    got = decay_histograms(rates.to(device), 1.0, 0.1, N_BINS, BIN).cpu()
    assert torch.allclose(got, ref, atol=1e-6)

    torch.manual_seed(0)
    module = FlimModule(n_bins=N_BINS, bin_width=BIN, latent_dim=4, hidden_dim=16,
                        decoder_dim=16, conv_channels="8,8").to(device)
    x = torch.poisson(torch.full((6, 1, N_BINS), 20.0)).to(device)
    out = module.model(x, sample=False)
    loss = poisson_half_deviance(x, out["expected"]).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(module.model.irf_t0.grad)


@pytest.mark.parametrize("tau,tail_tau", [(0.4, 0.25), (0.3, 0.3), (0.303, 0.3), (3.0, 0.25)])
def test_tailed_irf_matches_photon_simulation(tau, tail_tau):
    """Gaussian IRF with an exponential tail on 25% of photons, including a
    tail rate equal to the decay rate, where the closed form is 0/0."""
    rng = np.random.default_rng(1)
    n, w = 2_000_000, 0.25
    t = rng.normal(1.0, 0.1, n) + rng.exponential(tau, n)
    tail = rng.random(n) < w
    t[tail] += rng.exponential(tail_tau, tail.sum())
    h = np.histogram(np.mod(t, PERIOD), bins=N_BINS, range=(0, PERIOD))[0]
    p = decay_histograms(torch.tensor([1 / tau], dtype=torch.float64), 1.0, 0.1, N_BINS, BIN,
                         tail_weight=torch.tensor(w, dtype=torch.float64),
                         tail_rate=torch.tensor(1 / tail_tau, dtype=torch.float64))[0].numpy()
    mu = n * p
    ok = mu > 5
    assert abs(p.sum() - 1) < 1e-6
    assert ((h[ok] - mu[ok]) ** 2 / mu[ok]).sum() / ok.sum() < 1.6
