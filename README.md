# LUMOS

Physics-informed VAE that separates the static Raman spectrum from decaying
fluorescence in photobleaching time-series spectroscopy. The neural network
predicts the parameters of an analytical forward model - per-sample decay rates,
abundances and a Raman spectrum - rather than reconstructing the signal directly.
Training is unsupervised: the model is fit on early frames and validated on its
ability to extrapolate the rest.

## Install

```bash
pip install -e .
```

Python >= 3.10 with PyTorch.

If import fails with `GLIBCXX_3.4.31 not found` (an `optree`/`libstdc++` mismatch
in the conda environment, not a code fault), put the environment's own
`libstdc++` on the loader path, replacing `<env>` with your environment name:

```bash
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" -n <env>
conda deactivate && conda activate <env>
```

## Train

```bash
python -m lumos.train --data path/to/dataset.zarr
python -m lumos.train --data dataset.zarr --basis_mode mog --learning_rate 5e-4
```

The only input is a processed Zarr store. It must contain a `time_series`
`[N, W, T]` variable, a `split` variable (`train`/`val`/`test`), `lengths`,
`labels`, and `wavenumber` / `time` coordinates. Ground-truth variables
(`gt_raman`, `decay_rates_gt`, ...), when present, are used only for logging and
never enter any loss.

Checkpoints are written to `checkpoints/<run_name>/`, monitored by
`val_recon_loss`. Logs go to `logs/` via `CSVLogger`; pass `--wandb` to log to
Weights & Biases instead.

## The physics

The CCD integrates over each frame of duration `T`:

    S_n(nu) = S(nu)*T + sum_i w_i * B_i(nu) * e^(-lambda_i t_n) * (1 - e^(-lambda_i T)) / lambda_i

- `S(nu)` - static Raman spectrum (what we recover)
- `w_i`, `lambda_i` - per-sample abundance and decay rate of fluorophore `i`
- `B_i(nu)` - global fluorophore basis spectra

Decay rates are per-sample because the same fluorophore bleaches at different
rates depending on the local environment. The integration factor
`(1 - e^(-lambda*T))/lambda` matters: fast components contribute almost nothing to the
measured signal because they bleach within the frame.

## Layout

| Module | Role |
|--------|------|
| `physics.py` | Analytical forward model (three reconstruction variants) |
| `vae.py` | `VAE` module - encoder, decoder, physics forward pass, basis modes |
| `vae_module.py` | Lightning module - loss, optimiser, train/val steps |
| `dataset.py` | `BleachingDataset` - std-only normalisation |
| `datamodule.py` | `ZarrDataModule` - loads the processed store |
| `predict.py` | Inference: ensemble posterior sampling and decomposition |
| `callbacks.py` | Validation logging of recovered-vs-GT Raman similarity |
| `train.py` | CLI entry point |

Basis modes: `mog` (mixture of Gaussians, default) and `polynomial`. The
`dictionary` mode needs an external fluorophore dictionary and is not included.

Datasets are prepared offline into the Zarr layout described above.
