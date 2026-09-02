"""Validation-time logging callback.

Runs the model on a few validation samples and logs how well the recovered
Raman spectrum matches the ground truth. Ground truth is used only for this
logging and only when the store provides it; training is unsupervised and never
sees it.
"""

import numpy as np
import pytorch_lightning as pl

from lumos.predict import sample_posterior


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class DecompositionEvalCallback(pl.Callback):
    """Log recovered-vs-ground-truth Raman similarity on a few val samples.

    A no-op when the datamodule carries no ground truth (real data).

    Parameters
    ----------
    datamodule : ZarrDataModule
    n_samples : int
        Number of val samples to evaluate, evenly spaced.
    """

    def __init__(self, datamodule, n_samples: int = 4):
        self.dm = datamodule
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        gt = getattr(self.dm, "gt", None)
        if gt is None or "gt_raman" not in gt:
            return

        val_ds = self.dm.val_ds
        gt_raman = gt["gt_raman"]  # [n_val, W]
        n = min(self.n_samples, len(val_ds), len(gt_raman))
        if n == 0:
            return
        chosen = np.linspace(0, len(val_ds) - 1, n, dtype=int)

        cosines = []
        for idx in chosen:
            sample = val_ds[int(idx)][0].unsqueeze(0)  # [1, W, T]
            ensemble = sample_posterior(
                pl_module,
                sample,
                n_predictions=1,
            )
            pred_raman = ensemble["raman"].mean(axis=0)  # [W]
            cosines.append(_cosine(pred_raman, gt_raman[int(idx)]))

        pl_module.log("val_raman_cosine", float(np.mean(cosines)), prog_bar=True)
