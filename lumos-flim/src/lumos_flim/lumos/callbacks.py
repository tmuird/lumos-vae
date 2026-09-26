"""Validation logging against synthetic ground truth, as in LUMOS.

A no-op for real data. Ground truth is used only here and never in a loss.
"""

import numpy as np
import pytorch_lightning as pl
import torch

from lumos_flim.lumos.predict import run


class DecompositionEvalCallback(pl.Callback):
    """Log median lifetime and amplitude-fraction errors on the val pixels."""

    def __init__(self, datamodule):
        self.dm = datamodule

    def on_validation_epoch_end(self, trainer, pl_module):
        gt = self.dm.gt
        if not gt or "gt_tau" not in gt:
            return
        raw = self.dm.val_ds.data.numpy()
        was_training = pl_module.training
        with torch.no_grad():
            res = run(pl_module, raw, self.dm.full_ds.std)
        pl_module.train(was_training)
        order = np.argsort(-gt["gt_tau"], axis=1)
        gt_tau = np.take_along_axis(gt["gt_tau"], order, 1)
        gt_alpha = np.take_along_axis(gt["gt_alpha"], order, 1)
        for i in range(gt_tau.shape[1]):
            rel = np.abs(res["tau"][:, i] - gt_tau[:, i]) / gt_tau[:, i]
            pl_module.log(f"val_gt_tau{i}_rel_err", float(np.median(rel)))
            pl_module.log(f"val_gt_alpha{i}_abs_err",
                          float(np.median(np.abs(res["alpha"][:, i] - gt_alpha[:, i]))))
