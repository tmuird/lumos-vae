"""Device selection shared by training, inference and the baselines."""

import torch


def pick_device(name: str = "auto") -> torch.device:
    """CUDA if present, then Apple's MPS, then CPU. An explicit name wins."""
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
