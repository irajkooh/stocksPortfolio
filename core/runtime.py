"""Runtime environment helpers: device (cuda/mps/cpu) detection."""
from __future__ import annotations


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


DEVICE: str = detect_device()
