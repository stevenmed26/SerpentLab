# python/trainer/common/obs.py
from __future__ import annotations
import numpy as np


def one_hot_obs(obs: np.ndarray, num_channels: int = 4) -> np.ndarray:
    """
    obs: (H, W) int grid with values in {0..num_channels-1}
    returns: (C, H, W) float32
    """
    h, w = obs.shape
    out = np.zeros((num_channels, h, w), dtype=np.float32)
    for c in range(num_channels):
        out[c] = (obs == c)
    return out
