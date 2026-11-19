"""Regression inputs (paper 2602.15946 §1.1, §3.1, Eq 4)."""

import numpy as np

from smartpix_bench.common.features import add_noise, digitize

SLICES_2 = (0, 19)  # (16, 16, 2) configs c2-c4
SLICES_20 = tuple(range(20))  # config c1
NOISE_SIGMA = 80.0  # electrons, SoftQuantize stage only
TRANSFORMER_THRESHOLDS = (248.0, 668.0, 1663.0)  # §1.3: SoftQuantize init and baseline digitization



def standardize_stats(x: np.ndarray) -> list[float]:
    """Mean and std of the nonzero pixels (upstream OptimizedDataGenerator_v3 `to_standardize`)."""
    v = x[x != 0]
    return [float(v.mean(dtype=np.float64)), float(v.std(dtype=np.float64))]


def standardize(x: np.ndarray, stats) -> np.ndarray:
    """In place: nonzero pixels -> (x - mean) / std, zeros stay zero."""
    nz = x != 0
    np.subtract(x, stats[0], out=x, where=nz)
    np.divide(x, stats[1], out=x, where=nz)
    return x


__all__ = ["SLICES_2", "SLICES_20", "NOISE_SIGMA", "TRANSFORMER_THRESHOLDS", "add_noise", "digitize", "standardize",
           "standardize_stats"]
