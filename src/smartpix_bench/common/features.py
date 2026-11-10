"""Feature primitives shared by both tasks."""

import numpy as np


def digitize(x, thresholds) -> np.ndarray:
    """ADC code j for T_{j-1} <= x < T_j (regression paper Eq 4; filter Table 3), as float32.

    `thresholds` must be strictly increasing; codes run 0..len(thresholds).
    """
    t = np.asarray(thresholds, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.diff(t) > 0):
        raise ValueError(f"thresholds must be strictly increasing, got {thresholds}")
    return np.searchsorted(t, x, side="right").astype(np.float32)


def add_noise(x, rng: np.random.Generator, sigma: float = 80.0) -> np.ndarray:
    """x + N(0, sigma) electrons drawn independently per pixel per slice."""
    return (x + rng.normal(0.0, sigma, size=np.shape(x))).astype(np.float32)


def standardizer(x) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) over axis 0; zero std maps to 1 (sklearn StandardScaler convention)."""
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    return mean, np.where(std == 0, 1.0, std)
