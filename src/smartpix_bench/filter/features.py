"""Filter inputs (paper 2310.02474 §3.1, §3.3): y-profile, y-size, 2-bit ADC, 6-bit y0."""

import numpy as np

from smartpix_bench.common.features import digitize

PROFILE_SLICE = 19  # cumulative charge after 4 ns (20 x 200 ps)
TIMING_SLICES = tuple(range(8))  # Model 3: first eight 200 ps intervals
ADC_THRESHOLDS = (400.0, 1600.0, 2400.0)  # Table 3, electrons
Y0_HALF_RANGE = 8.0  # mm; 16 mm sensor
Y0_BINS = 64  # 6 bits of 250 µm


def y_profile(pixels) -> np.ndarray:
    """Charge summed over x per pixel row: (..., 13, 21) -> (..., 13)."""
    return np.asarray(pixels).sum(axis=-1)


def y_size(pixels) -> np.ndarray:
    """Number of pixel rows with nonzero net charge: (..., 13, 21) -> (...)."""
    return (y_profile(pixels) != 0).sum(axis=-1)


def adc_2bit(pixels) -> np.ndarray:
    """Per-pixel 2-bit ADC code, applied before the row sum."""
    return digitize(pixels, ADC_THRESHOLDS)


def y0_6bit(y_local_mm) -> np.ndarray:
    """Bin index 0..63 of y0 in 64 right-closed 250 µm bins over (-8, 8] mm, as the upstream converter bins; out-of-range values clip to the edge bins."""
    width = 2 * Y0_HALF_RANGE / Y0_BINS
    idx = np.ceil((np.asarray(y_local_mm, dtype=np.float64) + Y0_HALF_RANGE) / width) - 1
    return np.clip(idx, 0, Y0_BINS - 1).astype(np.float32)
