"""Regression inputs (paper 2602.15946 §1.1, §3.1, Eq 4)."""

from smartpix_bench.common.features import add_noise, digitize

SLICES_2 = (0, 19)  # (16, 16, 2) configs c2-c4
SLICES_20 = tuple(range(20))  # config c1
NOISE_SIGMA = 80.0  # electrons, SoftQuantize stage only
TRANSFORMER_THRESHOLDS = (248.0, 668.0, 1663.0)  # §1.3: SoftQuantize init and baseline digitization

__all__ = ["SLICES_2", "SLICES_20", "NOISE_SIGMA", "TRANSFORMER_THRESHOLDS", "add_noise", "digitize"]
