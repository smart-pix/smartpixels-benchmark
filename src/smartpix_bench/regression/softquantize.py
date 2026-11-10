"""SoftQuantize: learned 2-bit ADC thresholds (paper 2602.15946 §3.3, Eqs 4-9).

Forward is the hard Eq-4 digitization (T_{j-1} <= x < T_j); the backward pass uses the gradient of
D_soft(x) = sum_j sigmoid(k (x - T_j) / tau_j). Eq 7 as printed has (T_j - x), which counts down
from 4 and contradicts Eq 9; tau_j = (Delta_j + Delta_{j+1}) / 2 (Eq 8) with tau_2 = Delta_2, since
Delta_3 is undefined. Thresholds T_j = T_min + cumsum(softplus(theta)) (Eqs 5-6) stay strictly ordered.
"""

import math

import numpy as np
import tensorflow as tf

T_MIN = 0.0
K_INIT, K_MAX = 1.0, 67.0


def hard_digitize(x, thresholds):
    """Eq 4: code j for T_{j-1} <= x < T_j."""
    return tf.reduce_sum(tf.cast(x[..., None] >= thresholds, x.dtype), axis=-1)


class SoftQuantize(tf.keras.layers.Layer):
    def __init__(self, initial_thresholds=(248.0, 668.0, 1663.0), **kwargs):
        super().__init__(**kwargs)
        self.initial_thresholds = [float(t) for t in initial_thresholds]

    def build(self, input_shape):
        delta = np.diff(self.initial_thresholds, prepend=T_MIN)
        if not np.all(delta > 0):
            raise ValueError(f"thresholds must be strictly increasing above {T_MIN}: {self.initial_thresholds}")
        # Inverse softplus in the form that does not overflow for delta > ~709.
        theta = delta + np.log(-np.expm1(-delta))
        self.theta = self.add_weight(name="theta", shape=(len(delta),), initializer=tf.constant_initializer(theta))
        self.k = self.add_weight(name="k", shape=(), initializer=tf.constant_initializer(K_INIT), trainable=False)

    def deltas(self):
        return tf.nn.softplus(self.theta)

    def thresholds(self):
        return T_MIN + tf.cumsum(self.deltas())

    def call(self, x):
        d = self.deltas()
        t = T_MIN + tf.cumsum(d)
        tau = tf.concat([(d[:-1] + d[1:]) / 2, d[-1:]], axis=0)
        soft = tf.reduce_sum(tf.sigmoid(self.k * (x[..., None] - t) / tau), axis=-1)
        return soft + tf.stop_gradient(hard_digitize(x, t) - soft)

    def get_config(self):
        return {**super().get_config(), "initial_thresholds": self.initial_thresholds}


def k_at(epoch: int, epochs: int) -> float:
    """Cosine annealing of the temperature from K_INIT (first epoch) to K_MAX (last epoch)."""
    f = epoch / max(epochs - 1, 1)
    return K_MAX - (K_MAX - K_INIT) * (1 + math.cos(math.pi * f)) / 2


class AnnealK(tf.keras.callbacks.Callback):
    def __init__(self, layer: SoftQuantize, epochs: int):
        super().__init__()
        self.layer, self.epochs = layer, epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.layer.k.assign(k_at(epoch, self.epochs))
