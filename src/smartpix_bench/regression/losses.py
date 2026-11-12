"""Regression losses (paper 2602.15946 §3.2; upstream smart-pixels-ml-2bit-Synth loss.py).

Output layouts: Max = 14 (means at 0,2,4,6; scale-tril diagonal at 1,3,5,7 as max(., 0) + 1e-9; strict
lower triangle row-major at 8..13); Full = 8 (means 0..3; sigma = softplus(4..7) + 1e-9); Slim = 3 means, summed squared error.
"""

import math

import tensorflow as tf

MINVAL, MAXVAL = 1e-9, 1e9


def max_scale_tril(p):
    d = MINVAL + tf.maximum(p[:, 1:8:2], 0.0)
    o = p[:, 8:14]
    z = tf.zeros_like(d[:, 0])
    rows = [[d[:, 0], z, z, z], [o[:, 0], d[:, 1], z, z], [o[:, 1], o[:, 2], d[:, 2], z], [o[:, 3], o[:, 4], o[:, 5], d[:, 3]]]
    return tf.stack([tf.stack(r, axis=-1) for r in rows], axis=1)


def mvn_tril_nll(y, p):
    """Summed NLL of a 4-d Gaussian with scale-tril L; the likelihood is clipped to [1e-9, 1e9] as upstream."""
    y = tf.cast(y, p.dtype)
    L = max_scale_tril(p)
    r = tf.linalg.triangular_solve(L, (y - p[:, 0:8:2])[..., None], lower=True)[..., 0]
    nll = 0.5 * tf.reduce_sum(r * r, -1) + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), -1) + 2 * math.log(2 * math.pi)
    return tf.reduce_sum(tf.clip_by_value(nll, -math.log(MAXVAL), -math.log(MINVAL)))


def full_sigma(p):
    return tf.nn.softplus(p[:, 4:8]) + MINVAL


def diag_nll(y, p):
    """Summed NLL of independent Gaussians with sigma = softplus + 1e-9."""
    y = tf.cast(y, p.dtype)
    s = full_sigma(p)
    z = (y - p[:, 0:4]) / s
    return tf.reduce_sum(0.5 * z * z + tf.math.log(s) + 0.5 * math.log(2 * math.pi))


def sse(y, p):
    """Summed over the batch like the NLLs (upstream slim_sse_loss); a batch mean lets the regularizers zero the weights."""
    return tf.reduce_sum(tf.square(tf.cast(y, p.dtype) - p))


LOSSES = {"Max": mvn_tril_nll, "Full": diag_nll, "Slim": sse}
