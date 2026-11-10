"""Regression reference models (paper 2602.15946 §3.2, Fig 5).

Upstream: Conv2D and MLP from smart-pixels-ml-2bit-Synth@4983446 `models/{conv2d,mlp_encoder}_model*.py`,
Conv1D from smart-pixels-ml@1fb333f `paper-trainings/{full_1D/Full1D_model2,slim_1D/Slim1D_model}.py`.
Every model carries L1L2(0.01) kernel (as upstream writes it, so Keras l1 = 0.01, l2 = 0) and L2(0.01) activity
regularization (none on the output activity).
Quantized (config c4): conv weights fixed<4,1>, dense fixed<8,1>; conv-stage tanh 4-bit, other
activations fixed<8,1>.
"""

N_OUT = {"Max": 14, "Full": 8, "Slim": 3}
TARGETS = {"Max": ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"], "Full": ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"],
           "Slim": ["x-midplane", "y-midplane", "cotBeta"]}
# name: parameter count at 2 input slices
MODELS = {"Max_Conv2D": 1869, "Full_Conv2D": 1767, "Full_Conv1D": 2734, "Full_MLP": 2264,
          "Slim_Conv2D": 1682, "Slim_Conv1D": 2649, "Slim_MLP": 2179}
CONFIGS = ("c1", "c2", "c3", "c4")


def split_name(model: str) -> tuple[str, str]:
    head, arch = model.split("_")
    return head, arch


def build(model: str, n_slices: int = 2, quantized: bool = False, softquantize=None):
    """Keras model; `softquantize` = initial thresholds puts GaussianNoise(80 e) + SoftQuantize in front."""
    import tensorflow as tf
    from tensorflow.keras import layers as L

    from smartpix_bench.regression.features import NOISE_SIGMA
    from smartpix_bench.regression.softquantize import SoftQuantize

    head, arch = split_name(model)
    reg = dict(kernel_regularizer=tf.keras.regularizers.L1L2(0.01), activity_regularizer=tf.keras.regularizers.L2(0.01))
    if quantized:
        from qkeras import QActivation, QConv1D, QConv2D, QDense, QSeparableConv2D, quantized_bits
        q4, q8 = quantized_bits(4, 0, alpha=1), quantized_bits(8, 0, alpha=1)

        def dense(n, x, activity=True):
            r = reg if activity else {"kernel_regularizer": reg["kernel_regularizer"]}
            return QDense(n, kernel_quantizer=q8, bias_quantizer=q8, **r)(x)

        def act(kind, x, bits=8):
            spec = f"quantized_tanh({bits},0,1)" if kind == "tanh" else f"quantized_relu({bits},0)"
            return QActivation(spec)(x)

        def conv1d(x):
            return QConv1D(5, 3, kernel_quantizer=q4, bias_quantizer=q4, bias_regularizer=reg["kernel_regularizer"], **reg)(x)
    else:
        def dense(n, x, activity=True):
            r = reg if activity else {"kernel_regularizer": reg["kernel_regularizer"]}
            return L.Dense(n, **r)(x)

        def act(kind, x, bits=8):
            return L.Activation(kind)(x)

        def conv1d(x):
            return L.Conv1D(5, 3, bias_regularizer=reg["kernel_regularizer"], **reg)(x)

    x = x_in = L.Input((16, 16, n_slices))
    if softquantize is not None:
        x = L.GaussianNoise(NOISE_SIGMA)(x)
        x = SoftQuantize(softquantize, name="softquantize")(x)
    if arch == "Conv2D":
        sep = dict(depthwise_regularizer=reg["kernel_regularizer"], pointwise_regularizer=reg["kernel_regularizer"],
                   activity_regularizer=reg["activity_regularizer"])
        if quantized:
            x = QSeparableConv2D(5, 3, depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
                                 pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1), bias_quantizer=q4, **sep)(x)
            x = act("tanh", x, 4)
            x = act("tanh", QConv2D(5, 1, kernel_quantizer=q4, bias_quantizer=q4, **reg)(x), 4)
            x = QActivation("quantized_bits(8,0,alpha=1)")(L.AveragePooling2D(3)(x))
        else:
            x = act("tanh", L.SeparableConv2D(5, 3, **sep)(x))
            x = act("tanh", L.Conv2D(5, 1, **reg)(x))
            x = L.AveragePooling2D(3)(x)
        x = L.Flatten()(x)
    else:
        px = L.Reshape((16, n_slices))(L.AveragePooling2D((1, 16))(x))
        py = L.Reshape((16, n_slices))(L.AveragePooling2D((16, 1))(x))
        if arch == "Conv1D":
            x = L.Flatten()(act("tanh", L.Concatenate(axis=1)([conv1d(px), conv1d(py)]), 4))
        else:
            px, py = (act("relu", dense(16, L.Flatten()(p))) for p in (px, py))
            x = act("tanh", dense(16, L.Concatenate(axis=1)([px, py])))
    x = act("tanh", dense(16, x))
    x = act("tanh", dense(16, x))
    return tf.keras.Model(x_in, dense(N_OUT[head], x, activity=False), name=model)


def descriptor(model: str, config: str, n_params: int, thresholds=None) -> dict:
    """What the ASIC-track check reads (`regression.score.asic_violations`); only c4 claims the asic track."""
    d = {"model": model, "config": config, "track": "asic" if config == "c4" else "open", "n_params": int(n_params),
         "slices": [0, 19] if config != "c1" else list(range(20))}
    if config in ("c3", "c4"):
        d["thresholds"] = [float(t) for t in thresholds]
    if config == "c4":
        d.update(weight_bits=8, activation_bits=8)
    return d
