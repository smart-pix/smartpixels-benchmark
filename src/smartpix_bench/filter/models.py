"""Filter reference models (paper 2310.02474 §3.2, §3.3) and their inputs.

Upstream: jieunyoo/directional-pixel-detectors@4618ab2 — M2 `smartPixelsPaper/fullPrecisionBaseline/
SimpleNN_ylocalyprofile.ipynb`, M3 `smartPixelsPaper/timeSlices/neural.ipynb`, QM2
`qkeras_stuff/SimpleNN_ylocalyprofileQkeras1.ipynb`; M1 is built from the paper text (no upstream code).
"""

import numpy as np

from smartpix_bench.common.features import standardizer

# name: (input arrays from the cache, parameter count, track, auxiliary)
MODELS = {
    "M1": (["ysize", "y_local"], 771, "open", False),
    "M2": (["prof", "y_local"], 2307, "open", False),
    "M3": (["prof_ts", "y_local"], 83331, "open", False),
    "QM2": (["prof_adc", "y0_6bit"], 2819, "asic", False),
    "QM2_w4a8": (["prof_adc", "y0_6bit"], 2819, "asic", True),
}
AUXILIARY = {m for m, spec in MODELS.items() if spec[3]}


def raw_inputs(model: str, arrays: dict) -> list[np.ndarray]:
    """Unstandardized model inputs; M3 keeps (n, 8, 13) profiles and y0 as separate inputs."""
    a, b = (np.asarray(arrays[k], dtype=np.float32) for k in MODELS[model][0])
    if model == "M3":
        return [a, b[:, None]]
    return [np.column_stack([a, b])]


def fit_scaler(inputs: list[np.ndarray]) -> list[dict]:
    """Train-set mean/std per last-axis feature (M3 profiles pool the 8 slices, as upstream)."""
    out = []
    for x in inputs:
        m, s = standardizer(x.reshape(-1, x.shape[-1]))
        out.append({"mean": m.tolist(), "std": s.tolist()})
    return out


def apply_scaler(inputs: list[np.ndarray], scaler: list[dict]) -> list[np.ndarray]:
    xs = [(x - np.float32(s["mean"])) / np.float32(s["std"]) for x, s in zip(inputs, scaler)]
    return [xs[0][..., None], xs[1]] if xs[0].ndim == 3 else xs


def build(model: str):
    from tensorflow import keras
    L = keras.layers
    if model in ("M1", "M2"):
        x = x_in = keras.Input((2 if model == "M1" else 14,))
        x = L.Dense(128, activation="relu")(x)
        return keras.Model(x_in, L.Dense(3, activation="softmax")(x), name=model)
    if model == "M3":
        img, y0 = keras.Input((8, 13, 1)), keras.Input((1,))
        x = L.Conv2D(16, (3, 3), activation="relu")(img)
        x = L.Conv2D(64, (3, 3), activation="relu")(x)
        x = L.Concatenate()([L.Flatten()(x), y0])
        x = L.Dropout(0.1)(L.Dense(32, activation="relu")(x))
        return keras.Model([img, y0], L.Dense(3, activation="softmax")(x), name=model)
    from qkeras import QActivation, QDense, quantized_bits
    w, a = (5, 10) if model == "QM2" else (4, 8)
    q = quantized_bits(w, 0, alpha=1)
    x = x_in = keras.Input((14,))
    x = QDense(128, kernel_quantizer=q, bias_quantizer=q, name="dense1")(x)
    x = L.BatchNormalization()(x)
    x = QActivation(f"quantized_relu({a},0)", name="relu1")(x)
    x = QDense(3, kernel_quantizer=q, bias_quantizer=q, name="dense2")(x)
    return keras.Model(x_in, L.Activation("softmax")(x), name=model)


def descriptor(model: str, n_params: int) -> dict:
    """What the ASIC-track check reads (`filter.score.asic_violations`)."""
    _, _, track, _ = MODELS[model]
    d = {"model": model, "track": track, "n_params": int(n_params)}
    if track == "asic":
        from smartpix_bench.filter import features as ff
        d.update(adc_thresholds=[int(t) for t in ff.ADC_THRESHOLDS], y0_bits=int(np.log2(ff.Y0_BINS)))
    return d
