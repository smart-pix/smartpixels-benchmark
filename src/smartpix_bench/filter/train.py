"""Train one filter reference model for one seed.

    python -m smartpix_bench.filter.train --model M2 --seed 0 [--cache-dir C] [--out $SPB_WEIGHTS_DIR/filter]

Writes <out>/<model>_s<seed>.weights.h5, .scaler.json, .history.json and .val.json (validation
metrics, reweighted like test; `score.median_seed` over the seeds' PRIMARY picks the reference checkpoint).
The data subsample and train/val split are fixed (cache.DATA_SEED); the seed drives init, dropout
and batch order only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io
from smartpix_bench.filter import cache, models, score
from smartpix_bench.filter import data as fd

EPOCHS, BATCH, PATIENCE, LR = 200, 1024, 20, 1e-3


def weights_dir() -> Path:
    return io.env_path("WEIGHTS_DIR", "results/weights") / "filter"  # weights/ holds the committed reference checkpoints


def fit(model: str, x_tr, y_tr, x_va, y_va, seed: int, epochs: int = EPOCHS):
    """Adam 1e-3, SCCE, batch 1024, early stopping on val loss (patience 20, best weights restored)."""
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    m = models.build(model)
    m.compile(optimizer=keras.optimizers.Adam(LR), loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=BATCH, shuffle=True,
              callbacks=[es], verbose=2)
    return m, h.history


def predict_proba(m, x) -> np.ndarray:
    return m.predict(x, batch_size=1 << 15, verbose=0)


def evaluate(pt, proba, n_ref) -> dict:
    """Every filter metric: PRIMARY from p0, the rest from argmax decisions."""
    return {**score.rejection_at_efficiency(pt, proba[:, 0], n_ref),
            **score.reweighted_metrics(pt, proba.argmax(1), n_ref)}


def run(model: str, seed: int, cache_dir: Path, out: Path, epochs: int = EPOCHS) -> dict:
    arrays = cache.load("train", cache_dir)
    tr, va = (f["i"].to_numpy() for f in fd.train_val(pd.DataFrame({"i": np.arange(len(arrays["cls"]))}), cache.DATA_SEED))
    raw = models.raw_inputs(model, arrays)
    scaler = models.fit_scaler([x[tr] for x in raw])
    x = models.apply_scaler(raw, scaler)
    m, hist = fit(model, [a[tr] for a in x], arrays["cls"][tr], [a[va] for a in x], arrays["cls"][va], seed, epochs)
    if m.count_params() != models.MODELS[model][1]:
        raise RuntimeError(f"{model}: {m.count_params()} params, expected {models.MODELS[model][1]}")
    val = evaluate(arrays["pt"][va], predict_proba(m, [a[va] for a in x]), score.load_reference())
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{model}_s{seed}"
    m.save_weights(f"{stem}.weights.h5")
    Path(f"{stem}.scaler.json").write_text(json.dumps(scaler) + "\n")
    Path(f"{stem}.history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.items()}) + "\n")
    Path(f"{stem}.val.json").write_text(json.dumps({"epochs": len(hist["loss"]), **val}) + "\n")
    return val


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(models.MODELS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    a = p.parse_args(argv)
    print(json.dumps(run(a.model, a.seed, a.cache_dir or cache.cache_dir(), a.out or weights_dir(), a.epochs)))


if __name__ == "__main__":
    main()
