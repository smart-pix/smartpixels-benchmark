"""Train one regression model, configuration and seed.

    python -m smartpix_bench.regression.train --model Full_MLP --config sq|c1|c2|c3|c4 --seed 0 [--out $SPB_WEIGHTS_DIR/regression]

c1: 20 slices, c2: slices {0, 19}, both full precision. "sq" trains the model behind GaussianNoise(80 e) +
SoftQuantize (thresholds start at the transformer means, k annealed 1 -> 67) and writes the thresholds at
its best validation epoch; c3 (float) and c4 (QKeras) then train fresh models on inputs hard-digitized
(Eq 4) at those thresholds, noise-free. The sq stage validates on one fixed-seed noise draw (reused every epoch), so
best-validation selection sees the on-chip noise it is meant to be robust to. Nadam 1e-3, batch 5000, 1000 epochs, best-validation weights kept.
Targets are divided by the per-target 99th percentile |label| of the train split (stored in the meta file).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from smartpix_bench.common import io
from smartpix_bench.common.features import add_noise, digitize
from smartpix_bench.regression import data as rd
from smartpix_bench.regression import features as rf
from smartpix_bench.regression import models, predict, score

EPOCHS, BATCH, LR = 1000, 5000, 1e-3
VAL_NOISE_SEED = 0


def weights_dir() -> Path:
    return io.env_path("WEIGHTS_DIR", "weights") / "regression"


def load_train_val(slices, data_dir=None):
    files = rd.files(rd.TRAIN_SET, data_dir)
    sp = rd.splits(sorted(files))
    return rd.load(files, sp["train"], slices), rd.load(files, sp["val"], slices)


def fit(m, x_tr, y_tr, x_va, y_va, head: str, seed: int, epochs: int, ckpt: Path, callbacks=()):
    from tensorflow import keras

    from smartpix_bench.regression.losses import LOSSES
    m.compile(optimizer=keras.optimizers.Nadam(LR), loss=LOSSES[head])
    best = keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, save_weights_only=True)
    h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=BATCH, shuffle=True,
              callbacks=[best, *callbacks], verbose=2)
    m.load_weights(str(ckpt))
    return h.history


def run(model: str, config: str, seed: int, out: Path, epochs: int = EPOCHS, data_dir=None) -> dict:
    from tensorflow import keras

    from smartpix_bench.regression.softquantize import AnnealK
    keras.utils.set_random_seed(seed)
    head = models.split_name(model)[0]
    targets = models.TARGETS[head]
    (x_tr, l_tr), (x_va, l_va) = load_train_val(rf.SLICES_20 if config == "c1" else rf.SLICES_2, data_dir)
    scale = rd.label_scale(l_tr, targets)
    y_tr, y_va = (l[targets].to_numpy() / scale for l in (l_tr, l_va))
    stem = out / f"{model}_{config}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "config": config, "seed": seed, "targets": targets, "label_scale": scale.tolist()}
    if config == "sq":
        m = models.build(model, softquantize=rf.TRANSFORMER_THRESHOLDS)
        sq = m.get_layer("softquantize")
        x_va = add_noise(x_va, np.random.default_rng(VAL_NOISE_SEED), rf.NOISE_SIGMA)
        hist = fit(m, x_tr, y_tr, x_va, y_va, head, seed, epochs, Path(f"{stem}.weights.h5"), [AnnealK(sq, epochs)])
        meta["thresholds"] = sq.thresholds().numpy().tolist()
    else:
        if config in ("c3", "c4"):
            meta["thresholds"] = json.loads((out / f"{model}_sq_s{seed}.meta.json").read_text())["thresholds"]
            x_tr, x_va = digitize(x_tr, meta["thresholds"]), digitize(x_va, meta["thresholds"])
        m = models.build(model, x_tr.shape[-1], config == "c4")
        hist = fit(m, x_tr, y_tr, x_va, y_va, head, seed, epochs, Path(f"{stem}.weights.h5"))
    pred = predict.decode(head, m.predict(x_va, batch_size=1 << 15, verbose=0), scale)
    val = score.metrics(pred, l_va.reset_index(drop=True))
    Path(f"{stem}.meta.json").write_text(json.dumps(meta) + "\n")
    Path(f"{stem}.history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.items()}) + "\n")
    Path(f"{stem}.val.json").write_text(json.dumps({"epochs": len(hist["loss"]), "best_val_loss": float(np.min(hist["val_loss"])),
                                                    **val}) + "\n")
    return val


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(models.MODELS))
    p.add_argument("--config", required=True, choices=["sq", *models.CONFIGS])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--data-dir", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(run(a.model, a.config, a.seed, a.out or weights_dir(), a.epochs, a.data_dir)))


if __name__ == "__main__":
    main()
