"""Predict a regression test set with trained weights.

    python -m smartpix_bench.regression.predict --weights W/regression/Full_MLP_c3_s0 --split 3s --out RESULTS/regression/3s

Writes predictions_<name>.csv (sample_id, x-midplane, y-midplane, cotAlpha, cotBeta in physical units, plus
sigma_<target> for Max/Full; targets a model does not predict are NaN) and descriptor_<name>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common.features import digitize
from smartpix_bench.regression import data as rd
from smartpix_bench.regression import features as rf
from smartpix_bench.regression import models

SIGMA = [f"sigma_{t}" for t in rd.TARGETS]
OUTPUT_COLUMNS = [*rd.TARGETS, *SIGMA]


def decode(head: str, p: np.ndarray, scale) -> pd.DataFrame:
    """Network outputs in label-scaled units -> means and 1-sigma uncertainties in physical units."""
    s = np.asarray(scale, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out = pd.DataFrame(np.nan, index=range(len(p)), columns=OUTPUT_COLUMNS)
    if head == "Slim":
        out[models.TARGETS["Slim"]] = p * s
        return out
    if head == "Full":
        mu, sig = p[:, 0:4], np.logaddexp(0.0, p[:, 4:8]) + 1e-9
    else:
        mu, d, o = p[:, 0:8:2], 1e-9 + np.maximum(p[:, 1:8:2], 0.0), p[:, 8:14]
        L = np.zeros((len(p), 4, 4))
        L[:, range(4), range(4)] = d
        L[:, [1, 2, 2, 3, 3, 3], [0, 0, 1, 0, 1, 2]] = o
        sig = np.sqrt(np.einsum("nij,nij->ni", L, L))
    out[rd.TARGETS] = mu * s
    out[SIGMA] = sig * s
    return out


def load_meta(stem) -> dict:
    return json.loads(Path(f"{stem}.meta.json").read_text())


def inputs(x: np.ndarray, meta: dict) -> np.ndarray:
    return digitize(x, meta["thresholds"]) if meta["config"] in ("c3", "c4") else x


def predict_split(m, meta: dict, split: str, data_dir=None, batch=1 << 15) -> pd.DataFrame:
    """Predictions for every contained cluster of a test set, one file at a time."""
    files = rd.files(split, data_dir)
    slices = rf.SLICES_20 if meta["config"] == "c1" else rf.SLICES_2
    head = models.split_name(meta["model"])[0]
    parts = []
    for stem in sorted(files):
        x, lab = rd.load(files, [stem], slices)
        out = decode(head, m.predict(inputs(x, meta), batch_size=batch, verbose=0), meta["label_scale"])
        parts.append(pd.concat([lab[["sample_id"]], out], axis=1))
    return pd.concat(parts, ignore_index=True)


def load_model(stem):
    meta = load_meta(stem)
    m = models.build(meta["model"], 20 if meta["config"] == "c1" else 2, meta["config"] == "c4")
    m.load_weights(f"{stem}.weights.h5")
    return m, meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path stem: <stem>.weights.h5 and <stem>.meta.json")
    p.add_argument("--name", help="default: weights stem name")
    p.add_argument("--split", required=True, choices=rd.TEST_SETS)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    name = a.name or Path(a.weights).name
    m, meta = load_model(a.weights)
    df = predict_split(m, meta, a.split, a.data_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"predictions_{name}.csv", index=False)
    desc = models.descriptor(meta["model"], meta["config"], m.count_params(), meta.get("thresholds"))
    (a.out / f"descriptor_{name}.json").write_text(json.dumps(desc) + "\n")
    print(f"wrote {a.out}/predictions_{name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
