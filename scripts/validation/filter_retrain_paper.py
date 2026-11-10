"""Retrain M2 on the paper's own inputs (dec6_ds8 CSVs) and score its test set, for Table 1.

    python filter_retrain_paper.py --dec6 dec6_ds8 --ref-labels DS8_DIR --seed 0 --out OUT [--epochs 200]

Same recipe as `filter.train` (Adam 1e-3, SCCE, batch 1024, early stop on val loss, patience 20);
validation is the last 20 % after a seeded shuffle; standardization from the train part. Writes
OUT/M2_paper_s<seed>.json with aggregate metrics only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filter_golden import reference

from smartpix_bench.filter import cache, models, train
from smartpix_bench.filter import data as fd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dec6", type=Path, required=True)
    p.add_argument("--ref-labels", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=train.EPOCHS)
    a = p.parse_args()
    x = pd.read_csv(a.dec6 / "FullPrecisionInputTrainSet.csv").to_numpy(np.float32)
    y = pd.read_csv(a.dec6 / "TrainSetLabel.csv")["ptLabel"].to_numpy()
    xt = pd.read_csv(a.dec6 / "FullPrecisionInputTestSet.csv").to_numpy(np.float32)
    pt = pd.read_csv(a.dec6 / "TestSetTruePT.csv")["pt"].to_numpy()
    tr, va = (f["i"].to_numpy() for f in fd.train_val(pd.DataFrame({"i": np.arange(len(y))}), cache.DATA_SEED))
    sc = models.fit_scaler([x[tr]])
    (xs,), (xts,) = models.apply_scaler([x], sc), models.apply_scaler([xt], sc)
    m, hist = train.fit("M2", xs[tr], y[tr], xs[va], y[va], a.seed, a.epochs)
    res = {"seed": a.seed, "epochs": len(hist["loss"]), "best_val_loss": float(min(hist["val_loss"])),
           **train.evaluate(pt, train.predict_proba(m, xts), reference(a.ref_labels))}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"M2_paper_s{a.seed}.json").write_text(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
