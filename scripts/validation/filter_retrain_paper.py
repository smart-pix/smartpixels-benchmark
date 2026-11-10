"""Retrain M2 (dec6_ds8 CSVs) or M3 (dec8_ds8_ts CSVs) on the paper's own inputs and score its test set, for Table 1.

    python filter_retrain_paper.py --model M2 --paper dec6_ds8 --ref-labels DS8_DIR --seed 0 --out OUT [--epochs 200]
    python filter_retrain_paper.py --model M3 --paper dec8_ds8_ts --ref-labels DS8_DIR --seed 0 --out OUT

Same recipe as `filter.train` (Adam 1e-3, SCCE, batch 1024, early stop on val loss, patience 20);
validation is the last 20 % after a seeded shuffle; standardization from the train part. Writes
OUT/<model>_paper_s<seed>.json with aggregate metrics only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filter_golden import reference

from smartpix_bench.filter import cache, models, train
from smartpix_bench.filter import data as fd


def inputs(model: str, d: Path, split: str) -> list[np.ndarray]:
    """M2: dec6 FullPrecisionInput (13 profile values + y-local); M3: dec8 slice profiles (n, 8, 13) and y-local."""
    if model == "M2":
        return [pd.read_csv(d / f"FullPrecisionInput{split}Set.csv").to_numpy(np.float32)]
    df = pd.read_csv(d / f"Input{split}Set.csv")
    return [df.drop(columns="y-local").to_numpy(np.float32).reshape(len(df), 8, 13), df[["y-local"]].to_numpy(np.float32)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["M2", "M3"], default="M2")
    p.add_argument("--paper", type=Path, required=True, help="dec6_ds8 (M2) or dec8_ds8_ts (M3)")
    p.add_argument("--ref-labels", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=train.EPOCHS)
    a = p.parse_args()
    x, xt = (inputs(a.model, a.paper, split) for split in ("Train", "Test"))
    y = pd.read_csv(a.paper / "TrainSetLabel.csv")["ptLabel"].to_numpy()
    pt = pd.read_csv(a.paper / "TestSetTruePT.csv")["pt"].to_numpy()
    tr, va = (f["i"].to_numpy() for f in fd.train_val(pd.DataFrame({"i": np.arange(len(y))}), cache.DATA_SEED))
    sc = models.fit_scaler([v[tr] for v in x])
    xs, xts = models.apply_scaler(x, sc), models.apply_scaler(xt, sc)
    m, hist = train.fit(a.model, [v[tr] for v in xs], y[tr], [v[va] for v in xs], y[va], a.seed, a.epochs)
    res = {"seed": a.seed, "epochs": len(hist["loss"]), "best_val_loss": float(min(hist["val_loss"])),
           **train.evaluate(pt, train.predict_proba(m, xts), reference(a.ref_labels))}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.model}_paper_s{a.seed}.json").write_text(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
