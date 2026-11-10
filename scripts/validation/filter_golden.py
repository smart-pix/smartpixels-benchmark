"""Score the paper's stored Model 2 test predictions with filter.score (golden metric check).

    python filter_golden.py --pred predictionsFiles.csv --true true.csv --pt dec6_ds8/TestSetTruePT.csv --ref-labels DS8_DIR

The reweighting reference is the signed-pT histogram of every labels_*.parquet in DS8_DIR.
Prints aggregate numbers only (the inputs are CMS-restricted).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io
from smartpix_bench.filter import data as fd
from smartpix_bench.filter import score


def reference(labels_dir: Path) -> np.ndarray:
    files = io.parquet_files(labels_dir, "labels_")
    return score.histogram(np.concatenate([pd.read_parquet(f, columns=["pt"])["pt"].to_numpy() for f in files.values()]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("--pred", "--true", "--pt", "--ref-labels"):
        p.add_argument(a, type=Path, required=True)
    a = p.parse_args()
    pred = pd.read_csv(a.pred).iloc[:, 0].astype(int).to_numpy()
    true = pd.read_csv(a.true).iloc[:, 0].astype(int).to_numpy()
    pt = pd.read_csv(a.pt)["pt"].to_numpy()
    ref = reference(a.ref_labels)
    out = {"n_test": len(pt), "n_reference": int(ref.sum()), "true_matches_class_label": float(np.mean(fd.class_label(pt) == true)),
           **score.reweighted_metrics(pt, pred, ref)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
