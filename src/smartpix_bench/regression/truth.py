"""The single regression truth function: labels of the contained clusters of a test set.

    python -m smartpix_bench.regression.truth --split 3s --out RESULTS/regression/3s/truth.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from smartpix_bench.regression import data as rd

SPLITS = rd.TEST_SETS


def truth_frame(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    files = rd.files(split, data_dir)
    _, lab = rd.load(files, sorted(files), [rd.CONTAIN_SLICE])
    return lab[["sample_id", *rd.TARGETS]]


def scored_ids(split: str, data_dir: Path | None = None) -> pd.Series:
    return truth_frame(split, data_dir)["sample_id"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    truth_frame(a.split, a.data_dir).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
