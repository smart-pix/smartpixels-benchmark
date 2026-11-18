"""The single filter truth function: scored clusters of a split with signed pT and class.

    python -m smartpix_bench.filter.truth --split test --out RESULTS/filter/test/truth.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from smartpix_bench.filter import cache

SPLITS = ("test",)
CODABENCH_SPLITS = SPLITS


def truth_frame(split: str, cache_dir: Path | None = None) -> pd.DataFrame:
    a = cache.load(split, cache_dir)
    return pd.DataFrame({"sample_id": a["sample_id"], "pt": a["pt"], "cls": a["cls"]})


def scored_ids(split: str, cache_dir: Path | None = None) -> pd.Series:
    return truth_frame(split, cache_dir)["sample_id"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=SPLITS)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    truth_frame(a.split, a.cache_dir).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
