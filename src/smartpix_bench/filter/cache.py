"""Featurize each filter split once into $SPB_CACHE_DIR/filter/<split>.npz.

    python -m smartpix_bench.filter.cache [--data-dir D] [--cache-dir C]

Arrays per cluster: sample_id, pt, cls, y_local, ysize (slice 19), prof (slice-19 y-profile),
prof_ts (y-profiles of slices 0-7), prof_adc (y-profile of the 2-bit ADC codes at slice 19), y0_6bit.
"train" holds the equal-class subsample of the train files; validation is split off at train time.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io
from smartpix_bench.filter import data as fd
from smartpix_bench.filter import features as ff

DATA_SEED = 0
SLICES = [*ff.TIMING_SLICES, ff.PROFILE_SLICE]


def cache_dir() -> Path:
    return io.env_path("CACHE_DIR", "cache") / "filter"


def featurize(files: dict, df) -> dict:
    """Feature arrays for the rows of `df` (in order), read one file at a time to bound memory."""
    n, file_col = len(df), df["file"].to_numpy()
    out = {"ysize": np.empty(n, np.int64), "prof": np.empty((n, 13), np.float32),
           "prof_ts": np.empty((n, len(ff.TIMING_SLICES), 13), np.float32), "prof_adc": np.empty((n, 13), np.float32)}
    for stem in pd.unique(file_col):
        pos = np.flatnonzero(file_col == stem)
        px = fd.recon(files, df.iloc[pos], SLICES)
        p19 = px[:, -1]
        out["ysize"][pos], out["prof"][pos] = ff.y_size(p19), ff.y_profile(p19)
        out["prof_ts"][pos], out["prof_adc"][pos] = ff.y_profile(px[:, :-1]), ff.y_profile(ff.adc_2bit(p19))
    out.update(sample_id=df["sample_id"].to_numpy(str), pt=df["pt"].to_numpy(), cls=df["cls"].to_numpy(),
               y_local=df["y-local"].to_numpy(), y0_6bit=ff.y0_6bit(df["y-local"]))
    return out


def split_frames(files: dict) -> dict:
    sp = fd.splits(sorted(files))
    train = fd.equal_class_subsample(fd.labels(files, sp["train"]), DATA_SEED)
    return {"train": train, "test": fd.labels(files, sp["test"])}


def build(data_dir: Path, out: Path) -> None:
    files = fd.files(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, df in split_frames(files).items():
        np.savez(out / f"{split}.npz", **featurize(files, df))
        print(f"{split}: {len(df)} clusters, classes {np.bincount(df['cls']).tolist()}", flush=True)


def load(split: str, directory: Path | None = None) -> dict:
    with np.load(Path(directory or cache_dir()) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    a = p.parse_args(argv)
    build(a.data_dir or io.data_dir(), a.cache_dir or cache_dir())


if __name__ == "__main__":
    main()
