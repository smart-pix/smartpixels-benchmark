"""Filter task data: Zenodo 10783560 (v3) recon3D (20, 13, 21) = (time, y, x) cumulative charge + labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io

RECORD = "10783560"
SHAPE = (13, 21)
N_SLICES = 20
LABEL_COLUMNS = ["pt", "y-local"]
TEST_FRAC = 0.2
VAL_FRAC = 0.2
PT_CUT = 0.2  # GeV


def files(data_dir: Path | None = None) -> dict[str, tuple[Path, Path]]:
    root = Path(data_dir or io.data_dir()) / RECORD
    return io.pair_files(root / "labels" / "labels", root / "recon3D" / "recon3D")


def class_label(pt) -> np.ndarray:
    """0: |pT| > 0.2 GeV, 1: -0.2 <= pT < 0, 2: 0 < pT <= 0.2 (paper §3.1)."""
    pt = np.asarray(pt)
    if np.any(pt == 0):
        raise ValueError("pT == 0 has no class")
    return np.where(np.abs(pt) > PT_CUT, 0, np.where(pt < 0, 1, 2))


def splits(stems) -> dict[str, list[str]]:
    """File-disjoint `train` / `test` file lists."""
    train, test = io.split_files(stems, TEST_FRAC)
    return {"train": train, "test": test}


def labels(files_: dict, stems) -> pd.DataFrame:
    df = io.read_labels(files_, stems, LABEL_COLUMNS)
    df["cls"] = class_label(df["pt"])
    return df


def category(df: pd.DataFrame) -> np.ndarray:
    """Charge x pT category: 0 = pT < -0.2, 1 and 2 = the low-pT classes, 3 = pT > 0.2."""
    return np.where((df["cls"] == 0) & (df["pt"] > 0), 3, df["cls"])


def balanced_subsample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Every charge x pT category subsampled without replacement to the smallest one's count, so classes 0:1:2 = 2:1:1.

    Paper §3.1 "same number of clusters for each pT class" with its 800k = 4 x 203,800 total and its own train set.
    """
    rng = np.random.default_rng(seed)
    cat = category(df)
    n = np.bincount(cat, minlength=4).min()
    keep = np.concatenate([rng.choice(np.flatnonzero(cat == c), n, replace=False) for c in range(4)])
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def train_val(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded shuffle, then the last VAL_FRAC of rows is validation (Keras `validation_split` semantics)."""
    df = df.iloc[np.random.default_rng(seed).permutation(len(df))].reset_index(drop=True)
    n_train = int(len(df) * (1 - VAL_FRAC))
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def recon(files_: dict, df: pd.DataFrame, slices) -> np.ndarray:
    """(n, len(slices), 13, 21) pixels for the rows of `df`."""
    return io.read_recon(files_, df, slices, SHAPE)
