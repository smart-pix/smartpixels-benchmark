"""Regression task data: Zenodo 18472791 recon3D (20, 16, 16) = (time, y, x) cumulative charge + labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io

RECORD = "18472791"
SHAPE = (16, 16)
N_SLICES = 20
CONTAIN_SLICE = 19
BORDER_MAX = 50.0  # electrons; paper §3.1
TARGETS = ["x-midplane", "y-midplane", "cotAlpha", "cotBeta"]
LABEL_SCALE_PCTL = 99
VAL_FRAC = 0.2
TRAIN_SET = "3sExtended"
TEST_SETS = ("3s", "2s")
DATASET_DIRS = {ds: f"dataset_{ds}_16x16_50x12P5x100/dataset_{ds}_16x16_50x12P5x100" for ds in ("3sExtended", "3s", "2s")}


def files(dataset: str, data_dir: Path | None = None) -> dict[str, tuple[Path, Path]]:
    d = Path(data_dir or io.data_dir()) / RECORD / DATASET_DIRS[dataset]
    return io.pair_files(d, d)


def splits(train_stems) -> dict[str, list[str]]:
    """File-disjoint `train` / `val` lists of the 3sExtended files (80/20)."""
    train, val = io.split_files(train_stems, VAL_FRAC)
    return {"train": train, "val": val}


def border_charge(pixels) -> np.ndarray:
    """Charge summed over the outermost ring of pixels: (..., 16, 16) -> (...)."""
    p = np.asarray(pixels, dtype=np.float64)
    return p.sum(axis=(-2, -1)) - p[..., 1:-1, 1:-1].sum(axis=(-2, -1))


def contained(pixels19) -> np.ndarray:
    """Keep mask: border charge at slice 19 below 50 e."""
    return border_charge(pixels19) < BORDER_MAX


def load(files_: dict, stems, slices) -> tuple[np.ndarray, pd.DataFrame]:
    """Contained clusters of `stems`: pixels (n, 16, 16, len(slices)) in (y, x, t) and their labels with `sample_id`."""
    slices = list(slices)
    xs, labs = [], []
    for stem in stems:
        lab = io.read_labels(files_, [stem], TARGETS)
        order = sorted(set(slices) | {CONTAIN_SLICE})
        x = io.read_recon(files_, lab, order, SHAPE)
        keep = contained(x[:, order.index(CONTAIN_SLICE)])
        xs.append(x[keep][:, [order.index(t) for t in slices]].transpose(0, 2, 3, 1))
        labs.append(lab[keep])
    return np.concatenate(xs), pd.concat(labs, ignore_index=True)


def label_scale(train_labels: pd.DataFrame, targets=TARGETS) -> np.ndarray:
    """Per-target 99th percentile of |label| over the train split only."""
    return np.percentile(np.abs(train_labels[list(targets)].to_numpy()), LABEL_SCALE_PCTL, axis=0)
