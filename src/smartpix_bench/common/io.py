"""Paths from `SPB_*` environment variables, parquet listing, file splits, sample ids."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SEED = 0


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(f"SPB_{name}", default))


def data_dir() -> Path:
    return env_path("DATA_DIR", "data/zenodo")


def parquet_files(directory: Path, prefix: str) -> dict[str, Path]:
    """`{index: path}` for `<prefix><index>.parquet`; the anchored glob excludes macOS `._*` AppleDouble files."""
    out = {p.stem[len(prefix):]: p for p in sorted(Path(directory).glob(f"{prefix}*.parquet"))}
    if not out:
        raise FileNotFoundError(f"no {prefix}*.parquet in {directory}")
    return out


def pair_files(labels_dir: Path, recon_dir: Path) -> dict[str, tuple[Path, Path]]:
    """`{recon_stem: (labels_path, recon_path)}` for every labels/recon3D file pair."""
    labels = parquet_files(labels_dir, "labels_")
    recon = parquet_files(recon_dir, "recon3D_")
    if labels.keys() != recon.keys():
        raise ValueError(f"unpaired files: {sorted(labels.keys() ^ recon.keys())}")
    return {recon[k].stem: (labels[k], recon[k]) for k in labels}


def split_files(stems, frac: float, seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    """Disjoint `(rest, held_out)` file lists; held_out is `round(frac * n)` files chosen by a seeded permutation."""
    stems = sorted(stems)
    perm = np.random.default_rng(seed).permutation(len(stems))
    n_out = round(frac * len(stems))
    held = sorted(stems[i] for i in perm[:n_out])
    rest = sorted(stems[i] for i in perm[n_out:])
    return rest, held


def sample_ids(stem: str, rows) -> list[str]:
    return [f"{stem}:{r}" for r in rows]


def read_labels(files: dict[str, tuple[Path, Path]], stems, columns) -> pd.DataFrame:
    """Label columns of the given files plus `file`, `row` (position in the file) and `sample_id`."""
    frames = []
    for stem in stems:
        df = pd.read_parquet(files[stem][0], columns=list(columns)).reset_index(drop=True)
        df["file"] = stem
        df["row"] = np.arange(len(df))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["sample_id"] = out["file"] + ":" + out["row"].astype(str)
    return out


def read_recon(files: dict[str, tuple[Path, Path]], index: pd.DataFrame, slices, shape) -> np.ndarray:
    """Pixels of the rows in `index` (`file`, `row`), as float32 `(n, len(slices), *shape)` in `index` order.

    Recon columns are named by flat position `t * prod(shape) + pixel` (time-major, then y, then x).
    """
    n_pix = int(np.prod(shape))
    cols = [str(t * n_pix + i) for t in slices for i in range(n_pix)]
    out = np.empty((len(index), len(slices), *shape), dtype=np.float32)
    file_col, rows = index["file"].to_numpy(), index["row"].to_numpy()
    for stem in pd.unique(file_col):
        pos = np.flatnonzero(file_col == stem)
        x = pd.read_parquet(files[stem][1], columns=cols).to_numpy(np.float32)
        out[pos] = x[rows[pos]].reshape(-1, len(slices), *shape)
    return out
