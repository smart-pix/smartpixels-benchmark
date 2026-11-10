"""Rebuild the fixtures from a full Zenodo download: `python make_fixtures.py <data_dir> <out_dir>`.

Keeps the real file stems and row order; picks rows covering every filter class and both sides of
the regression containment cut. Row positions change, so fixture sample_ids differ from the full data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from smartpix_bench.filter import data as fd  # noqa: E402
from smartpix_bench.regression import data as rd  # noqa: E402


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd")


def copy_rows(pair, rows, out_labels: Path, out_recon: Path) -> None:
    """Labels as parquet; recon as npz, since a 5k-column parquet footer alone is MBs (conftest rebuilds the parquet)."""
    write(pd.read_parquet(pair[0]).iloc[rows], out_labels)
    recon = pd.read_parquet(pair[1]).iloc[rows]
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    assert list(recon.columns) == [str(i) for i in range(recon.shape[1])]
    np.savez_compressed(out_recon.with_suffix(".npz"), x=recon.to_numpy(), index=recon.index.to_numpy())


def main(src: Path, out: Path) -> None:
    files = fd.files(src)
    for stem in sorted(files)[:3]:
        cls = fd.class_label(pd.read_parquet(files[stem][0], columns=["pt"])["pt"])
        rows = np.sort(np.concatenate([np.flatnonzero(cls == c)[:2] for c in range(3)]))
        o = out / fd.RECORD
        copy_rows(files[stem], rows, o / "labels/labels" / files[stem][0].name, o / "recon3D/recon3D" / files[stem][1].name)
    (out / fd.RECORD / "labels/labels/._labels_d0.parquet").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    for ds, n_files in [("3sExtended", 2), ("3s", 1), ("2s", 1)]:
        files = rd.files(ds, src)
        for stem in sorted(files)[:n_files]:
            x = rd.io.read_recon(files, pd.DataFrame({"file": stem, "row": np.arange(200)}), [rd.CONTAIN_SLICE], rd.SHAPE)
            keep = rd.contained(x[:, 0])
            rows = np.sort(np.concatenate([np.flatnonzero(keep)[:3], np.flatnonzero(~keep)[:2]]))
            d = out / rd.RECORD / rd.DATASET_DIRS[ds]
            copy_rows(files[stem], rows, d / files[stem][0].name, d / files[stem][1].name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
