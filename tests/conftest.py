import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory) -> Path:
    """The fixture tree with each recon `.npz` written back as the Zenodo parquet layout."""
    root = tmp_path_factory.mktemp("zenodo")
    for rec in ("10783560", "18472791"):
        shutil.copytree(SRC / rec, root / rec)
    for npz in root.rglob("recon3D_*.npz"):
        z = np.load(npz)
        df = pd.DataFrame(z["x"], columns=[str(i) for i in range(z["x"].shape[1])], index=z["index"])
        df.to_parquet(npz.with_suffix(".parquet"))
        npz.unlink()
    return root
