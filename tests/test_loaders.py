import numpy as np
import pandas as pd
import pytest

from smartpix_bench.filter import data as fd
from smartpix_bench.filter import features as ff
from smartpix_bench.regression import data as rd




def test_filter_pairs_skip_appledouble(fixture_data):
    assert (fixture_data / fd.RECORD / "labels/labels/._labels_d0.parquet").exists()
    files = fd.files(fixture_data)
    assert len(files) == 3 and all(s.startswith("recon3D_d") for s in files)
    assert all(lab.name.replace("labels_", "recon3D_") == rec.name for lab, rec in files.values())


def test_filter_labels_and_recon(fixture_data):
    files = fd.files(fixture_data)
    df = fd.labels(files, sorted(files))
    assert df["sample_id"].is_unique
    assert df["sample_id"].iloc[0] == f"{sorted(files)[0]}:0"
    assert sorted(df["cls"].value_counts().tolist()) == [6, 6, 12]
    px = fd.recon(files, df, [0, ff.PROFILE_SLICE])
    assert px.shape == (len(df), 2, 13, 21)
    raw = pd.read_parquet(files[df.file[4]][1]).iloc[df.row[4]]
    np.testing.assert_array_equal(px[4, 1].ravel(), raw[[str(19 * 273 + i) for i in range(273)]].to_numpy(np.float32))
    assert np.all(px[:, 1].sum(axis=(1, 2)) >= px[:, 0].sum(axis=(1, 2)))  # cumulative charge
    assert np.all(ff.y_size(px[:, 1]) >= 1)


def test_regression_load_applies_containment(fixture_data):
    files = rd.files("3sExtended", fixture_data)
    stems = sorted(files)
    x, lab = rd.load(files, stems, (0, 19))
    raw = pd.concat([pd.read_parquet(files[s][1]) for s in stems], ignore_index=True)
    assert len(raw) == 10 and len(lab) == 6
    assert x.shape == (6, 16, 16, 2)
    assert np.all(rd.border_charge(x[..., 1]) < 50)
    assert lab["sample_id"].is_unique and set(lab.columns) >= set(rd.TARGETS)
    k = lab.index[-1]
    r = pd.read_parquet(files[lab.file[k]][1]).iloc[lab.row[k]]
    np.testing.assert_array_equal(x[k, :, :, 0].ravel(), r[[str(i) for i in range(256)]].to_numpy(np.float32))
    x20, _ = rd.load(files, stems, range(20))
    np.testing.assert_array_equal(x20[..., [0, 19]], x)


@pytest.mark.parametrize("ds", rd.TEST_SETS)
def test_regression_test_sets_load(ds, fixture_data):
    files = rd.files(ds, fixture_data)
    x, lab = rd.load(files, sorted(files), (0, 19))
    assert len(lab) == 3 and x.shape == (3, 16, 16, 2)


def test_label_scale_uses_given_split_only(fixture_data):
    files = rd.files("3sExtended", fixture_data)
    stems = sorted(files)
    _, a = rd.load(files, stems[:1], (19,))
    _, b = rd.load(files, stems[1:], (19,))
    sa = rd.label_scale(a)
    assert not np.allclose(sa, rd.label_scale(pd.concat([a, b])))
    np.testing.assert_allclose(sa, np.percentile(np.abs(a[rd.TARGETS].to_numpy()), 99, axis=0))
