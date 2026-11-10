import numpy as np
import pandas as pd
import pytest

from smartpix_bench.common import io
from smartpix_bench.filter import data as fd
from smartpix_bench.regression import data as rd


@pytest.mark.parametrize("n", [5, 80, 160])
def test_split_files_disjoint_and_deterministic(n):
    stems = [f"recon3D_d{i}" for i in range(n)]
    rest, held = io.split_files(stems, 0.2)
    assert set(rest).isdisjoint(held)
    assert set(rest) | set(held) == set(stems)
    assert len(held) == round(0.2 * n)
    assert (rest, held) == io.split_files(list(reversed(stems)), 0.2)
    assert held != stems[: len(held)]  # permuted, not the first files


def test_class_label_boundaries():
    pt = np.array([-5.0, -0.2001, -0.2, -0.0001, 0.0001, 0.2, 0.2001, 3.0])
    np.testing.assert_array_equal(fd.class_label(pt), [0, 0, 1, 1, 2, 2, 0, 0])
    with pytest.raises(ValueError):
        fd.class_label([0.0])


def _frame(counts):
    cls = np.repeat(np.arange(len(counts)), counts)
    return pd.DataFrame({"cls": cls, "sample_id": [f"f:{i}" for i in range(len(cls))]})


def test_equal_class_subsample():
    df = _frame([50, 7, 12])
    a = fd.equal_class_subsample(df, seed=3)
    assert a["cls"].value_counts().to_dict() == {0: 7, 1: 7, 2: 7}
    assert a["sample_id"].is_unique and set(a["sample_id"]) <= set(df["sample_id"])
    assert set(df.loc[df.cls == 1, "sample_id"]) <= set(a["sample_id"])  # smallest class kept whole
    assert a.equals(fd.equal_class_subsample(df, seed=3))
    assert not a.equals(fd.equal_class_subsample(df, seed=4))


@pytest.mark.parametrize("n", [10, 11, 99])
def test_train_val_keras_semantics(n):
    df = _frame([n])
    tr, va = fd.train_val(df, seed=0)
    assert len(tr) == int(n * 0.8) and len(tr) + len(va) == n
    assert set(tr.sample_id).isdisjoint(va.sample_id)
    shuffled = df.iloc[np.random.default_rng(0).permutation(n)]
    assert va.sample_id.tolist() == shuffled.sample_id.tolist()[int(n * 0.8):]


def test_label_scale_is_99th_percentile_abs():
    v = np.arange(1, 101, dtype=float)
    df = pd.DataFrame({t: v * (-1) ** i * (i + 1) for i, t in enumerate(rd.TARGETS)})
    np.testing.assert_allclose(rd.label_scale(df), [99.01 * (i + 1) for i in range(4)])
