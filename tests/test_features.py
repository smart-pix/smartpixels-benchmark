import numpy as np
import pytest

from smartpix_bench.common.features import add_noise, digitize, standardizer
from smartpix_bench.filter import features as ff
from smartpix_bench.regression import data as rd
from smartpix_bench.regression import features as rf


def test_filter_adc_boundaries_table3():
    x = np.array([-5, 0, 399.9, 400, 1599.9, 1600, 2399.9, 2400, 1e5])
    np.testing.assert_array_equal(ff.adc_2bit(x), [0, 0, 0, 1, 1, 2, 2, 3, 3])


def test_eq4_lower_boundary_inclusive_for_arbitrary_thresholds():
    t = (248.0, 668.0, 1663.0)
    x = np.array([247.99, 248, 667.99, 668, 1662.99, 1663])
    np.testing.assert_array_equal(digitize(x, t), [0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(digitize([10, 11, 29, 30], (11, 12, 30)), [0, 1, 2, 3])


@pytest.mark.parametrize("t", [(400, 400, 2400), (1600, 400, 2400), ((1, 2), (3, 4))])
def test_digitize_rejects_unordered_thresholds(t):
    with pytest.raises(ValueError):
        digitize([0.0], t)


def test_adc_applies_per_pixel_before_row_sum():
    px = np.zeros((13, 21))
    px[3, :2] = 400  # two pixels at 400 e: per-pixel sum 2, row-sum digitized would be 1
    px[5, 7] = 399  # below threshold
    prof = ff.y_profile(ff.adc_2bit(px))
    assert prof[3] == 2 and prof[5] == 0 and prof.sum() == 2


def test_y_profile_sums_over_x_per_row():
    px = np.arange(2 * 13 * 21, dtype=float).reshape(2, 13, 21)
    prof = ff.y_profile(px)
    assert prof.shape == (2, 13)
    np.testing.assert_array_equal(prof[1, 4], px[1, 4, :].sum())


def test_y_size_counts_rows_with_nonzero_net_charge():
    px = np.zeros((13, 21))
    px[2, 5] = 1.0
    px[6, [1, 9]] = [500.0, 3.0]
    px[8, [0, 1]] = [5.0, -5.0]  # net zero: not counted
    px[10, :] = 0.0
    assert ff.y_size(px) == 2
    assert ff.y_size(np.stack([px, np.zeros((13, 21))])).tolist() == [2, 0]


def test_y0_6bit_bins():
    y = np.array([-8.0, -7.75, -7.7499, -0.0001, 0.0, 0.0001, 7.75, 7.7501, 8.0, 8.2, -9.0])
    np.testing.assert_array_equal(ff.y0_6bit(y), [0, 0, 1, 31, 31, 32, 62, 63, 63, 63, 0])


def test_add_noise_seeded_per_pixel():
    x = np.zeros((2000, 16, 16, 2), dtype=np.float32)
    a = add_noise(x, np.random.default_rng(1))
    b = add_noise(x, np.random.default_rng(1))
    c = add_noise(x, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == x.shape and a.dtype == np.float32
    assert abs(a.std() - rf.NOISE_SIGMA) < 0.5 and abs(a.mean()) < 0.5
    assert abs(np.corrcoef(a[..., 0].ravel(), a[..., 1].ravel())[0, 1]) < 0.01


def test_standardizer_zero_std_maps_to_one():
    mean, std = standardizer(np.array([[1.0, 5.0], [3.0, 5.0]]))
    np.testing.assert_array_equal(mean, [2.0, 5.0])
    np.testing.assert_array_equal(std, [1.0, 1.0])


def test_border_charge_ring_only():
    px = np.zeros((16, 16))
    px[1:-1, 1:-1] = 1e4
    assert rd.border_charge(px) == 0
    px[0, 0] = px[0, 15] = px[15, 0] = px[15, 15] = 12.5  # corners counted once each
    assert rd.border_charge(px) == 50


@pytest.mark.parametrize("edge", [(0, 7), (15, 7), (7, 0), (7, 15)])
def test_containment_cut_at_50e(edge):
    px = np.zeros((3, 16, 16))
    px[:, 7, 7] = 5e4
    px[0][edge], px[1][edge], px[2][edge] = 49.9, 50.0, 50.1
    np.testing.assert_array_equal(rd.contained(px), [True, False, False])
