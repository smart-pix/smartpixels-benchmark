import numpy as np

from smartpix_bench.filter import cache, data as fd, features as ff


def test_cache_rows_match_direct_features(fixture_data, tmp_path):
    cache.build(fixture_data, tmp_path)
    files = fd.files(fixture_data)
    frames = cache.split_frames(files)
    assert set(frames["train"].file).isdisjoint(frames["test"].file)
    assert np.bincount(frames["train"].cls).tolist() == [4, 4, 4]
    for split, df in frames.items():
        a = cache.load(split, tmp_path)
        assert a["sample_id"].tolist() == df["sample_id"].tolist()
        for k in (len(df) - 1, 0):
            px = fd.recon(files, df.iloc[[k]], cache.SLICES)[0]
            np.testing.assert_array_equal(a["prof"][k], ff.y_profile(px[-1]))
            np.testing.assert_array_equal(a["prof_ts"][k], ff.y_profile(px[:8]))
            np.testing.assert_array_equal(a["prof_adc"][k], ff.y_profile(ff.adc_2bit(px[-1])))
            assert a["ysize"][k] == ff.y_size(px[-1]) and a["pt"][k] == df["pt"].iloc[k]
        assert a["y0_6bit"].tolist() == ff.y0_6bit(df["y-local"]).tolist()


def test_cache_train_is_equal_class(fixture_data, monkeypatch):
    labels = fd.labels
    monkeypatch.setattr(cache.fd, "labels", lambda f, s: labels(f, s).iloc[1:].reset_index(drop=True))
    frames = cache.split_frames(fd.files(fixture_data))
    assert np.bincount(frames["train"].cls).tolist() == [3, 3, 3]
    assert len(frames["test"]) == 5
