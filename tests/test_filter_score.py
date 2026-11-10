import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from smartpix_bench.common import report, submission
from smartpix_bench.filter import score as fs


def test_bin_index_right_closed_lowest_included():
    pt = [-10.0, -2.0, -1.99, -1.9, 0.0, 1e-9, 2.0, 2.0001, 10.0, 10.01, -10.01]
    np.testing.assert_array_equal(fs.bin_index(pt), [0, 0, 1, 1, 20, 21, 40, 41, 41, -1, -1])
    assert len(fs.EDGES) == 43 and fs.SIGNAL_BINS == [0, 41] and fs.BACKGROUND_BINS == list(range(1, 41))


def _ref(**bins):
    r = np.zeros(42)
    for b, n in bins.items():
        r[int(b[1:])] = n
    return r


def test_reweighted_metrics_hand_computed():
    # neg signal (bin 0): 4 test, 3 accepted, ref 400; pos signal (bin 41): 2 test, 1 accepted, ref 100
    # bkg bin 5 (-1.6, -1.5]: 2 test, 1 rejected, ref 30; bkg bin 31 (1.0, 1.1]: 1 test, rejected, ref 10
    pt = [-5, -5, -5, -5, 3, 3, -1.55, -1.55, 1.05]
    pred = [0, 0, 0, 1, 0, 2, 1, 0, 2]
    m = fs.reweighted_metrics(pt, pred, _ref(b0=400, b41=100, b5=30, b31=10))
    assert m["signal_efficiency"] == pytest.approx((3 * 100 + 1 * 50) / 500)
    assert m["signal_efficiency"] != pytest.approx((3 * 400 / 2 + 1 * 100 / 4) / 500)  # charge-swapped ratios
    assert m["background_rejection"] == pytest.approx((1 * 15 + 1 * 10) / 40)
    assert m["data_reduction"] == pytest.approx((1 * 100 + 1 * 50 + 1 * 15 + 1 * 10) / 540)


def test_reference_bin_without_test_clusters_raises():
    with pytest.raises(ValueError, match="no test clusters"):
        fs.reweighted_metrics([-5.0], [0], _ref(b0=10, b41=5))


def test_turn_on_binomial():
    t = fs.turn_on([0.25] * 4 + [-3.0], [0, 0, 0, 1, 1])
    b = {(r["lo"], r["hi"]): r for r in t}
    assert b[(0.2, 0.3)]["n"] == 4 and b[(0.2, 0.3)]["acceptance"] == 0.75
    assert b[(0.2, 0.3)]["error"] == pytest.approx(np.sqrt(0.75 * 0.25 / 4))
    assert b[(-3.1, -3.0)]["acceptance"] == 0.0 and np.isnan(b[(1.0, 1.1)]["acceptance"])


def test_asic_violations():
    ok = {"track": "asic", "n_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}
    assert fs.asic_violations(ok) == []
    for change in [{"n_params": 2820}, {"adc_thresholds": [400, 1600, 2500]}, {"y0_bits": 7}, {"n_params": None}]:
        d = {**ok, **change}
        if change.get("n_params", 0) is None:
            d.pop("n_params")
        assert len(fs.asic_violations(d)) == 1


def _truth():
    return pd.DataFrame({"sample_id": ["a:0", "a:1", "b:0"], "pt": [-5.0, 0.15, 3.0], "cls": [0, 2, 0]})


def _p(ids, preds):
    p0 = [0.9 if c == 0 else 0.1 for c in preds]
    return pd.DataFrame({"sample_id": ids, "pred": preds, "p0": p0, "p1": [(1 - x) / 2 for x in p0],
                         "p2": [(1 - x) / 2 for x in p0]})


def test_join_missing_prediction_is_read_out_and_counted():
    m, cov = fs.join(_p(["a:1", "b:0", "z:9"], [2, 1, 0]), _truth())
    assert m["pred"].tolist() == [0, 2, 1] and m["p0"].tolist() == [1.0, 0.1, 0.1]
    assert cov == {"n_truth": 3, "n_scored": 2, "n_truth_without_prediction": 1, "n_predictions_without_truth": 1}


@pytest.mark.parametrize("change", [{"sample_id": ["a:0", "a:0"]}, {"pred": [3, 0]}, {"p0": [1.2, 0.5]},
                                    {"p2": [np.nan, 0.1]}, {"p1": None}])
def test_join_rejects_invalid(change):
    pred = _p(["a:0", "a:1"], [0, 1])
    for k, v in change.items():
        pred = pred.drop(columns=k) if v is None else pred.assign(**{k: v})
    with pytest.raises(ValueError):
        fs.join(pred, _truth())


def _curve_case():
    sig = [(-5.0, p) for p in (0.9, 0.8, 0.6, 0.2)]
    bkg = [(-1.55, p) for p in (0.1, 0.3, 0.5, 0.6, 0.7)]
    pt, p0 = map(np.array, zip(*(sig + bkg)))
    return pt, p0, _ref(b0=4, b5=5)


def test_rejection_at_efficiency_interpolates_and_cuts_strictly_below():
    pt, p0, ref = _curve_case()
    r = fs.rejection_at_efficiency(pt, p0, ref, target=0.75)  # exact: cut 0.6, bkg p0 < 0.6 -> 3 of 5
    assert r == {fs.PRIMARY: pytest.approx(0.6), "p0_cut": pytest.approx(0.6)}
    r = fs.rejection_at_efficiency(pt, p0, ref, target=0.6)  # between cuts 0.6 (eff .75) and 0.7 (eff .5)
    assert r[fs.PRIMARY] == pytest.approx(0.6 + 0.6 * 0.2) and r["p0_cut"] == pytest.approx(0.66)
    perm = np.random.default_rng(0).permutation(len(pt))
    assert fs.rejection_at_efficiency(pt[perm], p0[perm], ref, target=0.6) == r


def test_rejection_at_efficiency_all_tied_is_defined():
    pt, p0, ref = _curve_case()
    r = fs.rejection_at_efficiency(pt, np.full(len(pt), 0.4), ref)
    assert r[fs.PRIMARY] == pytest.approx(1 - fs.TARGET_EFFICIENCY) and np.isfinite(r["p0_cut"])


def test_rejection_at_efficiency_without_signal_is_nan():
    r = fs.rejection_at_efficiency([-1.55, 1.05], [0.2, 0.8], _ref(b5=1, b31=1))
    assert np.isnan(r[fs.PRIMARY]) and np.isnan(r["p0_cut"])


def test_rejection_at_efficiency_cut_moves_with_reweighting():
    pt = np.array([-5.0, -5.0, 3.0, 3.0, -1.55, -1.55])
    p0 = np.array([0.9, 0.2, 0.8, 0.7, 0.1, 0.5])
    a = fs.rejection_at_efficiency(pt, p0, _ref(b0=1000, b41=10, b5=2), target=0.6)
    b = fs.rejection_at_efficiency(pt, p0, _ref(b0=10, b41=1000, b5=2), target=0.6)
    assert a["p0_cut"] < 0.5 < b["p0_cut"]


def test_median_seed():
    assert fs.median_seed({0: 0.2, 1: 0.4, 2: 0.3}) == 2
    assert fs.median_seed({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}) == 1
    assert fs.median_seed({0: 0.3, 1: 0.3, 2: 0.1}) == 0


def _metrics(tmp, split, name, bkg, track="open", violations=()):
    d = tmp / split / name
    d.mkdir(parents=True)
    m = {"coverage": {"n_scored": 3, "n_truth": 3}, "metrics": {fs.PRIMARY: bkg, "background_rejection": 0.1,
         "signal_efficiency": 0.9, "data_reduction": 0.2}, "descriptor": {"track": track}, "asic_violations": list(violations)}
    (d / "metrics.json").write_text(json.dumps(m))


def test_report_seed_aggregation_tracks_and_ranking(tmp_path):
    for s, v in enumerate([0.2, 0.3, 0.4]):
        _metrics(tmp_path, "test", f"M2_s{s}", v)
    _metrics(tmp_path, "test", "QM2_s0", 0.25, "asic")
    _metrics(tmp_path, "test", "QM2_w4a8_s0", 0.5, "asic")
    _metrics(tmp_path, "test", "BAD_s0", 0.9, "asic", ["y0_bits 7 != 6"])
    md = report.build(tmp_path, fs.METRICS, fs.PRIMARY, {"QM2_w4a8"}, "t")
    asic, open_ = md.split("## test — open track")
    assert "| M2 | 3 | 3/3 | 0.3000 ± 0.1000 |" in open_
    assert "BAD" not in asic and "BAD" in open_ and "| M2 |" not in asic
    rows = [line.split(" | ")[0] for line in open_.splitlines() if line.startswith("| ") and "seeds" not in line]
    assert rows == ["| BAD", "| *QM2_w4a8 (auxiliary)*", "| M2", "| QM2"]


@pytest.fixture
def results(tmp_path, monkeypatch):
    from smartpix_bench.filter import truth
    monkeypatch.setattr(truth, "scored_ids", lambda split, cache_dir=None: _truth()["sample_id"])
    (tmp_path / "test").mkdir()
    return tmp_path


def _pred(results, name, ids, preds, desc=None):
    _p(ids, preds).to_csv(results / "test" / f"predictions_{name}.csv", index=False)
    if desc:
        (results / "test" / f"descriptor_{name}.json").write_text(json.dumps(desc))


def test_submission_packages_scored_ids_in_truth_order(results):
    _pred(results, "ok", ["b:0", "x:1", "a:1", "a:0"], [1, 0, 2, 0],
          {"track": "asic", "n_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6})
    assert submission.package("filter", results, None, results / "cb") == 0
    with zipfile.ZipFile(results / "cb" / "ok_submission.zip") as z:
        assert sorted(z.namelist()) == ["descriptor.json", "predictions_test.csv"]
        df = pd.read_csv(z.open("predictions_test.csv"))
    assert df["sample_id"].tolist() == ["a:0", "a:1", "b:0"] and df["pred"].tolist() == [0, 2, 1]
    assert list(df.columns) == ["sample_id", *fs.OUTPUT_COLUMNS] and df["p0"].tolist() == [0.9, 0.1, 0.1]


@pytest.mark.parametrize("ids,preds,desc", [
    (["a:0", "a:1"], [0, 2], None),
    (["a:0", "a:1", "b:0"], [0, 5, 1], None),
    (["a:0", "a:1", "b:0"], [0, 2, 1], {"track": "asic", "n_params": 3000, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}),
])
def test_submission_refuses(results, ids, preds, desc):
    _pred(results, "bad", ids, preds, desc)
    assert submission.package("filter", results, None, results / "cb") == 1
    assert not (results / "cb" / "bad_submission.zip").exists()
