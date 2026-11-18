import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartpix_bench.common import submission
from smartpix_bench.filter import cache
from smartpix_bench.filter import score as fs
from smartpix_bench.filter import truth as ft
from smartpix_bench.regression import data as rd
from smartpix_bench.regression import score as rs
from smartpix_bench.regression import truth as rt

spec = importlib.util.spec_from_file_location("build_bundle", Path(__file__).parents[1] / "codabench" / "build_bundle.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

FILTER_ASIC = {"track": "asic", "n_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}
REG_ASIC = {"track": "asic", "n_params": 2264, "slices": [0, 19], "thresholds": [250.0, 700.0, 1700.0],
            "weight_bits": 8, "activation_bits": 8}


def filter_pred(t, noise, seed=0):
    p = np.eye(3)[t["cls"]] + np.random.default_rng(seed).uniform(0, noise, (len(t), 3))
    p /= p.sum(1, keepdims=True)
    return pd.DataFrame({"sample_id": t["sample_id"], "pred": p.argmax(1), "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]})


def synthetic_truth(n=400):
    """The fixture test split has no |pT| > 2 GeV cluster, so the primary metric needs a wider truth."""
    pt = np.random.default_rng(1).uniform(-4, 4, n)
    cls = np.where(np.abs(pt) > 0.2, 0, np.where(pt < 0, 1, 2))
    return pd.DataFrame({"sample_id": [f"s:{i}" for i in range(n)], "pt": pt, "cls": cls})


@pytest.fixture(scope="module")
def filter_bundle(fixture_data, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cb_filter")
    cache.build(fixture_data, tmp / "cache")
    t = synthetic_truth()
    mp = pytest.MonkeyPatch()
    mp.setattr(ft, "truth_frame", lambda split, cache_dir=None: t)
    ref = tmp / "pt_reference.json"
    ref.write_text(json.dumps({"edges": fs.EDGES.tolist(), "counts": fs.histogram(t["pt"]).tolist()}))
    res = tmp / "results"
    (res / "test").mkdir(parents=True)
    for name, noise, desc in (("good_s0", 0.5, FILTER_ASIC), ("noisy_s0", 3.0, None)):
        pred = filter_pred(t, noise)
        pred.to_csv(res / "test" / f"predictions_{name}.csv", index=False)
        if desc:
            (res / "test" / f"descriptor_{name}.json").write_text(json.dumps(desc))
        m = fs.score(pred, t, fs.load_reference(ref), desc)
        (res / "test" / name).mkdir()
        (res / "test" / name / "metrics.json").write_text(json.dumps(m))
    assert submission.package("filter", res, tmp / "cache", res / "codabench") == 0
    rc = bb.build("filter", res, "good_s0", tmp / "cache", ref, tmp / "build")
    yield tmp, res, rc, t
    mp.undo()


def run(bundle_dir, members: dict, tmp_path, name="sub.zip"):
    z = tmp_path / name
    bb.zip_files(z, members)
    return bb.run_scoring(bundle_dir / "bundle", z)


def test_filter_bundle_layout_and_scores_match_pipeline(filter_bundle):
    tmp, res, rc, t = filter_bundle
    b = tmp / "build" / "bundle"
    assert rc == 0
    pd.testing.assert_frame_equal(pd.read_csv(b / "reference_data" / "truth_test.csv", dtype={"sample_id": str}), t)
    with zipfile.ZipFile(b / "starting_kit" / "sample_submission.zip") as z:
        assert sorted(z.namelist()) == ["descriptor.json", "predictions_test.csv"]
    assert sorted(p.name for p in (b / "solution").iterdir()) == ["descriptor.json", "predictions_test.csv"]
    code, good = bb.run_scoring(b, res / "codabench" / "good_s0_submission.zip")
    _, base = bb.run_scoring(b, tmp / "build" / "extra" / "baseline_submission.zip")
    want = json.loads((res / "test" / "good_s0" / "metrics.json").read_text())["metrics"]
    assert code == 0 and good["asic"] == 1 and all(good[k] == pytest.approx(want[k], rel=1e-12) for k in fs.METRICS)
    assert base["asic"] == 0 and base[fs.PRIMARY] == pytest.approx(1 - fs.TARGET_EFFICIENCY) and base["data_reduction"] == 0
    assert good[fs.PRIMARY] > base[fs.PRIMARY] + 0.05


def test_filter_check_flags_a_score_that_differs(filter_bundle):
    tmp, res, _, _ = filter_bundle
    m = res / "test" / "noisy_s0" / "metrics.json"
    saved = m.read_text()
    d = json.loads(saved)
    d["metrics"]["data_reduction"] += 1e-3
    m.write_text(json.dumps(d))
    try:
        assert bb.check("filter", tmp / "build" / "bundle", res, [res / "codabench" / "noisy_s0_submission.zip"]) == 1
    finally:
        m.write_text(saved)
    assert bb.check("filter", tmp / "build" / "bundle", res, [res / "codabench" / "noisy_s0_submission.zip"]) == 0


@pytest.mark.parametrize("edit,needle", [
    (lambda p: p.iloc[1:], "no prediction for 1 of"),
    (lambda p: pd.concat([p, p.iloc[:1]]), "duplicate"),
    (lambda p: p.assign(p0=np.r_[np.nan, p["p0"].to_numpy()[1:]]), "finite"),
    (lambda p: p.assign(pred=np.r_[5, p["pred"].to_numpy()[1:]]), "outside"),
    (lambda p: p.drop(columns="p2"), "missing columns"),
])
def test_filter_malformed_submissions_fail_naming_the_problem(filter_bundle, tmp_path, edit, needle):
    tmp, _, _, t = filter_bundle
    code, err = run(tmp / "build", {"predictions_test.csv": edit(filter_pred(t, 0.5)).to_csv(index=False)}, tmp_path)
    assert code == 1 and "SUBMISSION ERROR" in err and needle in err


def test_filter_wrapping_folder_missing_file_and_asic_violation(filter_bundle, tmp_path):
    tmp, _, _, t = filter_bundle
    csv = filter_pred(t, 0.5).to_csv(index=False)
    code, flat = run(tmp / "build", {"predictions_test.csv": csv}, tmp_path, "a.zip")
    code2, wrapped = run(tmp / "build", {"sub/predictions_test.csv": csv}, tmp_path, "b.zip")
    assert code == code2 == 0 and flat == wrapped
    code, err = run(tmp / "build", {"predictions.csv": csv}, tmp_path, "c.zip")
    assert code == 1 and "'predictions_test.csv' not found" in err
    code, err = run(tmp / "build", {"predictions_test.csv": csv, "descriptor.json": json.dumps({**FILTER_ASIC, "y0_bits": 8})},
                    tmp_path, "d.zip")
    assert code == 1 and "y0_bits" in err


def reg_pred(t, scale, seed=0):
    rng = np.random.default_rng(seed)
    p = t.copy()
    for c, s in zip(rd.TARGETS, (10.0, 2.0, 0.05, 0.05)):
        p[c] = t[c] + rng.normal(0, s * scale, len(t))
        p[f"sigma_{c}"] = s * scale
    return p


@pytest.fixture(scope="module")
def regression_bundle(fixture_data, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cb_regression")
    truths = {s: rt.truth_frame(s, fixture_data) for s in rt.SPLITS}
    ref = tmp / "i68_reference.json"
    ref.write_text(json.dumps({"model": "ref", "seed": 0, **{
        s: {v: rs.metrics(reg_pred(t, 1.0), t)[f"I68_{v}"] for v in rs.RANK_VARIABLES} for s, t in truths.items()}}))
    res = tmp / "results"
    for name, scale, desc in (("ref_s0", 1.0, REG_ASIC), ("slim_s0", 2.0, None)):
        for s, t in truths.items():
            (res / s / name).mkdir(parents=True)
            pred = reg_pred(t, scale)
            if name == "slim_s0":
                pred = pred.assign(cotAlpha=np.nan).drop(columns=[f"sigma_{c}" for c in rd.TARGETS])
            pred.to_csv(res / s / f"predictions_{name}.csv", index=False)
            if desc:
                (res / s / f"descriptor_{name}.json").write_text(json.dumps(desc))
            m = rs.score(pred, t, rs.load_reference(s, ref), desc)
            (res / s / name / "metrics.json").write_text(json.dumps(m))
    assert submission.package("regression", res, fixture_data, res / "codabench") == 0
    rc = bb.build("regression", res, None, fixture_data, ref, tmp / "build")
    return tmp, res, rc, truths


def test_regression_bundle_scores_match_pipeline_and_discriminate(regression_bundle):
    tmp, res, rc, truths = regression_bundle
    b = tmp / "build" / "bundle"
    assert rc == 0 and sorted(p.name for p in (b / "solution").iterdir()) == ["descriptor.json", "predictions_3s.csv"]
    assert sorted(p.name for p in (b / "reference_data").iterdir()) == ["truth_3s.csv"]
    with zipfile.ZipFile(res / "codabench" / "ref_s0_submission.zip") as z:
        assert sorted(z.namelist()) == ["descriptor.json", "predictions_3s.csv"]
    code, ref = bb.run_scoring(b, b / "starting_kit" / "sample_submission.zip")
    _, slim = bb.run_scoring(b, res / "codabench" / "slim_s0_submission.zip")
    _, base = bb.run_scoring(b, tmp / "build" / "extra" / "baseline_submission.zip")
    assert code == 0 and ref["asic"] == 1 and ref["3s_rank_i68"] == pytest.approx(1.0)
    assert slim["3s_I68_alpha"] is None and slim["3s_rank_i68"] == pytest.approx(2.0, rel=0.3)
    assert base["3s_rank_i68"] > 3 and not any(k.startswith("2s_") for k in ref)


@pytest.mark.parametrize("edit,needle", [
    (lambda p: p.iloc[1:], "no prediction for 1 of"),
    (lambda p: p.assign(cotBeta=np.inf), "infinite"),
    (lambda p: p.drop(columns="cotBeta"), "missing columns"),
])
def test_regression_malformed_submissions_fail(regression_bundle, tmp_path, edit, needle):
    tmp, _, _, truths = regression_bundle
    code, err = run(tmp / "build", {"predictions_3s.csv": edit(reg_pred(truths["3s"], 1.0)).to_csv(index=False)}, tmp_path)
    assert code == 1 and "predictions_3s.csv" in err and needle in err
    code, err = run(tmp / "build", {"predictions_2s.csv": reg_pred(truths["2s"], 1.0).to_csv(index=False)}, tmp_path, "one.zip")
    assert code == 1 and "'predictions_3s.csv' not found" in err
