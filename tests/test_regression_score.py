import json

import numpy as np
import pandas as pd
import pytest

from smartpix_bench.common import report, submission
from smartpix_bench.regression import baselines, models, score, truth
from smartpix_bench.regression import data as rd


def test_i68_uses_ceil_of_68_percent():
    # 0.68 * 25 = 17 exactly: ceil keeps 17 points (width 16); floor + 1 would keep 18.
    assert score.i68(np.arange(25.0)) == 16
    assert score.i68([0.0, 1, 2, 3, 10]) == 3
    assert np.isnan(score.i68([]))


def test_angle_and_sigma_propagation():
    np.testing.assert_allclose(score.angle([1.0, 0.0, -1.0]), [45, 90, 135])
    df = pd.DataFrame({"cotBeta": [0.3], "sigma_cotBeta": [1e-6]})
    d = score.angle([0.3 + 1e-6]) - score.angle([0.3])
    np.testing.assert_allclose(score.physical(df, "beta", sigma=True), np.abs(d), rtol=1e-4)


def frames(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"sample_id": [f"f:{i}" for i in range(n)], "x-midplane": rng.normal(0, 100, n),
                      "y-midplane": rng.normal(0, 20, n), "cotAlpha": rng.normal(0, 1, n), "cotBeta": rng.normal(0, 1, n)})
    p = t.copy()
    for c, s in zip(rd.TARGETS, (10.0, 2.0, 0.05, 0.05)):
        p[c] = t[c] + rng.normal(0, s, n)
        p[f"sigma_{c}"] = s
    return p, t


def test_pulls_nll_and_coverage():
    p, t = frames()
    m = score.metrics(p, t)
    assert m["pull_std_x"] == pytest.approx(1, abs=0.1) and m["frac_1sigma_x"] == pytest.approx(0.68, abs=0.06)
    p2 = p.assign(**{"sigma_x-midplane": 20.0})
    m2 = score.metrics(p2, t)
    assert m2["pull_std_x"] == pytest.approx(0.5, abs=0.05) and m2["nll_x"] > m["nll_x"] + 0.1
    r = (p["x-midplane"] - t["x-midplane"]) / 10
    assert m["nll_x"] == pytest.approx(np.mean(0.5 * r**2 + np.log(10) + 0.5 * np.log(2 * np.pi)))


def test_join_excludes_missing_and_counts_them():
    p, t = frames()
    p.loc[0, "cotAlpha"] = np.nan
    res = score.score(p.iloc[5:], t)
    assert res["coverage"]["n_truth_without_prediction"] == 5 and res["coverage"]["n_scored"] == len(t) - 5
    assert res["metrics"]["n_x"] == len(t) - 5 and res["metrics"]["n_alpha"] == len(t) - 5
    res = score.score(p, t)
    assert res["metrics"]["n_alpha"] == len(t) - 1 and res["metrics"]["n_x"] == len(t)
    with pytest.raises(ValueError, match="duplicate"):
        score.score(pd.concat([p, p.iloc[:1]]), t)
    with pytest.raises(ValueError, match="infinite"):
        score.score(p.assign(cotBeta=np.inf), t)
    with pytest.raises(ValueError, match="missing"):
        score.score(p.drop(columns="cotBeta"), t)


def test_rank_is_mean_i68_ratio_over_x_y_beta():
    p, t = frames()
    m = score.metrics(p, t)
    ref = {"x": m["I68_x"], "y": m["I68_y"] / 2, "beta": m["I68_beta"] * 4}
    assert score.metrics(p, t, ref)[score.PRIMARY] == pytest.approx((1 + 2 + 0.25) / 3)
    assert np.isnan(m[score.PRIMARY])
    assert score.metrics(p.assign(cotAlpha=np.nan), t, ref)[score.PRIMARY] == pytest.approx((1 + 2 + 0.25) / 3)
    p.loc[3, "cotBeta"] = np.nan
    assert np.isnan(score.metrics(p, t, ref)[score.PRIMARY])


C4 = models.descriptor("Full_MLP", "c4", 2264, [250.0, 700.0, 1700.0])


@pytest.mark.parametrize("change,needle", [
    ({}, None), ({"n_params": 2735}, "n_params"), ({"slices": [0, 18]}, "slices"),
    ({"thresholds": [250.0, 250.0, 1700.0]}, "thresholds"), ({"thresholds": [250.0, 700.0]}, "thresholds"),
    ({"thresholds": [250.0, float("nan"), 1700.0]}, "thresholds"), ({"weight_bits": 9}, "weight_bits"),
    ({"activation_bits": None}, "activation_bits")])
def test_asic_violations(change, needle):
    d = {k: v for k, v in {**C4, **change}.items() if v is not None}
    v = score.asic_violations(d)
    assert v == [] if needle is None else (len(v) == 1 and needle in v[0])


def test_descriptor_tracks():
    assert C4["track"] == "asic" and models.descriptor("Full_MLP", "c3", 2264, [1, 2, 3])["track"] == "open"
    assert models.descriptor("Full_MLP", "c1", 9999)["slices"] == list(range(20))


def test_report_ranks_lower_first(tmp_path):
    for name, i68 in (("A", 2.0), ("B", 1.0)):
        d = tmp_path / "3s" / name
        d.mkdir(parents=True)
        d.joinpath("metrics.json").write_text(json.dumps({"coverage": {"n_scored": 1, "n_truth": 1},
                                                          "metrics": {score.PRIMARY: i68}}))
    md = report.build(tmp_path, [score.PRIMARY], score.PRIMARY, set(), "t", score.HIGHER_IS_BETTER)
    assert md.index("| B |") < md.index("| A |")
    md = report.build(tmp_path, [score.PRIMARY], score.PRIMARY, {"A": "A (optimistic)"}, "t", True)
    assert md.index("| *A (optimistic) (auxiliary)* |") < md.index("| B |")


def cluster(x_charge: dict, y_charge: dict):
    """One (1, 16 y, 16 x) cluster whose x / y profiles are the given {pixel: charge}."""
    a = np.zeros((1, 16, 16))
    for i, q in x_charge.items():
        a[0, 0, i] = q
    a[0, 0, 0] -= sum(x_charge.values())
    for j, q in y_charge.items():
        a[0, j, 0] += q
    return a


def test_baselines_on_a_synthetic_cluster():
    a = cluster({7: 1.0, 8: 3.0}, {3: 1.0, 4: 1.0, 5: 2.0})
    px, py = baselines.profiles(a)
    assert px[0, 7] == 1 and px[0, 8] == 3 and px[0, 0] == 0 and py[0, 5] == 2 and py[0, 4] == 1
    lab = pd.DataFrame({"sample_id": ["s:0"], "x-midplane": [0.0], "y-midplane": [0.0], "cotAlpha": [-1.0], "cotBeta": [0.2]})
    out = baselines.predict(a, lab)
    dy = 100 * 0.3046
    assert out["Barycenter"]["x-midplane"][0] == pytest.approx((0 * 1 + 1 * 3) / 4 * 50)
    assert out["Barycenter"]["y-midplane"][0] == pytest.approx((-4 + -3 + 2 * -2) / 4 * 12.5 + dy / 2)
    # Eq 11: x_F = x_L = 25 µm, W = |T cotα| = 100 µm.
    assert out["LocalReco"]["x-midplane"][0] == pytest.approx(25 + (3 - 1) / 4 * 100 / 2)
    y_f, y_l = -4 * 12.5 + 6.25, -2 * 12.5 - 6.25
    assert out["LocalReco"]["y-midplane"][0] == pytest.approx(
        (y_f + y_l) / 2 + (2 - 1) / 3 * (abs(100 * 0.2 + dy) - (y_l - y_f)) / 2 + dy / 2)
    assert out["Geometric"]["cotAlpha"][0] == out["GeometricCorrected"]["cotAlpha"][0] == pytest.approx(-0.5)
    assert out["GeometricCorrected"]["cotBeta"][0] == pytest.approx(2 * 12.5 / 100 - 0.3046)
    for cot_b, sign in ((-0.2, 1), (-0.5, -1)):
        got = baselines.predict(a, lab.assign(cotBeta=cot_b))
        assert got["GeometricCorrected"]["cotBeta"][0] == pytest.approx(sign * 2 * 12.5 / 100 - 0.3046)
        assert got["Geometric"]["cotBeta"][0] == pytest.approx(-2 * 12.5 / 100)
    assert out["Barycenter"]["cotAlpha"].isna().all() and out["Geometric"]["x-midplane"].isna().all()


def test_baselines_empty_cluster_is_nan():
    lab = pd.DataFrame({"sample_id": ["s:0"], "cotAlpha": [1.0], "cotBeta": [1.0]})
    out = baselines.predict(np.zeros((1, 16, 16)), lab)
    assert all(out[n][c].isna().all() for n, c in (("Barycenter", "x-midplane"), ("LocalReco", "y-midplane"),
                                                   ("Geometric", "cotBeta")))


def test_truth_and_submission_pipeline(fixture_data, tmp_path):
    res = tmp_path / "regression"
    for split in truth.SPLITS:
        t = truth.truth_frame(split, fixture_data)
        files = rd.files(split, fixture_data)
        _, lab = rd.load(files, sorted(files), [19])
        assert t["sample_id"].tolist() == lab["sample_id"].tolist() and list(t.columns) == ["sample_id", *rd.TARGETS]
        baselines.main(["--split", split, "--data-dir", str(fixture_data), "--out", str(res / split)])
        p = pd.read_csv(res / split / "predictions_LocalReco.csv", dtype={"sample_id": str})
        assert p["sample_id"].tolist() == t["sample_id"].tolist()
    assert submission.package("regression", res, fixture_data, tmp_path / "zips") == 0
    assert {f.name for f in (tmp_path / "zips").iterdir()} == {f"{n}_submission.zip" for n in baselines.NAMES}
    pd.read_csv(res / "3s" / "predictions_Geometric.csv").iloc[1:].to_csv(res / "3s" / "predictions_Geometric.csv", index=False)
    assert submission.package("regression", res, fixture_data, tmp_path / "zips2") == 1
