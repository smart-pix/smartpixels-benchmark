import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from smartpix_bench.common.features import add_noise, digitize  # noqa: E402
from smartpix_bench.regression import data as rd  # noqa: E402
from smartpix_bench.regression import features as rf  # noqa: E402
from smartpix_bench.regression import losses, models, predict, score, train  # noqa: E402
from smartpix_bench.regression.softquantize import AnnealK, SoftQuantize, k_at  # noqa: E402


@pytest.mark.parametrize("name", sorted(models.MODELS))
@pytest.mark.parametrize("quantized", [False, True])
def test_param_counts(name, quantized):
    assert models.build(name, 2, quantized).count_params() == models.MODELS[name]


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_regularizers(name):
    m = models.build(name)
    weighted = [layer for layer in m.layers if layer.weights]
    for layer in weighted:
        cfg = layer.get_config()
        ks = [cfg[k] for k in ("kernel_regularizer", "depthwise_regularizer", "pointwise_regularizer") if cfg.get(k)]
        assert ks and all(k["config"]["l1"] == pytest.approx(0.01) and k["config"]["l2"] == 0 for k in ks)
        act = cfg.get("activity_regularizer")
        if layer is weighted[-1]:
            assert act is None
        else:
            assert act["config"]["l2"] == pytest.approx(0.01)


@pytest.mark.parametrize("name", ["Full_Conv2D", "Full_Conv1D", "Full_MLP"])
def test_c4_quantizer_bits(name):
    m = models.build(name, 2, True)
    bits = {}
    for layer in m.layers:
        for k in ("kernel_quantizer", "depthwise_quantizer", "pointwise_quantizer"):
            if q := getattr(layer, f"{k}_internal", None):
                bits.setdefault(type(layer).__name__, set()).add(q.bits)
        if type(layer).__name__ == "QActivation":
            bits.setdefault("act", set()).add(layer.quantizer.bits)
    assert bits.pop("QDense") == {8}
    assert all(b == {4} for k, b in bits.items() if k != "act") and max(bits["act"]) <= 8


def test_softquantize_forward_is_eq4_and_gradient_reaches_thresholds():
    sq = SoftQuantize((248.0, 668.0, 1663.0))
    x = tf.constant([[0.0, 247.9, 248.0, 667.99, 668.0, 1662.0, 1663.0, 5000.0]])
    with tf.GradientTape() as g:
        y = sq(x)
        loss = tf.reduce_sum(y * tf.range(8, dtype=tf.float32))
    np.testing.assert_array_equal(y.numpy(), [[0, 0, 1, 1, 2, 2, 3, 3]])
    np.testing.assert_array_equal(y.numpy(), digitize(x.numpy(), [248, 668, 1663]))
    np.testing.assert_allclose(sq.thresholds().numpy(), [248, 668, 1663], rtol=1e-6)
    grad = g.gradient(loss, sq.theta).numpy()
    assert np.all(np.isfinite(grad)) and np.all(grad != 0)


def test_softquantize_tau_and_soft_gradient():
    sq = SoftQuantize((100.0, 300.0, 700.0))
    x = tf.constant([150.0, 650.0])
    sq(x)
    sq.k.assign(2.0)
    with tf.GradientTape() as g:
        g.watch(x)
        y = sq(x)
    # deltas 100, 200, 400 -> tau = 150, 300, 400
    t, tau = np.array([100.0, 300, 700]), np.array([150.0, 300, 400])
    s = 1 / (1 + np.exp(-2 * (x.numpy()[:, None] - t) / tau))
    np.testing.assert_allclose(g.gradient(y, x).numpy(), (s * (1 - s) * 2 / tau).sum(1), rtol=1e-4)


def test_k_schedule_and_callback():
    assert k_at(0, 10) == pytest.approx(1) and k_at(9, 10) == pytest.approx(67)
    assert k_at(1, 5) == pytest.approx(67 - 66 * (1 + np.cos(np.pi / 4)) / 2)
    assert all(np.diff([k_at(e, 50) for e in range(50)]) > 0)
    sq = SoftQuantize()
    sq.build(None)
    AnnealK(sq, 10).on_epoch_begin(9)
    assert sq.k.numpy() == pytest.approx(67) and not sq.k.trainable


def test_losses_match_numpy():
    rng = np.random.default_rng(1)
    p, y = rng.normal(size=(6, 14)).astype(np.float32), rng.normal(size=(6, 4)).astype(np.float32)
    y[0] = 100  # NLL far above -log(1e-9): exercises the clip
    want = 0.0
    for i in range(6):
        L = np.zeros((4, 4))
        L[range(4), range(4)] = 1e-9 + np.maximum(p[i, 1:8:2], 0) + 0.5
        L[[1, 2, 2, 3, 3, 3], [0, 0, 1, 0, 1, 2]] = p[i, 8:14]
        cov = L @ L.T
        r = y[i] - p[i, 0:8:2]
        nll = 0.5 * r @ np.linalg.solve(cov, r) + 0.5 * np.log(np.linalg.det(cov)) + 2 * np.log(2 * np.pi)
        want += np.clip(nll, -np.log(1e9), -np.log(1e-9))
    p[:, 1:8:2] = np.maximum(p[:, 1:8:2], 0) + 0.5
    assert losses.mvn_tril_nll(y, tf.constant(p)).numpy() == pytest.approx(want, rel=1e-4)
    q = rng.normal(size=(6, 8)).astype(np.float32)
    s = np.logaddexp(0, q[:, 4:]) + 1e-9
    z = (y - q[:, :4]) / s
    assert losses.diag_nll(y, tf.constant(q)).numpy() == pytest.approx(np.sum(0.5 * z * z + np.log(s) + 0.5 * np.log(2 * np.pi)), rel=1e-5)
    assert losses.sse(y[:, :3], tf.constant(q[:, :3])).numpy() == pytest.approx(np.sum((y[:, :3] - q[:, :3]) ** 2), rel=1e-5)
    assert losses.LOSSES["Slim"] is losses.sse


def test_decode_matches_loss_layout():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(5, 14)).astype(np.float32)
    L = losses.max_scale_tril(tf.constant(p)).numpy()
    scale = np.array([2.0, 3.0, 4.0, 5.0])
    d = predict.decode("Max", p, scale)
    np.testing.assert_allclose(d[predict.SIGMA].to_numpy(), np.sqrt(np.einsum("nij,nij->ni", L, L)) * scale, rtol=1e-5)
    np.testing.assert_allclose(d[rd.TARGETS].to_numpy(), p[:, 0:8:2] * scale, rtol=1e-6)
    f = predict.decode("Full", p[:, :8], scale)
    np.testing.assert_allclose(f[predict.SIGMA].to_numpy(), losses.full_sigma(tf.constant(p[:, :8])).numpy() * scale, rtol=1e-5)
    s = predict.decode("Slim", p[:, :3], scale[[0, 1, 3]])
    assert s["cotAlpha"].isna().all() and s["sigma_x-midplane"].isna().all()
    np.testing.assert_allclose(s["cotBeta"], p[:, 2] * 5, rtol=1e-6)


def test_sq_model_has_noise_in_front():
    m = models.build("Full_MLP", softquantize=(248.0, 668.0, 1663.0))
    assert [type(layer).__name__ for layer in m.layers[1:3]] == ["GaussianNoise", "SoftQuantize"]
    assert m.layers[1].stddev == 80


def test_two_stage_train_predict(fixture_data, tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "VAL_FRAC", 0.5)
    w = tmp_path / "w"
    fit, seen = train.fit, {}

    def spy(m, x_tr, y_tr, x_va, *a, **k):
        seen[m.name, len(m.layers)] = np.unique(x_tr)
        seen["x_va"] = x_va
        return fit(m, x_tr, y_tr, x_va, *a, **k)
    monkeypatch.setattr(train, "fit", spy)
    for config in ("sq", "c3", "c4", "c1"):
        seen.clear()
        val = train.run("Full_MLP", config, 0, w, epochs=2, data_dir=fixture_data)
        assert np.isfinite(val["I68_x"])
        x_va = seen.pop("x_va")
        (vals,) = seen.values()
        assert set(vals) <= {0, 1, 2, 3} if config in ("c3", "c4") else len(vals) > 4
        if config == "sq":
            clean = train.load_train_val([0, 19], fixture_data)[1][0]
            want = add_noise(clean, np.random.default_rng(train.VAL_NOISE_SEED), 80)
            np.testing.assert_allclose(x_va, want, rtol=1e-6)
            assert np.std(x_va - clean) == pytest.approx(80, rel=0.1)
    meta = predict.load_meta(w / "Full_MLP_c4_s0")
    sq = json.loads((w / "Full_MLP_sq_s0.meta.json").read_text())
    assert meta["thresholds"] == sq["thresholds"] and np.all(np.diff(sq["thresholds"]) > 0)
    assert sq["thresholds"] != pytest.approx([248, 668, 1663], abs=1e-4)
    _, lab = train.load_train_val([19], fixture_data)[0]
    np.testing.assert_allclose(meta["label_scale"], rd.label_scale(lab), rtol=1e-6)
    out = tmp_path / "res"
    for stem in ("Full_MLP_c4_s0", "Full_MLP_c1_s0"):
        predict.main(["--weights", str(w / stem), "--split", "3s", "--data-dir", str(fixture_data), "--out", str(out)])
    desc = json.loads((out / "descriptor_Full_MLP_c4_s0.json").read_text())
    assert desc["n_params"] == models.MODELS["Full_MLP"] and score.asic_violations(desc) == []
    p = (out / "predictions_Full_MLP_c4_s0.csv").read_text().splitlines()
    files = rd.files("3s", fixture_data)
    assert len(p) - 1 == len(rd.load(files, sorted(files), [19])[1])
    m, meta = predict.load_model(w / "Full_MLP_c4_s0")
    x, _ = rd.load(files, sorted(files), [0, 19])
    raw = m.predict(x, verbose=0)
    dig = m.predict(digitize(x, meta["thresholds"]), verbose=0)
    got = predict.predict_split(m, meta, "3s", fixture_data)
    np.testing.assert_allclose(got["x-midplane"], dig[:, 0] * meta["label_scale"][0], rtol=1e-5)
    assert not np.allclose(raw, dig)


def test_slim_mlp_c1_standardizes_its_inputs(fixture_data, tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "VAL_FRAC", 0.5)
    fit, seen = train.fit, {}

    def spy(m, x_tr, *a, **k):
        seen[m.name] = x_tr.copy()
        return fit(m, x_tr, *a, **k)
    monkeypatch.setattr(train, "fit", spy)
    w = tmp_path / "w"
    for model in ("Slim_MLP", "Full_MLP"):
        train.run(model, "c1", 0, w, epochs=1, data_dir=fixture_data)
    raw = train.load_train_val(rf.SLICES_20, fixture_data)[0][0]
    meta = predict.load_meta(w / "Slim_MLP_c1_s0")
    x = seen["Slim_MLP"]
    assert np.array_equal(x == 0, raw == 0)
    assert x[x != 0].mean() == pytest.approx(0, abs=1e-4) and x[x != 0].std() == pytest.approx(1, rel=1e-4)
    np.testing.assert_allclose(predict.inputs(raw.copy(), meta), x, rtol=1e-5, atol=1e-5)
    assert "input_standardize" not in predict.load_meta(w / "Full_MLP_c1_s0")
    np.testing.assert_array_equal(seen["Full_MLP"], raw)
