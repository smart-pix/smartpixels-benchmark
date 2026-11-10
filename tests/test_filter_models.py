import json

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from smartpix_bench.filter import cache, models, predict, score, train  # noqa: E402
from smartpix_bench.filter import data as fd  # noqa: E402


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_param_counts(name):
    assert models.build(name).count_params() == models.MODELS[name][1]


@pytest.mark.parametrize("name,w,a", [("QM2", 5, 10), ("QM2_w4a8", 4, 8)])
def test_qkeras_bits(name, w, a):
    m = models.build(name)
    cfg = {layer.name: layer.get_config() for layer in m.layers}
    for d in ("dense1", "dense2"):
        for q in ("kernel_quantizer", "bias_quantizer"):
            assert cfg[d][q]["config"]["bits"] == w and cfg[d][q]["config"]["integer"] == 0
    assert cfg["relu1"]["activation"] == f"quantized_relu({a},0)"


def test_m3_structure():
    m = models.build("M3")
    assert [layer.rate for layer in m.layers if isinstance(layer, tf.keras.layers.Dropout)] == [0.1]
    out = m([np.zeros((2, 8, 13, 1), np.float32), np.zeros((2, 1), np.float32)]).numpy()
    np.testing.assert_allclose(out.sum(1), 1, rtol=1e-6)


def test_train_predict_roundtrip(fixture_data, tmp_path, monkeypatch):
    cache.build(fixture_data, tmp_path / "c")
    tr = cache.load("train", tmp_path / "c")
    _, va = fd.train_val(pd.DataFrame({"i": np.arange(len(tr["cls"]))}), cache.DATA_SEED)
    ref = score.histogram(tr["pt"][va["i"].to_numpy()])
    monkeypatch.setattr(score, "load_reference", lambda: ref)
    val = train.run("QM2", 3, tmp_path / "c", tmp_path / "w", epochs=2)
    assert set(val) == {*score.METRICS, "p0_cut"}
    hist = json.loads((tmp_path / "w" / "QM2_s3.history.json").read_text())
    assert len(hist["loss"]) == 2
    predict.main(["--model", "QM2", "--weights", str(tmp_path / "w" / "QM2_s3"), "--cache-dir", str(tmp_path / "c"),
                  "--out", str(tmp_path / "r")])
    p = pd.read_csv(tmp_path / "r" / "predictions_QM2_s3.csv")
    te = cache.load("test", tmp_path / "c")
    assert p["sample_id"].tolist() == te["sample_id"].tolist() and set(p["pred"]) <= {0, 1, 2}
    assert score.check_values(p) is None
    np.testing.assert_array_equal(p["pred"], p[["p0", "p1", "p2"]].to_numpy().argmax(1))
    np.testing.assert_allclose(p[["p0", "p1", "p2"]].sum(1), 1, rtol=1e-5)
    d = json.loads((tmp_path / "r" / "descriptor_QM2_s3.json").read_text())
    assert d == {"model": "QM2", "track": "asic", "n_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}
    assert score.asic_violations(d) == []
    with pytest.raises(ValueError):
        models.build("M2").load_weights(str(tmp_path / "w" / "QM2_s3.weights.h5"))


def test_seed_reproducible(fixture_data, tmp_path):
    cache.build(fixture_data, tmp_path)
    a = cache.load("train", tmp_path)
    x = models.apply_scaler(models.raw_inputs("M2", a), models.fit_scaler(models.raw_inputs("M2", a)))
    w = [train.fit("M2", x, a["cls"], x, a["cls"], seed, epochs=1)[0].get_weights()[0] for seed in (1, 1, 2)]
    np.testing.assert_array_equal(w[0], w[1])
    assert not np.array_equal(w[0], w[2])


def test_training_recipe(monkeypatch):
    seen = {}

    def fake_fit(self, *args, **kw):
        seen.update(kw, optimizer=self.optimizer, loss=self.loss)
        return type("H", (), {"history": {"loss": [0.0]}})()

    monkeypatch.setattr(tf.keras.Model, "fit", fake_fit)
    x = np.zeros((4, 14), np.float32)
    train.fit("M2", x, np.zeros(4), x, np.zeros(4), seed=0)
    (es,) = seen["callbacks"]
    assert (es.monitor, es.patience, es.restore_best_weights) == ("val_loss", 20, True)
    assert (seen["epochs"], seen["batch_size"], seen["shuffle"]) == (200, 1024, True)
    assert isinstance(seen["optimizer"], tf.keras.optimizers.Adam)
    assert float(seen["optimizer"].learning_rate) == pytest.approx(1e-3)
    assert isinstance(seen["loss"], tf.keras.losses.SparseCategoricalCrossentropy) and not seen["loss"].get_config()["from_logits"]
