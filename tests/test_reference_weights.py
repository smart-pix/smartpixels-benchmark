import importlib.util
import json
from pathlib import Path

import pytest

W = Path(__file__).parents[1] / "weights"
spec = importlib.util.spec_from_file_location("fetch_weights", Path(__file__).parents[1] / "scripts" / "fetch_weights.py")
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)
MANIFEST = json.loads((W / "MANIFEST.json").read_text())["files"]


def test_manifest_lists_every_committed_file_with_its_checksum(tmp_path):
    on_disk = {str(f.relative_to(W)) for f in W.glob("*/*")}
    assert on_disk == set(MANIFEST)
    assert fw.main(["--out", str(tmp_path), "--manifest", str(W / "MANIFEST.json")]) == 0


def test_reference_checkpoints_load_into_their_builders():
    pytest.importorskip("tensorflow")
    from smartpix_bench.filter import models as fm
    from smartpix_bench.regression import models as rm
    from smartpix_bench.regression import predict

    stems = sorted({n.split(".")[0] for n in MANIFEST})
    assert {s.split("/")[0] for s in stems} == {"filter", "regression"}
    for stem in stems:
        task, name = stem.split("/")
        if task == "filter":
            model = name.rsplit("_s", 1)[0]
            m = fm.build(model)
            m.load_weights(str(W / f"{stem}.weights.h5"))
            assert m.count_params() == fm.MODELS[model][1]
        else:
            m, meta = predict.load_model(W / stem)
            assert f"{meta['model']}_{meta['config']}_s{meta['seed']}" == name
            if meta["config"] != "c1":
                assert m.count_params() == rm.MODELS[meta["model"]]
