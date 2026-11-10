import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("fetch_weights", Path(__file__).parents[1] / "scripts" / "fetch_weights.py")
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)


def manifest(tmp_path, files):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": files}))
    return tmp_path / "MANIFEST.json"


def test_copies_local_artifacts_and_checks_sha256(tmp_path):
    (tmp_path / "filter").mkdir()
    (tmp_path / "filter" / "M2_s1.scaler.json").write_text("{}")
    good = hashlib.sha256(b"{}").hexdigest()
    m = manifest(tmp_path, {"filter/M2_s1.scaler.json": {"sha256": good}})
    assert fw.main(["--out", str(tmp_path / "out"), "--manifest", str(m)]) == 0
    assert (tmp_path / "out" / "filter" / "M2_s1.scaler.json").read_text() == "{}"
    m = manifest(tmp_path, {"filter/M2_s1.scaler.json": {"sha256": "0" * 64}})
    assert fw.main(["--out", str(tmp_path / "out2"), "--manifest", str(m)]) == 1


def test_empty_or_missing_fails(tmp_path):
    assert fw.main(["--out", str(tmp_path / "o"), "--manifest", str(manifest(tmp_path, {}))]) == 1
    m = manifest(tmp_path, {"regression/x.weights.h5": {"sha256": "0" * 64}})
    assert fw.main(["--out", str(tmp_path / "o"), "--manifest", str(m), "--from-local", str(tmp_path / "nowhere")]) == 1
