"""Smart Pixels filter scoring program (results submission).

The upload is a zip with predictions_test.csv (sample_id, pred, p0, p1, p2 for every test cluster) and, for the
asic track, descriptor.json, at the zip root. Metrics are metrics.py, generated from the benchmark's own
smartpix_bench.filter.score, with the pT reference spectrum in reference.json.
"""

import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

ROOT = os.environ.get("CODABENCH_ROOT", "/app")
reference_dir = os.path.join(ROOT, "input", "ref")
prediction_dir = os.path.join(ROOT, "input", "res")
score_dir = os.path.join(ROOT, "output")

SUBMISSION_FILES = ["predictions_test.csv"]
PREDICTION_FILE = os.path.join(prediction_dir, "predictions_test.csv")
DESCRIPTOR_FILE = os.path.join(prediction_dir, "descriptor.json")


def fail(msg):
    print(f"SUBMISSION ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find(expected, required=True):
    """The file at the zip root, or inside one wrapping folder."""
    if os.path.exists(expected):
        return expected
    name = os.path.basename(expected)
    for dirpath, _, files in os.walk(prediction_dir):
        if name in files:
            return os.path.join(dirpath, name)
    if required:
        fail(f"'{name}' not found in the submission; zip {', '.join(SUBMISSION_FILES)} at the root of the archive")
    return None


def main():
    truth = pd.read_csv(os.path.join(reference_dir, "truth_test.csv"), dtype={"sample_id": str})
    pred = pd.read_csv(find(PREDICTION_FILE), dtype={"sample_id": str})
    if "sample_id" not in pred.columns:
        fail("predictions_test.csv is missing column sample_id")
    if pred["sample_id"].duplicated().any():
        fail("predictions_test.csv has duplicate sample_id values")
    missing = truth.loc[~truth["sample_id"].isin(pred["sample_id"]), "sample_id"]
    if len(missing):
        fail(f"predictions_test.csv has no prediction for {len(missing)} of {len(truth)} test clusters "
             f"(e.g. {missing.head(3).tolist()}); every test cluster is required")
    path = find(DESCRIPTOR_FILE, required=False)
    descriptor = json.load(open(path)) if path else None
    with open(os.path.join(HERE, "reference.json")) as f:
        n_ref = json.load(f)["counts"]
    try:
        result = metrics.score(pred, truth, n_ref, descriptor)
    except (ValueError, TypeError) as e:
        fail(f"predictions_test.csv: {e}")
    if result.get("asic_violations"):
        fail(f"descriptor.json declares the asic track but violates {result['asic_violations']}")

    scores = dict(result["metrics"])
    scores["asic"] = int(bool(descriptor) and descriptor.get("track") == "asic")
    scores = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
    print(f"[test] {result['coverage']}")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    os.makedirs(score_dir, exist_ok=True)
    with open(os.path.join(score_dir, "scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
