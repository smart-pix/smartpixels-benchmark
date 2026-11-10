"""Smart Pixels regression scoring program (results submission).

The upload is a zip with predictions_3s.csv and predictions_2s.csv (sample_id, x-midplane, y-midplane, cotAlpha,
cotBeta, optional sigma_<target>; empty where a target is not predicted) for every contained cluster and, for
the asic track, descriptor.json, at the zip root. Metrics are metrics.py, generated from the benchmark's own
smartpix_bench.regression.score, with the ranking reference I68 in reference.json.
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

SPLITS = ("3s", "2s")
SUBMISSION_FILES = ["predictions_3s.csv", "predictions_2s.csv"]
PREDICTION_FILES = {
    "3s": os.path.join(prediction_dir, "predictions_3s.csv"),
    "2s": os.path.join(prediction_dir, "predictions_2s.csv"),
}
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
        fail(f"'{name}' not found in the submission; zip {' and '.join(SUBMISSION_FILES)} at the root of the archive")
    return None


def main():
    path = find(DESCRIPTOR_FILE, required=False)
    descriptor = json.load(open(path)) if path else None
    with open(os.path.join(HERE, "reference.json")) as f:
        reference = json.load(f)
    scores = {}
    for split in SPLITS:
        name = os.path.basename(PREDICTION_FILES[split])
        truth = pd.read_csv(os.path.join(reference_dir, f"truth_{split}.csv"), dtype={"sample_id": str})
        pred = pd.read_csv(find(PREDICTION_FILES[split]), dtype={"sample_id": str})
        if "sample_id" not in pred.columns:
            fail(f"{name} is missing column sample_id")
        if pred["sample_id"].duplicated().any():
            fail(f"{name} has duplicate sample_id values")
        missing = truth.loc[~truth["sample_id"].isin(pred["sample_id"]), "sample_id"]
        if len(missing):
            fail(f"{name} has no prediction for {len(missing)} of {len(truth)} clusters "
                 f"(e.g. {missing.head(3).tolist()}); every contained cluster is required, empty where not predicted")
        try:
            result = metrics.score(pred, truth, reference.get(split), descriptor)
        except (ValueError, TypeError) as e:
            fail(f"{name}: {e}")
        if result.get("asic_violations"):
            fail(f"descriptor.json declares the asic track but violates {result['asic_violations']}")
        scores.update({f"{split}_{k}": v for k, v in result["metrics"].items()})
        print(f"[{split}] {result['coverage']}")
        for v in metrics.VARIABLES:
            print(f"  {v}: " + ", ".join(f"{k} = {result['metrics'][f'{k}_{v}']:.4g}"
                                         for k in ("I68", "mean", "pull_std", "nll") if f"{k}_{v}" in result["metrics"]))
        print(f"  {metrics.PRIMARY}: {result['metrics'][metrics.PRIMARY]}")

    scores["asic"] = int(bool(descriptor) and descriptor.get("track") == "asic")
    scores = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scores.items()}
    os.makedirs(score_dir, exist_ok=True)
    with open(os.path.join(score_dir, "scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
