"""Predict a split with trained filter weights.

    python -m smartpix_bench.filter.predict --model M2 --weights W/M2_s0 [--name M2_s0] --split test --out RESULTS/filter/test

Writes predictions_<name>.csv (sample_id, pred = argmax, p0, p1, p2) for every cluster of the split and
descriptor_<name>.json (track, parameter count, ASIC input declaration).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from smartpix_bench.filter import cache, models
from smartpix_bench.filter.train import predict_proba


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(models.MODELS))
    p.add_argument("--weights", required=True, help="path stem: <stem>.weights.h5 and <stem>.scaler.json")
    p.add_argument("--name", help="default: weights stem name")
    p.add_argument("--split", default="test")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    name = a.name or Path(a.weights).name
    arrays = cache.load(a.split, a.cache_dir)
    m = models.build(a.model)
    m.load_weights(f"{a.weights}.weights.h5")
    x = models.apply_scaler(models.raw_inputs(a.model, arrays), json.loads(Path(f"{a.weights}.scaler.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    proba = predict_proba(m, x)
    pd.DataFrame({"sample_id": arrays["sample_id"], "pred": proba.argmax(1),
                  **{f"p{i}": proba[:, i] for i in range(3)}}).to_csv(a.out / f"predictions_{name}.csv", index=False)
    (a.out / f"descriptor_{name}.json").write_text(json.dumps(models.descriptor(a.model, m.count_params())) + "\n")
    print(f"wrote {a.out}/predictions_{name}.csv ({len(arrays['sample_id'])} rows)")


if __name__ == "__main__":
    main()
