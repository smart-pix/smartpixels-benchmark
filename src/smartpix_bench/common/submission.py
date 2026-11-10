"""Package predictions as upload-ready Codabench zips, refusing incomplete or invalid ones.

    python -m smartpix_bench.common.submission --task filter --results RESULTS/filter

For each model with predictions for every split of the task, writes <results>/codabench/<model>_submission.zip
holding predictions_<split>.csv (exactly the scored ids, in truth order) and, if present, descriptor.json,
all at the zip root. An asic-track descriptor that fails its checks, a missing scored id, or an invalid
value writes no zip and makes the exit status non-zero.
"""

import argparse
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


class SubmissionError(ValueError):
    pass


def select_scored(pred: pd.DataFrame, ids: pd.Series, columns: list[str], check, label: str) -> pd.DataFrame:
    cols = ["sample_id", *columns]
    if missing := [c for c in cols if c not in pred.columns]:
        raise SubmissionError(f"{label}: missing columns {missing}")
    if pred["sample_id"].duplicated().any():
        raise SubmissionError(f"{label}: duplicate sample_id")
    pred = pred.set_index("sample_id")
    if n := int((~ids.isin(pred.index)).sum()):
        raise SubmissionError(f"{label}: no prediction for {n} of {len(ids)} scored samples")
    out = pred.loc[ids.to_numpy(), columns]
    if err := check(out):
        raise SubmissionError(f"{label}: {err}")
    return out.reset_index()[cols]


def package(task: str, results: Path, cache_dir: Path | None, out_dir: Path) -> int:
    truth = importlib.import_module(f"smartpix_bench.{task}.truth")
    sc = importlib.import_module(f"smartpix_bench.{task}.score")
    ids = {s: truth.scored_ids(s, cache_dir) for s in truth.SPLITS}
    failed = 0
    for path in sorted((results / truth.SPLITS[0]).glob("predictions_*.csv")):
        name = path.stem.removeprefix("predictions_")
        try:
            frames = {}
            for s in truth.SPLITS:
                pred = pd.read_csv(results / s / f"predictions_{name}.csv", dtype={"sample_id": str})
                frames[s] = select_scored(pred, ids[s], sc.OUTPUT_COLUMNS, sc.check_values, f"{name}/{s}")
            desc_path = results / truth.SPLITS[0] / f"descriptor_{name}.json"
            desc = json.loads(desc_path.read_text()) if desc_path.exists() else None
            if desc and desc.get("track") == "asic" and (v := sc.asic_violations(desc)):
                raise SubmissionError(f"{name}: asic track violations {v}")
        except (SubmissionError, FileNotFoundError) as e:
            print(f"{name}: NOT packaged: {e}", file=sys.stderr)
            failed += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_dir / f"{name}_submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for s, df in frames.items():
                z.writestr(f"predictions_{s}.csv", df.to_csv(index=False))
            if desc:
                z.writestr("descriptor.json", json.dumps(desc) + "\n")
        print(f"{name}: wrote {out_dir / f'{name}_submission.zip'}")
    return failed


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["filter"])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, help="default: <results>/codabench")
    a = p.parse_args(argv)
    sys.exit(1 if package(a.task, a.results, a.cache_dir, a.out or a.results / "codabench") else 0)


if __name__ == "__main__":
    main()
