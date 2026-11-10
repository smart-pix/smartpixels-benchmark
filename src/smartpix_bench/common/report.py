"""Collect <results>/<split>/<name>/metrics.json into LEADERBOARD.md, one table per split and track.

    python -m smartpix_bench.common.report --task filter|regression --results RESULTS/<task>

Names `<model>_s<seed>` are aggregated per model as mean ± std over seeds. The asic table holds
models whose descriptor declares the asic track and passes its checks; every model is in the open table.
Auxiliary models render in italics.
"""

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np

SEED = re.compile(r"^(?P<model>.+)_s(?P<seed>\d+)$")


def collect(results: Path) -> dict:
    """{split: {model: [metrics.json, ...]}} over every scored name."""
    out = {}
    for path in sorted(results.glob("*/*/metrics.json")):
        split, name = path.parts[-3], path.parts[-2]
        mo = SEED.match(name)
        out.setdefault(split, {}).setdefault(mo["model"] if mo else name, []).append(json.loads(path.read_text()))
    return out


def cell(vals) -> str:
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def build(results: Path, metrics: list[str], primary: str, auxiliary: set, title: str, higher_is_better: bool = True) -> str:
    lines = [f"# {title}", "", f"Ranked by {primary} (mean over seeds; ± is the sample std).",
             "Rows in *italics* are auxiliary comparison models, not reference solutions.", ""]
    for split, runs in collect(results).items():
        for track in ("asic", "open"):
            rows = []
            for model, ms in runs.items():
                on_asic = all(m.get("descriptor", {}).get("track") == "asic" and not m.get("asic_violations") for m in ms)
                if track == "asic" and not on_asic:
                    continue
                vals = {k: [m["metrics"][k] for m in ms] for k in metrics}
                cov = ms[0]["coverage"]
                name = f"*{model} (auxiliary)*" if model in auxiliary else model
                rows.append((np.mean(vals[primary]), f"| {name} | {len(ms)} | {cov['n_scored']}/{cov['n_truth']} | "
                             + " | ".join(cell(vals[k]) for k in metrics) + " |"))
            if rows:
                lines += [f"## {split} — {track} track", "", "| model | seeds | coverage | " + " | ".join(metrics) + " |",
                          "|---" * (3 + len(metrics)) + "|", *[r for _, r in sorted(rows, key=lambda r: -r[0] if higher_is_better else r[0])], ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["filter", "regression"])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: <results>/LEADERBOARD.md")
    a = p.parse_args(argv)
    sc = importlib.import_module(f"smartpix_bench.{a.task}.score")
    md = build(a.results, sc.METRICS, sc.PRIMARY, sc.AUXILIARY, f"smartpixels {a.task} leaderboard", sc.HIGHER_IS_BETTER)
    (a.out or a.results / "LEADERBOARD.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
