"""Regression metrics per variable x, y (µm), alpha, beta (degrees) on one test set.

    python -m smartpix_bench.regression.score --pred predictions_M.csv --truth truth.csv --out DIR [--descriptor d.json]
    python -m smartpix_bench.regression.score --write-reference src/smartpix_bench/regression/i68_reference.json \
        --results RESULTS/regression --weights WEIGHTS/regression

Per variable: residual mean, I68 (width of the shortest interval holding ceil(0.68 n) residuals), and for
models with uncertainties the mean sigma, pull mean and std, mean Gaussian NLL and the fraction of
|pull| < 1 and < 2. A prediction that is missing or NaN for a variable is excluded from that variable and
counted; the ranking score needs every cluster's x, y and beta. The ranking score is the mean over x, y, beta of I68 / I68 of the reference (`REFERENCE`, lower is
better): per test set, the test I68 of the `REFERENCE_MODEL` seed whose validation mean I68 over x, y, beta
is the median (`filter.score.median_seed`).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.regression import data as rd

VARIABLES = {"x": "x-midplane", "y": "y-midplane", "alpha": "cotAlpha", "beta": "cotBeta"}
RANK_VARIABLES = ("x", "y", "beta")
OUTPUT_COLUMNS = [*rd.TARGETS, *[f"sigma_{t}" for t in rd.TARGETS]]
PRIMARY = "rank_i68"
METRICS = [PRIMARY, *[f"I68_{v}" for v in VARIABLES], *[f"mean_{v}" for v in VARIABLES]]
HIGHER_IS_BETTER = False
AUXILIARY = {"Barycenter", "LocalReco", "Geometric"}
REFERENCE = Path(__file__).with_name("i68_reference.json")
REFERENCE_MODEL = "Full_MLP_c4"
ASIC = {"max_params": 2734, "slices": [0, 19], "n_thresholds": 3, "max_bits": 8}


def i68(r) -> float:
    """Width of the shortest interval containing ceil(0.68 n) of the values."""
    r = np.sort(np.asarray(r, dtype=np.float64))
    m = math.ceil(0.68 * len(r))
    if m == 0:
        return float("nan")
    return float(np.min(r[m - 1:] - r[:len(r) - m + 1]))


def angle(cot) -> np.ndarray:
    """Incidence angle in degrees in (0, 180) from its cotangent."""
    with np.errstate(divide="ignore"):
        a = np.arctan(1.0 / np.asarray(cot, dtype=np.float64))
    return np.degrees(np.where(a < 0, a + np.pi, a))


def physical(df: pd.DataFrame, v: str, sigma: bool = False) -> np.ndarray:
    col = VARIABLES[v]
    if not sigma:
        return angle(df[col]) if v in ("alpha", "beta") else df[col].to_numpy(dtype=np.float64)
    s = df[f"sigma_{col}"].to_numpy(dtype=np.float64)
    return np.degrees(s / (1 + df[col].to_numpy(dtype=np.float64) ** 2)) if v in ("alpha", "beta") else s


def variable_metrics(pred: pd.DataFrame, truth: pd.DataFrame, v: str) -> dict:
    r = physical(pred, v) - physical(truth, v)
    ok = np.isfinite(r)
    out = {"n": int(ok.sum()), "mean": float(np.mean(r[ok])) if ok.any() else float("nan"), "I68": i68(r[ok])}
    s = physical(pred, v, sigma=True)
    ok &= np.isfinite(s) & (s > 0)
    if ok.any():
        z = r[ok] / s[ok]
        out.update(mean_sigma=float(s[ok].mean()), pull_mean=float(z.mean()), pull_std=float(z.std()),
                   nll=float(np.mean(0.5 * z * z + np.log(s[ok]) + 0.5 * math.log(2 * math.pi))),
                   frac_1sigma=float(np.mean(np.abs(z) < 1)), frac_2sigma=float(np.mean(np.abs(z) < 2)))
    return out


def metrics(pred: pd.DataFrame, truth: pd.DataFrame, reference: dict | None = None) -> dict:
    """Flat metrics for row-aligned prediction and truth frames."""
    out = {}
    for v in VARIABLES:
        for k, val in variable_metrics(pred, truth, v).items():
            out[f"{k}_{v}"] = val
    # Unranked unless every cluster has finite x, y, beta, so dropping hard clusters cannot lower the score.
    complete = all(out[f"n_{v}"] == len(truth) for v in RANK_VARIABLES)
    out[PRIMARY] = (float(np.mean([out[f"I68_{v}"] / reference[v] for v in RANK_VARIABLES]))
                    if reference and complete else float("nan"))
    return out


def asic_violations(d: dict) -> list[str]:
    """ASIC-track constraints (PLAN §Metrics): parameters, slices {0, 19}, 3 strictly ordered thresholds, <= 8-bit."""
    v = []
    if not int(d.get("n_params", 1 << 62)) <= ASIC["max_params"]:
        v.append(f"n_params {d.get('n_params')} > {ASIC['max_params']}")
    if list(d.get("slices", [])) != ASIC["slices"]:
        v.append(f"slices {d.get('slices')} != {ASIC['slices']}")
    t = np.asarray(d.get("thresholds", []), dtype=np.float64)
    if not (t.shape == (ASIC["n_thresholds"],) and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)):
        v.append(f"thresholds {d.get('thresholds')} are not 3 strictly increasing values")
    for k in ("weight_bits", "activation_bits"):
        if not int(d.get(k, 1 << 62)) <= ASIC["max_bits"]:
            v.append(f"{k} {d.get(k)} > {ASIC['max_bits']}")
    return v


def check_values(out: pd.DataFrame) -> str | None:
    if missing := [c for c in rd.TARGETS if c not in out.columns]:
        return f"missing columns {missing}"
    if np.isinf(out.select_dtypes("number").to_numpy()).any():
        return "infinite predictions"
    return None


def join(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Predictions aligned to truth order (missing rows are NaN) and the coverage block."""
    if pred["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id in predictions")
    if err := check_values(pred):
        raise ValueError(err)
    cols = [c for c in OUTPUT_COLUMNS if c in pred.columns]
    p = pred.set_index("sample_id")[cols].reindex(truth["sample_id"]).reset_index(drop=True)
    p = p.reindex(columns=OUTPUT_COLUMNS)
    found = truth["sample_id"].isin(pred["sample_id"]).to_numpy()
    cov = {"n_truth": len(truth), "n_scored": int(found.sum()), "n_truth_without_prediction": int((~found).sum()),
           "n_predictions_without_truth": int((~pred["sample_id"].isin(truth["sample_id"])).sum())}
    return p, cov


def load_reference(split: str, path=REFERENCE) -> dict | None:
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text()).get(split)


def score(pred: pd.DataFrame, truth: pd.DataFrame, reference: dict | None = None, descriptor: dict | None = None) -> dict:
    p, cov = join(pred, truth)
    out = {"coverage": cov, "metrics": metrics(p, truth.reset_index(drop=True), reference)}
    if descriptor is not None:
        out["descriptor"] = descriptor
        out["asic_violations"] = asic_violations(descriptor) if descriptor.get("track") == "asic" else []
    return out


def write(result: dict, out: Path, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    m = result["metrics"]
    keys = sorted({k.rsplit("_", 1)[0] for k in m if k != PRIMARY})
    md = [f"# {title}", "", f"{PRIMARY}: {m[PRIMARY]:.4f}", "", "| metric | " + " | ".join(VARIABLES) + " |",
          "|---" * (1 + len(VARIABLES)) + "|"]
    md += [f"| {k} | " + " | ".join(f"{m.get(f'{k}_{v}', float('nan')):.4g}" for v in VARIABLES) + " |" for k in keys]
    md += ["", "Coverage: " + ", ".join(f"{k} = {v}" for k, v in result["coverage"].items())]
    if "descriptor" in result:
        md += [f"Track: {result['descriptor'].get('track')}; violations: {result['asic_violations'] or 'none'}"]
    (out / "METRICS.md").write_text("\n".join(md) + "\n")


def reference(results: Path, weights: Path, model: str = REFERENCE_MODEL) -> dict:
    from smartpix_bench.filter.score import median_seed
    val = {int(f.name.split("_s")[-1].split(".")[0]): json.loads(f.read_text()) for f in weights.glob(f"{model}_s*.val.json")}
    seed = median_seed({s: np.mean([v[f"I68_{r}"] for r in RANK_VARIABLES]) for s, v in val.items()})
    out = {"model": model, "seed": seed}
    for split in rd.TEST_SETS:
        m = json.loads((results / split / f"{model}_s{seed}" / "metrics.json").read_text())["metrics"]
        out[split] = {v: m[f"I68_{v}"] for v in RANK_VARIABLES}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path)
    p.add_argument("--truth", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--write-reference", type=Path, metavar="JSON")
    p.add_argument("--results", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--descriptor", type=Path)
    p.add_argument("--split", help="test set name for the ranking reference (default: the truth file's directory)")
    p.add_argument("--title", default="regression")
    a = p.parse_args(argv)
    if a.write_reference:
        a.write_reference.write_text(json.dumps(reference(a.results, a.weights)) + "\n")
        return
    desc = json.loads(a.descriptor.read_text()) if a.descriptor else None
    res = score(pd.read_csv(a.pred, dtype={"sample_id": str}), pd.read_csv(a.truth, dtype={"sample_id": str}),
                load_reference(a.split or a.truth.parent.name), desc)
    write(res, a.out, a.title)
    print(json.dumps({k: res["metrics"][k] for k in METRICS}))


if __name__ == "__main__":
    main()
