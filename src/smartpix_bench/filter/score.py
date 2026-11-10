"""Filter metrics, reweighted to the physical pT spectrum (upstream smartPixelsPaper/fullPrecisionBaseline/*_Reweighted.ipynb).

    python -m smartpix_bench.filter.score --pred predictions_M2.csv --truth truth.csv --out DIR [--descriptor d.json]
    python -m smartpix_bench.filter.score --write-reference src/smartpix_bench/filter/pt_reference.json   # from $SPB_DATA_DIR

A test cluster in signed-pT bin b counts N_ref[b] / N_test[b] times, so every metric estimates the
rate on the full-statistics pT spectrum N_ref. Bins are right-closed (pandas `value_counts(bins=...)`).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common.io import data_dir
from smartpix_bench.filter.models import AUXILIARY  # noqa: F401 -- re-exported for common.report

EDGES = np.round(np.concatenate([[-10.0], np.arange(-20, 21) / 10, [10.0]]), 1)
SIGNAL_BINS = [0, len(EDGES) - 2]  # [-10, -2] and (2, 10]
BACKGROUND_BINS = list(range(1, len(EDGES) - 2))  # 0.1 GeV bins over (-2, 2]
TURNON_EDGES = np.round(np.linspace(-5, 5, 101), 1)
CLASSES = (0, 1, 2)
OUTPUT_COLUMNS = ["pred", "p0", "p1", "p2"]
TARGET_EFFICIENCY = 0.933  # M2's Table 1 signal efficiency
PRIMARY = "background_rejection_at_eff"
METRICS = [PRIMARY, "background_rejection", "signal_efficiency", "data_reduction"]
ASIC = {"max_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}
REFERENCE = Path(__file__).with_name("pt_reference.json")


def bin_index(pt, edges=EDGES) -> np.ndarray:
    """Right-closed bin of each pT, lowest edge included; -1 outside [edges[0], edges[-1]]."""
    pt = np.asarray(pt, dtype=np.float64)
    i = np.searchsorted(edges, pt, side="left") - 1
    i = np.where(pt == edges[0], 0, i)
    return np.where((i >= 0) & (i < len(edges) - 1), i, -1)


def histogram(pt, edges=EDGES) -> np.ndarray:
    i = bin_index(pt, edges)
    return np.bincount(i[i >= 0], minlength=len(edges) - 1)


def sample_weights(pt, n_ref) -> tuple[np.ndarray, np.ndarray]:
    """(bin, weight N_ref[b] / N_test[b]) per cluster; clusters outside the edges get bin -1 and weight 0."""
    n_ref = np.asarray(n_ref, dtype=np.float64)
    b = bin_index(pt)
    n_test = np.bincount(b[b >= 0], minlength=len(n_ref))
    if np.any((n_ref > 0) & (n_test == 0)):
        raise ValueError(f"reference bins with no test clusters: {np.flatnonzero((n_ref > 0) & (n_test == 0)).tolist()}")
    w = np.divide(n_ref, n_test, out=np.zeros_like(n_ref), where=n_test > 0)
    return b, np.where(b >= 0, w[b], 0.0)


def reweighted_metrics(pt, pred, n_ref) -> dict:
    """Signal efficiency, background rejection and data reduction of class decisions on the N_ref spectrum."""
    n_ref = np.asarray(n_ref, dtype=np.float64)
    b, w = sample_weights(pt, n_ref)
    rejected = np.asarray(pred) != 0
    sig, bkg = np.isin(b, SIGNAL_BINS), np.isin(b, BACKGROUND_BINS)
    return {
        "signal_efficiency": (w * ~rejected * sig).sum() / n_ref[SIGNAL_BINS].sum(),
        "background_rejection": (w * rejected * bkg).sum() / n_ref[BACKGROUND_BINS].sum(),
        "data_reduction": (w * rejected).sum() / n_ref.sum(),
    }


def rejection_at_efficiency(pt, p0, n_ref, target: float = TARGET_EFFICIENCY) -> dict:
    """Background rejection when clusters with p0 < cut are rejected and the cut sets the reweighted signal
    efficiency to `target`, linearly interpolated between adjacent distinct p0 values.

    Cut candidates are the distinct p0 values plus one above all of them (reject everything), so the
    efficiency runs from 1 to 0 and every target is bracketed, including when all p0 are tied.
    NaN when the set has no signal or no background clusters.
    """
    b, w = sample_weights(pt, n_ref)
    ws, wb = w * np.isin(b, SIGNAL_BINS), w * np.isin(b, BACKGROUND_BINS)
    c, inv = np.unique(np.asarray(p0, dtype=np.float64), return_inverse=True)
    ws, wb = np.bincount(inv, ws, len(c)), np.bincount(inv, wb, len(c))
    if ws.sum() == 0 or wb.sum() == 0:
        return {PRIMARY: float("nan"), "p0_cut": float("nan")}
    eff = np.append(ws[::-1].cumsum()[::-1], 0.0) / ws.sum()
    rej = np.append(0.0, wb.cumsum()) / wb.sum()
    cuts = np.append(c, np.nextafter(max(c[-1], 1.0), np.inf))
    k = np.flatnonzero(eff >= target).max()
    f = 0.0 if eff[k] == target else (eff[k] - target) / (eff[k] - eff[k + 1])
    return {PRIMARY: float(rej[k] + f * (rej[k + 1] - rej[k])), "p0_cut": float(cuts[k] + f * (cuts[k + 1] - cuts[k]))}


def median_seed(primary_by_seed: dict) -> int:
    """Seed whose validation PRIMARY is the median (lower median for an even count; ties to the smaller seed)."""
    order = sorted(primary_by_seed, key=lambda s: (primary_by_seed[s], s))
    return order[(len(order) - 1) // 2]


def turn_on(pt, pred, edges=TURNON_EDGES) -> list[dict]:
    """Unweighted acceptance (pred == 0) per signed-pT bin with binomial errors."""
    b = bin_index(pt, edges)
    ok = b >= 0
    n = np.bincount(b[ok], minlength=len(edges) - 1)
    k = np.bincount(b[ok], weights=(np.asarray(pred)[ok] == 0), minlength=len(edges) - 1)
    eff = np.divide(k, n, out=np.full(len(n), np.nan), where=n > 0)
    err = np.sqrt(eff * (1 - eff) / np.where(n > 0, n, 1))
    return [{"lo": float(lo), "hi": float(hi), "n": int(c), "acceptance": float(e), "error": float(s)}
            for lo, hi, c, e, s in zip(edges[:-1], edges[1:], n, eff, err)]


def asic_violations(descriptor: dict) -> list[str]:
    """ASIC-track constraints (PLAN §Metrics): parameter budget, Table-3 ADC, 6-bit y0."""
    v = []
    if not int(descriptor.get("n_params", 1 << 62)) <= ASIC["max_params"]:
        v.append(f"n_params {descriptor.get('n_params')} > {ASIC['max_params']}")
    if list(descriptor.get("adc_thresholds", [])) != ASIC["adc_thresholds"]:
        v.append(f"adc_thresholds {descriptor.get('adc_thresholds')} != {ASIC['adc_thresholds']}")
    if descriptor.get("y0_bits") != ASIC["y0_bits"]:
        v.append(f"y0_bits {descriptor.get('y0_bits')} != {ASIC['y0_bits']}")
    return v


def check_values(out: pd.DataFrame) -> str | None:
    if missing := [c for c in OUTPUT_COLUMNS if c not in out.columns]:
        return f"missing columns {missing}"
    bad = out[~out["pred"].isin(CLASSES)]
    if len(bad):
        return f"predictions outside {CLASSES}: {bad['pred'].head(3).tolist()}"
    p = out[["p0", "p1", "p2"]].to_numpy(dtype=float)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        return "probabilities must be finite and in [0, 1]"
    return None


def join(pred: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Truth rows with their prediction; a missing prediction is read out (class 0, p0 = 1), never dropped."""
    if pred["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id in predictions")
    if err := check_values(pred):
        raise ValueError(err)
    p = pred.set_index("sample_id")[OUTPUT_COLUMNS]
    m = truth.join(p, on="sample_id")
    cov = {"n_truth": len(truth), "n_scored": int(m["pred"].notna().sum()),
           "n_truth_without_prediction": int(m["pred"].isna().sum()),
           "n_predictions_without_truth": int((~p.index.isin(truth["sample_id"])).sum())}
    m = m.fillna({"pred": 0, "p0": 1.0, "p1": 0.0, "p2": 0.0}).astype({"pred": int})
    return m, cov


def load_reference(path=REFERENCE) -> np.ndarray:
    ref = json.loads(Path(path).read_text())
    if not np.allclose(ref["edges"], EDGES):
        raise ValueError(f"{path}: bin edges differ from score.EDGES")
    return np.asarray(ref["counts"])


def score(pred: pd.DataFrame, truth: pd.DataFrame, n_ref, descriptor: dict | None = None) -> dict:
    m, cov = join(pred, truth)
    out = {"coverage": cov, "metrics": {**rejection_at_efficiency(m["pt"], m["p0"], n_ref),
                                        **reweighted_metrics(m["pt"], m["pred"], n_ref)},
           "turn_on": turn_on(m["pt"], m["pred"])}
    if descriptor is not None:
        out["descriptor"] = descriptor
        out["asic_violations"] = asic_violations(descriptor) if descriptor.get("track") == "asic" else []
    return out


def write(result: dict, out: Path, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    md = [f"# {title}", "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v:.4f} |" for k, v in result["metrics"].items()]
    md += ["", "Coverage: " + ", ".join(f"{k} = {v}" for k, v in result["coverage"].items())]
    if "descriptor" in result:
        d = result["descriptor"]
        md += [f"Track: {d.get('track')}; violations: {result['asic_violations'] or 'none'}"]
    (out / "METRICS.md").write_text("\n".join(md) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.DataFrame(result["turn_on"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar((t.lo + t.hi) / 2, t.acceptance, yerr=t.error, fmt=".", ms=3)
    ax.set(xlabel="signed $p_T$ [GeV]", ylabel="acceptance (class 0)", title=title, ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(out / "turn_on.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path)
    p.add_argument("--truth", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--descriptor", type=Path)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--title", default="filter")
    p.add_argument("--write-reference", type=Path, metavar="JSON")
    a = p.parse_args(argv)
    if a.write_reference:
        from smartpix_bench.filter import data as fd
        files = fd.files(data_dir())
        pt = fd.labels(files, sorted(files))["pt"]
        a.write_reference.write_text(json.dumps({"source": f"Zenodo {fd.RECORD}, all files", "edges": EDGES.tolist(),
                                                 "counts": histogram(pt).tolist()}) + "\n")
        return
    desc = json.loads(a.descriptor.read_text()) if a.descriptor else None
    res = score(pd.read_csv(a.pred, dtype={"sample_id": str}), pd.read_csv(a.truth, dtype={"sample_id": str}),
                load_reference(a.reference), desc)
    write(res, a.out, a.title)
    print(json.dumps(res["metrics"]))


if __name__ == "__main__":
    main()
