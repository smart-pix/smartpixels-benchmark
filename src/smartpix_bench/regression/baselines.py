"""Non-ML regression baselines (paper 2602.15946 §3.4) on slice 19 digitized at the transformer thresholds.

    python -m smartpix_bench.regression.baselines --split 3s --out RESULTS/regression/3s

Barycenter (Eq 10) and LocalReco (Eqs 11-12) predict x, y; Geometric (Eqs 13-14) predicts cotAlpha, cotBeta.
LocalReco and Geometric use the true angles / signs ("optimistic"), as the paper does. Pixel i of an axis
has its centre at (i - 7) * pitch (upstream summary-plots/local-pixel-reco.ipynb). Eq 14 as printed adds a
length to |cotβ|; this implements cotβ = s (w_y - 1) p_y / T - tanΘ_L with s = sign(cotβ_true + tanΘ_L),
the width relation Eq 12 uses.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common.features import digitize
from smartpix_bench.regression import data as rd
from smartpix_bench.regression import features as rf
from smartpix_bench.regression.predict import OUTPUT_COLUMNS

PITCH_X, PITCH_Y, THICKNESS, TAN_LORENTZ = 50.0, 12.5, 100.0, 0.3046
DY = THICKNESS * TAN_LORENTZ
NAMES = ("Barycenter", "LocalReco", "Geometric")


def profiles(adc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 16 y, 16 x) -> x profile (summed over y) and y profile (summed over x)."""
    return adc.sum(axis=1), adc.sum(axis=2)


def centre(i, pitch):
    return (np.asarray(i, dtype=np.float64) - 7) * pitch


def ends(prof):
    """First and last pixel with charge (NaN for an empty profile) and their charges."""
    hit = prof > 0
    any_ = hit.any(axis=1)
    f = np.where(any_, hit.argmax(axis=1), -1)
    l = np.where(any_, prof.shape[1] - 1 - hit[:, ::-1].argmax(axis=1), -1)
    rows = np.arange(len(prof))
    q_f, q_l = prof[rows, f].astype(np.float64), prof[rows, l].astype(np.float64)
    nan = np.where(any_, 0.0, np.nan)
    return f + nan, l + nan, q_f, q_l


def barycenter(prof, pitch):
    w = prof.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return centre(0, pitch) + pitch * (w * np.arange(w.shape[1])).sum(1) / w.sum(1)


def local_reco(prof, pitch, width):
    """Eq 11 with |T cot| replaced by `width` (Eq 12 passes |T cotβ + Δy|)."""
    f, l, q_f, q_l = ends(prof)
    x_f, x_l = centre(f, pitch) + pitch / 2, centre(l, pitch) - pitch / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x_f + x_l) / 2 + (q_l - q_f) / (q_l + q_f) * (width - (x_l - x_f)) / 2


def size(prof):
    f, l, _, _ = ends(prof)
    return l - f + 1


def predict(adc: np.ndarray, lab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    px, py = profiles(adc)
    cot_a, cot_b = lab["cotAlpha"].to_numpy(), lab["cotBeta"].to_numpy()
    cols = {
        "Barycenter": {"x-midplane": barycenter(px, PITCH_X), "y-midplane": barycenter(py, PITCH_Y) + DY / 2},
        "LocalReco": {"x-midplane": local_reco(px, PITCH_X, np.abs(THICKNESS * cot_a)),
                      "y-midplane": local_reco(py, PITCH_Y, np.abs(THICKNESS * cot_b + DY)) + DY / 2},
        "Geometric": {"cotAlpha": np.sign(cot_a) * (size(px) - 1) * PITCH_X / THICKNESS,
                      "cotBeta": np.sign(cot_b + TAN_LORENTZ) * (size(py) - 1) * PITCH_Y / THICKNESS - TAN_LORENTZ},
    }
    out = {}
    for name, c in cols.items():
        df = pd.DataFrame(np.nan, index=range(len(lab)), columns=OUTPUT_COLUMNS)
        for k, v in c.items():
            df[k] = v
        out[name] = pd.concat([lab[["sample_id"]].reset_index(drop=True), df], axis=1)
    return out


def predict_split(split: str, data_dir=None) -> dict[str, pd.DataFrame]:
    files = rd.files(split, data_dir)
    parts = {n: [] for n in NAMES}
    for stem in sorted(files):
        x, lab = rd.load(files, [stem], [rd.CONTAIN_SLICE])
        for n, df in predict(digitize(x[..., 0], rf.TRANSFORMER_THRESHOLDS), lab).items():
            parts[n].append(df)
    return {n: pd.concat(p, ignore_index=True) for n, p in parts.items()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=rd.TEST_SETS)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for n, df in predict_split(a.split, a.data_dir).items():
        df.to_csv(a.out / f"predictions_{n}.csv", index=False)
        print(f"wrote {a.out}/predictions_{n}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
