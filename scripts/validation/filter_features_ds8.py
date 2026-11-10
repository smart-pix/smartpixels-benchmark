"""Per-sample check of the filter features on dataset8 against the paper's preprocessed CSVs.

    python filter_features_ds8.py --ds8 DS8_DIR --dec6 dec6_ds8 --jan2 jan2_finalDS [--dec8 dec8_ds8_ts]

Rows are matched on (round(y-local, 5), round(pt, 5)), first occurrence in dataset8. Compares
slice-19 y-profile (dec6 FullPrecisionInput*), 2-bit y-profile and 6-bit y0 (jan2 Quantized*), and
slices 0-7 profiles (dec8 InputTrain/Test_noYlocal). Prints aggregate counts only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartpix_bench.common import io
from smartpix_bench.filter import cache
from smartpix_bench.filter import data as fd


def key(y, pt) -> pd.Series:
    return pd.Series([f"{a:.5f},{b:.5f}" for a, b in zip(np.round(np.asarray(y, float), 5), np.round(np.asarray(pt, float), 5))])


def compare(ours, theirs, collided) -> dict:
    """Row agreement; rows off by more than float32 rounding (0.01 e) are split by whether their key collides."""
    d = np.abs(np.asarray(ours, float).reshape(len(ours), -1) - np.asarray(theirs, float).reshape(len(theirs), -1)).max(1)
    off = d > 0.01
    return {"rows": len(d), "rows_equal": int((d == 0).sum()), "rows_off": int(off.sum()),
            "rows_off_at_colliding_key": int((off & collided).sum()), "max_abs_diff_noncolliding": float(d[~collided].max())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds8", type=Path, required=True)
    p.add_argument("--dec6", type=Path, required=True)
    p.add_argument("--jan2", type=Path, required=True)
    p.add_argument("--dec8", type=Path)
    a = p.parse_args()
    files = io.pair_files(a.ds8, a.ds8)
    lab = io.read_labels(files, sorted(files), ["pt", "y-local"])
    k = key(lab["y-local"], lab["pt"])
    out = {"ds8_rows": len(lab), "ds8_key_collisions": int(k.duplicated().sum())}
    first = pd.Series(np.arange(len(k)), index=k)[~k.duplicated().to_numpy()]
    for split in ("Train", "Test"):
        fp = pd.read_csv(a.dec6 / f"FullPrecisionInput{split}Set.csv")
        pt = pd.read_csv(a.dec6 / f"{split}SetTruePT.csv")["pt"].to_numpy()
        q = pd.read_csv(a.jan2 / f"Quantized{split}Set.csv")
        assert np.array_equal(pd.read_csv(a.jan2 / f"{split}SetTruePT.csv")["pt"].to_numpy(), pt), "dec6/jan2 row order differs"
        pk = key(fp["y-local"], pt)
        hit = pk.isin(first.index).to_numpy()
        rows = lab.iloc[first.loc[pk[hit]].to_numpy()].reset_index(drop=True)
        rows["cls"] = fd.class_label(rows["pt"])
        feats = cache.featurize(files, rows)
        dup = set(k[k.duplicated()]) | set(pk[pk.duplicated()])
        coll = pk[hit].isin(dup).to_numpy()
        r = {"paper_rows": len(fp), "paper_key_collisions": int(pk.duplicated().sum()), "unmatched": int((~hit).sum()),
             "y_local_equal": int((np.round(rows["y-local"].to_numpy(), 5) == np.round(fp["y-local"][hit].to_numpy(), 5)).sum()),
             "rows_at_colliding_key": int(coll.sum()),
             "fp_profile": compare(feats["prof"], fp.iloc[:, :13][hit], coll),
             "adc_profile": compare(feats["prof_adc"], q.iloc[:, :13][hit], coll),
             "y0_6bit": compare(feats["y0_6bit"], q["y-local"][hit], coll),
             "y_below_-8": int((rows["y-local"] < -8).sum()),
             "y0_6bit_mismatch_outside_y_below_-8": int(((feats["y0_6bit"] != q["y-local"][hit].to_numpy()) & (rows["y-local"].to_numpy() >= -8)).sum()),
             "y_below_-8_upstream_63_ours_0": int(((q["y-local"][hit].to_numpy() == 63) & (feats["y0_6bit"] == 0) & (rows["y-local"].to_numpy() < -8)).sum())}
        if a.dec8:
            ts = pd.read_csv(a.dec8 / f"Input{split}_noYlocal.csv").to_numpy()[hit]
            r["ts_profiles"] = compare(feats["prof_ts"], ts, coll)
        out[split.lower()] = r
        print(json.dumps({split: r}), flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
