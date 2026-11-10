# Data schema (draft)

Data are fetched by `scripts/fetch_data.py` into `$SPB_DATA_DIR/<record>/` (default `data/zenodo/`),
with the Zenodo revision, version and per-file md5 recorded in `<record>/REVISION.json`.
macOS `._*` AppleDouble files in the archives are skipped on extraction and by every loader.

Both records store clusters as row-aligned pairs `labels_d<N>.parquet` / `recon3D_d<N>.parquet`:
row *r* of the labels file describes row *r* of the recon file. Recon columns are named by flat
position `"t*H*W + y*W + x"`; values are cumulative collected charge in electrons, time slice
*t* at 200 ps steps (slice 19 = 4 ns).

**sample_id** = `<recon file stem>:<row>` (e.g. `recon3D_d17301:42`), unique within a split.

## Filter — Zenodo 10783560 (version 3)

Layout: `labels/labels/labels_d<N>.parquet`, `recon3D/recon3D/recon3D_d<N>.parquet`;
recon is (20, 13, 21) = (time, y, x).

Label fields (`labels_d<N>.parquet`, ~50k rows per file):

| field | type | meaning |
|---|---|---|
| `x-entry`, `y-entry`, `z-entry` | float64 | track entry point into the sensor (simulation truth) |
| `n_x`, `n_y`, `n_z` | float64 | track direction components (simulation truth) |
| `number_eh_pairs` | int64 | electron–hole pairs generated (simulation truth) |
| `y-local` | float64 | y0, mm from the sensor centre (v3 range −8.13 … 8.18; the 6-bit code clips to (−8, 8]) |
| `pt` | float64 | signed transverse momentum, GeV |
| `cotAlpha`, `cotBeta` | float64 | incidence angle cotangents |
| `x-midplane`, `y-midplane` | float64 | track position at the sensor mid-plane, µm |
| `__index_level_0__` | int64 | pandas index; equals the row position |

**Ground truth:** class from `pt` (GeV, signed by charge): 0 = |pT| > 0.2, 1 = −0.2 ≤ pT < 0,
2 = 0 < pT ≤ 0.2.

**Valid model inputs:** recon pixels and `y-local` only. Every other label column describes the
generated track and is forbidden as an input.

**Selection:** files are split file-disjointly, 20 % test (`common.io.split_files`, seed 0).
Train: every class is subsampled (seeded) to the smallest class's train count; validation is the
last 20 % of the train rows after a seeded shuffle. Test: all rows of the test files.

## Regression — Zenodo 18472791 (version 1)

Layout: `dataset_<set>_16x16_50x12P5x100/dataset_<set>_16x16_50x12P5x100/` for set ∈ {3sExtended, 3s, 2s};
recon is (20, 16, 16) = (time, y, x).

Label fields (`labels_d<N>.parquet`, ~25k rows per file): the filter fields plus

| field | type | meaning |
|---|---|---|
| `original_atEdge` | bool | upstream containment flag |
| `chargeOriginal_atEdge` | float64 | upstream border charge (upstream DG cuts `< 50`) |
| `nPixOriginalAbove1e_atEdge` | int64 | upstream border pixel count |

**Ground truth:** `x-midplane`, `y-midplane` (µm), `cotAlpha`, `cotBeta` (Slim models: x, y, cotβ).

**Valid model inputs:** recon pixels only, at slices {0, 19} (configs c2–c4) or all 20 (c1).
Every label column is forbidden as an input.

**Selection:** clusters with ≥ 50 e summed over the border pixels of the 16 × 16 array at slice 19
are dropped from every split (paper §3.1, computed from the pixels; the upstream
`chargeOriginal_atEdge < 50` cut selects the same clusters except 2 of 1,971,855 in 3sExtended). `3sExtended` files are split 80/20 file-disjointly into
train/val; `3s` and `2s` are separate test sets. Targets are divided by the per-target 99th
percentile of |label| over the train split only.
