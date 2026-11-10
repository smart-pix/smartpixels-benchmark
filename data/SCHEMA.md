# Data schema

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
Train: each (charge × pT) category — pT < −0.2, class 1, class 2, pT > 0.2 — is subsampled (seed 0) to the
smallest category's train count, so classes 0:1:2 are 2:1:1; validation is the last 20 % of the train rows after
a seeded shuffle. Test: all rows of the test files (32 of 160 files, 1,599,888 clusters).

**Predictions:** `predictions_<name>.csv` with `sample_id,pred,p0,p1,p2` for every test cluster: pred ∈ {0, 1, 2}
is the class decision, p0–p2 ∈ [0, 1] the class probabilities. A cluster without a prediction is scored as
class 0 with p0 = 1 (read out) and counted in the coverage block.

**Metrics:** the primary metric is the reweighted background rejection when clusters with p0 < cut are rejected
and the cut gives a reweighted signal efficiency of 0.933 (`score.rejection_at_efficiency`); signal efficiency,
background rejection and data reduction of the `pred` decisions are also reported.
**Descriptor:** `descriptor_<name>.json` with `track` (`asic` or `open`) and `n_params`; the asic track
also declares `adc_thresholds` (must be [400, 1600, 2400]) and `y0_bits` (must be 6), and needs
n_params ≤ 2819.

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

**Predictions:** `predictions_<name>.csv` with `sample_id` and `x-midplane`, `y-midplane` (µm), `cotAlpha`,
`cotBeta` for every contained cluster of a test set, plus `sigma_<target>` (1σ, same units) for models with
uncertainties. A target a model does not predict is empty (NaN) and is excluded from that variable's metrics.
**Metrics:** per test set and variable (x, y in µm; α, β in degrees from the cotangents): residual mean, I68 (width
of the shortest interval holding ⌈0.68 n⌉ residuals), and for σ-models mean σ, pull mean and width, Gaussian NLL
and the fraction inside ±1σ / ±2σ. The ranking score `rank_i68` is the mean over x, y, β of I68 divided by the
shipped reference (`regression/i68_reference.json`, the Full_MLP c4 reference model); lower is better, and a
model with any missing or NaN x, y or β is unranked.
**Descriptor:** `descriptor_<name>.json` with `track`, `n_params`, `slices`; the asic track also declares
`thresholds` (three strictly increasing ADC thresholds in e, applied as Eq 4 to slices 0 and 19) and
`weight_bits`, `activation_bits` (≤ 8), with n_params ≤ 2734 and slices [0, 19].

## Restricted data (CMS-internal)

Some validation uses data not on Zenodo: the filter paper's simulation (dataset8) and preprocessed CSVs, and the
tracked / untracked samples behind 2310.02474 Table 2 (DS14, DS15). They are available to CMS members on request
from the paper authors. Nothing from them is in this repository or its outputs beyond aggregate numbers in
`docs/VALIDATION.md`, and no benchmark score depends on them.
