# Smart Pixels filter: on-sensor low-pT cluster rejection

> Classify a single silicon-pixel cluster, as seen by a 13 × 21 pixel readout chip, as coming from a
> high-momentum track (|pT| > 0.2 GeV, keep) or a low-momentum one of either charge (reject), with a network small
> enough to run inside the pixel ASIC. Every cluster the network rejects is bandwidth the detector saves.

**Scientific Motif:** High-Energy Physics · **AI/ML Motif:** Classification · **Computing Motif:** Latency Bound

Card structure: MLCommons Science Benchmarks Ontology (arXiv:2511.05614). Sibling benchmark:
[regression](../regression/README.md). Validation record: [docs/VALIDATION.md](../../docs/VALIDATION.md) §1.

## Quick start

```bash
uv sync --extra tf                                      # Python 3.11, TensorFlow 2.15.1, QKeras 0.9.0
uv run python scripts/fetch_data.py --task filter       # Zenodo 10783560 v3 -> $SPB_DATA_DIR (default data/zenodo)
uv run python -m pytest tests -q
uv run python -m smartpix_bench.filter.cache            # features for the train / test splits
for m in M1 M2 M3 QM2; do for s in 0 1 2 3 4; do
  uv run python -m smartpix_bench.filter.train --model $m --seed $s
  uv run python -m smartpix_bench.filter.predict --model $m --weights weights/filter/${m}_s$s --out $SPB_RESULTS/filter/test
done; done
SPB_RESULTS=results uv run bash scripts/score_all.sh filter     # truth, metrics, LEADERBOARD.md, Codabench zips
```

On HiPerGator: `bash slurm/submit.sh -A <group> filter` ([docs/HIPERGATOR.md](../../docs/HIPERGATOR.md)).
To score your own model, write `results/filter/test/predictions_<name>.csv` (every test cluster; columns below)
and, for the asic track, `descriptor_<name>.json`, then rerun `scripts/score_all.sh filter`; it also writes your
Codabench zip.

## 1. Problem specification and constraints

**Task.** Given one cluster's charge profile and its position on the sensor, output class probabilities for
0 = |pT| > 0.2 GeV, 1 = −0.2 ≤ pT < 0, 2 = 0 < pT ≤ 0.2 GeV (signed by charge). Low-pT tracks curve in the 3.8 T
field and cross the 100 µm sensor at shallow angles, which leaves a longer cluster along y.

**Inputs.** Built from the cumulative charge (electrons) of the 13 × 21 array: the y-profile (charge summed over
x per row, 13 values) at slice 19 (4 ns); for M3 the y-profiles at slices 0–7; for M1 the y-size; and y0, the
cluster's y position on the sensor (mm). The asic track's inputs are fixed by the ASIC: each pixel digitized to
2 bits at 400 / 1600 / 2400 e before the row sum, and y0 as a 6-bit code over 250 µm bins
([data/SCHEMA.md](../../data/SCHEMA.md)). Simulation-truth label fields are not valid inputs.

**Outputs.** `predictions_<name>.csv`: `sample_id, pred, p0, p1, p2` for every test cluster (pred ∈ {0, 1, 2};
probabilities in [0, 1]).

**System constraints (asic track).** At most 2819 parameters (the paper's quantized model), the 2-bit / 6-bit
inputs above, declared in `descriptor_<name>.json` and checked by the scorer and the packager. The ASIC this
targets (2406.14860, 28 nm CMOS) runs the network as combinational logic at ~300 µW for 256 pixels, within
~6 µW per pixel overall, on 40 MHz collisions. The open track has no constraints.

## 2. Dataset

**Summary.** Zenodo [10.5281/zenodo.10783560](https://doi.org/10.5281/zenodo.10783560) v3, "Smart pixel dataset"
(Swartz, Dickinson; CC-BY-4.0): PixelAV simulation of CMS-like tracks in a 100 µm sensor with 50 × 12.5 µm²
pixels, 160 file pairs, 7,999,482 clusters, each a (20, 13, 21) time × y × x cumulative-charge array at 200 ps
steps plus truth labels. `scripts/fetch_data.py` downloads it, checks md5s and records the revision.

**Splits** (file-disjoint, seed 0, `common.io.split_files`; no overlap by construction):

| Split | Size | Purpose |
|---|---|---|
| Train | 2:1:1 per class, equal negative / positive high-pT (each charge × pT category subsampled to the smallest) | fit |
| Validation | last 20 % of train rows after a seeded shuffle | early stopping, median-seed choice |
| Test | 32 of 160 files, 1,599,888 clusters (all rows) | scoring |

**Schema.** Every field, the ground truth (class from signed `pt`) and the forbidden inputs:
[data/SCHEMA.md](../../data/SCHEMA.md).

**FAIR.** Findable: DOI-versioned record, per-cluster `sample_id = <recon file stem>:<row>`. Accessible: public
Zenodo, open HTTPS API. Interoperable: Parquet with documented columns. Reusable: versioned record, CC-BY-4.0,
selection and features in this repository.

**Bounded.** No augmentation or extra data for scored submissions.

## 3. Performance metrics

Each test cluster in signed-pT bin b is weighted N_ref[b] / N_test[b], N_ref being the v3 signed-pT histogram of all
7,999,482 clusters (0.1 GeV bins over (−2, 2], one bin per sign above 2 GeV), so every metric estimates the rate on
the physical pT spectrum.

| Metric | Definition | Captures |
|---|---|---|
| **background rejection @ eff 0.933** (primary) | reject clusters with p0 < cut, cut set (linear interpolation between distinct p0 values) so the reweighted fraction of \|pT\| > 2 GeV clusters kept is 0.933; reweighted fraction of \|pT\| < 2 GeV clusters rejected | rejection at the paper's M2 efficiency, so efficiency cannot be traded for rejection |
| signal efficiency | reweighted fraction of \|pT\| > 2 GeV clusters with pred = 0 | argmax operating point |
| background rejection | reweighted fraction of \|pT\| < 2 GeV clusters with pred ≠ 0 | argmax operating point |
| data reduction | reweighted fraction of all clusters with pred ≠ 0 | bandwidth saved |
| turn-on | unweighted acceptance vs signed pT, 0.1 GeV bins, binomial errors | plot only |

Edge cases: a cluster without a prediction is scored as read out (class 0, p0 = 1); the primary metric is NaN if
there is no signal or no background. Ranking is per track by the primary metric, mean over 5 seeds.

## 4. Reference solution

Code: `src/smartpix_bench/filter/` (models vendored from jieunyoo/directional-pixel-detectors @4618ab2; M1 from
the paper text). Softmax outputs, sparse categorical cross-entropy, Adam 1e-3, batch 1024, ≤ 200 epochs, early stop
on validation loss (patience 20, best weights), inputs standardized with train statistics.

| Model | Track | Inputs | Architecture | Parameters |
|---|---|---|---|---|
| M1 | open | y-size, y0 | Dense 128 relu → 3 | 771 |
| M2 | open | slice-19 y-profile, y0 | Dense 128 relu → 3 | 2307 |
| M3 | open | slices 0–7 y-profiles (8 × 13), y0 | Conv2D 16 (3×3) → Conv2D 64 (3×3) → Dense 32, dropout 0.1 → 3 | 83,331 |
| QM2 | asic | 2-bit y-profile, 6-bit y0 | QDense 128 (5-bit weights) → BN → 10-bit ReLU → QDense 3 | 2819 |
| *QM2_w4a8 (auxiliary)* | asic | as QM2 | 4-bit weights, 8-bit ReLU (the paper's stated hardware choice) | 2819 |

**Results** (HiPerGator `hpg-default`, AMD EPYC Rome; mean ± sample std over 5 seeds; test set as above):
the 2:1:1 reference run is *pending* (`docs/VALIDATION.md` §1.4). The superseded 1:1:1 ablation ranks
M3 0.332 ± 0.009, M2 0.223 ± 0.006, QM2 0.197 ± 0.002, M1 0.156 ± 0.000 on the primary metric.

**Verification.** Features match the paper's preprocessed inputs row by row; our scorer reproduces Table 1 from
the paper's stored M2 predictions (93.19 % / 25.13 %); M2 retrained on the paper's inputs gives 92.96 ± 0.74 % /
24.33 ± 0.89 % vs Table 1's 93.3 / 25.1 ([VALIDATION](../../docs/VALIDATION.md) §1.1–1.3).

**Requirements.** CPU only; Python 3.11, TensorFlow 2.15.1, QKeras 0.9.0 (`uv.lock`). Cache 5.5 min / 8 GB;
training 1–15 min per model-seed on 4 cores.

## 5. Documentation and reproducible protocol

**Reproduction.** The quick start above, or `slurm/submit.sh` on a cluster; `uv.lock` pins every package.

**Motivation.** Pixel detectors at the HL-LHC produce more data than can leave the detector at 40 MHz. Rejecting
clusters from low-pT tracks, which no trigger uses, inside the pixel chip cuts that data volume before it is
read out; the question is how much a network that fits on the chip can reject while keeping high-pT clusters.

**Background.** A charged particle's curvature in the magnetic field sets its crossing angle through the sensor,
and the angle sets the cluster's length along y and its charge-collection time profile. Yoo et al.,
[arXiv:2310.02474](https://arxiv.org/abs/2310.02474) define the task and models; Parpillon et al.,
[arXiv:2406.14860](https://arxiv.org/abs/2406.14860) put the quantized model on a 28 nm ASIC.

**Known limitations.**
- The paper's numbers come from a different simulation run (dataset8) than the public data; paper-input checks are
  CMS-restricted and reported only in aggregate.
- The paper's Table 2 estimate on tracked / untracked samples (CMS-restricted DS14 / DS15) is not part of this
  release: its tracking-efficiency correction was not found.
- The primary metric and the 2:1:1 balance are this benchmark's readings of the paper (VALIDATION §3 F3, F11).
- M1 has no upstream code.

**Citation.** Yoo et al., "Smart pixel sensors: towards on-sensor filtering of pixel clusters with deep
learning", [arXiv:2310.02474](https://arxiv.org/abs/2310.02474); dataset
[10.5281/zenodo.10783560](https://doi.org/10.5281/zenodo.10783560).

**License.** Dataset: CC-BY-4.0. Code: Apache-2.0 ([LICENSE](../../LICENSE)).
