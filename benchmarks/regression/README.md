# Smart Pixels regression: on-chip hit position and angle with uncertainties

> From one silicon-pixel cluster, as a 16 × 16 pixel readout chip sees it, estimate where the particle crossed the
> sensor (x, y) and at which angles (α, β), with calibrated uncertainties, using a network small enough to
> synthesize next to the pixels with 2-bit inputs.

**Scientific Motif:** High-Energy Physics · **AI/ML Motif:** Regression · **Computing Motif:** Latency Bound

Card structure: MLCommons Science Benchmarks Ontology (arXiv:2511.05614). Sibling benchmark:
[filter](../filter/README.md). Validation record: [docs/VALIDATION.md](../../docs/VALIDATION.md) §2.

## Quick start

```bash
uv sync --extra tf                                          # Python 3.11, TensorFlow 2.15.1, QKeras 0.9.0
uv run python scripts/fetch_data.py --task regression       # Zenodo 18472791 -> $SPB_DATA_DIR (default data/zenodo)
uv run python -m pytest tests -q
m=Full_MLP; s=0
for c in sq c1 c2 c3 c4; do uv run python -m smartpix_bench.regression.train --model $m --config $c --seed $s; done
for c in c1 c2 c3 c4; do for split in 3s 2s; do
  uv run python -m smartpix_bench.regression.predict --weights results/weights/regression/${m}_${c}_s$s --split $split --out results/regression/$split
done; done
SPB_RESULTS=results uv run bash scripts/score_all.sh regression
```

The median-seed reference checkpoints are committed in `weights/regression/`, so `predict` runs on them without
training (e.g. `--weights weights/regression/Full_MLP_c4_s2`);
`scripts/fetch_weights.py --out <dir>` copies them elsewhere and checks their sha256.

The full reference (7 models × 5 configurations × 5 seeds, about 1,300 core-hours) is one command on HiPerGator:
`bash slurm/submit.sh -A <group> regression` ([docs/HIPERGATOR.md](../../docs/HIPERGATOR.md)). To score your own
model, write `results/regression/<3s|2s>/predictions_<name>.csv` (every contained cluster; columns below) and, for
the asic track, `descriptor_<name>.json`, then rerun `scripts/score_all.sh regression`; it also writes your
Codabench zip, which carries 3s only (the competition does not score 2s, to keep uploads small).

## 1. Problem specification and constraints

**Task.** Given the cumulative charge (electrons) on a 16 × 16 array of 50 × 12.5 µm² pixels in a 100 µm sensor,
predict the track position at the sensor mid-plane, x and y (µm), and the incidence angles as cot α and cot β,
each with a 1σ uncertainty for probabilistic models.

**Inputs.** Pixel charges only, at time slices 0 and 19 (200 ps steps; 3.8 ns apart), or all 20 slices for the
idealized c1 configuration. Every label column is a forbidden input ([data/SCHEMA.md](../../data/SCHEMA.md)).

**Outputs.** `predictions_<name>.csv`: `sample_id, x-midplane, y-midplane, cotAlpha, cotBeta` and optionally
`sigma_<target>`, for every contained cluster of a test set; a target a model does not predict is left empty.

**System constraints (asic track).** At most 2734 parameters (the largest reference model); inputs at slices
{0, 19}, each pixel digitized to 2 bits by three strictly increasing thresholds the submission declares (Eq 4,
T_{j−1} ≤ Q < T_j; on-chip configurable); weights and activations ≤ 8-bit fixed point. Declared in
`descriptor_<name>.json` and checked by the scorer and the packager. The paper's designs synthesize in TSMC
28 nm at a 25 ns clock with 2-cycle latency and 0.29–0.64 mm² of area. The open track has no constraints.

## 2. Dataset

**Summary.** Zenodo [10.5281/zenodo.18472791](https://doi.org/10.5281/zenodo.18472791), "Smartpixels 16x16
datasets" (Shekar, Swartz, Dickinson, Wadud; CC-BY-4.0): PixelAV clusters, (20, 16, 16) time × y × x, in three sets:
3sExtended (|cot β| ≤ 10, 1,971,855 clusters), 3s (1,999,095) and 2s (2,040,253), the latter two with
|cot β| ≤ 1.2. Clusters with ≥ 50 e on the array's border pixels at slice 19 are dropped from every split.

**Splits** (file-disjoint, seed 0; no overlap by construction):

| Split | Size (contained clusters) | Purpose |
|---|---|---|
| Train | 80 % of 3sExtended files | fit; label scale (99th percentile \|label\| per target) |
| Validation | 20 % of 3sExtended files | best-epoch checkpoint, median-seed choice |
| Test 3s | 1,377,888 | scoring |
| Test 2s | 1,401,971 | scoring (a second simulated population) |

**Schema.** [data/SCHEMA.md](../../data/SCHEMA.md). **FAIR.** DOI-versioned public Zenodo record,
`sample_id = <recon file stem>:<row>`, Parquet with documented columns, CC-BY-4.0, selection code here.
**Bounded.** No extra data for scored submissions; training-time input noise, as the reference's SoftQuantize
stage uses to model readout noise, is permitted.

## 3. Performance metrics

Per test set and per variable v ∈ {x, y (µm), α, β (degrees, α = arctan(1/cot α), +180° if negative)}:

| Metric | Definition | Captures |
|---|---|---|
| I68_v | width of the shortest interval holding ⌈0.68 n⌉ residuals (prediction − truth) | resolution, robust to tails |
| mean_v | mean residual | bias |
| mean_sigma_v, pull_mean_v, pull_std_v | mean predicted σ; mean and std of residual / σ | uncertainty scale and calibration |
| nll_v | mean Gaussian negative log-likelihood | joint accuracy and calibration |
| frac_1sigma_v, frac_2sigma_v | fraction with \|residual\| < σ, < 2σ | interval coverage (ideal 0.683, 0.954) |
| **rank_i68** (primary) | mean over x, y, β of I68_v / I68_v of the reference | one resolution score; lower is better |

The reference I68 per test set is the Full_MLP c4 seed with the median validation I68
(`regression/i68_reference.json`, written once from the reference run). α is reported but not ranked, since
Slim models do not predict it. A NaN prediction is excluded from that variable and counted; a model missing any
x, y or β is unranked. σ for an angle is propagated from σ(cot) by the derivative.

## 4. Reference solution

Code: `src/smartpix_bench/regression/` (builders vendored from smart-pixels-ml-2bit-Synth @4983446 and
smart-pixels-ml @1fb333f). Seven networks, every one with L1L2(0.01) kernel and L2(0.01) activity
regularization, Nadam 1e-3, batch 5000, 1000 epochs, best-validation checkpoint:

| Model | Head | Parameters (2 slices) |
|---|---|---|
| Max_Conv2D | 4 means + full covariance (MVN, Cholesky) | 1869 |
| Full_Conv2D / Full_Conv1D / Full_MLP | 4 means + 4 σ (diagonal Gaussian NLL) | 1767 / 2734 / 2264 |
| Slim_Conv2D / Slim_Conv1D / Slim_MLP | x, y, cot β (squared error summed over the batch) | 1682 / 2649 / 2179 |

Configurations: c1 20 slices, float; c2 slices {0, 19}, float; c3 slices {0, 19} digitized at the thresholds the
model's own SoftQuantize stage learned (straight-through 2-bit ADC with a sigmoid surrogate gradient, k annealed
1 → 67, thresholds from 248 / 668 / 1663 e, N(0, 80 e) noise, noisy fixed-seed validation); c4 = c3 with QKeras
weights (conv 4-bit, dense 8-bit) and activations (asic track). Auxiliary rows: *Barycenter*, *LocalReco
(optimistic)*, *Geometric (optimistic)*, *Geometric + correction (optimistic)* (paper §3.4; the last three use truth
angles or signs).

**Results** (HiPerGator `hpg-default`; mean ± sample std over 5 seeds; 3s, 2s within 0.02; per variable and
both sets in [reference_results/regression/LEADERBOARD.md](../../reference_results/regression/LEADERBOARD.md)):

| rank_i68 (3s) | c1 | c2 | c3 | c4 (asic) | c4 I68 x / y µm, β ° |
|---|---|---|---|---|---|
| Full_MLP | 0.667 ± 0.030 | 0.800 ± 0.021 | 0.927 ± 0.017 | 1.021 ± 0.035 | 15.18 / 2.89 / 5.80 |
| Slim_MLP | 0.617 ± 0.008 | 0.892 ± 0.011 | 0.940 ± 0.028 | 1.052 ± 0.048 | 14.18 / 2.77 / 6.92 |
| Slim_Conv1D | 0.639 ± 0.021 | 0.743 ± 0.021 | 0.918 ± 0.018 | 1.148 ± 0.027 | 16.25 / 2.93 / 7.43 |
| Full_Conv1D | 0.490 ± 0.022 | 0.703 ± 0.027 | 0.925 ± 0.020 | 1.163 ± 0.052 | 16.90 / 3.11 / 7.09 |
| Full_Conv2D | 0.611 ± 0.032 | 0.962 ± 0.070 | 1.087 ± 0.017 | 1.349 ± 0.041 | 21.44 / 3.33 / 8.11 |
| Slim_Conv2D | 0.763 ± 0.016 | 0.990 ± 0.017 | 1.146 ± 0.026 | 1.395 ± 0.031 | 19.88 / 3.15 / 9.78 |
| Max_Conv2D | 0.673 ± 0.010 | 1.056 ± 0.081 | 2.892 ± 2.916 | 2.844 ± 2.915 | 45.76 / 9.73 / 11.63 |

Against the paper's stated relations (VALIDATION §2.1): two frames cost 23–69 % in I68 for the Conv2D models and
Full_Conv1D (paper 30–40 %, some angles > 60 %), less for Full_MLP and Slim_Conv1D; for Slim_MLP x is
unchanged and y, β cost 53–86 %; 2-bit inputs change y and β by −10 to +30 % and cost 15–65 % in x (paper 5–10 %); the Conv2D models rank
last at c2–c4; Full and Slim agree within 6 % at c3 and c4. Baselines on 3s: Barycenter x / y I68 19.6 / 2.47 µm;
LocalReco 13.2 / 3.22 µm; corrected Geometric α / β I68 3.76° / 6.79°, against Full_MLP c3 14.3 / 2.53 µm and 2.90°
/ 5.25°.

**Verification.** Parameter counts, losses, SoftQuantize forward / gradient, the k schedule, decoding and
every metric are unit-tested with discriminating controls; the full chain ran on real data
([VALIDATION](../../docs/VALIDATION.md) §2.3). The paper publishes its resolutions only as figures, so the comparison
is against its stated relations between configurations (§2.1).

**Requirements.** CPU only; Python 3.11, TensorFlow 2.15.1, QKeras 0.9.0 (`uv.lock`); 16 GB per training (48 GB
for c1); 1000-epoch sq stage 5–11 h on 4 cores.

## 5. Documentation and reproducible protocol

**Reproduction.** The quick start above, or `slurm/submit.sh` on a cluster; `uv.lock` pins every package.

**Motivation.** A track's position and angles measured in one layer, with an uncertainty, let the detector
select or seed tracks at the sensor edge instead of shipping every pixel. The benchmark asks how well a network
within an ASIC's area, latency and precision budget can do this, and how much the 2-bit, two-frame readout costs.

**Background.** The charge a crossing particle leaves drifts through the sensor over a few ns; the cluster's
shape and its development in time encode where and at which angles it crossed, with a Lorentz drift in y from
the magnetic field. Das et al., [arXiv:2602.15946](https://arxiv.org/abs/2602.15946), define the task, the models
and the four configurations.

**Known limitations.**
- The paper's evaluation set is not public; this benchmark tests on the 3s and 2s sets (VALIDATION §3 R3).
- 3sExtended keeps 591,562 clusters after containment, more than the paper's ~520k (R2).
- Max_Conv2D has 1869 parameters, the paper 1898 (R1). Its clipped likelihood traps or collapses 7 of its 15
  sq / c3 / c4 trainings (R15); all seeds stay in the mean, and its c3 reference checkpoint (seed 1, the median
  by validation) is one of the angle-collapsed ones.
- Slim_MLP c1 deviates from the paper's recipe: its inputs are standardized (nonzero pixels to zero mean, unit
  std). On raw 20-frame inputs its training was unstable and it ranked worse than at 2 frames (0.977 vs 0.892);
  standardized, it ranks 0.617 (R16).
- The ranking score and the SoftQuantize details where the paper's equations are inconsistent are this
  benchmark's choices (R4, R13).

**Citation.** Das et al., "On-chip probabilistic inference for charged-particle tracking at the sensor edge",
[arXiv:2602.15946](https://arxiv.org/abs/2602.15946); dataset
[10.5281/zenodo.18472791](https://doi.org/10.5281/zenodo.18472791).

**License.** Dataset: CC-BY-4.0. Code: Apache-2.0 ([LICENSE](../../LICENSE)).
