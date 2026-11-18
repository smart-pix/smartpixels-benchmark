# Validation

How the reference solutions of both benchmarks were checked, with the numbers, dates and hardware. Paper text is
the specification; upstream code fills its silences; every place this benchmark departs from either is listed in
§3. All runs: UF HiPerGator `hpg-default` CPU nodes (AMD EPYC Rome, 128 cores, ~1 TB), TensorFlow 2.15.1, QKeras 0.9.0,
NumPy 1.26.4, Python 3.11 (`uv.lock`); job IDs in §6.

Some checks use CMS-internal data — the paper authors' own preprocessed CSVs and the simulation they came from
(dataset8; DS14/DS15 for Table 2). These are available to CMS members on request from the authors; they are not
redistributed, and only aggregate numbers derived from them appear here.

Contents: 1 Filter · 2 Regression · 3 Discrepancies · 4 Training-label audit · 5 Metric checks · 6 Runs · 7 Tools · 8 History

## 1. Filter (2310.02474; data Zenodo 10783560 v3)

### 1.1 Preprocessing equivalence with the paper's inputs (CMS-restricted)

`scripts/validation/filter_features_ds8.py` recomputes our features on dataset8 (the simulation behind the
paper's CSVs) and compares them row by row with the paper's preprocessed files, matching rows on
(round(y-local, 5), round(pT, 5)), first occurrence. A row differs when any value is off by more than 0.01 e.

| Paper file | Feature | Train rows (off / at colliding key) | Test rows (off / at colliding key) |
|---|---|---|---|
| dec6_ds8 FullPrecisionInput | slice-19 y-profile | 652,160 (30 / 30) | 163,040 (0 / 0) |
| jan2_finalDS QuantizedInput | 2-bit y-profile (Table 3 ADC) | 652,160 (29 / 29) | 163,040 (0 / 0) |
| jan2_finalDS QuantizedInput | 6-bit y0 | 3,349 off, all y < −8 mm (§3 F4) | 832 off, all y < −8 mm |
| dec8_ds8_ts Input | slices 0–7 y-profiles (8 × 13, slice-major) | 652,160 (81 / 37; others ≤ 0.014 e) | 163,040 (8 / 0; all ≤ 0.019 e) |

Every row off by more than float32 rounding sits at a (y-local, pT) key that occurs more than once, where the
first-occurrence match can pick the wrong cluster. dataset8 has 385 such keys, the paper train set 13. The
transposed (y-major) reading of the dec8 columns mismatches every row, which fixes the layout M3 consumes.

### 1.2 Metric reproduction of Table 1 from stored predictions

`scripts/validation/filter_golden.py` scores the paper's stored M2 predictions
(`smartPixelsPaper/baselineModelFP/predictionsFiles.csv`, argmax only, CMS-restricted) with our scorer and the
dataset8 pT reference (7,999,509 clusters):

| | Signal efficiency | Background rejection | Data reduction |
|---|---|---|---|
| Paper Table 1, M2 | 93.3 % | 25.1 % | — |
| Our scorer on the paper's predictions | 93.19 % | 25.13 % | 24.50 % |

The 0.1-point efficiency gap is the upstream notebook swapping the per-charge reweighting ratios (§3 F6); the
notebook reports 93.28 %. The stored predictions carry no probabilities, so the fixed-efficiency primary metric
cannot be evaluated on them.

### 1.3 Retraining on the paper's own inputs (CMS-restricted)

`scripts/validation/filter_retrain_paper.py` trains our model code with our recipe on the paper's preprocessed
train set (2:1:1, 652,160 clusters) and scores the paper's test set. Mean ± sample std over 5 seeds.

| Model | Inputs | Signal eff. (argmax) | Bkg. rejection (argmax) | Bkg. rejection @ eff 0.933 | Paper Table 1 |
|---|---|---|---|---|---|
| M2 | dec6_ds8 | 92.96 ± 0.74 % | 24.33 ± 0.89 % | 23.40 ± 0.21 % | 93.3 / 25.1 |
| M3 | dec8_ds8_ts | 97.44 ± 0.46 % | 21.96 ± 0.58 % | 33.96 ± 0.89 % | 97.6 / 21.7 |

M2 and M3 reproduce Table 1 within their seed spread, which validates the training, prediction and metric code
independently of our data pipeline.

### 1.4 Reference models on Zenodo v3

Test set: 32 of 160 files (1,599,888 clusters), reweighted to the v3 signed-pT spectrum. Mean ± sample std over
5 seeds; the primary metric is background rejection at a reweighted signal efficiency of 0.933.

**2:1:1 training balance (reference)** (run `runs/filter_d74c027`, jobs in §6; `reference_results/filter/`):

| Model | Track | Bkg. rej. @ eff 0.933 | Bkg. rej. (argmax) | Signal eff. (argmax) | Data reduction | Val-median seed | Paper eff. / rej. |
|---|---|---|---|---|---|---|---|
| M3 | open | 0.3365 ± 0.0031 | 0.2127 ± 0.0085 | 0.9794 ± 0.0023 | 0.2061 ± 0.0083 | 1 | 97.6 / 21.7 (Table 1) |
| M2 | open | 0.2314 ± 0.0035 | 0.2379 ± 0.0090 | 0.9315 ± 0.0059 | 0.2320 ± 0.0088 | 0 | 93.3 / 25.1 (Table 1) |
| QM2 | asic | 0.1989 ± 0.0011 | 0.2426 ± 0.0092 | 0.9070 ± 0.0066 | 0.2375 ± 0.0091 | 4 | 87.3 / 28.2 (Table 4; §3 F16) |
| M1 | open | 0.1563 ± 0.0003 | 0.2639 ± 0.0056 | 0.8420 ± 0.0055 | 0.2602 ± 0.0056 | 0 | 84.8 / 26.6 (Table 1) |
| *QM2_w4a8 (auxiliary)* | asic | 0.1919 ± 0.0017 | 0.2559 ± 0.0126 | 0.8930 ± 0.0088 | 0.2508 ± 0.0124 | 1 | — |

On the public v3 data the argmax operating points land within 0.6 points of Table 1's efficiencies and within
1.4 points of its rejections (M2 93.15 / 23.79 %, M3 97.94 / 21.27 %, M1 84.20 / 26.39 %), although the paper's
numbers come from a different simulation run (§3 F2).

**1:1:1 training balance (ablation, superseded; §8):**

| Model | Track | Bkg. rej. @ eff 0.933 | Bkg. rej. (argmax) | Signal eff. (argmax) | Data reduction | Val-median seed |
|---|---|---|---|---|---|---|
| M3 | open | 0.3317 ± 0.0091 | 0.3053 ± 0.0099 | 0.9467 ± 0.0045 | 0.2966 ± 0.0096 | 1 |
| M2 | open | 0.2225 ± 0.0062 | 0.3848 ± 0.0160 | 0.8242 ± 0.0166 | 0.3776 ± 0.0160 | 1 |
| QM2 | asic | 0.1971 ± 0.0019 | 0.4354 ± 0.0160 | 0.7420 ± 0.0160 | 0.4293 ± 0.0160 | 0 |
| M1 | open | 0.1561 ± 0.0004 | 0.5662 ± 0.0051 | 0.5246 ± 0.0059 | 0.5631 ± 0.0052 | 4 |

With 1:1:1 the argmax operating point sits at much lower signal efficiency than the paper's (M2 82 % vs 93 %;
M1 52.5 / 56.6, M3 94.7 / 30.5, QM2 74.2 / 43.5 % efficiency / rejection);
the paper's own train set is 2:1:1, which moved the reference to that balance (§3 F3). M2 reached the 200-epoch
cap without early stopping in 4 of 5 seeds.

## 2. Regression (2602.15946; data Zenodo 18472791)

### 2.1 Reference models

7 models × {c1, c2, c3, c4} × 5 seeds, each c3/c4 on the thresholds its own SoftQuantize (sq) stage learned;
test sets 3s (1,377,888 contained clusters) and 2s (1,401,971), every cluster predicted by every model. Run
`runs/regression_0731db5` (§6); the Slim models were retrained with the summed loss (§3 R6, §8). Per-variable
numbers: `reference_results/regression/LEADERBOARD.md`.

rank_i68 on 3s (mean ± sample std over 5 seeds; lower is better; 2s is within 0.02 of 3s in every cell):

| Model | c1 | c2 | c3 | c4 (asic) |
|---|---|---|---|---|
| Full_Conv1D | 0.490 ± 0.022 | 0.703 ± 0.027 | 0.925 ± 0.020 | 1.163 ± 0.052 |
| Full_MLP | 0.667 ± 0.030 | 0.800 ± 0.021 | 0.927 ± 0.017 | 1.021 ± 0.035 |
| Slim_Conv1D | 0.639 ± 0.021 | 0.743 ± 0.021 | 0.918 ± 0.018 | 1.148 ± 0.027 |
| Slim_MLP | 0.977 ± 0.040 | 0.892 ± 0.011 | 0.940 ± 0.028 | 1.052 ± 0.048 |
| Full_Conv2D | 0.611 ± 0.032 | 0.962 ± 0.070 | 1.087 ± 0.017 | 1.349 ± 0.041 |
| Slim_Conv2D | 0.763 ± 0.016 | 0.990 ± 0.017 | 1.146 ± 0.026 | 1.395 ± 0.031 |
| Max_Conv2D | 0.673 ± 0.010 | 1.056 ± 0.081 | 2.892 ± 2.916 | 2.844 ± 2.915 |

Max_Conv2D c3/c4 include failed seeds, kept in the mean (§3 R15). The paper gives its resolutions only in
figures (Fig 3; its data file was never committed upstream), so the comparison is against what its text states.
Ratios are of seed-mean I68 on 3s (2s agrees within 0.04 outside Max_Conv2D c3):

| Paper statement (§1.3) | This benchmark | Verdict |
|---|---|---|
| Two frames (c2) are 30–40 % worse than 20 (c1), a few angles more than 60 % | x / y / α / β: Max_Conv2D +63 / +40 / +49 / +67 %, Full_Conv2D +60 / +43 / +46 / +69 %, Full_Conv1D +23 / +42 / +29 / +66 %, Slim_Conv2D x / y / β +43 / +27 / +24 %, Slim_Conv1D +15 / +6 / +27 %, Full_MLP +5 / +16 / 0 / +36 %, Slim_MLP −17 / −3 / −7 % | close for the Conv2D models and Full_Conv1D, β the largest; smaller for the MLPs and Slim_Conv1D; inverted for Slim_MLP (§3 R16) |
| Two-bit inputs at the optimal thresholds (c3) are typically 5–10 % worse than c2 | y +7 to +23 %; β −10 to +30 %; x +15 to +65 % (Full_Conv2D 15, Slim_Conv2D 25, Slim_MLP 29, Slim_Conv1D 46, Full_MLP 50, Full_Conv1D 65 %); Max_Conv2D dominated by failed seeds | close for y and β, x degrades more than stated |
| Conv2D models have the largest resolution on most parameters; Conv1D and MLP are similar | the three Conv2D models rank last at c2, c3 and c4; at c1 Full_Conv2D ranks second; Conv1D and MLP models lie within 0.03 of each other at c3 and 0.15 at c4 | reproduced for c2–c4 |
| Little dependence on the number of outputs (Full vs Slim) | Full / Slim rank at c3: 1.087 / 1.146, 0.925 / 0.918, 0.927 / 0.940; at c4: 1.349 / 1.395, 1.163 / 1.148, 1.021 / 1.052 (Conv2D, Conv1D, MLP); at c1 Slim is 25–46 % worse | reproduced for c3 and c4 |
| Quantization (c4) shifts some residual means, the Conv2D models the most | c3 → c4 shift: β̄ +0.51° Full_Conv2D, +0.49° Slim_Conv2D, ≤ 0.30° for Conv1D / MLP; x̄ −0.51 µm Slim_Conv2D, but also +0.44 Full_MLP and −0.40 Slim_Conv1D | partly: largest β shift is Conv2D, x shifts are comparable across architectures |
| σ close to but greater than I68 for x, y, β; σ_α significantly over-estimated | 2 × mean σ / I68 (1 for a Gaussian), Full models c1–c4: x 1.05–1.43, y 1.25–1.71, β 1.16–1.71, α 2.03–3.22; Max_Conv2D c1 / c2 x 1.05–1.16, β 1.12–1.24, α 2.00–2.06 | reproduced, y and β over-estimated more than "close" |

### 2.2 Baselines (auxiliary)

Slice 19 digitized at the transformer thresholds 248/668/1663 e; all test-set clusters; I68 and mean residual.

| Baseline | Set | x I68 / mean (µm) | y I68 / mean (µm) | α I68 / mean (°) | β I68 / mean (°) | Paper ȳ / β̄ |
|---|---|---|---|---|---|---|
| Barycenter | 3s | 19.64 / −0.00 | 2.47 / −0.98 | — | — | −0.70 µm |
| LocalReco (optimistic) | 3s | 13.24 / −0.01 | 3.22 / −1.19 | — | — | −0.81 µm |
| Geometric (optimistic) | 3s | — | — | 3.76 / −0.01 | 22.75 / −6.32 | — |
| Geometric + correction (optimistic) | 3s | — | — | 3.76 / −0.01 | 6.79 / −2.01 | −1.1° |
| Barycenter | 2s | 19.37 / −0.01 | 2.56 / −0.97 | — | — | |
| LocalReco (optimistic) | 2s | 13.15 / −0.00 | 3.23 / −1.17 | — | — | |
| Geometric (optimistic) | 2s | — | — | 3.76 / −0.00 | 21.64 / −6.69 | |
| Geometric + correction (optimistic) | 2s | — | — | 3.76 / −0.00 | 6.88 / −1.84 | |
| Full_MLP c3 (5 seeds) | 3s | 14.32 / +0.08 | 2.53 / −0.05 | 2.90 / +0.07 | 5.25 / +0.36 | −0.1 µm / +0.6° |

The sign of the Lorentz term in Eq 14 was chosen on data (§3 R12): the other sign gives β I68 = 37.5° (3s).
The biases have the paper's sign and size; the paper does not say which test set or input precision its
Fig 4 used.

Against the paper's Fig 4 statements, with Full_MLP c3 as the ML model: x is comparable to LocalReco (14.3 vs
13.2 µm) and much better than Barycenter (19.6 µm), as stated; y is better than LocalReco (2.53 vs 3.22 µm) but
equal to Barycenter (2.47 µm), where the paper says the ML model is significantly better. α and β beat the
corrected Geometric method (2.90 vs 3.76°, 5.25 vs 6.79°), and β without the Lorentz correction is poor (22.8°),
as stated. The baselines' y biases are larger than the paper's (−0.98 / −1.19 vs −0.70 / −0.81 µm) and the ML bias
is as small (−0.05 vs −0.1 µm).

### 2.3 Pipeline smoke on real data

Full_MLP sq → c3 → c4, 5 epochs each, then predict, baselines and `score_all.sh regression` on 3s and 2s
(job 50653219, 14 min, exit 0). Validation pull widths after the 5-epoch sq stage: x 0.94, y 0.96, α 0.93, β 0.89,
with 76–82 % of clusters inside ±1σ.

## 3. Discrepancies

Each line: what the paper or upstream says, what this benchmark does, and the evidence.

### Filter

- **F1 Dataset text.** The Zenodo 10783560 description gives a pT range of 0.2–0.4 GeV; the v3 files span
  ±5 GeV with about 5 % |pT| < 0.2 GeV in every file and no flattened-pT subset. The data are used as they are.
- **F2 Paper data ≠ Zenodo.** The paper's clusters (dataset8) are a different PixelAV run from Zenodo v1/v3:
  kinematics match but no v1 slice reproduces dec6_ds8's pixels. The preprocessing check (§1.1) therefore runs on
  the CMS-restricted dataset8; the reference models train on public v3.
- **F3 Train balance.** "The same number of clusters for each pT class" is ambiguous. The paper's own train
  labels count 326,080 / 163,040 / 163,040 and its 800,000-cluster total is 4 × 203,800, so each
  (charge × pT) category is subsampled to the smallest one: classes 0:1:2 = 2:1:1 with equal negative and
  positive high-pT clusters (`filter.data.balanced_subsample`). The 1:1:1 run is kept as an ablation (§1.4).
- **F4 6-bit y0.** Codes follow the upstream converter, clip(ceil((y + 8)/0.25) − 1, 0, 63) (0 mismatches with
  jan2_finalDS outside y < −8 mm; floor gives mismatches). Upstream sends y < −8 mm to 63; this benchmark
  clips to 0 (3,349 train / 832 test rows of the paper's sets).
- **F5 M1.** No upstream code; the input y-size counts rows with nonzero net charge at slice 19 (paper §3.1),
  and the 771-parameter count comes from the description.
- **F6 Signal-efficiency reweighting.** The upstream notebook applies each charge's ratio to the other charge
  (93.28 % vs 93.19 % charge-consistent on the same predictions); this benchmark uses charge-consistent ratios.
- **F7 2-bit boundary.** Table 3's ≥ boundaries (≥ 400 → 1, …), applied per pixel before the row sum; one
  upstream converter uses strict >. Exact on every jan2_finalDS row.
- **F8 Quantized model.** The paper's hardware choice is w4/a8, the notebook behind its Fig 8 is w5/BN/a10
  (2819 parameters). QM2 is the notebook model (asic track); the w4/a8 variant is auxiliary.
- **F9 Recipes.** Upstream M3 trains 100 epochs with patience 10 on logits; every filter model here uses one
  recipe (softmax, SCCE, Adam 1e-3, batch 1024, ≤ 200 epochs, early stop patience 20 on validation loss).
- **F10 Seeds.** Upstream shuffles are unseeded. Here the split, subsample and validation shuffle are fixed
  (seed 0); the training seed drives initialisation, dropout and batch order only.
- **F11 Primary metric.** Not in the paper: background rejection at a reweighted signal efficiency of 0.933 (M2's
  Table 1 value), cutting on p0, so a model cannot buy rejection with efficiency. Argmax efficiency, rejection
  and data reduction are also reported.
- **F12 Reweighting reference.** The v3 signed-pT histogram of all 7,999,482 clusters (`filter/pt_reference.json`)
  replaces the paper's dataset8 histogram (7,999,509 clusters).
- **F13 Missing predictions.** A test cluster without a prediction is scored as read out (class 0, p0 = 1) and
  counted in the coverage block.
- **F14 Paper-internal.** Table 5's 400 e row (87.7 / 27.0) disagrees with Table 4's matching row (87.3 / 28.2);
  Table B2 prints "2000−24000". Neither is used.
- **F15 Not implemented.** The Table 2 data-rejection estimate on tracked/untracked samples (CMS-restricted DS14 /
  DS15) needs a tracking-efficiency correction whose code was not found, and the pixel-weighted bandwidth
  likewise; both are left out of this release.
- **F16 Table 4's quantized row.** QM2, the w5/BN/a10 notebook model behind Fig 8, gives 90.70 ± 0.66 % /
  24.26 ± 0.92 % (argmax, v3) against Table 4's "quantized weights & inputs" row, 87.3 / 28.2 %. No committed
  upstream model or notebook output reproduces that row, so the model behind it is unidentified; the w4/a8
  variant (the paper's stated hardware choice, auxiliary here) gives 89.30 / 25.59 %.

### Regression

- **R1 Max Conv2D size.** The builder gives 1869 parameters; the paper's Table 1 says 1898. Unexplained.
- **R2 Containment.** Clusters with ≥ 50 e on the 16 × 16 border at slice 19 are dropped (paper §3.1), computed
  from the pixels. Upstream's `chargeOriginal_atEdge < 50` selects the same clusters except 2 of 1,971,855 in
  3sExtended. 3sExtended keeps 591,562 clusters; the paper reports about 520k.
- **R3 Test sets.** Upstream evaluates on `dataset_3sr_..._centeredIncidence` (108,222 clusters), which is not on
  Zenodo; this benchmark scores 3s and 2s separately and trains on 3sExtended (80/20 by file).
- **R4 SoftQuantize.** Eq 7 as printed counts down and contradicts Eq 9, so D_soft = Σ σ(k(x − T_j)/τ_j);
  Eq 8's τ for the last threshold needs an undefined Δ, so τ_2 = Δ_2; T_min = 0; the hard forward pass is Eq 4
  with T_{j−1} ≤ x < T_j (upstream strict >). Upstream has no τ scaling, initialises at [400, 1000, 2000] and
  its one complete notebook trains 2000 epochs; this benchmark follows the paper (248/668/1663, 1000 epochs,
  k cosine 1 → 67). Max_Conv2D's upstream SoftQuantize builder raises a TypeError.
- **R5 Noise.** N(0, 80 e) per pixel per slice is never enabled in committed upstream notebooks. Here it is
  drawn per batch in the sq stage's training and once (fixed seed) for its validation, so best-validation
  thresholds are chosen under noise; c1–c4 and every test set are noise-free.
- **R6 Slim loss.** The paper names a mean-squared-error loss without its batch reduction. Upstream sums the
  squared errors over the batch: smart-pixels-ml's `slim_sse_loss` over batch and outputs, as this benchmark does
  (`regression.losses.sse`), smart-pixels-ml-2bit-Synth's `custom_sse_loss` the same divided by 3. The reduction
  sets the weight of the fixed L1L2(0.01) / L2(0.01) regularizers: with a batch mean the activity term outweighed
  the data term about 26,000× at initialisation and every Slim model trained to a constant (I68_x 128.3 µm, the
  truth spread; §8).
- **R7 Slim Conv1D.** Upstream has no regularizers and a 4-bit quantized dense layer; here every model carries
  the paper's regularization and the paper's 8-bit dense weights. Full Conv1D gets a QKeras builder (upstream has
  float only).
- **R8 Regularization.** The paper and upstream write `L1L2(0.01)`, which in Keras is l1 = 0.01, l2 = 0; kept.
- **R9 Label scale.** Upstream scales by the 99th percentile |label| of each generator, so train and validation
  scales differ; here the scale comes from the train split only and ships with each checkpoint.
- **R10 Paper-internal precision.** §1.3 says all weights are 8-bit, §3.2 gives conv weights fixed<4,1>; the
  QKeras models follow §3.2 and the ASIC check allows ≤ 8 bits.
- **R11 I68.** Shortest interval holding ⌈0.68 n⌉ residuals; the upstream notebook keeps ⌊0.68 n⌋ + 1.
- **R12 Baselines.** Input is slice 19 digitized at 248/668/1663 (paper §1.3); the upstream notebook uses full
  precision. Pixel centres (i − 7)·pitch and y + Δy/2 follow that notebook. Eq 14 adds a length to |cotβ|; the
  implementation is cotβ = s (w_y − 1) p_y / T − tanΘ_L with s = sign(cotβ_true + tanΘ_L), the relation Eq 12
  uses. LocalReco and both Geometric rows take signs and angles from truth, as the paper assumes.
- **R13 Ranking.** Not in the paper: the mean over x, y, β of I68 relative to the Full_MLP c4 seed with the
  median validation I68 (α is excluded since Slim models do not predict it); a model is unranked unless every
  test cluster has finite x, y and β.
- **R14 Upstream coverage.** Only the Slim MLP 2-bit pipeline is committed upstream; the other configurations are
  built from the paper's text.
- **R15 Max_Conv2D training failures.** The MVN likelihood is clipped to [1e-9, 1e9] as upstream does, so a
  cluster whose NLL reaches the clip (20.7) has zero gradient. Two failure modes appear (probe on 20,000
  validation clusters per checkpoint, a HiPerGator-only probe script):
  - *clip trap*: every cluster at the clip from the first epoch (validation loss flat at 102,767), so the
    model keeps a constant output (I68_x 128.4 µm, the truth spread): sq s4 (its thresholds never move from
    248 / 668 / 1663), c3 s4 (trained on those thresholds) and c4 s2;
  - *angle collapse*: α and β predictions constant (std of the predicted means 0.002 in scaled units, 0.08 % of
    clusters at the clip), x and y still learned: c3 s0 and c3 s1, and their sq stages.

  Per seed on 3s (rank_i68 / I68_x µm / I68_α ° / I68_β °):

  | Seed | sq thresholds (e) | c1 | c2 | c3 | c4 |
  |---|---|---|---|---|---|
  | 0 | 284 / 712 / 1719 | 0.67 / 10.9 / 2.1 / 3.4 | 1.14 / 19.5 / 3.2 / 6.3 | 2.05 / 20.5 / 147.1 / 21.0 (collapse) | 1.47 / 25.4 / 5.1 / 8.2 |
  | 1 | 279 / 715 / 1712 | 0.69 / 10.5 / 2.3 / 3.9 | 1.01 / 18.0 / 3.3 / 5.4 | 1.98 / 18.2 / 146.4 / 20.9 (collapse) | 1.50 / 25.2 / 5.5 / 8.8 |
  | 2 | 305 / 704 / 1716 | 0.67 / 10.5 / 2.1 / 3.8 | 1.13 / 18.7 / 3.9 / 6.8 | 1.16 / 20.1 / 3.7 / 6.3 | 8.06 / 128.4 / 146.8 / 21.0 (trap) |
  | 3 | 311 / 698 / 1714 | 0.67 / 11.3 / 2.2 / 3.5 | 0.95 / 15.9 / 3.3 / 5.5 | 1.22 / 20.8 / 4.2 / 6.7 | 1.58 / 25.1 / 5.7 / 9.4 |
  | 4 | 248 / 668 / 1663 (trap) | 0.67 / 11.5 / 2.5 / 3.4 | 1.04 / 17.3 / 2.9 / 6.2 | 8.06 / 128.4 / 146.8 / 21.0 (trap) | 1.61 / 24.7 / 4.8 / 10.8 |

  All five seeds stay in the mean ± std. The median-validation rule picks c3 **s1, an angle-collapsed
  checkpoint**, as the c3 reference weight (flagged in `weights/MANIFEST.json`); c2 s4 and c4 s3 are healthy.
  Even the healthy c4 seeds (rank 1.47–1.61) rank behind every other c4 model, unlike the paper's Fig 3.
- **R16 Slim_MLP at 20 frames.** Slim_MLP c1 is worse than its c2 in every seed (rank 0.94–1.03 vs 0.88–0.91 on
  3s), the one model where the paper's "two frames are worse" does not hold here. It is not a seed failure or
  early stop: best epochs 729–993 of 1000, no dead ReLUs. On a validation batch the regularizers are 44 % of its
  loss (data 48, regularizers 38) against 20 % at c2 (37 and 9), and on the raw-electron 20-frame inputs its
  training loss spikes: in the last 500 epochs 132–175 epochs exceed twice the minimum training loss (none at c2;
  57–116 for Slim Conv1D / Conv2D c1, which still beat their c2). The recipe is kept as the paper states it (no
  input standardization, Nadam 1e-3); upstream's data generator can standardize but does so off by default.

## 4. Training-label audit

- Filter: the class comes from the signed `pt` label (§3.1 boundaries 0.2 GeV) for both training and scoring;
  `filter.data.class_label`; the paper's train labels (dec6_ds8 TrainSetLabel) follow the same rule (§1.1 row
  match on pT).
- Regression: every model trains on `x-midplane`, `y-midplane`, `cotAlpha`, `cotBeta` divided by the train-split
  scale and is scored on the same columns in physical units (`regression.predict.decode`); no other label column
  is an input.

## 5. Metric checks

- Filter: the scorer reproduces Table 1 from the paper's stored predictions (§1.2).
- Regression: I68, pulls, NLL and calibration are unit-tested against hand-computed values; no second
  implementation (e.g. the genesis `uq-metrics-evaluator` skill) has been run yet.

## 6. Runs on HiPerGator

| Run | Commit | Jobs | Outcome |
|---|---|---|---|
| Filter 1:1:1 (ablation) | 5faa4ba | cache 50647290, train 50647291_[0-19], score 50647292 | all exit 0 |
| Filter M2 paper-input retrain | 5faa4ba | 50647293_[0-4] | all exit 0 |
| Filter 2:1:1 (reference) | d74c027 | cache 50655964, train 50655965_[0-19], score 50655966 | all exit 0 |
| Filter QM2_w4a8 (auxiliary) | afd4a9a (filter code = d74c027) | train + predict 51318558_[0-4], rescore 51318559 | all exit 0 |
| Filter M3 paper-input retrain | 0731db5 | 50657322_[0-4] | all exit 0 |
| dec6/jan2/dec8 feature check | 0731db5 | 50656085 | exit 0 |
| Regression baselines | 560ee15 | srun 50648773 | exit 0 |
| Regression smoke | 560ee15 | 50653219 | exit 0 |
| Regression reference | 0731db5 | sq 50657941, c1 50657942, c2 50657943, c3c4 50657944, predict 50657945, score 50657946 | all exit 0; Slim rows superseded (§8) |
| Regression Slim retrain | afd4a9a | sq 51054890, c1 51054891, c2 51054892, c3c4 51054893, predict 51054894, score 51054895 (tasks 20–34) | all exit 0 |
| Regression Codabench bundle (3s and 2s), tiers 1–4 | afd4a9a | build + tier 4 51318708, tiers 1–3 51902398 | exit 0; 144 zips match; superseded by 3s only |
| Regression Codabench bundle (3s only), tiers 1–4 | 75ebfca | 51905203 | exit 0; 144 zips match on 3s; zip 265 MB (codabench/README.md) |

## 7. Tools

benchmark-builder skill @cf43c37 (layout, rubric, HPC templates). No other external checker has been run.

## 8. History

- 2026-11-10: the filter reference moved from a 1:1:1 to a 2:1:1 train balance (F3); the 1:1:1 run
  (`runs/filter_5faa4ba`, commit 5faa4ba) is kept as the ablation in §1.4.
- 2026-11-12: the Slim loss changed from a batch mean to a batch sum (R6, afd4a9a). With the mean every Slim
  model of run 0731db5 predicted a constant (I68_x 128.3 µm); those 75 checkpoints were retrained and their
  rows replaced (outputs kept on HiPerGator in `runs/regression_0731db5/superseded_slim_mean/`).
