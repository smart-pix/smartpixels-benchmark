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
| dec8_ds8_ts Input | slices 0–7 y-profiles (8 × 13, slice-major) | 652,160 (81 / 37; others ≤ 0.014 e) | *pending* |

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
| M3 | dec8_ds8_ts | *pending* | *pending* | *pending* | 97.6 / 21.7 |

M2 reproduces Table 1 within its seed spread, which validates the training, prediction and metric code
independently of our data pipeline.

### 1.4 Reference models on Zenodo v3

Test set: 32 of 160 files (1,599,888 clusters), reweighted to the v3 signed-pT spectrum. Mean ± sample std over
5 seeds; the primary metric is background rejection at a reweighted signal efficiency of 0.933.

**2:1:1 training balance (reference):** *pending* (run `runs/filter_d74c027`, jobs in §6).

**1:1:1 training balance (ablation, superseded; §8):**

| Model | Track | Bkg. rej. @ eff 0.933 | Bkg. rej. (argmax) | Signal eff. (argmax) | Data reduction | Val-median seed |
|---|---|---|---|---|---|---|
| M3 | open | 0.3317 ± 0.0091 | 0.3053 ± 0.0099 | 0.9467 ± 0.0045 | 0.2966 ± 0.0096 | 1 |
| M2 | open | 0.2225 ± 0.0062 | 0.3848 ± 0.0160 | 0.8242 ± 0.0166 | 0.3776 ± 0.0160 | 1 |
| QM2 | asic | 0.1971 ± 0.0019 | 0.4354 ± 0.0160 | 0.7420 ± 0.0160 | 0.4293 ± 0.0160 | 0 |
| M1 | open | 0.1561 ± 0.0004 | 0.5662 ± 0.0051 | 0.5246 ± 0.0059 | 0.5631 ± 0.0052 | 4 |

With 1:1:1 the argmax operating point sits at much lower signal efficiency than the paper's (M2 82 % vs 93 %);
the paper's own train set is 2:1:1, which moved the reference to that balance (§3 F3). M2 reached the 200-epoch
cap without early stopping in 4 of 5 seeds.

## 2. Regression (2602.15946; data Zenodo 18472791)

### 2.1 Reference models

7 models × {c1, c2, c3, c4} × 5 seeds, each c3/c4 on the thresholds its own SoftQuantize (sq) stage learned;
test sets 3s (1,377,888 contained clusters) and 2s (1,401,971). Run `runs/regression_0731db5` (§6): *pending*.
The paper gives its resolutions only in figures (Fig 3; its data file was never committed upstream), so the
comparison is against the paper's stated relations:

| Paper statement (§1.3) | This benchmark |
|---|---|
| two frames are 30–40 % worse than 20 frames (c2 vs c1), some angles > 60 % | *pending* |
| two-bit inputs at the optimal thresholds are 5–10 % worse (c3 vs c2) | *pending* |
| predicted σ is close to but greater than I68 for β | *pending* |
| MLP Full ȳ bias −0.1 µm | *pending* |

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

The sign of the Lorentz term in Eq 14 was chosen on data (§3 R12): the other sign gives β I68 = 37.5° (3s).
The biases have the paper's sign and size; the paper does not say which test set or input precision its
Fig 4 used.

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
- **R6 Slim loss.** Upstream sums squared errors / 3; the paper's mean squared error is used.
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
| Filter 2:1:1 (reference) | d74c027 | cache 50655964, train 50655965_[0-19], score 50655966 | *pending* |
| Filter M3 paper-input retrain | 0731db5 | 50657322_[0-4] | *pending* |
| dec6/jan2/dec8 feature check | 0731db5 | 50656085 | *pending* |
| Regression baselines | 560ee15 | srun 50648773 | exit 0 |
| Regression smoke | 560ee15 | 50653219 | exit 0 |
| Regression reference | 0731db5 | sq 50657941, c1 50657942, c2 50657943, c3c4 50657944, predict 50657945, score 50657946 | *pending* |

## 7. Tools

benchmark-builder skill @cf43c37 (layout, rubric, HPC templates). No other external checker has been run.

## 8. History

- 2026-11-10: the filter reference moved from a 1:1:1 to a 2:1:1 train balance (F3); the 1:1:1 run
  (`runs/filter_5faa4ba`, commit 5faa4ba) is kept as the ablation in §1.4.
