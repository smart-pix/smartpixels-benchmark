# Evaluation

Upload a zip with `predictions_test.csv` at its root: columns `sample_id, pred, p0, p1, p2` for every test
cluster, where pred ∈ {0, 1, 2} (0 = |pT| > 0.2 GeV, 1 = −0.2 ≤ pT < 0, 2 = 0 < pT ≤ 0.2 GeV) and p0–p2 are class
probabilities in [0, 1]. For the asic track, add `descriptor.json`:
`{"track": "asic", "n_params": N, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}` with N ≤ 2819.

Each test cluster in signed-pT bin b is weighted N_ref[b] / N_test[b], N_ref being the signed-pT histogram of the
whole dataset, so every metric estimates the rate on the physical pT spectrum.

- **Background rejection @ eff 0.933** (ranking): clusters with p0 below a cut are rejected, and the cut is set
  so that 93.3 % of |pT| > 2 GeV clusters are kept. The score is the fraction of |pT| < 2 GeV clusters rejected.
- Signal efficiency, background rejection and data reduction of the `pred` decisions.
- **ASIC track**: 1 if `descriptor.json` declares the asic track and passes its checks. A descriptor that
  declares the asic track and fails them is rejected with the violations named.

A submission fails, with a message naming the problem, on a missing file or column, a duplicate or missing
`sample_id`, or an invalid value.
