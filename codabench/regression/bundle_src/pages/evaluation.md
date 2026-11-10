# Evaluation

Upload a zip with `predictions_3s.csv` and `predictions_2s.csv` at its root. Each has columns
`sample_id, x-midplane, y-midplane, cotAlpha, cotBeta` and optionally `sigma_<target>`, for every contained
cluster of that test set; leave a target you do not predict empty. For the asic track, add `descriptor.json`:
`{"track": "asic", "n_params": N, "slices": [0, 19], "thresholds": [T1, T2, T3], "weight_bits": W,
"activation_bits": A}` with N ≤ 2734, T1 < T2 < T3 in electrons, and W, A ≤ 8.

Each test set and variable v ∈ {x, y (µm), α, β (degrees)} is scored on the residual (prediction − truth):
- **I68_v**: the width of the shortest interval holding 68 % of the residuals;
- the mean residual;
- with σ: the pull mean and std, the Gaussian NLL and the 1σ / 2σ coverage.

**rank_i68** (ranking, on 3s; lower is better) is the mean over x, y and β of I68_v divided by the I68_v of the
reference model (the median seed of Full_MLP c4). It is empty unless every cluster has x, y and cot β. α is
reported but not ranked.

**ASIC track** is 1 if `descriptor.json` declares the asic track and passes its checks. A descriptor that
declares the asic track and fails them is rejected with the violations named. A submission also fails, with a
message naming the problem, on a missing file or column, a duplicate or missing `sample_id`, or an infinite
value.
