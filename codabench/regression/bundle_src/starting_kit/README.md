# Starting kit

- `3s_sample_ids.csv`, `2s_sample_ids.csv`: the ids your predictions must cover.
- `sample_submission.zip`: the ranking reference's predictions (Full_MLP c4, median seed) and asic descriptor,
  in the upload format.

Running the benchmark repository's `scripts/score_all.sh regression` on your `predictions_<name>.csv` writes the
upload as `results/regression/codabench/<name>_submission.zip`. By hand:
`zip submission.zip predictions_3s.csv predictions_2s.csv [descriptor.json]` (no wrapping folder).
