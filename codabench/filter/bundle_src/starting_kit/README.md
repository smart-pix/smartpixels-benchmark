# Starting kit

- `test_sample_ids.csv`: the ids your predictions must cover.
- `sample_submission.zip`: the reference QM2 model's predictions and asic descriptor, in the upload format.

Running the benchmark repository's `scripts/score_all.sh filter` on your `predictions_<name>.csv` writes the
upload as `results/filter/codabench/<name>_submission.zip`. By hand:
`zip submission.zip predictions_test.csv [descriptor.json]` (no wrapping folder).
