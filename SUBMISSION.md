# Submitting a model

Both benchmarks take results, not code: you train and run your model anywhere, write one predictions CSV per
test split (plus a descriptor for the asic track), and score it with the same scripts that score the reference.
Formats are defined in [data/SCHEMA.md](data/SCHEMA.md); the cards define the metrics
([filter](benchmarks/filter/README.md), [regression](benchmarks/regression/README.md)).

## The contract

| | filter | regression |
|---|---|---|
| Splits | `test` | `3s`, `2s` (both required) |
| File per split | `$SPB_RESULTS/filter/test/predictions_<name>.csv` | `$SPB_RESULTS/regression/<split>/predictions_<name>.csv` |
| Columns | `sample_id, pred, p0, p1, p2` | `sample_id, x-midplane, y-midplane, cotAlpha, cotBeta` and optionally `sigma_<target>`; a target you do not predict is left empty |
| Rows | every test cluster | every contained cluster of the split |
| Primary metric | background rejection @ efficiency 0.933 (higher is better) | rank_i68 (lower is better; needs x, y and cot β) |

`<name>` is free; `<model>_s<seed>` names are aggregated as mean ± std over seeds, and the reference reports 5
seeds. The `sample_id`s are those of `truth.csv`, which `scripts/score_all.sh` writes into each split directory from
the fetched data (`scripts/fetch_data.py`; the filter also needs its feature cache).

**Track.** Without a descriptor a model is on the open track. For the asic track, put
`descriptor_<name>.json` next to every predictions file of that model:

```json
{"track": "asic", "n_params": 2819, "adc_thresholds": [400, 1600, 2400], "y0_bits": 6}
```

```json
{"track": "asic", "n_params": 2264, "slices": [0, 19], "thresholds": [250.0, 700.0, 1700.0],
 "weight_bits": 8, "activation_bits": 8}
```

(filter, then regression.) The limits are filter n_params ≤ 2819 at exactly those ADC thresholds and a 6-bit y0;
regression n_params ≤ 2734, slices [0, 19], three strictly increasing thresholds, ≤ 8-bit weights and
activations. A descriptor that fails these is listed in the open table only, and gets no Codabench zip.
The descriptor is a declaration: the scorer checks its values, not your network.

**Score.**

```bash
SPB_RESULTS=results uv run bash scripts/score_all.sh filter|regression
```

This writes `<split>/<name>/{metrics.json, METRICS.md}`, rebuilds `LEADERBOARD.md` with your rows, and packages
`results/<task>/codabench/<name>_submission.zip` (all splits at the zip root, plus `descriptor.json`). It refuses
to package a model with a missing id, a duplicate id, a missing column or an invalid value (filter: pred ∉ {0, 1, 2}
or a probability not finite or outside [0, 1]; regression: an infinite value), and names the problem. That zip is the file to
upload to Codabench ([codabench/README.md](codabench/README.md)).

**Rules.** Train on the train split only (validation is yours to use). The test labels are public, so the
leaderboard relies on you not fitting, tuning or selecting on them. Label columns are not inputs
(data/SCHEMA.md lists them). Filter: no augmentation or extra data. Regression: no extra data; training-time
input noise is permitted.

## What to report with it

Required:
- the predictions and descriptor above, on every split;
- `metrics.json` from `score_all.sh` for each split, unmodified;
- a comparison of prediction against truth: for the filter, the turn-on curve and the confusion matrix at your
  operating point; for regression, the residual distribution of x, y, α, β and, with σ, the pull distribution.

Strongly recommended:
- the architecture, inputs and training recipe, enough to reimplement it without your code;
- code and trained weights;
- for the asic track, how the descriptor's numbers were obtained (parameter count, quantizer settings) and, if
  you have one, a synthesis estimate (area, power, latency) with the tool and technology.

As applicable:
- anything beyond the dataset and the card's rules that the result depends on (pretraining, other simulation,
  a different input digitization on the open track), so the result is not silently non-comparable.
