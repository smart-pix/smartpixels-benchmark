# Codabench bundles

One results-submission competition per benchmark. Participants run their model offline and upload the zip that
`scripts/score_all.sh` writes (`results/<task>/codabench/<name>_submission.zip`; contract in
[SUBMISSION.md](../SUBMISSION.md)). The ground truth is public, so there is nothing a code submission would hide.

```
codabench/
├── build_bundle.py            <task>/bundle_src + generated files -> build/<task>/
├── filter/bundle_src/         competition.yaml, pages/, scoring_program/{scoring.py, metadata.yaml}, starting_kit/README.md
├── regression/bundle_src/
└── build/<task>/              generated, git-ignored
    ├── bundle/                validate this
    ├── competition_bundle.zip upload this
    └── extra/baseline_submission.zip
```

`build_bundle.py` generates everything that depends on the data or the metric from the package:
- `reference_data/truth_<split>.csv` from `truth.truth_frame`;
- `starting_kit/<split>_sample_ids.csv`;
- `scoring_program/metrics.py`, the scoring functions of `smartpix_bench.<task>.score` copied by source, with
  the metric's reference JSON (the filter's pT spectrum, regression's ranking I68);
- `solution/` and `starting_kit/sample_submission.zip`, from a pipeline zip;
- the weak baseline: training class priors for the filter, training mean and std for regression.

After building, it scores every pipeline zip and the baseline with the bundle's own scoring program and fails if
any score differs from the pipeline's `metrics.json`.

## Build

On HiPerGator, after the reference run (`source slurm/env.sh && spb_activate`; 16 GB):

```bash
python codabench/build_bundle.py --task filter --results $SPB_RESULTS/filter --solution QM2_s4 --data $SPB_CACHE_DIR/filter
python codabench/build_bundle.py --task regression --results $SPB_RESULTS/regression --data $SPB_DATA_DIR
python <benchmark-builder>/scripts/validate_codabench_bundle.py codabench/build/<task>/bundle \
    --submission codabench/build/<task>/bundle/starting_kit/sample_submission.zip
```

`tests/test_codabench.py` builds both bundles from the fixtures and checks scores, malformed uploads and the
baseline.

## Design

| | filter | regression |
|---|---|---|
| Ranking key (index 0) | `background_rejection_at_eff`, desc; bounded in [0, 1] | `3s_rank_i68`, asc; bounded below by 0 |
| Other columns | argmax rejection, efficiency, data reduction | 2s rank, I68 of x, y, β, α on both sets |
| Solution / sample submission | QM2 seed with the median validation primary metric (asic track) | the ranking reference seed (Full_MLP c4, from `i68_reference.json`) |
| Weak baseline | constant training class priors | training mean, σ = training std |

- **One phase.** The test sets are public, so a held-out final phase would test nothing new.
- **One leaderboard for both tracks.** The `asic` column is 1 when `descriptor.json` declares the asic track.
  An asic descriptor that fails the checks is rejected with its violations named, as the packager does.
- **Strict uploads.** A missing scored id fails the upload, where the standalone scorer counts it as read out
  (filter) or not predicted (regression). This matches the packager, which never writes such a zip.
- **Unranked regression rows.** A regression target left empty is legal, and `rank_i68` is then null unless x, y
  and β are complete.
- **Image.** `docker_image: codalab/codalab-legacy:py312`; the programs need only numpy and pandas.

## Validation

**Filter, real data.** Built on HiPerGator on 2026-11-10 from the 2:1:1 reference run (`runs/filter_d74c027`).
The bundle is 194 MB, the zip 70 MB.
- All 20 pipeline zips score identically to their `metrics.json`.
- Discrimination: QM2_s4 rejects 0.1976 against the baseline's 0.0670 (= 1 − 0.933, every p0 tied). M3 seeds
  reach 0.333–0.342.
- Validator tiers 1–3: pass except the missing logo (below). The validator warns that `SUBMISSION_FILES`
  omits `descriptor.json`; that file is optional.
- Malformed uploads fail and name the problem:
  - missing sample: "no prediction for 1 of 1599888 test clusters";
  - NaN p0: "probabilities must be finite and in [0, 1]";
  - missing file: "'predictions_test.csv' not found".
- A wrapping folder is flagged by the validator, and the scoring program still scores it.
- Tier 4: `codalab/codalab-legacy:py312` (Python 3.12.13, numpy 1.26.4, pandas 2.2.3), run with Apptainer on
  HiPerGator. The sample and baseline scores are identical to the host run. There is no local Docker engine.

**Regression.** Fixture build and checks only (`tests/test_codabench.py`). The real build needs
`i68_reference.json`, which the reference run writes. It is rebuilt and recorded here with the reference results.

## Before upload

- [ ] `logo.png` in each `bundle_src/`: none exists, and none is fabricated here.
- [ ] Replace `pages/terms_and_conditions.md`, which is a marked placeholder, with the organizers' terms.
- [ ] Set the phase `start` (placeholder 1-1-2027), and optionally `contact_email`.
- [ ] Confirm the titles, `max_submissions_per_day` (5) and `execution_time_limit` (600 s filter, 1200 s regression).
- [ ] Replace "the benchmark repository" in `pages/` with its URL once it is public.
