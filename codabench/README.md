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

**Regression, real data.** Built on HiPerGator on 2026-11-14 from the reference run (`runs/regression_0731db5`,
code afd4a9a, job 51318708) against the shipped `i68_reference.json` (Full_MLP c4 seed 2). The bundle is 963 MB,
the zip 511 MB, and the sample submission 207 MB.
- All 148 pipeline zips (140 model-seeds, 8 baselines) score identically to their `metrics.json`.
- Discrimination on 3s: the solution, Full_MLP_c4_s2, scores `3s_rank_i68` = 1.0. The constant baseline (training
  mean, σ = training std) scores 8.06, and reference models span 0.47–1.61 outside the failed Max_Conv2D seeds
  (1.98–8.06; docs/VALIDATION.md R15).
- Validator tiers 1–3 (job 51902398): pass except the missing logo. As for the filter, the validator warns that
  `SUBMISSION_FILES` omits the optional `descriptor.json`.
- Malformed uploads fail and name the problem:
  - missing sample: "predictions_3s.csv has no prediction for 1 of 1377888 clusters";
  - infinite x: "predictions_3s.csv: infinite predictions";
  - missing file: "'predictions_2s.csv' not found in the submission".
- A wrapping folder is flagged by the validator, and the scoring program still scores it.
- Tier 4: `codalab/codalab-legacy:py312` (Python 3.12.13, numpy 1.26.4, pandas 2.2.3) run with Apptainer on
  HiPerGator gives exit 0 and the same `scores.json` as the host for the sample and the baseline.

## Before upload

- [ ] `logo.png` in each `bundle_src/`: none exists, and none is fabricated here.
- [ ] Replace `pages/terms_and_conditions.md`, which is a marked placeholder, with the organizers' terms.
- [ ] Set the phase `start` (placeholder 1-1-2027), and optionally `contact_email`.
- [ ] Confirm the titles, `max_submissions_per_day` (5) and `execution_time_limit` (600 s filter, 1200 s regression).
- [ ] Regression size: the bundle zip is 511 MB (truth for 2.8 M clusters plus the solution), and a submission is
  about 207 MB. Check both against the Codabench instance's upload limits, or host the reference data separately.
- [ ] Replace "the benchmark repository" in `pages/` with its URL once it is public.
