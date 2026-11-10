# Smart Pixels benchmarks

Two ML benchmarks for neural networks that run inside a silicon-pixel readout chip, built on public PixelAV
simulation of a 100 µm sensor with 50 × 12.5 µm² pixels:

| Benchmark | Task | Card |
|---|---|---|
| **filter** | classify a 13 × 21 pixel cluster as high- or low-pT, to reject low-pT clusters on the sensor (arXiv:2310.02474) | [benchmarks/filter](benchmarks/filter/README.md) |
| **regression** | regress hit position and angles with uncertainties from a 16 × 16 pixel cluster (arXiv:2602.15946) | [benchmarks/regression](benchmarks/regression/README.md) |

Each has an **asic** track (the chip's input precision, parameter and bit budget, checked from a submitted
descriptor) and an **open** track, reference models trained from scratch with 5 seeds, a leaderboard and
Codabench packaging. Each card follows the MLCommons Science Benchmarks Ontology (arXiv:2511.05614) and has its own
`rubric.yaml` and `SCORE_REPORT.md`.

## Quick start

```bash
uv sync --extra tf                          # Python 3.11; TensorFlow 2.15.1 + QKeras 0.9.0 via the `tf` extra
uv run python scripts/fetch_data.py         # both Zenodo records -> $SPB_DATA_DIR (default data/zenodo), md5-checked
uv run python -m pytest tests -q
```

Then each card's quick start, or on a Slurm cluster (`slurm/`, HiPerGator profile and notes in
[docs/HIPERGATOR.md](docs/HIPERGATOR.md)):

```bash
SPB_MACHINE=hipergator bash slurm/setup.sh
SPB_MACHINE=hipergator bash slurm/submit.sh -A <group> filter|regression
```

Pipeline, identical for both tasks: `fetch_data → [filter cache] → train → predict → scripts/score_all.sh <task>`
(truth, metrics, `LEADERBOARD.md`, Codabench zips). A submission is a predictions CSV per test split (plus a
descriptor for the asic track) dropped into `$SPB_RESULTS/<task>/<split>/` before `score_all.sh`
([SUBMISSION.md](SUBMISSION.md)).

## Layout

```
src/smartpix_bench/{common,filter,regression}/   data, features, models, train, predict, truth, score, report, submission
scripts/                fetch_data.py, score_all.sh, validation/ (checks against the papers' own inputs)
slurm/                  env.sh, stack.sh, setup.sh, submit.sh, profiles/, jobs/
data/SCHEMA.md          every field, ground truth, forbidden inputs, prediction and descriptor formats
docs/VALIDATION.md      how the reference solutions were checked, and every departure from paper and upstream
benchmarks/<task>/      benchmark card, rubric.yaml, SCORE_REPORT.md
codabench/              Codabench bundle builder and per-task bundle sources
weights/MANIFEST.json   reference checkpoints with sha256 (scripts/fetch_weights.py)
tests/                  unit and pipeline tests on real fixture clusters
SUBMISSION.md           how to submit a model and what to report with it
```

Some validation uses CMS-internal data (the filter paper's simulation and preprocessed inputs); it is available
to CMS members on request from the authors, and only aggregate numbers derived from it appear here.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Datasets: CC-BY-4.0 (Zenodo 10783560, 18472791). To cite, see
[CITATION.cff](CITATION.cff).
