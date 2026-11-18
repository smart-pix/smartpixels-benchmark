# Running on UF HiPerGator

Both benchmarks' full reference runs (training included) on HiPerGator's Slurm CPU partition. The scripts in
`slurm/` are generic; HiPerGator's facts live in `slurm/profiles/hipergator.sh` and this benchmark's stack hooks
in `slurm/stack.sh`.

## Where each setting came from

| Setting | Value | Source |
|---|---|---|
| Partition | `hpg-default`: 233 AMD EPYC Rome nodes, 128 cores, ~1 TB, 31-day wall limit | `sinfo -p hpg-default` |
| Account | your investment group, `-A <group>` on the command line (required) | UFRC docs, https://help.rc.ufl.edu |
| Filesystem | `/blue/<group>/$USER/smartpixels` (project storage; home is too small for caches) | UFRC docs |
| Python stack | `uv` user install (`~/.local/bin/uv`), env from `uv.lock` with the `tf` extra | this benchmark |
| GPUs | not used: TF 2.15 ships no Blackwell kernels and JIT-compiles on B200 / RTX PRO 6000 (a 3-epoch probe fit took 77 s / 53 s vs 3 s on CPU); on L4 3.8 s vs 1.5 s. These ~2k-parameter models train faster on CPU | probe jobs 50458655–57 |

## One-time setup (login node)

```bash
git clone <repo> && cd smartpixels-benchmark
export SPB_MACHINE=hipergator SPB_GROUP=<group>          # SPB_ROOT defaults to /blue/$SPB_GROUP/$USER/smartpixels
bash slurm/setup.sh envs    # uv sync --frozen --extra tf into $SPB_ROOT/venv, then an import check
bash slurm/setup.sh data    # Zenodo 10783560 + 18472791 (~37 GB; about 2 h at the 5.5 MB/s measured), md5-checked
```

## Tests before submitting

```bash
source slurm/env.sh && spb_activate
python -m pytest tests -q --ignore tests/test_filter_models.py --ignore tests/test_regression_models.py   # login node
```

The TensorFlow tests train tiny models on the fixtures; run them on a compute node:
`srun -A <group> -p hpg-default -c 4 --mem=16G -t 30 python -m pytest tests -q` (114 passed, about 1 min).

## Submitting

```bash
bash slurm/submit.sh -A <group> filter
bash slurm/submit.sh -A <group> regression [--write-reference]
```

`--write-reference` regenerates the regression ranking reference (`src/smartpix_bench/regression/i68_reference.json`,
from the run's Full_MLP c4 seeds) and rescores; use it only when rebuilding the shipped reference. Array tasks run
model × seed (`i / SPB_SEEDS`, `i % SPB_SEEDS`, 5 seeds); `SPB_MAX_*` in the profile cap how many run at once.

| Job | Array | Resources (per task) | Measured wall time |
|---|---|---|---|
| filter cache | — | 2 cores, 16 GB | 5.5 min, 7.9 GB |
| filter train | 5 models × 5 seeds | 4 cores, 8 GB | M1 1–1.5 min, M2 2–3.5, QM2 2–4, M3 10–15 (1:1:1 balance); QM2_w4a8 2–5.5 incl. predict |
| filter predict + score | 25, then 1 | 4 cores, 24 GB | 12 min for 20 predictions serially; rescoring all 25 9 min |
| regression sq | 7 × 5, ≤ 12 at once | 4 cores, 16 GB | 1000 epochs: Conv1D/MLP ~5 h (18–19 s/epoch), Conv2D ~10.6 h (38 s) |
| regression c1 (20 frames) | 7 × 5, ≤ 3 | 4 cores, 48 GB | Conv1D 5 s/epoch |
| regression c2 | 7 × 5, ≤ 3 | 4 cores, 16 GB | Conv1D 3 s/epoch |
| regression c3 + c4 | 7 × 5, ≤ 6, each after its own sq | 4 cores, 16 GB | MLP 2 s/epoch each (8 cores) |
| regression predict | 7 × 5, ≤ 6 | 4 cores, 40 GB | 18–61 min per model-seed (c1–c4, both test sets) |
| regression score | — | 4 cores, 48 GB | 5.1–6.2 h (144 prediction files per test set) |

A full regression run is about 1,300 core-hours; at the default throttles (≤ 96 cores) it takes about 34 h.

## Results layout

```
$SPB_ROOT/
├── cache/filter/{train,test}.npz
├── weights/<task>/<model>[_<config>]_s<seed>.{weights.h5, scaler.json | meta.json, history.json, val.json}
├── results/<task>/<split>/{truth.csv, predictions_<name>.csv, descriptor_<name>.json, <name>/METRICS.md, metrics.json}
├── results/<task>/{LEADERBOARD.md, codabench/<name>_submission.zip}
└── logs/
```

## Scoring your own model

Write `results/<task>/<split>/predictions_<name>.csv` (columns in `data/SCHEMA.md`, every test cluster) and,
for the asic track, `descriptor_<name>.json`, then `bash scripts/score_all.sh <task>` inside an activated
environment. It rescores everything, rebuilds `LEADERBOARD.md` and writes your Codabench zip.

## Interactive use

`srun -A <group> -p hpg-default -c 4 --mem=24G -t 60 --pty bash`, then `source slurm/env.sh && spb_activate`.

## Troubleshooting

- `stack.sh` sets `OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS` to the job's cores and
  `TF_NUM_INTEROP_THREADS=1`: TensorFlow otherwise sizes its pools to the node's 128 cores.
- `Unable to register cuFFT/cuBLAS factory` lines on CPU nodes come from the CUDA-enabled TensorFlow wheel and
  are harmless.
- A training that dies leaves its downstream predict/score jobs running (`afterany`); the predict job logs
  `MISSING <checkpoint>` and the leaderboard simply lacks that seed. Resubmit the failed array index with
  `sbatch --array=<i> ...` from `slurm/submit.sh`'s command lines.
- uv's cache defaults to home; the profile moves it to `$SPB_ROOT/.cache/uv`.
