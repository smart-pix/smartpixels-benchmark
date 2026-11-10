#!/bin/bash
# score_all.sh <task>: truth, metrics, LEADERBOARD.md, Codabench zips. With SPB_WRITE_REFERENCE=1 (regression),
# first regenerates the package's i68_reference.json from this run's Full_MLP_c4 seeds, then rescores.
# Resources: SPB_SB_*_SCORE.
#SBATCH --job-name=spb-score
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SPB_REPO/slurm/env.sh" && spb_activate
task=$1
cd "$SPB_REPO"
bash scripts/score_all.sh "$task"
if [ "$task" = regression ] && [ "${SPB_WRITE_REFERENCE:-0}" = 1 ]; then
    ref=src/smartpix_bench/regression/i68_reference.json
    python -m smartpix_bench.regression.score --write-reference "$ref" \
        --results "$SPB_RESULTS/regression" --weights "$SPB_WEIGHTS_DIR/regression"
    cat "$ref"
    bash scripts/score_all.sh regression
fi
