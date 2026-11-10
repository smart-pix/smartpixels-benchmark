#!/bin/bash
# Array over models x seeds: sbatch --array=0-<n-1> train.sbatch <task> [<config> ...].
# Regression configs run in order inside one task, so c3/c4 can follow their own sq with aftercorr.
# Resources: SPB_SB_FILTER_TRAIN / SPB_SB_REG_TRAIN[_C1].
#SBATCH --job-name=spb-train
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SPB_REPO/slurm/env.sh" && spb_activate
task=$1; shift
read -r model seed < <(spb_task_index "$task")
if [ "$task" = filter ]; then
    python -m smartpix_bench.filter.train --model "$model" --seed "$seed"
fi
for config in "$@"; do
    echo "== $model $config s$seed $(date -Is)"
    python -m smartpix_bench.regression.train --model "$model" --config "$config" --seed "$seed"
done
