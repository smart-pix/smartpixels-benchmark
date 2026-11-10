#!/bin/bash
# Array over models x seeds: predictions for every test split; regression also c1-c4, and task 0 the baselines.
# A missing checkpoint is logged and skipped so one failed training does not block scoring. Resources: SPB_SB_*_PREDICT.
#SBATCH --job-name=spb-predict
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
source "$SPB_REPO/slurm/env.sh" && spb_activate
task=$1
read -r model seed < <(spb_task_index "$task")
W=$SPB_WEIGHTS_DIR/$task R=$SPB_RESULTS/$task
if [ "$task" = filter ]; then
    stem=$W/${model}_s$seed
    if [ -f "$stem.weights.h5" ]; then
        python -m smartpix_bench.filter.predict --model "$model" --weights "$stem" --out "$R/test"
    else echo "MISSING $stem"; fi
    exit 0
fi
for config in c1 c2 c3 c4; do
    stem=$W/${model}_${config}_s$seed
    [ -f "$stem.weights.h5" ] || { echo "MISSING $stem"; continue; }
    for split in 3s 2s; do python -m smartpix_bench.regression.predict --weights "$stem" --split "$split" --out "$R/$split"; done
done
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    for split in 3s 2s; do python -m smartpix_bench.regression.baselines --split "$split" --out "$R/$split"; done
fi
