#!/bin/bash
# Filter feature cache (train/test splits). Resources: SPB_SB_FILTER_CACHE.
#SBATCH --job-name=spb-filter-cache
#SBATCH --output=%x-%j.out
set -euo pipefail
# Slurm runs a spooled copy, so env.sh is found through the exported repo path.
source "$SPB_REPO/slurm/env.sh" && spb_activate
python -m smartpix_bench.filter.cache
