#!/bin/bash
# Pipeline contract, last step: truth -> score every predictions_*.csv -> LEADERBOARD.md -> Codabench zips.
# Usage: score_all.sh [filter|regression]. Needs the package importable, $SPB_RESULTS, and
# $SPB_CACHE_DIR (filter) or $SPB_DATA_DIR (regression), else their defaults.
set -euo pipefail
task=${1:-filter}
res="$SPB_RESULTS/$task"
for split in $(python -c "from smartpix_bench.$task.truth import SPLITS; print(*SPLITS)"); do
    dir="$res/$split"
    mkdir -p "$dir"
    python -m smartpix_bench.$task.truth --split "$split" --out "$dir/truth.csv"
    shopt -s nullglob
    for pred in "$dir"/predictions_*.csv; do
        name=$(basename "$pred" .csv); name=${name#predictions_}
        args=()
        [ -f "$dir/descriptor_$name.json" ] && args=(--descriptor "$dir/descriptor_$name.json")
        python -m smartpix_bench.$task.score --pred "$pred" --truth "$dir/truth.csv" --out "$dir/$name" \
            --title "$name on $split" ${args[@]+"${args[@]}"} > /dev/null
        echo "scored $split/$name"
    done
done
python -m smartpix_bench.common.report --task $task --results "$res" > /dev/null
python -m smartpix_bench.common.submission --task $task --results "$res"
