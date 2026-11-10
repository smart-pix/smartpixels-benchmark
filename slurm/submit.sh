#!/bin/bash
# Submit one task's full reference run as a Slurm dependency graph:
#   filter:     cache -> train (models x seeds) -> predict (models x seeds) -> score
#   regression: sq, c1, c2 (models x seeds, independent); c3+c4 per task after its own sq (aftercorr);
#               predict after all training -> score
# Usage, from the repo root on a login node, after setup.sh and the tests:
#     SPB_MACHINE=<machine> bash slurm/submit.sh -A <account> filter|regression [--write-reference] [sbatch args...]
# Resources and array throttles come from the profile; #SBATCH lines cannot expand variables.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

TASK="" WRITE_REF=0 SB_ARGS=()
for a in "$@"; do
    case "$a" in
        filter|regression) TASK=$a ;;
        --write-reference) WRITE_REF=1 ;;
        *) SB_ARGS+=("$a") ;;
    esac
done
[ -n "$TASK" ] || { echo "usage: $0 -A <account> filter|regression [--write-reference] [sbatch args]" >&2; exit 2; }
if [ "$SPB_REQUIRE_ACCOUNT" = 1 ] && ! printf '%s\n' "${SB_ARGS[@]:-}" | grep -qE '^(-A|--account)'; then
    echo "error: pass your allocation, e.g. bash slurm/submit.sh -A <account> $TASK" >&2; exit 2
fi
[ -e "$SPB_DATA_DIR" ] || { echo "error: missing $SPB_DATA_DIR -- run: bash slurm/setup.sh" >&2; exit 1; }
mapfile -t MODELS < <(spb_models "$TASK")
N=$(( ${#MODELS[@]} * SPB_SEEDS - 1 ))
mkdir -p "$SPB_ROOT/logs"

# shellcheck disable=SC2086
sb() { local res="$1"; shift; sbatch --parsable --chdir="$SPB_ROOT/logs" \
    --export=ALL,SPB_REPO="$PWD",SPB_MACHINE="$SPB_MACHINE",SPB_WRITE_REFERENCE="$WRITE_REF" $res "${SB_ARGS[@]}" "$@"; }

if [ "$TASK" = filter ]; then
    cache=$(sb "$SPB_SB_FILTER_CACHE" slurm/jobs/filter_cache.sbatch)
    train=$(sb "$SPB_SB_FILTER_TRAIN" --array=0-$N --dependency=afterok:$cache slurm/jobs/train.sbatch filter)
    pred=$(sb "$SPB_SB_FILTER_SCORE" --array=0-$N --dependency=afterany:$train slurm/jobs/predict.sbatch filter)
    score=$(sb "$SPB_SB_FILTER_SCORE" --dependency=afterany:$pred slurm/jobs/score.sbatch filter)
    echo "cache=$cache train=$train predict=$pred score=$score"
else
    sq=$(sb "$SPB_SB_REG_TRAIN" --job-name=spb-sq --array=0-$N%$SPB_MAX_SQ slurm/jobs/train.sbatch regression sq)
    c1=$(sb "$SPB_SB_REG_TRAIN_C1" --job-name=spb-c1 --array=0-$N%$SPB_MAX_C1 slurm/jobs/train.sbatch regression c1)
    c2=$(sb "$SPB_SB_REG_TRAIN" --job-name=spb-c2 --array=0-$N%$SPB_MAX_C2 slurm/jobs/train.sbatch regression c2)
    c34=$(sb "$SPB_SB_REG_TRAIN" --job-name=spb-c3c4 --array=0-$N%$SPB_MAX_C34 --dependency=aftercorr:$sq \
        slurm/jobs/train.sbatch regression c3 c4)
    pred=$(sb "$SPB_SB_REG_PREDICT" --array=0-$N%$SPB_MAX_PREDICT --dependency=afterany:$sq:$c1:$c2:$c34 \
        slurm/jobs/predict.sbatch regression)
    score=$(sb "$SPB_SB_REG_SCORE" --dependency=afterany:$pred slurm/jobs/score.sbatch regression)
    echo "sq=$sq c1=$c1 c2=$c2 c3c4=$c34 predict=$pred score=$score"
fi
echo "logs -> $SPB_ROOT/logs   results -> $SPB_RESULTS/$TASK/LEADERBOARD.md   (watch: squeue --me)"
