# Source before any benchmark step, on a login or compute node:
#     SPB_MACHINE=<machine> source slurm/env.sh
# Cluster facts come from profiles/$SPB_MACHINE.sh, this benchmark's stack hooks from stack.sh.
# Override any path by exporting it before sourcing.

SPB_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SPB_MACHINE:=generic}"
if [ ! -f "$SPB_HPC_DIR/profiles/$SPB_MACHINE.sh" ]; then
    echo "env.sh: no profile $SPB_HPC_DIR/profiles/$SPB_MACHINE.sh" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$SPB_HPC_DIR/profiles/$SPB_MACHINE.sh"
# shellcheck disable=SC1091
source "$SPB_HPC_DIR/stack.sh"

: "${SPB_DATA_DIR:=$SPB_ROOT/data/zenodo}"     # Zenodo records (+ REVISION.json each)
: "${SPB_CACHE_DIR:=$SPB_ROOT/cache}"          # filter feature cache
: "${SPB_WEIGHTS_DIR:=$SPB_ROOT/weights}"      # <task>/<model>[_<config>]_s<seed>.*
: "${SPB_RESULTS:=$SPB_ROOT/results}"          # <task>/<split>/predictions_*.csv, metrics, LEADERBOARD.md, codabench/
: "${SPB_SEEDS:=5}"
: "${UV_PROJECT_ENVIRONMENT:=$SPB_ROOT/venv}"  # uv's project venv, kept off the repo checkout
export SPB_MACHINE SPB_ROOT SPB_DATA_DIR SPB_CACHE_DIR SPB_WEIGHTS_DIR SPB_RESULTS SPB_SEEDS UV_PROJECT_ENVIRONMENT

: "${SPB_REPO:=$(cd "$SPB_HPC_DIR/.." && pwd)}"
export SPB_REPO PYTHONUNBUFFERED=1
spb_stack_env

# Reference model lists; task i of an array over models x seeds is model i / SPB_SEEDS, seed i % SPB_SEEDS.
SPB_FILTER_MODELS=(M1 M2 M3 QM2)
SPB_REGRESSION_MODELS=(Full_Conv1D Full_Conv2D Full_MLP Max_Conv2D Slim_Conv1D Slim_Conv2D Slim_MLP)

spb_activate() {
    spb_load_stack
    # shellcheck disable=SC1091
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
}

spb_models() {  # spb_models <task>: the model list of a task, one per line
    local -n list="SPB_${1^^}_MODELS"
    printf '%s\n' "${list[@]}"
}

spb_task_index() {  # spb_task_index <task>: "model seed" for $SLURM_ARRAY_TASK_ID
    local models; mapfile -t models < <(spb_models "$1")
    local i=${SLURM_ARRAY_TASK_ID:?not an array task}
    echo "${models[i / SPB_SEEDS]} $((i % SPB_SEEDS))"
}
