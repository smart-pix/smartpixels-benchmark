#!/bin/bash
# One-time setup on a LOGIN node: the uv environment and the Zenodo data. Jobs never download.
#     SPB_MACHINE=<machine> bash slurm/setup.sh [all|envs|data]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$SPB_ROOT"

step_envs() {
    echo "== uv environment ($UV_PROJECT_ENVIRONMENT) on $SPB_MACHINE"
    spb_load_stack
    (cd "$SPB_REPO" && uv sync --frozen --extra tf)
    spb_post_install
    spb_check_env
}

step_data() {
    echo "== Zenodo records -> $SPB_DATA_DIR"
    "$UV_PROJECT_ENVIRONMENT/bin/python" "$SPB_REPO/scripts/fetch_data.py" --data-dir "$SPB_DATA_DIR"
}

case "${1:-all}" in
    envs) step_envs ;;
    data) step_data ;;
    all) step_envs; step_data ;;
    *) echo "usage: $0 [all|envs|data]"; exit 2 ;;
esac
echo "done. Next: the tests (docs/HIPERGATOR.md), then bash slurm/submit.sh -A <account> filter|regression"
