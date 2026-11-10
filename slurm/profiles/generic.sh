# Machine profile TEMPLATE: copy to profiles/<machine>.sh and fill every value (docs/HIPERGATOR.md shows
# where HiPerGator's came from). Job sizes below are this benchmark's measured needs on HiPerGator CPUs.

SPB_MACHINE="<machine>"
: "${SPB_ROOT:=<project filesystem>/smartpixels}"
SPB_REQUIRE_ACCOUNT=1
P="--partition=<cpu partition>"
SPB_SB_FILTER_CACHE="$P --cpus-per-task=2 --mem=16G --time=01:00:00"
SPB_SB_FILTER_TRAIN="$P --cpus-per-task=4 --mem=8G --time=03:00:00"
SPB_SB_FILTER_SCORE="$P --cpus-per-task=4 --mem=24G --time=02:00:00"
SPB_SB_REG_TRAIN="$P --cpus-per-task=4 --mem=16G --time=16:00:00"
SPB_SB_REG_TRAIN_C1="$P --cpus-per-task=4 --mem=48G --time=16:00:00"
SPB_SB_REG_PREDICT="$P --cpus-per-task=4 --mem=40G --time=06:00:00"
SPB_SB_REG_SCORE="$P --cpus-per-task=4 --mem=48G --time=08:00:00"
: "${SPB_MAX_SQ:=12}" "${SPB_MAX_C1:=3}" "${SPB_MAX_C2:=3}" "${SPB_MAX_C34:=6}" "${SPB_MAX_PREDICT:=6}"
spb_load_stack() { :; }   # e.g. module load <python>, if uv is not on PATH
