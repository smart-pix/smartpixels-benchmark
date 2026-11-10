# UF HiPerGator (Slurm). Verified 2026-11-10 with this benchmark (jobs in docs/HIPERGATOR.md).
# Sources: UFRC docs (https://help.rc.ufl.edu) and the jobs run there; the account is the user's investment group.

SPB_MACHINE=hipergator
: "${SPB_ROOT:=/blue/${SPB_GROUP:-$(id -gn)}/$USER/smartpixels}"   # /blue: project storage, not purged
SPB_REQUIRE_ACCOUNT=1
P="--partition=hpg-default"    # CPU nodes: 128 cores, ~1 TB; 31-day wall limit
SPB_SB_FILTER_CACHE="$P --cpus-per-task=2 --mem=16G --time=01:00:00"
SPB_SB_FILTER_TRAIN="$P --cpus-per-task=4 --mem=8G --time=03:00:00"
SPB_SB_FILTER_SCORE="$P --cpus-per-task=4 --mem=24G --time=02:00:00"
SPB_SB_REG_TRAIN="$P --cpus-per-task=4 --mem=16G --time=16:00:00"
SPB_SB_REG_TRAIN_C1="$P --cpus-per-task=4 --mem=48G --time=16:00:00"
SPB_SB_REG_PREDICT="$P --cpus-per-task=4 --mem=40G --time=06:00:00"
SPB_SB_REG_SCORE="$P --cpus-per-task=4 --mem=48G --time=08:00:00"
# Concurrent array tasks; the defaults keep a full regression run near 96 cores of a group allocation.
: "${SPB_MAX_SQ:=12}" "${SPB_MAX_C1:=3}" "${SPB_MAX_C2:=3}" "${SPB_MAX_C34:=6}" "${SPB_MAX_PREDICT:=6}"
# Home quota is small: keep uv's cache and temporaries on /blue and local disk.
: "${UV_CACHE_DIR:=$SPB_ROOT/.cache/uv}"
export UV_CACHE_DIR TMPDIR=${TMPDIR:-/var/tmp}
spb_load_stack() { :; }        # uv is a user install; no site module needed
