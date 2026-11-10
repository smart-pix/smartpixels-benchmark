# This benchmark's software-stack hooks, sourced by env.sh after the machine profile.
# Fill them only with what this stack needs; record each in docs/<MACHINE>.md.

spb_stack_env() {
    # TensorFlow sizes its thread pools to the node's cores, not the job's allocation.
    local n=${SLURM_CPUS_PER_TASK:-1}
    export OMP_NUM_THREADS=$n TF_NUM_INTRAOP_THREADS=$n TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
}

spb_post_install() { :; }

spb_check_env() {
    "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import tensorflow as tf, qkeras, smartpix_bench; print('tf', tf.__version__, 'qkeras', qkeras.__version__, 'smartpix_bench ok')"
}
