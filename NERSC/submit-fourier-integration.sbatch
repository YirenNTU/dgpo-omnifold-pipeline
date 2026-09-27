#!/bin/bash
# User-submitted Fourier integration arm; all backbone parameters remain trainable.
#SBATCH --job-name=fourier-integration
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5019_g
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.3
#SBATCH --output=fourier_integration_%j.out
#SBATCH --error=fourier_integration_%j.out
#SBATCH --export=ALL

set -euo pipefail
# Slurm spools this file: BASH_SOURCE does NOT point to the repository at runtime.
REPO_ROOT="${DGPO_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the ml_pipeline repository}}"
cd "$REPO_ROOT"
[[ -f scripts/train_fourier_integration.py ]] || { echo 'Wrong repository directory' >&2; exit 1; }
[[ "${SLURM_JOB_NUM_NODES:?}" == 4 ]] || { echo 'This config requires 4 nodes / 16 GPUs' >&2; exit 1; }

ARM="${1:?Pass a Fourier arm (none, input, input_slow, input_fast, input_mlp, output, readout_k4, readout_multiscale)}"
CHECKPOINT_ARGS=()
# Omit the second argument to use the shared YAML checkpoint.
if [[ -n "${2:-}" ]]; then CHECKPOINT_ARGS=(--checkpoint "$2"); fi
OUTPUT_ROOT="${3:-/pscratch/sd/y/yiren/Ztautau/fourier_integration}"
case "$ARM" in none|input|input_slow|input_fast|input_mlp|output|readout_k4|readout_multiscale) ;; *) echo "Invalid arm: $ARM" >&2; exit 1 ;; esac
RUN_ROOT="$REPO_ROOT/launch_logs/$SLURM_JOB_ID"
mkdir -p "$RUN_ROOT"

export PYTHONPATH="$REPO_ROOT/evenet_dgpo:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export WANDB_MODE=online
# Let the trainer generate a fresh ID; never reuse the submitting shell's run ID.
unset WANDB_RUN_ID WANDB_RESUME
export TORCH_NCCL_TIMEOUT=180
export DGPO_RAY_WAIT_S=600
export RAY_TMPDIR="/tmp/dgpo_${SLURM_JOB_ID}"
# Ray must mask one GPU per Train worker, not expose all four to each process.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
# Use inherited WANDB_API_KEY or the existing W&B/netrc login. No keys in this file.
shifter python3 scripts/train_fourier_integration.py --arm "$ARM" "${CHECKPOINT_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT" --check-only > "$RUN_ROOT/resolved-config.yaml"
# Loads the matched arms on CPU and requires identical initial velocity outputs.
# No optimizer steps, Ray cluster, or W&B training run are started by preflight.
shifter python3 scripts/train_fourier_integration.py --arm "$ARM" "${CHECKPOINT_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT" --preflight > "$RUN_ROOT/preflight.log" 2>&1

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 --gpus-per-task=4 \
    --cpus-per-task=128 -w "$head_node" hostname --ip-address | awk '{print $1}')
[[ -n "$head_node_ip" ]] || { echo 'Could not resolve Ray head IP' >&2; exit 1; }
export RAY_ADDRESS="$head_node_ip:6379"

HEAD_PID= WORKER_PID=
cleanup() {
    # These are only the two srun steps created by THIS job, never another cluster.
    for pid in "$WORKER_PID" "$HEAD_PID"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

srun --overlap --nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=128 \
    -w "$head_node" shifter ray start --head --node-ip-address="$head_node_ip" \
    --port=6379 --num-cpus=128 --num-gpus=4 --temp-dir="$RAY_TMPDIR" \
    --include-dashboard=false --block > "$RUN_ROOT/ray-head.log" 2>&1 &
HEAD_PID=$!

# Wait for actual readiness instead of assuming a fixed sleep is enough.
ready=0
for ((attempt=0; attempt<120; attempt++)); do
    if grep -q 'Ray runtime started' "$RUN_ROOT/ray-head.log"; then ready=1; break; fi
    if ! kill -0 "$HEAD_PID" 2>/dev/null; then break; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then
    tail -60 "$RUN_ROOT/ray-head.log" >&2
    echo 'Ray head failed to become ready' >&2
    exit 1
fi

srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 \
    --gpus-per-task=4 --cpus-per-task=128 --exclude="$head_node" \
    shifter ray start --address="$RAY_ADDRESS" --num-cpus=128 --num-gpus=4 \
    --temp-dir="$RAY_TMPDIR" --block > "$RUN_ROOT/ray-workers.log" 2>&1 &
WORKER_PID=$!

# Fail closed on an incomplete cluster (the trainer's legacy timeout only warns).
shifter python3 - <<'PY'
import os
import time
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
deadline = time.monotonic() + float(os.environ["DGPO_RAY_WAIT_S"])
while True:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpu_nodes = [node for node in nodes if node.get("Resources", {}).get("GPU", 0) >= 4]
    if len(gpu_nodes) == 4 and ray.cluster_resources().get("GPU", 0) >= 16:
        print("Ray ready: 4 GPU nodes / 16 GPUs", flush=True)
        break
    if time.monotonic() >= deadline:
        raise SystemExit("Ray startup timed out: refusing to train without 4 GPU nodes / 16 GPUs")
    print(f"Waiting for Ray: {len(gpu_nodes)}/4 GPU nodes", flush=True)
    time.sleep(5)
ray.shutdown()
PY

echo "Ray: $RAY_ADDRESS; scheduler logs: $RUN_ROOT"
shifter python3 scripts/train_fourier_integration.py --arm "$ARM" "${CHECKPOINT_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT"
