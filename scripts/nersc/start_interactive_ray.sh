#!/bin/bash
# start_interactive_ray.sh
#
# Start a Ray cluster across all nodes in the current SLURM allocation,
# interactively. Designed to be SOURCED so it exports head_node /
# head_node_ip / port into your shell:
#
#     source NERSC/start_interactive_ray.sh
#
# Then verify with:
#
#     shifter ray status --address=$head_node_ip:$port
#
# Requires: $SLURM_JOB_ID set (i.e. you are inside an salloc).
#           start-head.sh and start-worker.sh in the same dir.

set -u

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: not inside an salloc. Run salloc first." >&2
    return 1 2>/dev/null || exit 1
fi

CALLER_DIR="$PWD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Leave this variable absent by default so Ray assigns CUDA_VISIBLE_DEVICES to
# each one-GPU Train worker. Ray checks presence/truthiness, so the string "0"
# must not be exported as a false value. Set it explicitly to 1 only for the
# legacy Shifter/NCCL workaround.
if [ "${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-}" = "0" ]; then
    unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
fi

# ---------------------------------------------------------------------------
# 1. Preflight: prepare a job-specific Ray temp dir on every node. Avoid long
#    paths under $SCRATCH: Ray's plasma_store socket path must stay under the
#    Linux AF_UNIX limit (~107 bytes) including ray/session_.../sockets/... .
#    Use short node-local /tmp (default); override with RAY_TMPDIR if needed.
# ---------------------------------------------------------------------------
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r${UID}_${SLURM_JOB_ID}}"

echo "[1/4] Preparing Ray temp dir on $SLURM_JOB_NUM_NODES nodes: $RAY_TMPDIR ..."
prepared_nodes=$(srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --overlap \
                 bash -c 'rm -rf "$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR" && chmod 700 "$RAY_TMPDIR" && hostname' \
                 2>/dev/null | sort -u)

if [ -z "$prepared_nodes" ]; then
    echo "ERROR: failed to prepare $RAY_TMPDIR on allocated nodes." >&2
    return 1 2>/dev/null || exit 1
fi

head_node=$(echo "$prepared_nodes" | head -1)
echo "  Prepared nodes   : $(echo $prepared_nodes | tr '\n' ' ')"
echo "  Selected head    : $head_node"

# Always take the FIRST IP — Perlmutter compute nodes are multi-NIC and
# secondary IPs are not routable between nodes.
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" --overlap \
               hostname --ip-address | awk '{print $1}')
port=6379

if [ -z "$head_node_ip" ]; then
    echo "ERROR: could not resolve IP for $head_node" >&2
    return 1 2>/dev/null || exit 1
fi
echo "  Head address     : $head_node_ip:$port"

export head_node head_node_ip port
export RAY_ADDRESS="$head_node_ip:$port"

# ---------------------------------------------------------------------------
# 2. Launch head
# ---------------------------------------------------------------------------
# IMPORTANT: interactive shells are usually already inside an ``srun --pty`` step
# (see NERSC/salloc_4node_16gpu_ray.sh). Nested srun for Ray MUST use --overlap,
# otherwise SLURM cannot place the head (GPU/CPU already held by the pty) and
# HEAD_LOG stays empty / "Ray runtime started" never appears.
HEAD_LOG="/tmp/ray_head_${USER}_${SLURM_JOB_ID}.log"
: > "$HEAD_LOG"
echo "[2/4] Launching Ray head on $head_node (log: $HEAD_LOG) ..."
srun --overlap --nodes=1 --ntasks=1 --gpus=0 --gpus-per-task=4 --cpus-per-task=128 \
     -w "$head_node" shifter ./start-head.sh "$head_node_ip" \
     > "$HEAD_LOG" 2>&1 &
HEAD_PID=$!

# Poll: Ray head can take >25s on a cold Shifter pull / busy node.
ready=0
for _i in $(seq 1 24); do
    if grep -q "Ray runtime started" "$HEAD_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$HEAD_PID" 2>/dev/null; then
        break
    fi
    sleep 5
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: head did not start. Tail of $HEAD_LOG :" >&2
    if [ -s "$HEAD_LOG" ]; then
        tail -40 "$HEAD_LOG" >&2
    else
        echo "(log empty — nested srun likely blocked without --overlap, or shifter failed)" >&2
        echo "Also check: echo \$SLURM_JOB_ID; squeue -j \$SLURM_JOB_ID; shifter which ray" >&2
    fi
    kill -9 "$HEAD_PID" 2>/dev/null
    wait "$HEAD_PID" 2>/dev/null
    return 1 2>/dev/null || exit 1
fi
echo "  Head OK."

# ---------------------------------------------------------------------------
# 3. Launch workers
# ---------------------------------------------------------------------------
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ "$worker_num" -gt 0 ]; then
    echo "[3/4] Launching $worker_num worker node(s) ..."
    srun --overlap -n "$worker_num" --nodes="$worker_num" --ntasks-per-node=1 \
         --gpus-per-task=4 --cpus-per-task=128 --gpus=0 \
         --exclude "$head_node" \
         shifter ./start-worker.sh "$head_node_ip:$port" &
    sleep 15
else
    echo "[3/4] Single-node allocation: skipping worker launch."
fi

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo "[4/4] Cluster status:"
shifter ray status --address="$head_node_ip:$port" || {
    echo "WARNING: ray status failed. Cluster may still be initialising — retry in a few seconds." >&2
}

echo
echo "Variables exported in your shell:"
echo "  head_node    = $head_node"
echo "  head_node_ip = $head_node_ip"
echo "  port         = $port"
echo "  RAY_ADDRESS  = $RAY_ADDRESS"
echo
echo "To check status later:"
echo "  shifter ray status --address=\$head_node_ip:\$port"

cd "$CALLER_DIR" || true
