#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Choose a dashboard port (or randomize it safely)
PORT=8265
RAY_TEMP_DIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "$RAY_TEMP_DIR"
NODE_IP="${1:-}"

# Absence means Ray masks GPUs per worker. Never export the string "0": Ray
# treats a present non-empty value as enabling the opt-out.
if [ "${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-}" = "0" ]; then
    unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
fi

NODE_IP_ARGS=()
if [ -n "$NODE_IP" ]; then
    NODE_IP_ARGS=(--node-ip-address="$NODE_IP")
fi

echo "Ray will start on $(hostname) at port $PORT" > "$SCRATCH"/ray_dashboard_info.txt
# Launch the head node
ray start --head "${NODE_IP_ARGS[@]}" --dashboard-host 0.0.0.0 --port=6379 --dashboard-port=$PORT --temp-dir="$RAY_TEMP_DIR"
sleep infinity
