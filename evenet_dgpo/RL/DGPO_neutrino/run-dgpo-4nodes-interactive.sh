#!/bin/bash
# Interactive 4-node DGPO training on NERSC Perlmutter.
#
# Step 1 — allocate nodes (run on login node):
#   salloc --nodes 4 --qos interactive --time 04:00:00 \
#          --constraint gpu --gpus 16 --account m5019_g \
#          --image=registry.nersc.gov/m2616/avencast/evenet:1.3
#          (account is m5019, not m5019_g)
#
# Step 2 — inside the allocation, from repo root:
#   bash RL/DGPO_neutrino/run-dgpo-4nodes-interactive.sh
set -euo pipefail

# ── Verify we are inside a Slurm allocation ────────────────────────────────
if [[ -z "${SLURM_JOB_NUM_NODES:-}" ]]; then
  echo "ERROR: not inside a Slurm allocation. Run salloc first." >&2
  exit 1
fi
echo "Nodes: $SLURM_JOB_NUM_NODES | List: $SLURM_JOB_NODELIST"

# ── Paths & environment ────────────────────────────────────────────────────
REPO_ROOT="/global/homes/y/yiren/EveNet-private"
NERSC_DIR="$REPO_ROOT/NERSC"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-2a0d7e3df0ce2abff1442fc210a5da54c19ea850}"
export RAY_TMPDIR="${SCRATCH}/ray_tmp"
mkdir -p "$RAY_TMPDIR"

export RAY_raylet_start_wait_time_s=60
export TORCH_NCCL_TIMEOUT=180
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── Resolve head node IP ───────────────────────────────────────────────────
head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi
port=6379

# ── Clean up stale Ray sockets on ALL nodes (prevents "Address already in use") ──
echo "Cleaning up stale Ray/Plasma sockets on all nodes..."
shifter ray stop --force 2>/dev/null || true
rm -rf /tmp/ray* /tmp/plasma* 2>/dev/null || true
SHIFTER_IMAGE="registry.nersc.gov/m2616/avencast/evenet:1.3"
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -v "^${head_node}$"); do
  ssh "$node" "shifter --image=${SHIFTER_IMAGE} ray stop --force 2>/dev/null; rm -rf /tmp/ray* /tmp/plasma* 2>/dev/null" &
done
wait
sleep 3

# ── Start Ray HEAD directly (no srun — keeps all step slots free for workers) ──
echo "STARTING HEAD at $head_node ($head_node_ip:$port)"
shifter ray start --head \
  --node-ip-address="$head_node_ip" \
  --port=$port \
  --dashboard-host=0.0.0.0

# Poll until GCS is ready before launching workers.
echo "Waiting for Ray GCS to be ready..."
for i in $(seq 1 60); do
  if shifter ray status 2>/dev/null | grep -q "ray.head.node"; then
    echo "Ray head is ready after ${i}s."
    break
  fi
  sleep 2
done

# ── Start Ray WORKERS via srun (head not using srun, so slots are free) ────
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [[ "$worker_num" -gt 0 ]]; then
  # Use ssh + shifter --image explicitly — when ssh-ing to worker nodes there is
  # no container context, so we must specify the image.
  SHIFTER_IMAGE="registry.nersc.gov/m2616/avencast/evenet:1.3"
  echo "Starting $worker_num worker(s) via ssh + shifter..."
  for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -v "^${head_node}$"); do
    echo "  Starting worker on $node ..."
    ssh "$node" "export RAY_TMPDIR=$RAY_TMPDIR; export RAY_raylet_start_wait_time_s=60; \
      shifter --image=${SHIFTER_IMAGE} ray start \
        --address=${head_node_ip}:${port} \
        --node-ip-address=\$(hostname --ip-address | awk '{print \$1}'); \
      sleep infinity" &
  done

  # Wait until all nodes are registered — check every 5s up to 3 minutes.
  echo "Waiting for all $SLURM_JOB_NUM_NODES nodes to register with Ray..."
  for i in $(seq 1 36); do
    connected=$(shifter ray status 2>/dev/null | grep -c "node_" || true)
    echo "  [${i}] Ray nodes connected: $connected / $SLURM_JOB_NUM_NODES"
    if [[ "$connected" -ge "$SLURM_JOB_NUM_NODES" ]]; then
      echo "All $SLURM_JOB_NUM_NODES nodes ready!"
      break
    fi
    if [[ "$i" -eq 36 ]]; then
      echo "WARNING: only $connected/$SLURM_JOB_NUM_NODES nodes connected after 3 minutes."
      echo "Check worker logs: ls $SCRATCH/ray_tmp/ray/session_latest/logs/"
    fi
    sleep 5
  done
fi

# ── Verify cluster ─────────────────────────────────────────────────────────
echo "Ray cluster status:"
shifter ray status 2>/dev/null
echo ""

# ── Launch DGPO training ───────────────────────────────────────────────────
echo "=========================================="
echo "Starting DGPO training on $SLURM_JOB_NUM_NODES nodes"
echo "=========================================="

shifter python3 RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml
