#!/bin/bash
# Interactive 4-node DGPO training on NERSC Perlmutter.
#
# Step 1 — allocate nodes (run on login node):
#   salloc --nodes 4 --qos interactive --time 04:00:00 \
#          --constraint gpu --gpus 16 --account m5019_g \
#          --image=registry.nersc.gov/m2616/avencast/evenet:1.3
#          (account is m5019, not m5019_g)
#
# Step 2 — inside the allocation, from ml_pipeline root:
#   bash evenet_dgpo/RL/DGPO_neutrino/run-dgpo-4nodes-interactive.sh
set -euo pipefail

# ── Verify we are inside a Slurm allocation ────────────────────────────────
if [[ -z "${SLURM_JOB_NUM_NODES:-}" ]]; then
  echo "ERROR: not inside a Slurm allocation. Run salloc first." >&2
  exit 1
fi
echo "Nodes: $SLURM_JOB_NUM_NODES | List: $SLURM_JOB_NODELIST"

# ── Paths & environment ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/evenet_dgpo:$REPO_ROOT:${PYTHONPATH:-}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before launching DGPO}"

export RAY_raylet_start_wait_time_s=60
export TORCH_NCCL_TIMEOUT=180
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Start the self-contained cluster helper unless the caller already did so.
if [[ "${DGPO_SKIP_RAY_START:-0}" != "1" ]]; then
  # shellcheck source=../../../NERSC/start_interactive_ray.sh
  source "$REPO_ROOT/NERSC/start_interactive_ray.sh"
fi

# ── Launch DGPO training ───────────────────────────────────────────────────
echo "=========================================="
echo "Starting DGPO training on $SLURM_JOB_NUM_NODES nodes"
echo "=========================================="

shifter python3 evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py \
  config/dgpo_omnifold_ztautau.yaml \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_fresh_residual_v2/ray_results
