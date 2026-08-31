#!/bin/bash
# DGPO neutrino on 4 Perlmutter GPU nodes.
# Submit from the ml_pipeline repo root:
#   sbatch evenet_dgpo/RL/DGPO_neutrino/submit-dgpo-4nodes.sbatch

#SBATCH -C gpu
#SBATCH --time=12:00:00
#SBATCH -q regular
#SBATCH -A m5019

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.3
#SBATCH --output=dgpo_4nodes_%j.out   # stdout+stderr → dgpo_4nodes_<jobid>.out

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NERSC_SCRIPTS="${REPO_ROOT}/NERSC"

# ── WandB ──────────────────────────────────────────────────────────────────
: "${WANDB_API_KEY:?Set WANDB_API_KEY before submitting DGPO}"

# ── Ray cluster setup ──────────────────────────────────────────────────────
head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# Convert multi-IP output to a single IPv4 address
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi
port=6379

echo "STARTING HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 --gpus=0 --gpus-per-task=4 --cpus-per-task=128 -w "$head_node" \
  shifter "${NERSC_SCRIPTS}/start-head.sh" "$head_node_ip" &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
srun -n "$worker_num" --nodes="$worker_num" --ntasks-per-node=1 \
  --gpus-per-task=4 --cpus-per-task=128 --gpus=0 --exclude "$head_node" \
  shifter "${NERSC_SCRIPTS}/start-worker.sh" "$head_node_ip:$port" &
sleep 5

# ── Environment ────────────────────────────────────────────────────────────
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/evenet_dgpo:$REPO_ROOT:${PYTHONPATH:-}"

# NCCL settings for multi-node GPU communication
export TORCH_NCCL_TIMEOUT=180
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── Launch ─────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Starting DGPO training on $SLURM_JOB_NUM_NODES nodes"
echo "Repo: $REPO_ROOT"
echo "=========================================="

shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_overlay.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_v1/ray_results

exit
