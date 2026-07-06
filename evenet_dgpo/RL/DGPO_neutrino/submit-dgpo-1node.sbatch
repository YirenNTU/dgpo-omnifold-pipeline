#!/bin/bash
# DGPO neutrino on 1 Perlmutter GPU node (4 GPUs).
# Submit from the EveNet repo root:
#   cd /global/homes/y/yiren/EveNet-private && sbatch RL/DGPO_neutrino/submit-dgpo-1node.sbatch

#SBATCH -C gpu
#SBATCH --time=12:00:00
#SBATCH -q regular
#SBATCH -A m5019

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.3
#SBATCH --output=dgpo_1node_%j.out

# ── Environment ────────────────────────────────────────────────────────────
REPO_ROOT="/global/homes/y/yiren/EveNet-private"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_API_KEY="2a0d7e3df0ce2abff1442fc210a5da54c19ea850"
export RAY_TMPDIR="${SCRATCH}/ray_tmp"
mkdir -p "$RAY_TMPDIR"

echo "=========================================="
echo "Starting DGPO training on 1 node (4 GPUs)"
echo "=========================================="

shifter python3 RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml --num-gpus 4

exit
