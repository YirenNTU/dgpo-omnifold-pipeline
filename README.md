# Ztautau ML Pipeline

This repository contains the self-contained Ztautau EveNet workflow:

1. supervised diffusion training
2. one-candidate-per-event OmniFold fitting
3. OmniFold-guided DGPO with `K` diffusion candidates per event

Run commands from the `ml_pipeline` root unless stated otherwise.

## Repository layout

```text
ml_pipeline/
├── NERSC/                         interactive multi-node Ray helpers
├── config/                        diffusion, OmniFold, and DGPO YAMLs
│   └── evenet_defaults/           vendored EveNet options/network defaults
├── evenet_dgpo/
│   ├── evenet/                    EveNet training implementation
│   └── RL/DGPO_neutrino/          sampling, OmniFold, rewards, and DGPO
├── preprocessing/                 EveNet preprocessing helpers
├── scripts/
│   ├── run_omnifold_stage.py
│   └── train_neutrino_backend.py
└── generate_event_info_yaml.py
```

The active workflow does not require a separate EveNet-Full or EveNet-private
checkout. Data, checkpoints, and generated training outputs remain on CFS or
pscratch and are not stored in Git.

## Current NERSC campaign paths

| Item | Path |
| --- | --- |
| Full supervised training data | `/global/cfs/cdirs/m5019/tihsu/Ztautau/evenet-input/train-diffusion` |
| DGPO/OmniFold 40% training subset | `/pscratch/sd/y/yiren/Ztautau/dgpo_post_training_40pct/train` |
| Validation data | `/global/cfs/cdirs/m5019/tihsu/Ztautau/evenet-input/val-diffusion` |
| Normalization | `/global/cfs/cdirs/m5019/tihsu/Ztautau/evenet-input/normalization.pt` |
| Shape metadata | `/global/cfs/cdirs/m5019/tihsu/Ztautau/evenet-input/shape_metadata.json` |
| Diffusion checkpoints | `/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/checkpoints` |
| OmniFold output | `/pscratch/sd/y/yiren/Ztautau/omnifold_ztautau_v1` |
| DGPO output | `/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_fresh_residual_v2` |

The active configurations are:

- `config/train_diffusion_nersc.yaml`: cold-start supervised diffusion
- `config/train_diffusion_resume_nersc.yaml`: exact Lightning resume
- `config/omnifold_ztautau.yaml`: standalone `K=1` OmniFold stage
- `config/dgpo_omnifold_ztautau.yaml`: single-control `K=8` OmniFold-guided DGPO

## Upload from a Mac

Run this command on the Mac, not on Perlmutter. The trailing slashes copy the
contents into the existing remote `ml_pipeline` directory. Do not add
`--delete`; NERSC-only files such as generated event info may exist remotely.

```bash
rsync -avz --progress --partial \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='wandb/' \
  --exclude='.DS_Store' \
  /Users/yirenwu/Ztautau/ml_pipeline/ \
  yiren@perlmutter.nersc.gov:/global/homes/y/yiren/ml_pipeline/
```

## NERSC environment setup

```bash
ssh yiren@perlmutter.nersc.gov
cd /global/homes/y/yiren/ml_pipeline

export PYTHONPATH="$PWD/evenet_dgpo:$PWD:${PYTHONPATH:-}"
export TORCH_NCCL_TIMEOUT=180
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

source NERSC/start_interactive_ray.sh
```

Do not store the W&B key in YAML, README, or shell scripts.

Generate the event schema after each relevant analysis/schema change:

```bash
shifter python3 generate_event_info_yaml.py \
  --analysis-config config/analysis.yaml \
  --evenet-config config/evenet_schema.yaml \
  --output config/generated_event_info.yaml
```

The generated file is intentionally NERSC-local and may be regenerated after
each rsync.

## Interactive 4-node × 4-GPU Ray cluster

The training YAMLs use 16 Ray Train workers with one GPU per worker. Therefore
the allocation and Ray status must expose four nodes and sixteen GPUs.

### Option A: allocate and source Ray manually

On a Perlmutter login node:

```bash
salloc --nodes 4 --qos interactive --time 04:00:00 \
  --constraint gpu --gpus-per-node=4 --account m5019_g \
  --image=registry.nersc.gov/m2616/avencast/evenet:1.3
```

Check the allocation:

```bash
scontrol show job "$SLURM_JOB_ID" | grep -E 'TRES|NumNodes'
```

From the shell holding that allocation:

```bash
cd /global/homes/y/yiren/ml_pipeline
export PYTHONPATH="$PWD/evenet_dgpo:$PWD:${PYTHONPATH:-}"
export TORCH_NCCL_TIMEOUT=180

source NERSC/start_interactive_ray.sh
```

`source` is required so `head_node`, `head_node_ip`, `port`, and `RAY_ADDRESS`
remain in the current shell. The shorter compatibility command is also valid:

```bash
source NERSC/interactive.sh
```

Verify the cluster before training:

```bash
shifter ray status --address="$RAY_ADDRESS"
```

Expected result: four active Ray nodes and `16.0 GPU` total.

### Option B: allocate and start Ray with one command

From a login node:

```bash
cd /global/homes/y/yiren/ml_pipeline
export WANDB_API_KEY="<your-current-key>"
bash NERSC/salloc_4node_16gpu_ray.sh
```

This opens an interactive compute-node shell, starts Ray, and exports
`RAY_ADDRESS`. Exit that shell to release the allocation.

Leave `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` unset for normal training.
Ray then assigns exactly one visible GPU to each worker.

## Stage 1: supervised diffusion

### Start or continue the cold-start configuration

```bash
shifter python3 evenet_dgpo/evenet/train.py \
  config/train_diffusion_nersc.yaml \
  --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/ray_results
```

### Resume an interrupted run

The resume YAML restores `last.ckpt`, including live weights, optimizer state,
epoch/global step, and the separately saved EMA state. It continues toward the
same configured total epoch horizon.

```bash
shifter python3 evenet_dgpo/evenet/train.py \
  config/train_diffusion_resume_nersc.yaml \
  --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/ray_results_resume
```

Before the next stage, confirm:

```bash
test -f /pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/checkpoints/last.ckpt
```

## Optional standalone OmniFold diagnostic

This standalone path remains useful for inspecting fixed K=1 pools, but it is
not a prerequisite for the active DGPO launch. The active DGPO trainer now
generates its own K=1 population, trains frozen-backbone OmniFold PEFT banks,
and installs the resulting reward before its first policy update.

Check paths and checkpoint compatibility:

```bash
shifter python3 scripts/run_omnifold_stage.py \
  --train-config config/train_diffusion_nersc.yaml \
  --omnifold-config config/omnifold_ztautau.yaml \
  --stage check \
  --device cuda
```

Build the fixed train/validation pools and fit the reward:

```bash
shifter python3 scripts/run_omnifold_stage.py \
  --train-config config/train_diffusion_nersc.yaml \
  --omnifold-config config/omnifold_ztautau.yaml \
  --stage all \
  --device cuda
```

Expected outputs:

```text
/pscratch/sd/y/yiren/Ztautau/omnifold_ztautau_v1/
├── train_k1_pool.pt
├── validation_k1_pool.pt
└── omnifold_reward.pt
```

Use `--stage pool` or `--stage fit` to run only one part. Add
`--rebuild-pools` if the supervised checkpoint, input data, or sampler settings
changed.

## Stage 3: OmniFold-guided DGPO

Start DGPO from the same shell where Ray was sourced:

```bash
shifter python3 evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py \
  config/dgpo_omnifold_ztautau.yaml \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_fresh_residual_v2/ray_results
```

Like EveNet-private, DGPO is controlled by one complete YAML. The diffusion
pretraining config is not merged at launch time.

The active setup uses:

- 16 distributed workers, one GPU each
- 150 DGPO epochs (independent of the base diffusion YAML's 1500 epochs)
- `1e-4` projector/head and `1e-5` body learning rates
- `dgpo.K: 8` candidates per event
- 1024 DGPO events per GPU/update, evaluated as four sequential gradient-bearing
  256-event microbatches; rollout evaluates all K=8 chains together under no-grad
- OmniFold/audit effective training batch 8192 rows per class/GPU, accumulated
  from 2048-row gradient microbatches before one optimizer step
- the EveNet-private shared-noise round-reference trust term (`coefficient: 1`)
- live-policy DDIM sampling; EMA is still updated every step and checkpointed
- an in-process, fail-closed K=1 OmniFold bootstrap before round 1
- adaptive staleness probes and training-time OmniFold refits
- live rank-0 console/W&B OmniFold fit metrics during classifier fitting
- a recovery `last.ckpt` immediately after bootstrap and every 25 optimizer steps
- TARP diagnostics and targeted one-dimensional physics distributions
- W&B project `nu2flow-RL`

For each OmniFold residual iteration, the frozen-backbone PEFT classifier must
reach saturation before its weights are stored as a reward snapshot. The next
iteration trains against the newly reweighted Gen population. The final closure
classifier is logged but deliberately not stored in the reward.

After a refit, installation no longer compares the candidate with a separately
audited incumbent. The candidate is installed when its fresh held-out weighted
balanced accuracy is strictly below `0.51` and both the residual fits and audit
have reached their configured saturation condition.

During each residual reward fit, acceptance audit, and staleness audit, rank 0
logs `omnifold_live/*` periodically. Each classifier epoch now consumes the
entire fit split exactly once, including a short final batch.

The adaptive fresh audit runs every 10 DGPO logical epochs. Its retraining
margin is a configurable hyperparameter, currently set to a 1 percentage-point
AUC gap. The active trigger is
`current_weighted_gap > installed_baseline_gap + 0.01`. This is the complete
retraining rule; there is no statistical uncertainty guard or absolute floor.
The single weighted audit judge checks its early-stop split after every complete
epoch. If its AUC gap crosses the threshold, training stops and the untouched
final split must confirm the crossing. Otherwise training continues until ten
complete epochs in a row show no validation-loss improvement. Candidate
installation always uses the saturated path; the old separate raw audit has
been removed. The fresh PEFT audit logs separation and precision:

- `staleness/judge_auc_weighted` and `staleness/weighted_auc_gap` test closure
  after the installed OmniFold weights.
- `staleness/audit_power_at_retrain_margin`,
  `staleness/audit_minimum_detectable_auc_gap`, and
  `staleness/audit_power_sufficient` report whether the held-out population
  can resolve the configured `0.01` retraining margin. These are monitoring
  metrics, not a hidden retraining gate.

The active routine-audit cap is 600k events; its untouched 20% audit split gives
about 120k examples per class. At ideal weight ESS this provides more than 85%
power for a two-sided `|AUC-0.5|=0.01` test at `alpha=0.05`. W&B recomputes the
actual power using the observed weight ESS. Ordinary K=16 validation remains
capped near 53k events by `dgpo.validation_max_batches: 13`.

Alternatively, this script starts Ray and launches the same DGPO command:

```bash
bash evenet_dgpo/RL/DGPO_neutrino/run-dgpo-4nodes-interactive.sh
```

If Ray is already running in the current shell:

```bash
DGPO_SKIP_RAY_START=1 \
  bash evenet_dgpo/RL/DGPO_neutrino/run-dgpo-4nodes-interactive.sh
```

## DGPO resume

DGPO checkpoints are written to:

```text
/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_fresh_residual_v2/checkpoints
```

To resume, change `options.Training.model_checkpoint_load_path` in
`config/dgpo_omnifold_ztautau.yaml` from the supervised checkpoint to:

```text
/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_fresh_residual_v2/checkpoints/last.ckpt
```

Then run the same Stage 3 command. The checkpoint carries optimizer/EMA state,
DGPO counters, active adaptive OmniFold reward stack, controller state, and the
paired policy reference.

## Validation

Run the active Ztautau/OmniFold/DGPO tests locally:

```bash
PYTHONPATH="$PWD/evenet_dgpo:${PYTHONPATH:-}" python3 -m pytest -q \
  evenet_dgpo/RL/DGPO_neutrino/test_dgpo_utils.py \
  evenet_dgpo/RL/DGPO_neutrino/test_model_utils.py \
  evenet_dgpo/RL/DGPO_neutrino/test_normalizer_denormalize_grad_parity.py \
  evenet_dgpo/RL/DGPO_neutrino/test_rewards.py \
  evenet_dgpo/RL/DGPO_neutrino/diagnostics/test_ztautau_validation.py \
  evenet_dgpo/RL/DGPO_neutrino/omnifold_ztautau/test_adaptive.py \
  evenet_dgpo/RL/DGPO_neutrino/omnifold_ztautau/test_ztautau_omnifold.py
```

## Common pitfalls

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: evenet` | Export `PYTHONPATH="$PWD/evenet_dgpo:$PWD:${PYTHONPATH:-}"` from `ml_pipeline`. |
| Included options/network YAML is missing | Re-rsync `config/evenet_defaults`; EveNet-Full is not required. |
| `generated_event_info.yaml` is missing | Run the generator command in the NERSC setup section. |
| `AF_UNIX path length cannot exceed 107 bytes` | Use the default short `/tmp/r${UID}_${SLURM_JOB_ID}` Ray directory. |
| Ray cannot find a running instance | Source `NERSC/start_interactive_ray.sh` in the same shell used for training. |
| Ray reports fewer than 16 GPUs | Reallocate with four nodes and `--gpus-per-node=4`; do not start training. |
| Training waits for workers | Ensure `platform.number_of_workers: 16` and Ray exposes 16 GPUs. |
| `omnifold_reward.pt` is missing | Complete Stage 2 before DGPO. |
| `shifter: command not found` | Run on a Perlmutter host shell, not from inside an already-entered container shell. |

## Cleanup

Exit the interactive shell or cancel the allocation:

```bash
scancel "$SLURM_JOB_ID"
```

Ray logs for the head process are written to
`/tmp/ray_head_${USER}_${SLURM_JOB_ID}.log` on the allocation.
