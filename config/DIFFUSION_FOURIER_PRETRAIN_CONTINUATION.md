# Add Fourier after 10pct diffusion pretraining

The initial input/no-Fourier comparison is complete. The newest prepared
experiment is the matched direct-readout pair (`readout_k4` and
`readout_multiscale`); see [FOURIER_DIRECT_READOUT.md](FOURIER_DIRECT_READOUT.md).
The earlier input MLP and LR experiments remain documented below.

## Earlier experiment: nonlinear input adapter

Run `input_mlp` from the original no-Fourier pretraining checkpoint stored in
the common YAML. It inherits the `input` arm, with only the projection changed
to the existing MLP adapter: Linear(16,64), GELU, LayerNorm(64), then zero-init
Linear(64,256) without output bias. Keep PET input injection, harmonics k=1..4,
branch LR 2e-5, all backbone parameters unfrozen, seed 42 and 50 epochs.
The adapter includes normalization and additional parameters, so improvement
would support this adapter package rather than nonlinearity alone.

Compare the mean validation loss of epochs 45--49 with the linear-input run
`98q3imcl` and no-Fourier run `mpvmx945`. Do not initialize from `s93qlh9m`.
Zero output initialization preserves initial predictions; the hidden encoder
starts receiving gradients once the output projection becomes nonzero.

In an existing 4-node / 16-GPU Ray allocation:

```bash
shifter python3 scripts/train_fourier_integration.py --arm input_mlp --check-only
shifter python3 scripts/train_fourier_integration.py --arm input_mlp --preflight
shifter python3 scripts/train_fourier_integration.py --arm input_mlp
```

Alternatively the user can submit
`sbatch NERSC/submit-fourier-integration.sbatch input_mlp`; all Python and Ray
processes in that launcher use Shifter. Preflight checks no-Fourier, linear
input and MLP input initial prediction equivalence. Output directory:
`/pscratch/sd/y/yiren/Ztautau/fourier_integration/seed42/input_mlp/`.

## Current follow-up: branch LR 3x

The completed controls are `98q3imcl` (input) and `mpvmx945` (none).
Their last-five-epoch mean validation velocity losses are 0.1274616003 and
0.1274362385 respectively: no material improvement demonstrated by input in
this single-seed comparison. Input's final validation residual/base RMS is
approximately 0.21%; this alone does not establish insufficient learning.

Run `input_fast` from the original YAML pretraining checkpoint, not either
control's final checkpoint. Only the Fourier peak LR changes, from 2e-5 to
6e-5. Keep the backbone fully trainable and use the same 50-epoch budget,
seed, data, batch size, schedule and head/backbone learning rates. Compare
epochs 45--49 against both completed controls, along with `lr-fourier` and
`val/fourier/residual_to_base_rms`. Increased branch scale without improved
loss does not support simply strengthening Fourier as the solution.

In an existing 4-node / 16-GPU Ray allocation, run from `ml_pipeline`:

```bash
shifter python3 scripts/train_fourier_integration.py --arm input_fast --check-only
shifter python3 scripts/train_fourier_integration.py --arm input_fast --preflight
shifter python3 scripts/train_fourier_integration.py --arm input_fast
```

Alternatively, submit `sbatch NERSC/submit-fourier-integration.sbatch input_fast`.
That script performs the checks, starts Ray, and runs Python through Shifter.
The assistant prepares and validates the experiment; the user submits it.

## Question

Can the existing input-linear k4 Fourier branch improve supervised velocity
prediction when added to an already trained no-Fourier diffusion model?
The branch starts at zero, so initial predictions must match the baseline.
All backbone and generation-head parameters remain unfrozen throughout.
No DGPO/reward loss, auxiliary regularization, new gate, MLP or injection-site
change is included in the primary comparison.

## Source

All arms use raw weights from the same best 10pct diffusion checkpoint:

```text
/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/epoch=190_train=0.1381_val=0.1263.ckpt
```

Source run: `ytchou97-university-of-washington/EveNet/bvp5rn76`.
The exact filename was confirmed from the W&B save log; its current remote
availability is checked by the launcher, not assumed. This is the trained
diffusion checkpoint, not `checkpoints.20M.a4.last.ckpt` foundation weights and
not a DGPO checkpoint. Optimizer, scheduler and loop counters restart; EMA is off.

## Arms and execution order

| Arm | Fourier | Fourier peak LR | Purpose |
| --- | --- | ---: | --- |
| `none` | Off | — | Matched no-Fourier continuation control |
| `input` | Existing PET-input linear k4 | 2e-5 | First comparison: does adding Fourier help from a good solution? |
| `input_slow` | Same as input | 2e-6 | Optional follow-up: 0.1x branch learning rate |
| `input_fast` | Same as input | 6e-5 | Optional follow-up: 3x branch learning rate |

Run `none` and `input` as parallel, independent continuation jobs from the
same YAML checkpoint. They use separate output directories and fresh W&B runs.
If needed, compare the two LR follow-ups against these controls. Only branch LR changes in the sweep;
weight decay, schedule, source, data, bandwidth and backbone/head LR stay fixed.
The previously implemented `output` adapter remains available but is not part
of this first experiment.

Common settings: seed 42, 50 epochs, 5-epoch warmup then cosine decay, FP32,
16 GPUs, batch 2048/GPU, same fixed 10pct training and cleaned validation data.
Body/ObjectEncoder peak LR is 2e-5; head/input-projector peak LR is 1e-4.
Separate Fourier optimizer ownership prevents double updates through PET.
The inherited dropout/stochastic depth and weight-decay rules are identical
across arms. Early-stopping patience 51 cannot shorten the planned finite-loss
50-epoch budget. This controlled continuation is not an exact rerun of the
historical baseline optimizer recipe.

Primary endpoint: mean validation velocity loss over epochs 45--49.
Inspect training/validation learning curves and initial loss as well; best
validation loss alone is secondary. Compare `input` against the matched `none`
continuation at the same epoch/update budget to separate the Fourier addition
from ordinary additional training. The two-event preflight is only an
initial-equivalence check, not a full validation benchmark. The historical
0.1263 loss also used a different validation setup; do not use it as a matched
threshold. Interpret the Fourier arms as follows:

- Input improves over none: supports a useful Fourier contribution for this
  starting point, training recipe and budget.
- Input ties: no benefit demonstrated within the tested budget.
- Input worsens: investigate branch learning rate/scale with the two follow-ups.
  A slower arm winning supports an update-scale issue, not necessarily a
  frequency-bandwidth issue.

Record `lr-fourier` and `train/fourier/residual_to_base_rms` plus its validation
equivalent. A large residual is a diagnostic, not proof of causality. All main
weights remain trainable, and both positive and negative co-adaptation remain
possible. Confirm a promising result with further seeds before generalizing.

## Checks and user launch commands

Use the existing remote `ml_pipeline` checkout. Sync updated code and configs
with the usual exclusions; repository-root rsync must include
`--exclude-from=NERSC/upload-excludes.txt`. No toy code/outputs or excluded
classifier checkpoints should be uploaded. Keep the existing generated event
schema, dataset and normalization files available.

```bash
# Checkpoint is read from train_diffusion_fourier_integration_common.yaml.

# Validate configuration without starting training.
shifter python3 scripts/train_fourier_integration.py --arm input --check-only

# Optional standalone CPU preflight; the batch scripts also perform this check.
shifter python3 scripts/train_fourier_integration.py --arm input --preflight

# Parallel matched comparison. Each job uses 4 nodes / 16 GPUs (32 GPUs if concurrent).
# Slurm decides when each independent job can start; no dependency is imposed.
sbatch NERSC/submit-fourier-integration.sbatch none
sbatch NERSC/submit-fourier-integration.sbatch input

# Follow-ups if the initial comparison warrants a branch-LR test.
sbatch NERSC/submit-fourier-integration.sbatch input_slow
sbatch NERSC/submit-fourier-integration.sbatch input_fast
```

Preflight evaluates the untouched source and all three Fourier arms on the same checkpoint and
two validation events, requiring equal initial train/eval predictions with
matched RNG. It rejects frozen model parameters and incompatible shared weights.
It does not update weights or initialize a W&B training run.

W&B project: `EveNet`; group: `Fourier pretrained continuation`; fresh run IDs.
Outputs: `/pscratch/sd/y/yiren/Ztautau/fourier_integration/seed42/<arm>/`.
Choose a new output root (third sbatch argument) for a rerun; prior runtime
configs cannot be overwritten. All compared arms must use the same checkpoint.

Ordered data and per-rank/batch RNG isolate draws from model initialization and
unrelated diagnostics. This requires identical data sharding and does not
promise event-identity pairing or bitwise reproducibility across distributed
executions. Keep GPU count/batch/data fixed. Local tests cannot replace NERSC
checkpoint/data preflight or distributed verification.

See [DIFFUSION_FOURIER_INTEGRATION.md](DIFFUSION_FOURIER_INTEGRATION.md) for the
optional output-adapter design and literature context.
