# Supervised angular-Fourier fine-tuning on 10% data

Prepared experiment; no remote training submitted by the assistant.

Use the raw EveNet foundation weights from
`/pscratch/sd/y/yiren/checkpoints.20M.a4.last.ckpt`, the same pre-diffusion source
configured in `train_diffusion_nersc.yaml`. This is NOT the already fine-tuned
10% diffusion checkpoint or the DGPO step-1110 policy, not EMA, and not a full
resume. Optimizer, scheduler and epoch start fresh. Compatible source weights
are loaded; missing or shape-incompatible parameters retain fresh initialization.
The missing angular projection is initialized to zero, so attaching it does not
change the initial network output; this does not imply an already trained
neutrino generator. All PET weights and the generation head remain trainable;
no classifier is fitted or used. Inspect the loader's missing/shape-mismatch
report against the actual remote foundation checkpoint on launch.

The overlay inherits the established supervised low-noise experiment's setup,
but explicitly disables low-noise weighting. Train data is the pre-existing
`diffusion_train_10pct_seed42/train` population (dataset_limit=1); validation is
`diffusion_val_20pct_seed42_stic_filtered_test1/val` (val_dataset_limit=1).
Normalization is unchanged. Do not substitute the OmniFold subset silently.

Settings: Lightning `32-true` (float32 model/training, not mixed precision),
16 GPU workers, batch_size 2048 per GPU (32768 per full distributed batch), fixed
unscaled peak body LR 2e-5 and generation/projector LR 1e-4, AdamW and the existing
EveNet scheduler. Budget and scheduler horizon 500 epochs, early stopping on val/loss with patience 75
validation checks and min_delta 1e-4. EMA is disabled.

W&B: nu2flow-RL / diffang10p1, display name
"Angular Fourier fine-tuning | diffusion 10% | FP32 raw pretrain".
Precision follows the original EveNet FP32 setup. FP64 compatibility fixes remain
available in the code, but this experiment no longer enables FP64.
The per-GPU batch is 2048; no successful 16-GPU memory benchmark was run locally.
Inspect train/validation diffusion losses and existing time-stratified loss
diagnostics. JointCoverage is disabled: no pre-training coverage baseline or
periodic joint-coverage evaluation runs, and no coverage panel is required.
Ordinary diffusion validation and the existing generation metrics remain enabled.
These are supervised-generation diagnostics, not fresh-classifier closure.
No matched no-Fourier retraining arm is included, so improvements cannot alone
establish a causal Fourier advantage.

Run from the updated repository inside the user's existing 16-GPU Ray allocation:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_angular_10pct_nersc.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_seed42/ray_results
```

Do not pass this overlay directly to evenet/train.py; the wrapper resolves its
inheritance. Do not add --resume_checkpoint for this fresh fine-tuning run.

## Continue the interrupted/early-stopped run

To continue instead of starting fresh, append to the command above:
`--resume_checkpoint /pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_seed42/checkpoints/last.ckpt`.
This restores raw model, optimizer, scheduler step, epoch and early-stopping
best/wait state. The new patience is 75 (the prior wait count is not reset),
and 500 is the total epoch limit, not 500 additional epochs. The cosine horizon
is rebuilt from the new 500-epoch configuration while retaining scheduler step;
LR can increase on the next scheduler update relative to the old 350-epoch curve.
The CLI clears the foundation initialization path when full resume is requested.
The wrapper writes each resolved runtime YAML beneath the experiment's shared
`runtime_configs/` directory next to `checkpoints/`, not node-local `/tmp`.
All Ray nodes must see this directory and the referenced default configuration files.
The remote checkpoint must be available; local validation does not
verify remote files or execute a 16-GPU training smoke test.
