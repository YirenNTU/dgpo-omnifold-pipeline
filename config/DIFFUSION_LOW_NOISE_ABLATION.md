# Low-noise diffusion fine-tuning ablation

## Corrected LR pilot (use this command)

The initial `fjpj85rx` run shared the body optimizer group across GlobalEmbedding
and PET, using the first component's LR, then multiplied it by sqrt(16).
Its observed body and head LR were both about 8e-4. The corrected config uses
the same 10% diffusion training population and raw step-1110 source, with
actual peak body LR 2e-5 and head/projector LR 1e-4. Both body components now
specify the same LR; ObjectEncoder uses 2e-5. GPU-count LR scaling is disabled.
AdamW weight-decay scaling retains the existing behavior.

```bash
shifter env PYTHONPATH="$PWD/evenet_dgpo${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u evenet_dgpo/evenet/train.py \
  config/train_diffusion_low_noise_10pct_nersc.yaml \
  --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_lr_10pct_seed42/ray_results
```

Both training and cosine schedule run for 250 epochs, with one epoch of warmup.
EMA is disabled, optimizer state is fresh, and validation uses the cleaned
20% directory. Standard generation validation runs every 5 epochs; the fixed-noise
joint-coverage audit runs before training and every 25 completed epochs. Early-stopping
patience covers the whole pilot. W&B project is `nu2flow-RL`, group
`Diffusion low-noise coverage`. No existing live run is altered by these edits.
The change is a corrected training pilot, not a one-factor comparison with
`fjpj85rx` because LR, validation cleaning, and schedule also differ.

`lr-body` should peak at 0.00002 and `lr-generation` at 0.0001, then decay.
The disabled FAMO optimizer is no longer registered, eliminating its unused
`lr-AdamW=0.025` curve. Existing noise-band MSE diagnostics remain available.

### Joint coverage validation

`Training.JointCoverage` enables a Lightning callback on every rank. It reuses
the exploratory held-out panel from `h4_spike_coverage_1110/panel.pt` (1024
events), K=32, legacy DDIM20, batch size 16 per GPU. It does not consume the
training dataloader or pass truth target values to the sampler. Event-specific
initial noise is identical at every checkpoint, independent of rank partition.
The raw model runs in eval mode; module modes and Torch RNG states are restored
after sampling. The initial baseline is evaluated after model loading and DDP
synchronization, before the first update. A fresh run uses raw step-1110; a
resumed checkpoint without callback state would establish a new baseline at
that checkpoint. Normal callback-state resume preserves the original baseline.

W&B keys under `joint_coverage/` include:

- `completed_epochs`: 0, 25, 50, ..., 250; use this to align evaluations.
- `joint/0.0001/generated_draw_fraction` and `truth_fraction`, with analogous
  1e-5 and 1e-6 thresholds; event-hit rates are separate from probability mass.
- `topology/joint_radius/w1_radians` and `cdf_max_gap`, plus acoplanarity,
  acollinearity and the four target marginal W1 values.
- `paired/joint/<threshold>/after_minus_before_draw_fraction`, its 95% CI,
  and absolute truth-gap change/CI, using 500 whole-event bootstrap resamples.
- `change_from_baseline/*`, invalid-direction counts, duration, events and K.
- `plots`: joint-radius CDF and matched-scale 2D angular histograms.

JSON reports, generated candidates, and PNG plots are saved under
`checkpoints/joint_coverage/epoch-XXXX.*`. W1 changes are point estimates;
bootstrap intervals apply to the reported threshold probabilities/gaps.
Zero hits/zero empirical CI do not establish zero support. This reused panel
is a monitoring panel; an independent test panel is required after model
selection. No K=128 endpoint is automatically run.

Updating code cannot attach a callback to an already running Python process.
The callback takes effect at the next launch with this config. No running
NERSC job is stopped or restarted by editing these files.

## Original pilot (historical command)

This ablation fine-tunes the raw step-1110 EveNet checkpoint with the existing
truth-supervised v-prediction target. It changes only the timestep loss weight:

\[
w(t)=\begin{cases}4/1.3&t<0.1\\1/1.3&t\ge0.1\end{cases}
\]

so the expected weight remains one under uniform `t`. It does not add a physics
loss, angle prior, classifier reward, EMA replacement, or sampler change.

Use the existing NERSC diffusion config and override the source checkpoint,
output paths, and the two `TruthGeneration` keys:

```bash
shifter python3 -u evenet_dgpo/evenet/train.py \
  config/train_diffusion_10pct_nersc.yaml \
  --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_10pct_seed42/ray_results \
  --pretrain_model_load_path /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt \
  --low_noise_cutoff 0.1 \
  --low_noise_weight 4.0 \
  --epochs 250 \
  --model_checkpoint_save_path /pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_10pct_seed42/checkpoints \
  --wandb_run_name Ztautau-Diffusion-LowNoise-10pct-seed42 \
  --local_save_dir /pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_10pct_seed42/logs \
  --disable_ema
```

Run the matched control with `low_noise_cutoff=0.0` and
`low_noise_weight=1.0`, using a separate output directory and W&B run. Keep
the checkpoint raw (do not select EMA) and keep the same dataset, seed,
optimizer, epochs, and validation set.

For the control, use the same command but replace the two low-noise flags with
`--low_noise_cutoff 0.0 --low_noise_weight 1.0`, and use distinct
`--ray_dir`, `--model_checkpoint_save_path`, `--wandb_run_name`, and
`--local_save_dir` values.

## W&B decision diagnostics

Every train/validation step logs `generation-neutrino-*` (and the analogous
global/point-cloud keys when enabled):

- `unweighted`: the historical MSE, comparable between arms;
- `low_noise_mse`, `mid_noise_mse`, `high_noise_mse`: unweighted per-band MSE;
- `low_noise_fraction`: realized fraction of samples with `t<0.1`;
- `effective_weight_mean`: sanity check for the normalized weighting.

The primary check is held-out generated coverage with identical initial noise
and sampler settings. A lower weighted loss alone is not success. Require the
held-out topology/joint-radius W1 and coverage metrics to improve without a
large regression in the marginal W1 metrics. Log the original and fine-tuned
models under separate W&B IDs; never overwrite the raw source checkpoint.
