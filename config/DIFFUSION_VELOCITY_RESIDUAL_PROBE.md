# Frozen-backbone output velocity residual probe

The matched supervised result through epoch30 showed `residual_all - global`
validation loss around `+9e-6` on average: no useful hidden-residual advantage.
Its residual-to-hidden RMS was only0.08–0.20%. This probe removes full-backbone
continued fine-tuning as an explanation and bypasses the final LayerNorm.

The frozen source predicts `v_base`. A new observed-only branch predicts:

```text
v = v_base + delta_v(final invisible state, visible Fourier/kinematic tokens)
```

The branch uses the existing unpooled visible nonlinear encoder output as memory;
the final invisible hidden state supplies query/noisy-state/time/slot information.
Four-head cross-attention, a two-layer SiLU MLP and LayerNorm feed a zero-initialized
four-coordinate output. Padded invisible slots and events without visible memory
receive exactly zero correction. No clean invisible truth enters the branch.

All original parameters are frozen and held in eval mode during training. Only
`TruthGeneration.visible_conditioning.token_readout` owns trainable parameters.
The probe loads raw weights only from the same step-zero source used by
`diffresall1` and `difftokglobal1`; optimizer/scheduler start fresh. It runs20
epochs on16 GPUs, FP32,2048 events/GPU, ordered data and the same paired noise.
Branch AdamW LR5e-4, two-epoch warmup, cosine decay; no early stop before epoch20.

Primary comparison is within this run because the zero-output epoch0 function is
the frozen source: fixed-noise `val/loss` at epoch19 and the minimum after warmup
versus epoch0. Also inspect `velocity_residual_rms` and
`velocity_residual_to_base_rms`. A reproducible loss reduction larger than the
rough `1e-5` paired variation seen in the hidden-residual/global comparison is a
screening pass; use `1e-4` as the practical material scale. A pass authorizes a
separate joint-finetune ablation; it does not authorize DGPO automatically.

User launch on an existing16-GPU Ray allocation:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_velocity_residual_probe.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_velocity_residual_probe/ray_results
```

W&B project `EveNet`, ID `diffvelres1`, group
`Diffusion conditioning architecture`. This command does not submit a job.
