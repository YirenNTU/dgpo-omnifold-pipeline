# Higher-LR angular diffusion fine-tuning

User-launched experiment; no remote job submitted by the assistant.
Overlay: `train_diffusion_angular_10pct_highlr.yaml`.

Source is the raw best validation checkpoint from diffang10r130:
`/pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_resume130/checkpoints/epoch=390_train=0.1764_val=0.1745.ckpt`.
The filename was verified from the W&B save log (val/loss 0.1745055).
Remote filesystem availability is not independently verified.

Load model weights including learned angular projection, not optimizer/loop state.
Fresh AdamW, fresh stopping history, epochs 0--199. No EMA, DGPO or classifier.
All backbone weights stay trainable. Ordinary diffusion loss; joint coverage off.
FP32, 16 workers, 2048 events per GPU. Same 10% training/clean validation datasets.

Actual peak LRs with world-size scaling disabled:
- body (PET + GlobalEmbedding) and ObjectEncoder: 1e-4;
- generation head and both input projectors: 2e-4.

Existing EveNet cosine scheduler: 5 epochs linear warmup, cosine decay to zero
over the remainder of a 200-epoch total horizon. For 416708 events and 16x2048,
the configured estimate is 13 updates/epoch, 65 warmup updates, 2600 total.
Early stopping remains patience 75, min_delta 1e-4; may stop before epoch 200.
Use lr-body/lr-generation/lr-projector plus validation loss to monitor the run.
This tests optimization headroom, not Fourier causality. The original checkpoint
is preserved and new output goes to diffusion_angular_10pct_highlr.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_angular_10pct_highlr.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_highlr/ray_results
```

Do not add --resume_checkpoint: that would restore the old optimizer and counters.
W&B ID: diffang10lr1, in nu2flow-RL. No joint-coverage baseline is run.

## Higher peak: 5e-4, 500 epochs

Overlay `train_diffusion_angular_10pct_lr5e4.yaml` loads raw weights from the
last checkpoint of diffang10lr1, not the original epoch-390 checkpoint:
`/pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_highlr/checkpoints/last.ckpt`.
This alias is read at launch time. W&B reported the source run still running
when configured; the remote checkpoint target and epoch have not been verified.
Check the alias before launch (including whether a newer last-vN.ckpt exists).
Prefer finishing/stopping the source run first and pinning its resolved file
if an immutable source is needed. Do not confuse latest with best validation.
Body (PET + GlobalEmbedding) and ObjectEncoder use peak LR 2.5e-4;
generation head and both input projectors use 5e-4. This preserves the previous
1:2 LR ratio while increasing both groups by 2.5x.
World-size scaling remains off. Five epochs warmup then cosine to zero at the
500-epoch horizon (65 warmup / 6500 total estimated updates). Fresh AdamW and
epoch zero. Early stopping uses patience 50, min_delta 1e-4 on val/loss,
so 500 is a cap, not a guaranteed duration.
FP32, 16 GPUs, 2048 per GPU, unfrozen backbone, no EMA or joint coverage.
Source weights, LR and horizon change; this is sequential fine-tuning,
not a matched one-variable causal comparison.
No existing run is stopped or modified and no job is submitted automatically.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_angular_10pct_lr5e4.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_lr5e4/ray_results
```

W&B ID `diffang10lr5e4`. Do not add `--resume_checkpoint`.

## Continue with the historical non-Fourier optimization recipe

Overlay `train_diffusion_angular_10pct_legacy_setup.yaml` loads the last raw
checkpoint of completed `diffang10lr5e4` (W&B final epoch 296), keeping Fourier
weights. The remote last.ckpt alias target has not been independently inspected.
Fresh AdamW/scheduler/epoch zero; not a full-state Lightning resume.
Historical `EveNet/bvp5rn76` effective group LR was 8e-4 (nominal 2e-4 with
sqrt(16) scaling); explicitly use 8e-4 without LR world-size scaling for all
active groups. Weight-decay scaling stays unchanged. One epoch warmup,
1000 total epochs cosine (user override of the historical 1500), early stopping
patience 25/min_delta 0 on val/loss.
Generation diagnostics every 20 epochs, loss validation every epoch.
FP32, 16 GPUs, 2048/GPU, unfrozen backbone, ordinary diffusion objective.
Intentional differences from the historical run: learned Fourier architecture
and starting weights, cleaned validation, fixed seed, and raw/no EMA per user
preference (historical EMA tracking was on but training evaluation used raw).
No joint coverage. Independent W&B ID `diffang10oldopt1`; no job submitted.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_angular_10pct_legacy_setup.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_legacy_setup/ray_results
```
