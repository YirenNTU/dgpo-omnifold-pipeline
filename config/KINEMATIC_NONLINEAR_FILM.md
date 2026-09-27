# Observed angle + momentum Fourier with nonlinear FiLM

The user narrowed the proposed full-condition expansion to visible kinematics.
Only energy, pT, theta and phi receive the new/retained Fourier expansion.
The full-feature/global-Fourier draft was removed before any training.

## Two configurations

For the new **three-block diffusion + OmniFold/audit** variants, see
[NONLINEAR_CONDITIONING_DEPTH3.md](NONLINEAR_CONDITIONING_DEPTH3.md).
The configurations described below remain the unchanged one-block controls.

- `dgpo_h4_visible_adaln.yaml`: original angle-only branch, unchanged.
- `dgpo_h4_kinematic_adaln.yaml`: adds energy/pT scalars and Fourier to that branch.

For every valid visible particle, use observed eta -> theta and phi in radians
with k=1,2,3,4. Energy/pT use the existing sequential normalizer output **before
grouped projection**, with frequencies 0.25,0.5,1,2 radians per normalized unit.
Retain the two normalized scalar values as well, so periodic encoding does not
replace magnitude information. Existing upstream log transforms and checkpoint
normalization are inherited, not recomputed or applied twice.

Concatenate these with the existing pre-PET visible representation. Two SiLU
layers width64 -> masked attention pooling -> two-layer SiLU event encoder with
encoder-only LayerNorm -> separate zero-initialized per-block FiLM/AdaLN heads.
No second zero gate, no change to existing backbone LayerNorm or residual gates.
New branch attachment leaves initial policy outputs unchanged.

No Fourier expansion of IDs, charge, shower details, multiplicities, hemisphere
flags, or global quantities. These inputs are **not removed**: their original
conditioning paths remain. Candidate coordinates, invisible truth and labels
never enter the new branch. Existing classifier H4 candidate-pair late fusion
is separate and remains unchanged.

Both diffusion and fresh OmniFold/audit classifiers use this branch, with
independent weights. Supervised training, sampling/reference/DGPO velocity and
classifier forwards use the same normalized observed inputs. Checkpoints save
the numerical frequencies and branch specification; legacy angle/off classifier
payloads keep their original architecture when loaded under a new live config.

## Unchanged experimental controls

- Raw weights-only from the same `diffang10lr5e4/checkpoints/last.ckpt`, never EMA.
- Same nonlinear width/depth, one generation block and existing classifier depth.
  This is not an increase to the toy's three conditioned denoiser layers.
- Same AdamW/LRs and coefficient-1 velocity-MSE **surrogate**, no objective change.
- Two folds, one repeat, two iterations; refit every20 policy epochs; cold audit
  every5; validation every10; sixteen Ray GPU workers.
- Select best validation BCE; patience25 epochs from step0; no forced minimum
  or max fit steps. Inspect saturation before treating chance AUC as closure.
- Fresh classifier fitting and optimizer state, not a full-state old resume.

The token encoder grows from 23 to41 inputs for the current seven-slot projected
basis: 16 angle features plus 2 numerical scalars plus16 numerical Fourier
features. Event encoder input stays65, width64. This adds1,152 trainable weights
per branch; a positive result is not a capacity-matched proof of complementarity.
Both policy/reward representations change; compare cold audit BCE/AUC at matched
policy epochs, not absolute rewards across independently fitted critics. Reward
gain within each fixed round, reference velocity MSE and ESS are diagnostics.

Existing cheap scale/shift/context RMS and encoder/modulation gradient logging
remain; `numerical_input_rms` is added. No extra inference, classifier fits or
research probes are introduced. All metrics describe the ordinary latest batch.

## Launch (user submits)

With an existing sixteen-GPU Ray cluster:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_kinematic_adaln.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln/ray_results
```

W&B ID `h4kfilm1`, isolated from control `h4vadaln1`; separate outputs.
No remote job, allocation, upload or W&B run was started by implementation.
The remote source checkpoint was not inspected; it must exist before launch.

## Local validation (2026-09-25)

251 related CPU tests passed, 2 skipped, 29 subtests passed: visible/kinematic
conditioning, existing angular/Fourier integration, OmniFold, model loading,
DGPO trainer and classifier-LR contracts. Checks include exact zero-start
prediction equivalence; supervised/sampling normalization before projection;
masking/empty events and float32/64; nonzero gradients; candidate/truth
independence; legacy/new and two-fold/two-iteration frozen reward payloads;
actual classifier optimizer groups; and lightweight W&B forwarding.
Sixteen-worker configuration is validated, not a sixteen-GPU execution test.
