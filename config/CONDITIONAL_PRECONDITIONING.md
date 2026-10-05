# Conditional preconditioning: real Ztautau experiment

One sequential two-stage experiment, **16 GPUs in each stage**, on the existing
Ray allocation. This prepares code and commands only; the user starts the run.

| Stage | W&B ID | Budget | Trainable parameters |
|---|---|---|---|
| Coordinate fit | `condcoordfit02` | 20 epochs | Separate observed-event mean/covariance network |
| Diffusion | `condpre02` | 50 epochs | Existing context-control backbone and FiLM; coordinates frozen |

Overlay: `config/train_diffusion_conditional_preconditioning.yaml`.
Output root: `/pscratch/sd/y/yiren/Ztautau/diffusion_conditional_preconditioning_02`.
Both stages use 4 nodes × 4 GPUs, 2048 events/GPU, the established ordered10%
training split, and a separate validation directory. No allocation is requested
by the launcher. It connects to the user's existing Ray cluster.

## Filtered-data correction (2026-09-29)

Use the **existing** cleaned datasets for every current conditioning arm:

- Train: `/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`
  — 416,701 events; the seven STIC outlier events were already removed.
- Validation: `/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`
  — 119,002 events; the two STIC outlier events were already removed.

Both completed filter manifests and actual parquet row counts are checked before
use; there is no raw-data fallback. The source checkpoint normalization stays
fixed. No dataset is regenerated or overwritten. The shared low-noise parent
now supplies these paths to context, relations, shift-only, pair-attention and
preconditioning configurations. Historical W&B runs retain their original data.

`condcoordfit01` / `condpre01` used unfiltered training inputs. The default now
starts **fresh** as `condcoordfit02` / `condpre02` under the `_02` output root;
do not resume the old calibration or optimizer. No running job is changed by
this local configuration edit. Sync the code before launching the new run.

For a matched physical comparison, use the prepared filtered context control
`config/train_diffusion_relation_context_filtered.yaml`, run ID `relcontext02`.
The older `relcontext01` remains a historical reference because its training
inputs differ. Launch the fresh control on a separate 16-GPU allocation:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_relation_context_filtered.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_relation_context_filtered_02/ray_results
```

## Scientific question and coordinates

Can event-dependent scale and cross-tau covariance improve narrow joint physics
after the condition-injection experiments plateau?

Let `u = existing_invisible_normalizer(y)` for the ordered four coordinates
`[tau A delta theta, tau A delta phi, tau B delta theta, tau B delta phi]`.
Fit `mu(c)` and `L(c)` from observed particles/global conditions only, then freeze:

```
z0 = solve(L(c), u - mu(c))
zt = alpha(t) * z0 + sigma(t) * epsilon
v_target = alpha(t) * epsilon - sigma(t) * z0
generated_y = existing_denormalizer(mu(c) + L(c) @ generated_z)
```

Training, generation metrics, engine prediction and the shared candidate sampler
use the same coordinate adapter. DDIM runs entirely in z, then applies the inverse
once. Normalized/noisy invisible tokens seen by PET are z coordinates. Existing
FiLM scale/shift and the context adapter remain enabled. No pair-attention or
shift-only intervention is mixed into this arm.

The separate coordinate encoder pools masked observed tokens (mean/max), includes
observed global conditions and particle count, and uses sin/cos for observed phi.
It reads no clean invisible target, tau truth assignment, candidate or timestep
to determine coordinates. Targets enter only its supervised Gaussian NLL.
It does not share trainable embeddings with the diffusion backbone: freezing its
parameters really fixes the coordinate system during the second stage.

Four means and ten lower-triangular entries parameterize a full4D covariance:
`Sigma(c) = A(c) A(c)^T + 0.01^2 I`, `L = chol(Sigma)`.
Diagonal entries of A are bounded by5; off-diagonal entries by2. The small4x4
factorization uses float64 to preserve the eigenvalue floor; the networks use
FP32. The floor is in the **existing normalized target chart**, not radians.
It protects the coordinate inverse; it is not a lower bound on final posterior
width. Initial coordinates are identity and initialization preserves shared RNG.
After fitting, the initial physical generator function deliberately changes.

The Gaussian is an auxiliary coordinate estimator. Diffusion learns the full
residual distribution, including non-Gaussian/multimodal structure. The affine
transform does not itself guarantee recovery of curved/thin physics manifolds.

## Fitting, checkpointing and provenance

The coordinate network uses AdamW, LR0.001, weight decay0.0001, gradient clipping5,
seed42 and fixed20 epochs. Only training batches backpropagate; validation Gaussian
NLL selects the best coordinate checkpoint. The held-out joint-coverage panel is
never used to fit or select coordinates. Each epoch uses the same full-batch DDP
step count on all16 ranks, dropping the final incomplete distributed batch.

The fitter saves `calibration/last-fit.pt` with optimizer state each epoch and
`calibration/best-fit.pt` for validation selection. After all20 epochs it publishes
`calibration/calibration.pt` plus `COMPLETE`. Incomplete fits cannot initialize
diffusion. Artifact metadata records the architecture, fit/data protocol, selected
epoch and SHA256 of the pinned raw source.

The source remains
`/pscratch/sd/y/yiren/Ztautau/film_strength_real_01/raw_policy.pt`
(supervised epoch286/global_step22386), identical to `relcontext01`. The strict
loader preserves every shared weight and source normalization buffer; only the
new identity-initialized coordinate estimator and zero-output context adapter are
added. The second stage starts a fresh diffusion optimizer/scheduler. It saves
full-state `last.ckpt` and the best-three validation checkpoints.

All coordinate weights, normalizers and fitted/configuration markers are stored
inside diffusion checkpoints. Full-state diffusion resume needs the matching
preconditioning architecture config but no external calibration artifact.
Weights-only loading cannot silently discard coordinates or their normalizers.
This first experiment supports supervised training and physics evaluation.
DGPO training transfer is explicitly rejected until its loss/reference-coordinate
paths are adapted; the old FiLM-strength diagnostics also assume the old chart.

## Upload and launch

From the Mac, upload the repository with the standing exclusions. The old limited
`sync_relation_film.sh` does not include all files needed for this experiment.

```bash
rsync -avz --progress \
  -e "ssh -i ~/.ssh/nersc" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude-from='/Users/yirenwu/Ztautau/ml_pipeline/NERSC/upload-excludes.txt' \
  /Users/yirenwu/Ztautau/ml_pipeline/ \
  yiren@perlmutter.nersc.gov:/global/homes/y/yiren/ml_pipeline/
```

Within your **16-GPU allocation**, with its Ray cluster and the usual W&B
environment initialized, run from `/global/homes/y/yiren/ml_pipeline`:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/run_conditional_preconditioning.py --stage all
```

This fits coordinates and, only after successful completion, starts50-epoch
diffusion. It does not run both stages simultaneously or request another16 GPUs.
Both stages fail on existing incompatible outputs instead of overwriting them.

Read-only input/configuration preflight (writes only a resolved runtime YAML,
starts no Ray workers):

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/run_conditional_preconditioning.py --stage check
```

If calibration was interrupted after an epoch checkpoint, restore it and continue:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/run_conditional_preconditioning.py --stage all --resume-fit
```

If calibration completed but diffusion has not started, reuse it:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/run_conditional_preconditioning.py --stage train
```

If diffusion was interrupted, restore its full state and finish the original50
epochs (the optimizer/scheduler and coordinate model resume together):

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_conditional_preconditioning.yaml \
  -- --resume_checkpoint /pscratch/sd/y/yiren/Ztautau/diffusion_conditional_preconditioning_02/checkpoints/last.ckpt \
  --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_conditional_preconditioning_02/ray_results
```

An independent rerun needs new output directories and W&B IDs for both stages.

## Evaluation and interpretation

The unchanged callback evaluates before diffusion and at completed epoch50:
1024-event panel, K32, legacy DDIM20, fixed event-wise noise, all draws retained.
Both reports and generated samples are in `checkpoints/joint_coverage/`.

Primary comparison: absolute epoch50 joint-radius W1 minus the fresh same-data
context control `relcontext02` (prepare/run separately). Also examine joint probability gaps at1e-6,1e-5,1e-4 rad,
fine target marginals, CDFs and invalid-direction counts. A W1-only improvement
without narrow-region mass recovery is not resolution of the bottleneck.

Use the existing paired whole-event bootstrap on saved physical samples:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/compare_conditioning_endpoints.py \
  --control /pscratch/sd/y/yiren/Ztautau/diffusion_relation_context_filtered_02/checkpoints/joint_coverage/epoch-0050.json \
  --candidate /pscratch/sd/y/yiren/Ztautau/diffusion_conditional_preconditioning_02/checkpoints/joint_coverage/epoch-0050.json \
  --output /pscratch/sd/y/yiren/Ztautau/diffusion_conditional_preconditioning_02/context_comparison.json
```

The panel is exploratory and reused. Noise is paired in different coordinate
charts, not identical physical perturbations. Compare absolute endpoints, not
improvement relative to different initial physical generators. Whitened velocity
MSE has different units/weighting from the historical baseline and must not rank
the arms. The extra20 coordinate-fit epochs make this a package comparison,
not total-compute matched. A promising result requires independent-panel and
training-seed confirmation, then a fresh physics-classifier audit.

## Local validation

CPU tests cover invertibility, full covariance and eigenvalue floor, observed-only
conditioning, periodicity/masks, Gaussian-fit gradients, frozen second-stage
coordinates, source/artifact provenance, full checkpoint restoration, real EveNet
training/sampling velocity agreement, multi-chain inverse mapping, and existing
shift-only/pair/FiLM behavior. These do not establish real-case performance or
validate a live16-GPU NERSC execution. No jobs were submitted or launched.
