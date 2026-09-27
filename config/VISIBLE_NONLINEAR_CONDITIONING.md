# Visible-only nonlinear AdaLN conditioning

The new branch is opt-in; existing configs/checkpoints keep their old function.
`network.VisibleConditioning.diffusion_enabled` and `classifier_enabled` are
independent switches (default false). `width: 64`, `harmonics: [1, 2, 3, 4]`.

An optional angle-plus-energy/pT variant is now in `dgpo_h4_kinematic_adaln.yaml`;
see `KINEMATIC_NONLINEAR_FILM.md`. This angle-only config is unchanged. The new
variant does not Fourier-expand auxiliary detector features or global inputs.

Observed raw particle eta -> theta, and phi -> Fourier sin/cos features. For
each valid visible particle, concatenate these with a normalized copy of its
existing **pre-PET** projected features. Two SiLU MLP layers, masked learned
attention pooling, then another nonlinear event encoder produce the context.
The count of valid visibles is supplied to the event encoder. Variable particle
multiplicity is supported; empty visible events have exactly zero modulation.
No candidate, invisible truth, label, or physics target enters this encoder.
The projected features retain the existing observed particle-identity inputs;
pooling does not invent particle slots or average raw angles across the seam.

Separate zero-initialized linear heads map the context to per-layer delta-scale
and delta-shift. The encoder is normally initialized. There is no second new
zero gate. The formula is `LN(h) * (1 + delta_scale) + delta_shift`, composed with
the existing modulation where present. Existing LayerNorms are not replaced.

- Diffusion: modulate the attention and FFN normalized inputs in every existing
  TruthGeneration block, only on valid invisible slots. Existing timestep/global
  conditioning, PET/additive angular path, masks, layer scales and residuals stay.
  Both supervised forward and sampling/DGPO/reference velocity use this path.
- Classifier: add deltas to the existing candidate decoder's three AdaLN
  scale/shift pairs; preserve its GlobalEmbedding-driven residual gates and H4
  candidate-pair late fusion. This is **not** `topology_conditioning: true`.
  Fresh reward folds and cold audits use independent branch parameters.
- Only new modulation heads are zero initialized: attaching to an existing
  model initially leaves its function unchanged. The encoder's first gradient
  is zero until these heads move. Fresh classifier output heads and existing
  AdaLN gates also start at zero, so their usual opening phase still applies.

## Optimization and diagnostics

Classifier branch has a named `visible_conditioning` AdamW group, using the
existing fresh-head learning rate (2e-4 in this overlay), with the same scheduler
and weight decay. Existing decoder/adapter/body rates remain unchanged. Diffusion
branch belongs to TruthGeneration, using the existing policy optimizer in DGPO
or the generation optimizer in supervised training. A separate supervised group
can target `TruthGeneration.visible_conditioning`; nested ownership is supported.

Three cheap forward diagnostics: context RMS, modulation scale RMS, shift RMS.
Classifier progress and DGPO also report encoder/modulation gradient norms after
clipping. These do not add classifier fits, inference passes, or probe batches.
They describe the latest forward batch, not a fixed held-out diagnostic panel.
Classifier forward RMS values are averaged across ranks at progress reports.

## Checkpoints

The supplied DGPO overlay is a **fresh weights-only** run from the same raw
`diffang10lr5e4/checkpoints/last.ckpt` used by the preceding visible-Fourier
experiment. Its remote existence/content must be checked on NERSC before launch.
It fits fresh classifiers because their requested architecture has changed.
Do not resume an old optimizer with a different parameter layout.

Frozen classifier payloads own their saved branch configuration: an old payload
remains old even if the new run enables classifier conditioning, while a new
payload reconstructs its branch even if the live setting is off. The policy,
frozen reference, folds and audits never share the new branch's live parameters.
Keep the same switches/schema for full-state continuation of this new run.

## Launch (user submits; no job started by implementation)

With the existing 16-GPU Ray allocation already running:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_visible_adaln.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_visible_adaln/ray_results
```

Retains two iterations, two folds, one repeat, refits every 20 policy epochs,
cold audits every 5, physics validation every 10, and velocity MSE coefficient 1.
No EMA. Both conditioning paths change in this run; better results alone cannot
identify which side caused improvement. For a causal follow-up, compare separate
diffusion-only and classifier-only switches with matched samples/checkpoints.
Do not reuse this run ID/output directory for different switch combinations.
Relation-latent decoders are deliberately unsupported by this new branch.

## Local validation (2026-09-25)

239 related CPU tests passed, 2 skipped: the new branch tests plus existing
Fourier integration, angular conditioning, OmniFold, model loading, DGPO trainer,
and classifier-LR tests. This includes actual supervised/sampling entrypoints,
nonzero gradients through both new encoders, empty/padded inputs, unchanged
zero-start predictions, legacy/new payloads, and frozen two-fold/two-iteration
reward round trips. Optimizer coverage and lightweight W&B forwarding are tested.
The overlay resolves to 16 GPU workers; no multi-node GPU execution was performed.

A broader run also exposed the existing, independently reproducible
`TestBestDecayTrustRadius.test_best_point_restart_opts_inherited_reference_in_at_age_zero`
failure in the untouched adaptive-trust module. This overlay keeps that adaptive
boundary disabled; no unrelated trust logic was changed to make the suite green.
## Three-block variants

The original configuration above remains a one-block control. New three-block
diffusion and OmniFold/audit variants for both angle-only and angle+momentum
conditioning are documented in [NONLINEAR_CONDITIONING_DEPTH3.md](NONLINEAR_CONDITIONING_DEPTH3.md).
