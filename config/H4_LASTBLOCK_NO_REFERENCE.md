# One-iteration H4 DGPO pilot without reference trust

Run ID: `h4lbnr01`. The first attempt completed classifier fitting but failed
before policy updates: the one-iteration cap still required residual closure.
The corrected configuration explicitly installs one accepted increment.

- Same 10% data lineage and pinned DGPO step-1110 policy; weights-only start.
- Last-block Fourier classifier from the classifier test: internal adapters,
  final PET block and classifier heads train; earlier pretrained blocks and
  input projectors stay frozen. Head/topology dropout 0.25, AdamW decay 0.001.
- One reward bootstrap: **2 folds x 1 repeat x 1 iteration**. No residual
  stacking, repeated ensemble, or policy-loop refits.
- Freeze installed reward and likelihood-ratio reference after bootstrap.
  Beta=1, fixed tempering alpha=1, additive reference trust disabled, no hard
  boundary. Native policy AdamW settings remain unchanged.
- 100 policy updates (10 policy epochs); no step-0 cold audit, including hidden
  bootstrap baseline fits. Score the unchanged frozen ensemble and each fold
  at steps 0/50/100 on the same external validation event/noise panel.
  Cold audits at steps 50 and 100 are secondary diagnostics on a disjoint final set.
- Both reward folds and audits have no maximum step/epoch budget. Train at
  least 1000 optimizer updates, then stop after ten classifier epochs without
  validation-BCE improvement (min delta 0.0001); validate once per actual
  fold/audit epoch and restore the minimum-validation-BCE checkpoint.
- Keep W&B classifier curves, policy update metrics and installed/fresh
  gradient-direction diagnostics at 50/100. No TARP or staleness controller.
- Joint-topology diagnostics are enabled for future runs at steps 0/10/.../100:
  `val_ztautau/jsd/current/topology/{cos_opening,delta_phi_to_pi,back_to_back_loss,calibration_deltaR_sum}`,
  with truth/current/frozen-reference overlays and per-leg target marginals.
  They use candidate zero, never the reward-best candidate. These are
  physically motivated joint projections, not an exhaustive joint-distribution test.
  The completed h4lbnr01 run did not record these panels; enabling them cannot
  reconstruct its missing historical values.

W&B gradient panels (policy clock is `global_step`, not classifier updates):

- Primary: `frozen_classifier/ensemble/auc` and `/auc_gap`, with
  `frozen_classifier/fold01/auc` and `fold02/auc` to separate member behavior.
  These scores train no classifier and do not update the installed reward.

- Every policy update: `train/grad/global_norm_pre_clip`,
  `train/grad/clip_active`, and `train/parameter_update_rms` (actual AdamW
  displacement, not inferred from gradient magnitude).
- At 50/100: `gradient_conflict/omnifold/norm`, `staleness/norm`,
  `omnifold_staleness/cosine`, `omnifold_staleness/conclusive`, and each
  gradient's `split_cosine`. Interpret alignment only with reliability.
- Classifier fit charts retain gradient/clipping diagnostics on their own
  classifier-update clock. These are not policy-gradient measurements.

These probes are measurement-only; no new gradient transformation or
optimizer control is applied. Raw-gradient cosine is not an AdamW-update
cosine and does not, by itself, establish distribution improvement.

## Retained classifier for later ablations

Immediately after the initial OmniFold bootstrap succeeds—and before any
policy update—the trainer writes the unpruned checkpoint:

```text
/pscratch/sd/y/yiren/Ztautau/h4_lastblock_no_ref_100step/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt
```

Its `dgpo_omnifold_reward_stack` payload contains the complete installed
two-fold, one-repeat, one-iteration classifier stack. Later ablations should
pin this explicit snapshot rather than `last.ckpt`, whose symlink advances
during policy training. Keep the whole two-fold stack; the first-fold held-out
metric is only the initial diagnostic reference, not the installed reward by
itself.

Compare frozen-classifier `abs(AUC-0.5)` at steps 0/50/100. Keep the fresh-audit
series separate: there is no step-zero cold-audit baseline. Inspect audit learning curves:
a weak/underfit judge cannot establish closure. This trajectory changes classifier design
relative to older trust-on runs, so it cannot isolate trust removal causally.

Within the existing NERSC Shifter/Ray allocation (4 nodes, 16 GPUs):

```bash
shifter python3 scripts/train_dgpo_h4_lastblock_no_ref.py
```

Add `--validate-only` for a read-only configuration check. Outputs use the
separate `h4_lastblock_no_ref_100step` scratch directory. The launcher does
not allocate nodes, start a Ray cluster, stop existing runs or submit a job.
