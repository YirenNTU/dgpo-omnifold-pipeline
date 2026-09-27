# BA55 two-fold/one-repeat DGPO radius diagnostic

## Question

Does the locally useful BA55 H4 reward retain the correct DGPO sign at a
larger policy-parameter radius, consistently across fresh rollout noise, when
the reward uses only the two classifiers required for OOF cross-fitting?

This is a read-only artifact replay. It performs zero classifier fits, zero
policy optimizer steps, installs zero rewards, and leaves the `c4a91e07`
checkpoint unchanged.

## Why this test

The completed full-H4 epsilon sweep improved its fixed judge most at parameter
RMS `1e-6`.  Its response weakened at `3e-6`; at `1e-5` the direction left the
local linear regime and produced little classifier-gap improvement.  The
separate weak-classifier trajectory used RMS `1e-5` and found its largest local
H4-judge improvement at BA55.  That BA55 result used a four-member reward and
one rollout seed.  It does not establish that the wider BA55 step is
reproducible or that extra classifier repeats are needed.

## Frozen contract

- source: completed `aqbszk1r` artifacts in
  `c4a91e07_h4_weak_classifier_trajectory_v1`;
- policy: `c4a91e07` step 1110, weights only;
- reward: saved seed `20260913` H4 fold 1 and fold 2 logits
  (`2 folds x 1 repeat`);
- checkpoints: the saved first states whose held-out BA 95% lower bound reached
  0.55 (fold 1 step 725, fold 2 step 650);
- reward mapping: unchanged raw LOO with tempering 0.75;
- gradient: unchanged production DGPO loss with K=8 and the exact anchor
  reference;
- evaluation: the saved independent H4 judge, previously saturation-certified
  after 1,975 updates;
- events: one fixed 1,024-event panel;
- radii: `1e-6`, `3e-6`, `1e-5`, `3e-5`, and `1e-4` parameter RMS;
- replication: eight fresh common-random-number rollout seeds.

The radius is a normalized read-only parameter perturbation.  It is not called
an AdamW learning rate until a later short training arm measures the actual
native optimizer displacement.

## Primary endpoint

For every radius and rollout seed, compare H4 judge gaps for plus, zero, and
minus.  A radius is reliable when:

- its mean `plus - zero` gap is negative;
- the upper end of its paired 90% bootstrap interval is negative;
- plus beats zero in at least 75% of seeds;
- plus beats minus in all seeds.

Among reliable radii, select the largest mean gap improvement.  A selected
radius above `1e-6` returns `larger_ba55_radius_supported` and licenses a short
two-fold/one-repeat DGPO pilot.  No result from this diagnostic establishes
closed-loop fresh-H4 closure.

## W&B

The run streams every completed rollout-seed/radius point to:

```text
project: ytchou97-university-of-washington/nu2flow-RL
id: h4b55lr1
name: Can a larger policy step escape the plateau? | BA55 two-fold H4 | checkpoint replay
group: H4 policy projection
```

There are no new classifier curves because this run fits no classifier. The
original training curves remain in `aqbszk1r`. The replay uses the dedicated
`actionability/rollout_index` x axis, independent of W&B's transport step.

## Run

Inside the initialized 16-GPU Ray allocation:

```bash
shifter python3 scripts/diagnose_ba55_actionability_replay.py \
  config/dgpo_10pct_c4a91e07_h4_ba55_m2_lr_actionability.yaml
```

Preflight only:

```bash
shifter python3 scripts/diagnose_ba55_actionability_replay.py \
  config/dgpo_10pct_c4a91e07_h4_ba55_m2_lr_actionability.yaml \
  --check-only
```
