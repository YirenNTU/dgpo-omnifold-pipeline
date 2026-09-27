# EveNet-style classifier scheduler

The shared `fit_density_ratio` loop now supports `lr_scheduler: cosine` for
both `dgpo.adaptive_omnifold.recalibration.fit` (OmniFold) and `audit_fit`.
Historical overlays default to `constant` and remain unchanged.

New settings: `lr_warmup_epochs: 1`, `lr_cosine_epochs: 250`,
`lr_min_ratio: 0.1`. The cosine horizon INCLUDES warmup. These are explicit
experiment choices; the 250-epoch horizon follows the user's requested budget.
The warmup and half-cosine shape match EveNet/HuggingFace; unlike finite
EveNet pretraining, the unbounded classifier keeps a 10% LR floor after
epoch 250 and never restarts the cosine. Set the floor to 0 to match EveNet
through its horizon. This does not cap fit steps or change early stopping.

Epochs count actual fit rows / GLOBAL per-class batch (respecting drop_last),
including each cross-fit fold's smaller population. Microbatching and world
size do not change this clock or scale peak LR. All parameter-group ratios
are preserved. Cold fits restart their own schedule; recovery checkpoints
resume from their completed-update cursor and saved initial group LRs.
Loading an old constant-LR recovery into a cosine fit fails explicitly;
do not silently mutate an in-flight experiment.

`learning_rate` in existing logs remains the nominal base LR. The named
`optimizer_group_lr_*` metrics report the LR actually used for that update.
`scheduler_multiplier`, `scheduler_epoch`, `scheduler_warmup_steps` and
`scheduler_total_steps` are retained by compact/critical W&B profiles.
The first update uses LR=0, matching EveNet's HF scheduler convention.

## Classifier-only test (same step-50 checkpoint, 10% data, 16 GPUs)

```bash
shifter python3 scripts/train_h4_classifier_lr_stability.py --scheduler
```

New run `h4clfcos1`, with diagnostic probes enabled and separate output paths.
No policy updates. Compare with constant-LR `h4clfd1`. Both scheduler config
blocks are enabled, but this classifier-only entry point executes only audit.

## Full DGPO (reward OmniFold and audits)

Use `config/dgpo_h4_lastblock_classifier_cosine.yaml` with the generic backend
launcher, as specified in its `nersc.execution.command`. New run `h4lbcos1`.
It preserves its parent DGPO source checkpoint, classifier peak LRs, 2 folds,
1 repeat, 1 iteration, 100 policy updates and audits at 50/100. This is not
the classifier-only step-50 test and does not reduce its parent's peak LRs.
Neither experiment has been submitted automatically.

## Same classifier on the original old-DGPO endpoint

```bash
shifter python3 scripts/train_h4_classifier_lr_stability.py --scheduler --old-checkpoint
```

New run `h4clfold1` uses `c4a91e07`'s
`dgpo-epoch=110-next_ep=111-step=1110.ckpt`, weights-only. It inherits the
classifier-only cosine test unchanged except for the policy source and
provenance/output locations. It keeps the 10% dataset, 16 GPUs, seeds,
architecture, group peak LRs, diagnostics, 250-epoch schedule, minimum 1000
updates and ten-epoch validation BCE patience. It performs no DGPO updates
and does not reuse the trained step-50 classifier or generated sample pool.
Compare final held-out AUC and fit trajectories with `h4clfcos1`; differences
measure discrimination on different policies, not a classifier-design ablation.
The source path is pinned by config; file existence is checked on NERSC
before launch, not by `--validate-only` on a workstation.
