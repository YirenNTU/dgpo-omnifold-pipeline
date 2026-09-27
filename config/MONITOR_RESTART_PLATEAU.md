# Cold versus warm classifier restart, trained to plateau

Run `scripts/test_monitor_restart_plateau.py config/dgpo_10pct_monitor_restart_plateau.yaml`.
Use `--check-only` for read-only preflight before starting Ray/W&B.

This is the old clean 128-wide, one-decoder raw monitor on the verified step260
replay pool. It does NOT load the latest policy, train diffusion, install a reward,
or change any production checkpoint. A dedicated output directory and fresh W&B run
are used. Existing output is refused.

Two seeds each compare A_cold with C_full_warm. Both use the same frozen pretrained
backbone, trainable scope, AdamW configuration, dropout, shuffled batch sequence,
fixed train/validation identities and generated samples. Cold initialization uses
the original pretrained values for trainable body modules and fresh classifier banks;
warm initialization uses the complete saved monitor, including its decoder/output,
GroupedSequentialEmbedding, invisible projector and PET adapters. Optimizers start fresh.
The saved warm monitor must first reproduce its verified replay AUC within 0.005.

Training uses the historical old TRAIN fold, never the complement of the common
validation set. The established replay has approximately 199,754 training events and
10,041 common validation events; actual counts and identity hashes are checked.
All pretrained backbone and normalization files are fingerprinted and rechecked.

Each arm has a minimum 1,000 updates, validation every 40 updates, and patience of
10 validation evaluations with BCE min-delta 0.0001. It restores the numerically best
validation checkpoint. The cap is 6,000 updates; reaching the cap without plateau
is explicitly inconclusive, not evidence against cold initialization.

W&B has separate seed/arm axes for fit losses, sampled live metrics, and selected-best
metrics. Saved logits allow paired event-wise BCE comparisons. Recorded ESS and
weight tails use exp(raw logits), with no clipping; these are ratio diagnostics,
not a demonstration that this monitor would make a better reward.

Each seed writes selected checkpoints, logits, summaries, and training duration.
The root summary calls cold restart promising only if BOTH seeds reached plateau,
cold BCE beats warm by >0.002, and cold AUC is no more than 0.005 worse in BOTH seeds.
These are screening thresholds, not proof. Reported paired normal intervals reuse
the validation data used for selection and are labeled exploratory.

Historical validation is not untouched test data. This experiment tests classifier
optimization at one fixed policy, not the cause of a 5%-versus-10% DGPO difference.
A promising result needs confirmation at another policy point, followed by matched
DGPO experiments evaluated with common judges and physics observables.

On NERSC with the existing 16-GPU Ray allocation:

```bash
shifter python3 scripts/test_monitor_restart_plateau.py \
  config/dgpo_10pct_monitor_restart_plateau.yaml --check-only
shifter python3 scripts/test_monitor_restart_plateau.py \
  config/dgpo_10pct_monitor_restart_plateau.yaml
```

Output: `/pscratch/sd/y/yiren/Ztautau/raw_monitor_restart_plateau_step260_v1/summary.json`.
The running residual diagnostic hashes shared scripts; use a separate checkout or
wait until it finishes before syncing changes to its source files.
