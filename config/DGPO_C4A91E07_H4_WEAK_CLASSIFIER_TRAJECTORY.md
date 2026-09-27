# c4a91e07 H4 weak-classifier trajectory ablation

This fixed-policy experiment asks one question: does deliberately stopping the
H4 reward classifier early produce a DGPO direction that agrees better with an
independent fully trained H4 judge than the fully trained reward classifier?

The policy anchor is the exact `c4a91e07/checkpoints/last.ckpt` at global step
1110. It is loaded weights-only. The run never installs a reward, resumes an
optimizer, or updates the policy with AdamW.

## Controlled arms

Every one of the four reward members follows one cold H4 training trajectory.
The run saves the first checkpoint whose held-out balanced-accuracy 95% lower
confidence bound reaches 55%, 60%, and 70%, then continues the same optimizer
trajectory to the usual best-validation checkpoint. Thus the four arms are:

- `ba_lcb_55`
- `ba_lcb_60`
- `ba_lcb_70`
- `fully_trained`

All arms use the same four members, identity partitions, fixed K=8 candidates,
diffusion times/noise, raw LOO reward mapping, and logit tempering 0.75. Truth
and generated populations are balanced at K=1 during classifier fitting. The
independent H4 judge uses a disjoint fit partition; the final audit partition is
untouched by reward or judge fitting.

## Run

From the repository root on the 16-GPU NERSC Ray allocation:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h4_weak_classifier_trajectory.yaml \
  --check-only
```

Then launch the experiment:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h4_weak_classifier_trajectory.yaml
```

The output directory is created exclusively. To rerun, pass a new path:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h4_weak_classifier_trajectory.yaml \
  --output-dir /pscratch/sd/y/yiren/Ztautau/c4a91e07_h4_weak_classifier_trajectory_v2
```

W&B creates the separate run
`c4a91e07_h4_weak_classifier_trajectory_v1`. Classifier loss, balanced
accuracy, and validation results stream every 25 steps, and each
threshold capture is logged immediately under `trajectory/<member>/<stage>/`.

## Decision rule

For a stage to pass, all of the following must hold on the fixed panel:

1. its centered candidate advantage has positive cosine with the judge;
2. its DGPO policy gradient has positive cosine with the judge gradient;
3. its normalized `+epsilon` step lowers the independent-judge AUC gap relative
   to both zero and the paired `-epsilon` control.

Response matrices, physics metrics, and paired response bootstrap intervals are
reported for every signed endpoint but do not silently replace this declared
direction test. If an early stage passes and `fully_trained` fails, the report
returns `early_stopping_repairs_classifier_to_dgpo_direction`. If all weak
stages fail, deliberate undertraining is ruled out for this local checkpoint.

The complete result is written to `trajectory_report.json` and uploaded as a
W&B artifact. `trajectory_fixed_k8_panel.pt` stores the exact candidates, judge
logits, and all member logits for independent replay.
