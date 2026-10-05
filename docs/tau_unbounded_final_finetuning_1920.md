# Unbounded ratio for final pathwise finetuning

This opt-in ablation fits one fresh classifier against the user-selected policy
at step 1920, then freezes that classifier and the matching step1920 velocity
reference throughout the final trajectory finetuning. The inherited bound30
classifier was trained against step1880; its latent readout is **not** substituted
for an unbounded density ratio. Both new arms cold-fit their classifier with the
same architecture, initialization seed, data, weighted paired BCE, internal
event split and best-validation fit protocol. Only `ratio_bound` differs.

## Inputs and update

- All generation, classifier fitting and actor updates use the existing 16-GPU
  worker group. No nested trainer or new allocation is created.
- Use the complete verified filtered train population of 416,701 events. Each
  truth event has exactly one step1920 generated negative with the same observed
  condition and event weight. Each internal split therefore has equal truth and
  anchor counts and equal class weight.
- The filtered 119,002-event validation population remains disjoint. Preserve
  checkpoint normalization and the frozen raw1110 feature trunk.
- The unbounded head returns its BCE logit directly (`ratio_bound: null`), without
  a ratio cap, logit clipping, reward normalization or reward tempering. Finite
  checks remain. Equal class weights eliminate a class-prior odds correction;
  classifier approximation/calibration and feature limitations still remain.
- Refitting does not update the actor, its inherited AdamW moments or scheduler.
  After an adequately trained finite head is available, install it and copy the
  source1920 actor state into the frozen velocity reference once.
- Actor updates use K=8 terminal scores and full 20-step DDIM backpropagation,
  event microbatch64, inherited AdamW and the shared native recorded batches.
  The loss remains negative mean reward plus coefficient1 velocity reference
  loss (0.5 active-coordinate velocity MSE with native timestep sampling).

This is **not** an exact endpoint KL, a hard trust region or a periodic
finetune/refit cycle. Removing the ratio bound alone does not establish truth as
the global optimum. With an exact unbounded truth/anchor log ratio and exact
coefficient1 endpoint KL, the sum would equal reverse KL to truth; the current
velocity penalty does not supply that identity.

## User launch commands

Run from the existing NERSC repo inside the user's 16-GPU allocation:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920_mb64_fresh_unbounded.yaml trajectory --method pathwise
```

Matched freshly fitted bound30 control:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920_mb64_fresh_bound30.yaml trajectory --method pathwise
```

Each invocation uses a new output directory and W&B ID under
`/pscratch/sd/y/yiren/Ztautau/tau_full_trajectory_1920_fresh_ratio_mb64`.
The shared recorded training data stay at the existing
`tau_full_trajectory_1920/native_training_full_rows_v2` location. Original
inherited-head/ref1, ref5 and dynamic configs are unchanged.

Display names:

- `Does removing the ratio bound help? | pathwise | unbounded | anchor 1920`
- `Does removing the ratio bound help? | pathwise | bound30 control | anchor 1920`

The default endpoint remains +50 accepted updates (1920 to1970), with the
existing 0/1/5/10/20/35/50 measurements and fresh audits at0/+50. Optional
predeclared +200/+500 budgets remain fresh starts from step1920. A startup fit
must satisfy the configured minimum budget and at least1,000 fitting updates.
This launcher still requires the original step1920 source; no general resume
command for an already finetuned checkpoint is added.

## Evidence and review

Use full polarimeter Cij total/diagonal/offdiagonal/component closure and the
independent adequately trained fresh audit to assess transfer. Preserve the
fixed-energy reconstruction scope: direction outputs do not reconstruct full
tau rest frames or establish entanglement closure on their own.

The two arms have different fitted training heads. Their raw rewards and reward
gains are not interchangeable. The artifact cross-arm summary deliberately
does not pair different `reward_refit` contracts as if they shared a reward
evaluator. Spin/audit comparisons must align policy update counts and evaluation
events/noise. Earlier fe1dc5a4/d3e1fdbb runs also differ in teacher and reference
anchor; they are contextual observations, not a clean ratio-bound control.

Startup saves `mechanisms/startup-reward-refit/reward_refit.json` with paired
class counts, fit status and the new anchor. W&B logs
`tau/trajectory_reward_refit/*`; actor checkpoints preserve the installed head's
unbounded/bounded setting and denominator1920. Prepare-only mode validates inputs
and writes a runtime, but does not generate samples or fit a classifier.

Validation on the implementation turn:87 CPU tests and40 subtests passed,
including actual balanced-BCE fitting, score parity above log30, nonzero frozen
classifier input gradients, checkpoint reload, startup fit guards, matching
reference installation and preserved actor/AdamW state. No production training
or GPU performance claim follows from these checks.

## Executed run: a8db4239, early snapshot 2026-10-05 17:49 Taipei

Run `a8db4239` (`Does removing the ratio bound help? | pathwise | unbounded |
anchor 1920`) was running at this snapshot, with training through step1929
(+9) and complete physics measurements through step1925 (+5). Source:
`dgpo_tau_attention_1780/checkpoints/dgpo-epoch=191-next_ep=192-step=1920.ckpt`.
Startup performed a fresh balanced unbounded fit with416,701 truth and anchor
events,5,500 optimizer steps and best validation BCE0.601105, then recentered
the frozen velocity reference to1920. Actor updates remain pathwise with
reference coefficient1. Snapshot evidence is saved in
`artifacts/a8db4239_review/snapshot_20261005_1749.json`.

| Metric on the complete119,002-event validation | +0 | +1 | +5 |
|---|---:|---:|---:|
| Own-head mean reward | -0.585587 | -0.512397 | -0.257396 |
| Cij total error | 0.257478 | 0.308943 | 0.525400 |
| Cij diagonal error | 0.206892 | 0.265419 | 0.475346 |
| Cij offdiagonal error | 0.153267 | 0.158109 | 0.223810 |
| Generated nn (truth=-0.454640) | -0.248042 | -0.189461 | -0.021713 |
| Absolute nn error | 0.206598 | 0.265179 | 0.432927 |

At+5 the paired-bootstrap total-error increase is0.267922 with95% interval
[0.159971,0.347804]; nn-error increase is0.226329 with interval
[0.144539,0.300262]. Reward gain is0.328190 with interval
[0.323321,0.333393]. These support reward transfer accompanied by worsening
spin-observable closure on the measured early trajectory. The latest training
velocity penalty is0.071045; this is an active soft velocity-MSE penalty,
not a certified distribution KL or hard trust boundary.

Baseline fresh-audit AUC is0.595105 after1,120 total fitting updates, but the
selected best head is from700 updates; no post-update audit has yet been
recorded. The predeclared+50 audit/physics endpoint remains unresolved.
Removing the ratio bound alone is not identified causally: a new teacher and
reference recentering accompany this intervention, and the fresh bound30
matched-control result has not been established by this snapshot. Earlier
inherited-head runs must not be compared through raw reward values.
