# Step1920: can the refitted reward be absorbed and retained?

Run in the user's existing sixteen-GPU Ray allocation and Shifter environment:

```bash
shifter python3 -u scripts/diagnose_tau_reward_transfer.py config/tau_reward_transfer_1920.yaml
```

Preparation without launching the trainer or connecting to Ray:

```bash
shifter python3 -u scripts/diagnose_tau_reward_transfer.py config/tau_reward_transfer_1920.yaml --prepare-only
```

Each invocation creates a fresh `probe-*` subfolder under
`/pscratch/sd/y/yiren/Ztautau/tau_reward_transfer_1920`, pins the exact source
checkpoint and production runtime, and creates a new readable W&B run. Nothing
in the production checkpoint/output folder is overwritten. The preparation
prints the actual source/head/reference clocks, preserved optimizer LRs, output
folder and full trainer command. No allocation or job submission is performed.

## What is fixed, and what changes once

The exact source is
`dgpo_tau_attention_1780/checkpoints/dgpo-epoch=191-next_ep=192-step=1920.ckpt`.
Its path and resume counters are verified by W&B run `6d8bb6c2`; its current file
availability and contents are checked on NERSC when the command runs. We clone
the full resolved `dgpo_tau_attention_1780/runtime.yaml`, not an old H4 overlay.

The source is expected to contain the original no-policy-Fourier diffusion,
three-block FiLM plus cross-attention tau classifier, and a reward fitted at
step1880 (round20). The launcher records the actual saved denominator/round;
it does not mislabel the inherited head as freshly fitted at step1920.

At startup, before any actor update, fit **one** fresh best-internal-validation
cross-attention classifier on current step1920 K1 negatives from the filtered
416701-event training population. Score inherited and fresh heads on identical
held-out candidates, then install the fresh head and recenter the velocity-MSE
reference to the unchanged step1920 actor **once**. Freeze this head and
reference for the next 50 native optimizer updates. No periodic reward refit or
reference recentering occurs inside this window.

Full actor AdamW moments and cosine-scheduler clock are inherited. Architecture,
raw weights, pinned normalization, label0 process condition, K8, DDIM20,
unscaled leave-one-out advantage, eight timestep draws and the original
`t in [0,0.7]` distribution are unchanged. Keep velocity-MSE reference
coefficient **1**. This is a counterfactual continuation, not a bitwise replay:
the historical checkpoint did not store its RNG/data-iterator state.

The launcher requires the existing completed filtered manifests for train
416701 events and external validation119002 events. It never falls back to raw
data. Classifier fitting retains 1024 paired events/GPU, minimum1000 fit steps,
250-epoch horizon, best-validation BCE selection and ratio bound30. External
validation is not used to refit/select the reward.

## Measurements and interpretation

Evaluate at relative updates **0, 1, 5, 10, 20, 35, 50**, corresponding to native
steps1920–1970. Evaluation uses the complete119002-event external validation
panel and allK8 candidates, with common generation seeds. Report paired
event-bootstrap intervals from300 resamples; candidates are not treated as
independent events.

The primary endpoint is the **paired all-K held-out mean reward change at +50**
under the same frozen installed head. Best-of-K is supporting information, not
the criterion for absorption. Record reward mean/median/best/worst and
within-condition spread/ESS, plus raw unweighted Cij total/diagonal/offdiagonal
and component errors. These Cij diagnostics use the existing matched
fixed-energy tau reconstruction and TT2L convention; they are not full
tau-energy unfolding or a direct entanglement measurement.

Fresh best-validation classifiers at0 and50 are **diagnostic audits only**:
they never replace the reward or reference. Compare held-out BCE/AUC to test
whether distribution differences shrink; no near0.5 interpretation is valid
without verifying the recorded fit budget. Native gradient tracing at the
positive endpoints records main/reference gradient alignment and actual AdamW
displacement at fixed coefficient1. Some gradient keys retain the legacy `h4`
name in the shared diagnostic code; the actual reward backend is tau, not H4.

Decision rules:

- Fresh-head fixed-candidate reweighting does not improve Cij: do not blame
  diffusion absorption alone; the new ratio itself has a physics tradeoff.
- Held-out frozen reward rises and stays improved: native updates can absorb
  this head; inspect Cij/fresh audit before claiming physics closure.
- Initial reward gains disappear: local transfer exists but accumulation or
  retention is failing under a fixed teacher/reference.
- Native surrogate or gradients improve while independently regenerated
  reward does not: investigate the surrogate-to-generation interface.
- Reward improves but Cij worsens: reward uptake is not equivalent to spin
  closure; retain the component-level tradeoff rather than declaring success.

Head replacement and reference recentering are a joint startup intervention.
This test cannot causally distinguish their individual effects. Training stops
after exactly50 successful native updates through `--max-steps1970`; the
inherited1500-epoch cosine horizon is not reset or shortened.

Local checks cover source-state/config guards, alias-safe new W&B logging,
absolute step cap and prepare-only no-launch behavior. Real sixteen-GPU runtime
integration still requires the user's production environment; no remote
training is started by implementation or local tests.
