# Fixed step1920 classifier-KL trust-region ablation

Status: implemented; prepare-only and import checks do not constitute a GPU
training run. The user submits the 16-GPU job personally.

Question: does a hard acceptance gate on estimated cumulative anchor KL reduce
reward-driven drift beyond the coefficient-one classifier-KL soft objective?
Matched control: `config/tau_full_trajectory_1920_mb64_classifier_kl.yaml`.
Only the hard gate and output directory differ in the new settings file.

## Objective and feasibility

Use the same unbounded, frozen truth/step1920 reward and the same feature map
for the freshly fitted current/step1920 KL classifier. With exact ratios,

```
r(z,c) = log p_truth(z|c) / q_anchor(z|c)
k(z,c) = log q_current(z|c) / q_anchor(z|c)
L = E_current [k-r] = KL(q_current || p_truth)
```

The coefficient remains exactly 1. Both components use the same candidates,
all-K/event averaging and full 20-step DDIM input VJP. Velocity penalties and
dynamic gradient balancing are disabled. The additional constraint is

```
KL(q_current || q_anchor) <= 0.01 nats
```

Here the conditioning marginal is the positive base-event-weight population.
It is an average over conditioning events, not an individual classifier score
bound. The anchor is permanently fixed at the start of final finetuning,
step1920; it is not recentered after acceptance. If truth is outside this
region, the constrained optimum cannot equal truth. This experiment preserves
the soft objective while deliberately restricting its feasible distributions.

## Executable acceptance rule

The finite classifier implements an estimated **feature-space** mean KL, not
an exact full conditional KL certificate.

1. Form one ordinary clipped-gradient AdamW proposal from the coefficient-one
   pathwise objective. Retain the incumbent parameters, buffers and AdamW
   moments/counters until the proposal passes.
2. For the actual tentative actor, generate one draw for each of the complete
   filtered 416,701 training conditioning events. Use the same fixed training
   noise and paired anchor samples as the control. Fit current as positive,
   step1920 anchor as negative, with identical class counts, conditioning,
   base weights and internal splits. Select only heads with at least 1,000
   classifier optimizer steps; preserve the existing longer fit budget.
3. Generate one independent draw for each of all 119,002 filtered validation
   events, disjoint from the classifier training/selection population. The
   same validation seed is used across a proposal's five predeclared scales;
   the seed changes across policy proposals. Evaluate only current features
   with the proposed actor's newly fitted head. Truth candidates never enter
   this trust estimate.
4. Let `mu = sum(w_i * log_ratio_i) / sum(w_i)`. Compute the weighted event
   standard error, with effective-event correction. Accept only when
   `0 <= mu + z*SE <= 0.01`. `z = Phi^-1(1-0.05/5)` gives a large-sample,
   one-sided 95% upper bound with Bonferroni adjustment over this proposal's
   five-scale grid. Keep all signed logits: negative individual logits are
   valid, but an entirely negative upper bound is rejected as an invalid KL
   estimate. No logit clipping or taking an absolute value is used.
5. Test scales `1, 0.5, 0.25, 0.125, 0.0625` on the **same AdamW displacement**.
   Each tentative policy gets a new adequately trained head. A stale incumbent
   head cannot approve a changed policy. The exact parameter point scored is
   retained on acceptance; there is no subsequent repair or interpolation.
6. Reuse the accepted head for the next gradient update. This is equivalent
   to the control's next refit: same accepted actor, training samples, noise,
   fit seed, selection contract and feature map. Keep one AdamW moment update
   and one scheduler/accepted-policy tick. Shrinking displacement is equivalent
   to reducing all AdamW group learning rates for that proposal.
7. If every positive scale fails, restore incumbent parameters/buffers and
   all AdamW moments/counters. Do not tick the scheduler or accepted policy
   clock. Save the restored checkpoint, perform an additional cold audit and
   full K8 physics evaluation at that incumbent, then finish the pilot. This
   early stop is explicitly an **incomplete predeclared endpoint**, not +50.
   Fit/scoring exceptions also roll back the proposed actor/AdamW state, then
   propagate the error; they are not silently accepted.

The SE interval covers event sampling conditional on the fitted classifier;
it excludes classifier approximation/calibration bias and adaptation across
updates. Equal class sizes remove class-prior offset, not these biases.
Underfitting or a compressed feature map can underestimate true distribution
KL even when the numerical gate passes. The validation set is used for
acceptance and the existing physics/audit diagnostics, so it is not a pristine
final test set. No inference of exact KL follows from AUC near 0.5.

## Data, state and endpoints

- Source: `/pscratch/sd/y/yiren/Ztautau/dgpo_tau_attention_1780/checkpoints/dgpo-epoch=191-next_ep=192-step=1920.ckpt`.
- Preserve its normalizers, inherited AdamW and LR scheduler. Startup refits
  the truth/anchor reward once and explicitly establishes the matching frozen
  step1920 reference. Subsequent trust fits never install a reward head or
  advance reward-round clocks.
- Filtered train: `/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`.
- Filtered validation: `/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`.
- Keep 16 GPUs, per-GPU batch512, K8 policy updates, event microbatch64,
  DDIM20, original replay, predeclared +50/+200/+500 budgets and measurement
  points. Completed filter manifests remain required by the launcher.
- Assess fresh independent H4 audit at the declared endpoint and polarimeter,
  Cij and entanglement-observable closure for physics usefulness. Own-head
  reward is supporting evidence. Direction-only fixed-energy tau reconstruction
  retains its existing limitation and is not a full tau-rest-frame analysis.

Saved `mechanisms/classifier-trust/from-step-*/trust_proposal.json` records all
attempts, fits, signed means, SE bounds, scale and acceptance. Accepted-policy
heads and the latest trust report are included in the reward-stack checkpoint
payload. Rejected heads never become the current-gradient critic. The launch
remains explicitly pinned to source1920; continuing a later trust checkpoint
requires a separately supported resume configuration.

Key W&B series:

```
tau/classifier_trust/accepted_mean
tau/classifier_trust/accepted_upper
tau/classifier_trust/max_kl
tau/classifier_trust/accepted_scale
tau/classifier_trust/trials
tau/classifier_trust/optimizer_state_restored
tau/classifier_trust/stopped
tau/classifier_trust/accepted_updates
tau/classifier_trust/trial/*
tau/classifier_trust_fit/*
```

Classifier fitting adds compute. A first accepted candidate avoids a duplicate
next-step refit; every failed backtrack requires another fit/generation.
There is no claim of equal wall time to the soft control.

## Launch inside the user's existing 16-GPU allocation

```bash
cd ~/ml_pipeline
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py \
  config/tau_full_trajectory_1920_mb64_classifier_kl_trust001.yaml \
  trajectory --method pathwise
```

Default cap is 50 accepted updates, step1920 to1970, or an earlier explicit
trust-boundary stop. No ensemble selection is required. Prepare-only:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py \
  config/tau_full_trajectory_1920_mb64_classifier_kl_trust001.yaml \
  prepare --method pathwise
```

Validation completed 2026-10-05:

- CPU suite: **110 passed, 43 subtests passed**, including real AdamW
  displacement equivalence, complete rollback, decision-publication failures,
  accepted-head reuse, held-out scoring and the production scheduler/terminal
  control blocks. Production modules compile and `git diff --check` passes.
- NERSC prepare-only validates both completed filtered manifests (416,701 train,
  119,002 validation), source1920, workers16, maxstep1970 and maxKL0.01.
  Prepared runtime:
  `/pscratch/sd/y/yiren/Ztautau/tau_classifier_kl_trust001_1920_mb64/trajectory-20261005T102339-j2950n8w/runtime.yaml`.
- Production trainer/controller/critic/trust imports and resolved configuration
  pass inside `registry.nersc.gov/m2616/avencast/evenet:1.3`.
- Code/configuration/documentation synced to the existing NERSC checkout using
  the required upload exclusions. No allocation, remote policy generation,
  training or job submission was started. Distributed GPU behavior and physics
  outcomes remain untested until the user runs this ablation.
