# Coefficient-one classifier KL from step1920

This opt-in pathwise experiment replaces the velocity penalty with a separate
current-policy/anchor classifier. The frozen reward teacher is freshly fitted
once against truth and the source step1920 policy. The KL head is freshly fitted
before each subsequent actor update, using the same416,701 filtered training
conditions, class weights, split and frozen feature map as the reward teacher.
Each class has one generated candidate per conditioning event. Current samples
are the positive BCE class; the fixed step1920 samples are the negative class.
Both logits are unbounded; coefficient1 and identical all-K/event averaging
are fixed. There is no dynamic norm balancing or hard trust-region rejection.

For exact ratios in this common feature space, write the frozen reward as
`r(z,c)=log(p_truth(z|c)/q_anchor(z|c))` and the current KL logit as
`k_t(z,c)=log(q_t(z|c)/q_anchor(z|c))`. The functional objective is
`E_q[k-r]=KL(q||p_truth)` when the current classifier is exact atq. A freshly
fitted KL head is frozen during an actor update. Its input derivative is
propagated through the same complete DDIM trajectory as reward. At the current
policy the missing explicit derivative of the fitted numerator integrates to
`E_q[partial_theta log q]=0`; hence the frozen exact head gives the correct
local KL gradient. The identity requires ratios and derivatives to be accurate.

The implementation estimates **feature-space** KL. The existing context/hidden/
coordinate representation is not proved sufficient or injective, so this does
not certify full-output conditional KL, truth closure or entanglement closure.
Equal class counts remove class-prior corrections; they do not establish ratio
calibration. Head capacity, optimization, finite sampling and fit selection can
all introduce error. Signed empirical mean KL is logged without clipping;
finite estimates can be negative. New KL heads are not installed as the reward,
and never change reward rounds or recenter the anchor.

At the initial current=anchor point, the exact log-ratio and its gradient are
zero. A zero head implements this identity directly rather than fitting noise.
From step1921 onward, the independent head is cold-fit before every update.
Sampling uses eval mode and the same fixed noise seed as startup anchor samples;
fitting and sampling restore actor mode/RNG and never touch actor AdamW or its
scheduler. Selected reward/KL heads in this experiment must each reach1,000
optimizer updates; total training alone is not a sufficient selection gate.
The original full classifier fit budget remains250 epochs with early stopping.
This adds substantial per-update sampling/fitting cost and is not a speed test.

The startup anchor-candidate shards are reused as fixed denominator data.
Classifier-fit artifacts are written to
`mechanisms/classifier-kl/policy-step-XXXXXXXX/`. They record class direction,
event counts, coefficient, current-policy step, anchor step and fit status.
Checkpoints additionally store `trajectory_classifier_kl_state` in the tau
reward-stack payload. Its head was fitted **before** the checkpointed actor
update; it must be refitted before another update. The launcher supports only
fresh continuations from the pinned1920 source, not arbitrary resumed pilots.

The actor keeps16 workers,512 events per rank,K8,DDIM20,mb64, inherited AdamW,
pinned normalization and the existing recorded native batch order. Velocity
regularization is disabled, while the native DGPO surrogate is a forward-only
monitor with zero backward weight. The complete119,002-event filtered validation
panel and predeclared+50/+200/+500 audit endpoints remain unchanged.

## User launch

From the existing NERSC checkout:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py \
  config/tau_full_trajectory_1920_mb64_classifier_kl.yaml trajectory --method pathwise
```

For preflight only, replace `trajectory` with `prepare`. Preflight reads the
source/checkpoint/manifests and writes a runtime; it does not train classifiers
or start GPU inference. The output root is
`/pscratch/sd/y/yiren/Ztautau/tau_classifier_kl_1920_mb64`.

Compare classifier KL with the fresh-unbounded velocity control at matched
updates/events, using Cij total/diagonal/offdiagonal/component closure and the
independent adequately trained audit. Own-head reward improvement alone does
not demonstrate truth closure. Preserve the fixed-energy reconstruction scope.

## Implementation verification

95 CPU tests and40 subtests passed, including exact Gaussian log-ratio
objective cancellation, full20-step DDIM combined-gradient parity, actual
frozen-classifier value/input-gradient parity, fresh-fit label orientation,
minimum selected-fit budget, actor/AdamW isolation, and legacy trajectory
compatibility. Production GPU sampling/fitting has not been executed by the
assistant; user submission remains required.

NERSC prepare-only validation passed: source step1920, full filtered manifests,
16 workers,K8 and stop1970. The runtime disables native reference-trust
regularization and declares classifier-KL refresh every update. No GPU job
was submitted.
