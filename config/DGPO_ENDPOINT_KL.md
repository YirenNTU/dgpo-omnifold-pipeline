# Endpoint conditional KL switch

This experiment adds an online estimate of
`E_c KL(q_current(x|c) || q_reference(x|c))` to the existing DGPO training
loss. `x` is the complete generated candidate, and `c` contains the same
visible event inputs as H4. It uses raw deterministic DDIM endpoints.

## Run on NERSC

From the repository, with the existing 16-GPU Ray allocation running:

```bash
shifter python3 scripts/train_dgpo_endpoint_kl.py --endpoint-kl on
```

Switch off (no KL critic, extra sampling, or KL backward):

```bash
shifter python3 scripts/train_dgpo_endpoint_kl.py --endpoint-kl off
```

Optional coefficient and run/output overrides:

```bash
shifter python3 scripts/train_dgpo_endpoint_kl.py \
  --endpoint-kl on --coefficient 0.1 \
  --run-id h4epkl01 \
  --output /pscratch/sd/y/yiren/Ztautau/h4_endpoint_kl_01
```

`--validate-only` prints the resolved switch and destinations without launching.
Default on/off runs use distinct W&B IDs `h4epkl1on` / `h4epkl1off` and
directories `h4_endpoint_kl_on` / `h4_endpoint_kl_off`. They both start raw
step1110 weights with fresh experiment state. This is a switch, not an automatic
two-arm launch. Give a new ID/output when starting another independent run.

Any compatible DGPO YAML can use:

```yaml
dgpo:
  endpoint_kl:
    enabled: true  # false, or omit the section, to disable
    coefficient: 1.0
```

The supplied config inherits the 100-policy-step H4 pilot: 10pct train,
cleaned validation, two-fold/one-repeat/one-iteration frozen H4 reward,
raw generation, no velocity/path trust and no legacy denoising anchor.
OmniFold and fresh audits now share the latest `h4ratio1` classifier setup:
last PET block + internal adapters + H4 Fourier encoder/late fusion, fixed
training-fold Fourier output standardization, dropout 0.25, and AdamW decay
0.001. Constant LRs are backbone 1e-5, adapters/entire decoder 5e-5, other
heads (including Fourier encoder/fusion) 2e-4. There is no direct-logit bypass,
small standalone model, rest-frame branch, or added theta-pair feature.

Per the latest user instruction, each OmniFold fold and each fresh audit has
a **300-update minimum**, 3000-update maximum, ten-epoch validation-BCE
patience and min_delta 0.001. The strict best BCE checkpoint is restored.
The OmniFold maximum is explicit per fold, not halved by cross-fit scaling.
This changes the minimum from h4ratio1's 1000; a 300-update fit is not
automatically saturated, and near-chance AUC alone does not establish closure.
Cold audits and policy validation occur every ten DGPO epochs. With ten
policy updates per epoch and a 100-update run, this means step 100 only;
initial baseline validation remains, with no initial cold audit. The
gradient-conflict probe is aligned at step 100. Classifier-internal validation
still occurs every classifier epoch for early stopping. Grouped LR/update,
gradient, and representation diagnostics are enabled in both fitting paths;
classifier-only panel/snapshot/ratio-export directories are not reused.
The separate online endpoint-KL critic retains its own 200/20 update budgets
and optimizer; it is not the truth-vs-generator audit classifier.

The critic is initialized from the first installed H4 fold with an independent
body/bank and zeroed scalar output. Its new balanced labels are current=1 and
reference=0; it never modifies the installed truth-vs-reference classifier.
The same frozen round reference that generated the reward bootstrap population
is used for the KL samples. Refits/rollback/EMA generation are disallowed for
this initial protocol so that reference provenance cannot drift.

## Estimator and gradient

Each policy step selects disjoint local event groups: 128 fit, 32 validation,
16 actor events by default. Both classifier labels use the same condition
distribution. Current and reference draws use independent random noise.
The first critic fit takes 200 optimizer updates; subsequent fits take 20.
These are exploratory online budgets, **not a claim of critic saturation**.
Validation every 10 updates selects the best BCE checkpoint, including the
pre-fit model. Critic parameters and its AdamW state are restored together.
Fit budgets, event counts and learning rates are configurable in
`dgpo_endpoint_kl_10pct.yaml`.

The critic estimates `log(q_current/q_reference)`. During the actor phase its
parameters are frozen, but its input gradient propagates through candidate
normalization and all DDIM steps. Activation checkpointing and event
microbatches bound activation memory. No truth values are passed into sampling.
The actor loss is `coefficient * mean(current_logit)` on fresh samples and
event identities excluded from that step's critic fitting/selection. There is
no clamping of negative estimates: finite-capacity or finite-sample estimates
may be negative even though true KL is nonnegative. Suppressing them would
hide estimator error and alter the gradient.

With an exact current/reference log ratio, holding the critic fixed gives the
correct pathwise KL gradient under the usual differentiability/integrability
conditions (the expected explicit density-score derivative is zero).
With a fitted critic it is an approximate plug-in gradient. Its BCE,
normalization identity and input sensitivity must be monitored.

**The existing DGPO main term remains a velocity-loss surrogate with a
detached gate. Adding this endpoint estimator does not make the combined loss
an exact distribution-level `E[reward] - KL` objective.** In particular,
coefficient=1 is an experimental relative loss scale, not a theorem that the
current implementation now has truth as its exact optimum.

## W&B

All `endpoint_kl/*` metrics survive both simplified and critical logging:

- `estimate`, `weighted_loss`, `actor_logit_std`: current-policy KL estimate
  and actual scalar added once per policy update, not per diffusion timestep.
- `train_bce_last`, `validation_bce`, `validation_bce_before_fit`,
  `validation_balanced_accuracy`, `selected_fit_step`, `fit_updates_total`:
  whether the critic tracks current/reference differences and generalizes.
- `validation_log_mean_ratio_reference`: log E_ref[exp(logit)], ideally 0;
  this is a calibration diagnostic, not a sufficient accuracy test.
- `input_gradient_norm`: whether the critic produces a usable endpoint gradient.
- `mean_rank_main_gradient_norm`, `mean_rank_kl_gradient_norm`,
  `mean_rank_gradient_cosine`: mean local-rank gradient geometry before policy
  all-reduce. These are **not** norms/cosines of the global mean gradient.
- `coefficient`, both learning rates, sample counts, DDIM steps,
  `reference_round`, `policy_step`, `seconds`.

If a critic remains indistinguishable from the balanced null, a small KL
estimate does not prove that current and reference distributions match.
Likewise, a large negative estimate or a drifting reference normalization
diagnostic indicates an unreliable critic. Judge this together with unchanged
frozen-H4 reward mean and classifier measurements, not physics metrics alone.

## Distributed execution and recovery

Unbounded full-state continuation (no step or epoch limit):

```bash
shifter python3 -u scripts/train_dgpo_endpoint_kl.py --endpoint-kl on --resume
```

The trainer uses an open-ended epoch iterator, not a large artificial maximum.
It stops on manual interruption, job walltime, or failure. Existing periodic
checkpoint saving and audit/validation every ten logical epochs remain active.
An abrupt stop may lose updates since the last saved checkpoint.

Full-state continuation of the existing run to **1000 total updates** (not
1000 additional updates):

```bash
shifter python3 -u scripts/train_dgpo_endpoint_kl.py --endpoint-kl on --resume --total-steps 1000
```

This reads `h4_endpoint_kl_on/checkpoints/last.ckpt`, checks that critic and
optimizer/reference/reward state exist, disables bootstrap, and continues the
same W&B ID. Run only after the previous process stops. The endpoint is an
explicit user choice; the example does not claim that longer training is
validated by the pilot. Audit/validation remain every 100 policy updates.

All ranks start from synchronized critic weights, average critic gradients,
and select checkpoints using globally averaged validation BCE. Local event
budgets are equal. Policy DGPO and endpoint gradients are accumulated locally
then averaged together once before clipping and AdamW. This supports arbitrary
world sizes, including the configured 16 GPU workers.

Checkpoints include `dgpo_endpoint_kl_state` (critic, critic optimizer, config,
fit counters, reference round) along with existing policy/reference/reward
state. Full resume restores this state lazily after the reward is available.
Switching the coefficient or enabled flag is allowed; changing critic settings
on full resume requires a new weights-only experiment. Resuming a checkpoint
without a critic starts a new online critic fit. Turning KL off ignores saved
critic state. Use the project's ordinary full-resume configuration for recovery;
the launcher starts fresh unless `--resume --total-steps N` is supplied.

## Verification scope

Tests cover learned Gaussian KL sign/magnitude, full DDIM finite-difference
gradients, gradient-bearing normalization through the EveNet classifier,
critic/reference isolation, RNG preservation, identical resume continuation,
checkpoint/logging integration, off-path behavior, and distributed averaging.
Local CPU tests do not establish NERSC memory/throughput or 16-GPU runtime.
