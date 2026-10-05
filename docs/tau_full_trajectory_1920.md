# Full DDIM trajectory reward uptake from step1920

Prepared, not launched. This compares native DGPO with the negative mean of the
same frozen tau reward differentiated through all twenty DDIM steps. Both arms
inherit the actor, classifier, velocity reference, AdamW moments and cosine
scheduler from `dgpo-epoch=191-next_ep=192-step=1920.ckpt`. The reward denominator
is still step1880/round20; it is not relabeled as a fresh step1920 ratio.

The question is whether changing the reward update improves persistent uptake
under the inherited reward head and downstream polarimeter spin-Cij closure. This
is a new pathwise objective, not DPO and not a claim of exact distribution KL.

## User launch commands

Use the existing remote `ml_pipeline` checkout and an existing sixteen-GPU
Shifter/Ray allocation. These commands do not submit or request an allocation.

No ensemble is needed. Both arms use the single frozen classifier inherited
from step1920. On the first real launch, all sixteen workers record one complete
pass of native training batches to the shared `native_training_directory` in
the YAML. Training ingestion uses a non-dropping Ray split, preserving the
13 rows that an equal 16-way split would omit. The corrected default directory
is `native_training_full_rows_v2`; earlier incomplete captures remain untouched.
Subsequent arms reuse these exact batches; this capture does not fit
any classifier. Let the first capture finish before launching the other arm.
An incomplete capture is rejected; finish the first launch or select a new
shared directory for both arms, preserving the incomplete artifact for diagnosis.

Validate the source, completed filter manifests, and an existing replay if present, pin the full
checkpoint and write an isolated resolved runtime **without starting training**:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920.yaml prepare --method pathwise
```

Run the two initial +50-update arms separately, each from the original step1920:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920.yaml trajectory --method native
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920.yaml trajectory --method pathwise
```

Both stop at absolute step1970. `--prepare-only` works on either trajectory
command. `--updates 200` or `--updates 500` explicitly requests a longer arm
from the original source, not a continuation of a +50 output. Each invocation
creates its own folder under `/pscratch/sd/y/yiren/Ztautau/tau_full_trajectory_1920`.
No source checkpoint or production output is overwritten.

Read completed or partial results:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920.yaml summarize
```

The inventory labels prepared-only invocations and pairs cross-method results
only with matching source, replay, inherited head, validation identities/noise and rollout
contract. The reported difference is explicitly **after minus before**, with
both method labels; inspect their order rather than assuming a sign convention.

## Fixed controls and changed loss

- All sixteen ranks replay the same original per-rank native tensor batches,
  retaining their original ordering and partial tails; 512 events/GPU maximum.
- The filtered train population is416701 at
  `/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`;
  validation is119002 at
  `/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`.
  Completed manifests and exact counts are required; no raw fallback.
- Preserve pinned normalization, label0 conditioning, architecture, DDIM20
  arithmetic, K8, dropout mode, optimizer/scheduler state, coefficient1 reference
  and eight native reference-time draws in `[0,0.7]`. No EMA, refit or recenter.
- Both arms generate and score one candidate at a time in event chunks of16.
  The original frozen reward scores each chunk, matching the differentiable
  reward batch layout; this avoids comparing 256-event and 16-event kernels.
  The saved `reward_layout` contract prevents pairing with older scoring layouts. This common
  memory layout changes the exact RNG assignment from the historical native
  launch, so the native arm must be rerun. It retains the native DGPO loss.
- Pathwise removes the DGPO main gradient and adds
  `-sum(training_time_bounded_log_ratio)/(K*local_batch_events)` once. All K8
  candidates and partial chunks contribute with their proper mass. The existing
  invalid-event mask is retained; no best-of-K, rescaling or extra clipping.
- Reference backward remains the original velocity-MSE computation on
  **detached** candidates, with its native timestep and mask normalization.
  It does not backpropagate through the sampling distribution or frozen reference.
- The reward backward replays each recorded chunk's original noise and dropout,
  with activation checkpointing at every DDIM step. It retains gradients through
  denormalization, the frozen raw1110 feature trunk, relative angles, directional
  products and the frozen attention reward head. There is no timestep truncation.
- Every chunk checks endpoint replay and differentiable reward against the
  original evaluation path (`atol=rtol=5e-5`). Nonfinite values/gradients, value
  mismatch or skipped native loss substeps abort before committing that update.
  Reward parity failures save per-rank tensors under `mechanisms/trajectory-failures`
  and report fixed-endpoint implementation error separately from endpoint
  perturbation amplification and original scoring-layout differences.
  Training-mode BatchNorm is rejected because replay would mutate running state.
- Distributed pathwise and reference gradients accumulate locally and are
  averaged together exactly once, before native clipping and one AdamW step.
  Native control retains its normal distributed reduction.

This is equal event/update budget, not equal compute or policy displacement.
Retaining AdamW moments is intentional full-state continuation. The pathwise
loss has its own scale; the logged update RMS and clipping must accompany any
claim that its trajectory is more actionable.

## Pathwise reference scale ablation (2026-10-05)

The user requested calibration of reward/reference gradient magnitudes, rather
than projection of their directions. The prepared fixed-coefficient trial is
`config/tau_full_trajectory_1920_mb64_ref5.yaml`, compared with pathwise mb64/ref1
(`fe1dc5a4`). It restarts at the original step1920 with the same inherited actor,
head, reference, full AdamW state, scheduler, DDIM20/K8, microbatch64, seeds,
complete training replay, filtered populations and evaluation schedule. Only
`dgpo.reference_trust.coefficient` changes from1 to5. The source runtime must
still pass the original coefficient1 preflight before applying this explicit
override; using the trial config with `--method native` is rejected.

This tests whether a stronger velocity-reference gradient controls drift while
retaining reward uptake. Five is a screening coefficient motivated by the
rank-local reward/reference norm ratio around+20; it does not equalize every
update or the globally averaged gradients. There is no adaptive rescaling,
gradient projection, or extra backward, and it is not exact distribution KL.
The fixed regularization tradeoff changes; truth is not guaranteed to be the
optimizer's target. Do not judge success solely by slower reward growth. Inspect
all Cij components, total/offdiagonal closure, displacement/reference loss, and
the predeclared fresh audits at0/50.

On an existing sixteen-GPU allocation, after the current run releases its GPUs:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920_mb64_ref5.yaml prepare --method pathwise
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920_mb64_ref5.yaml trajectory --method pathwise
```

The trial stops at1970 and uses the same mb64 series root, with a new invocation
directory and fresh W&B ID. Its display name is
`Does stronger reference control drift? | pathwise | ref=5 DDIM20 | step 1920`.
The inventory can pair same-method coefficient1/5 endpoints and labels the
coefficient intervention explicitly; it refuses comparisons changing both method
and coefficient or the rollout contract.

New pathwise metrics are rank-local, before distributed averaging/clipping/AdamW:

- `trajectory/reference_coefficient`: actual fixed multiplier.
- `trajectory/local_reference_gradient_norm`: weighted reference norm (existing).
- `trajectory/local_unweighted_reference_gradient_norm`: norm divided by the
  declared multiplier.
- `trajectory/local_reward_to_reference_gradient_ratio`: reward-loss norm over
  weighted reference norm; the zero-reference flag disambiguates a zero denominator.
- `trajectory/local_total_on_reward_projection_ratio`: raw total projection on
  reward-loss gradient divided by reward-loss norm squared. Negative reward/reference
  cosine can reduce this fraction; this is not the AdamW displacement projection.

The trainer checks the runtime coefficient against the controller declaration.
The coefficient is applied once in the native reference backward, before the
unchanged pathwise reward gradient is added.

## Dynamic reference scale ablation (2026-10-05)

`config/tau_full_trajectory_1920_mb64_dynamic.yaml` keeps the same source1920,
filtered populations, mb64, K8/DDIM20, head/reference, full optimizer/scheduler,
saved training replay, and validation/fresh-audit endpoints. The only intervention
is a detached, current-step dynamic reference coefficient. It is pathwise-only.

Let `g_r` be the negative mean reward gradient and `g_ref` the unweighted native
detached-candidate velocity-reference gradient, each averaged across all ranks.
The controller maintains EMA norms (decay0.9) and proposes
`lambda = EMA(norm(g_r)) / (EMA(norm(g_ref)) + 1e-8)`. The target reference/reward
norm ratio is1; clamp lambda to[1,30] and limit its change to a factor2 per update.
The previous coefficient starts at5. The first valid observation initializes
the EMA directly, rather than averaging against artificial zeros. A current
component norm <=1e-8 holds the previous coefficient and both averages.
NaN/Inf component norms abort the update. This balances magnitudes; it does not
project directions, enforce a hard distance budget, or guarantee a truth optimum.

Native reference backward uses base coefficient1, then full reward backward is
added. After the existing total-gradient all-reduce, one additional flattened
reference-gradient all-reduce produces the unweighted global reference gradient.
Subtract it from the total to recover the global reward component, choose lambda,
and add `(lambda-1)*g_ref` before clipping and AdamW. A tiny rank0 state broadcast
keeps all workers on the same EMA/coefficient. There is no extra model forward or
backward, no per-parameter extra communication, and no higher-order differentiation
through coefficient selection. An extra flattened buffer is about63MiB for the
current16.46M FP32 trainable parameters; real-GPU overhead remains to be measured.

The pending EMA transaction is committed only after a successful optimizer step.
Checkpoint reward-stack payloads include `trajectory_reference_balance_state`
with the recipe, EMA values, coefficient and update count; incompatible state
is rejected on load. This does not add a general resume command to the diagnostic
launcher, which still starts each declared trial from the pinned step1920.

Launch on an existing sixteen-GPU allocation after the active run finishes:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_full_trajectory_1920_mb64_dynamic.yaml trajectory --method pathwise
```

The fresh W&B name is
`Does dynamic reference control drift? | pathwise | EMA norms DDIM20 | step 1920`.
The trial stops at1970, preserves all evaluation points, and shares the mb64
inventory with ref1/ref5. Paired comparisons label `reference_balance` as the
intervention and reject simultaneous method/layout changes.

Use `trajectory/reference_coefficient`, `trajectory/balance/*`, and
`trajectory/global_{reward,reference,unweighted_reference}_gradient_norm` for
the effective update. `trajectory/global_reward_to_reference_gradient_ratio`,
`trajectory/global_reward_reference_cosine`, and
`trajectory/global_total_on_reward_projection_ratio` describe the averaged,
unclipped gradients. Existing `trajectory/local_*` describe the base-coefficient1
components before distributed reduction and dynamic adjustment; they are not
the applied weighted reference norm. `reference_trust/coefficient` and the
logged total/weighted-reference losses reflect the effective detached coefficient.

Validation includes EMA/bounds/rate-limit/zero-gradient and malformed-state tests,
checkpoint continuation equivalence, exact component recombination, launcher
guards and comparison labels, a two-rank CPU cancellation test, and the actual
production train_step on a small local policy. These do not establish real-GPU
performance or physics improvement. No training job was submitted.

## Measurements and interpretation

Both arms generate raw full-validation K8 samples at relative
0/1/5/10/20/35/50, with adequate fresh classifier audits at0/50. Longer arms keep
the existing declared endpoints and audits. The primary numerical endpoint is
the shared inherited head's all-K mean reward gain at+50, with paired event-bootstrap
intervals; the intermediate points assess whether that improvement persists.
This is the training reward, not independent evidence of physics improvement.
Fresh classifier audits at0/50 are never installed as rewards.

Reward gains and physics results remain distinct: examine polarimeter spin Cij
total/diagonal/offdiagonal/component closure. The inherited reconstruction fixes
tau energy; direction reconstruction alone does not establish fully reconstructed
tau rest frames or a direct entanglement result. Angular plots support the analysis.

W&B uses fresh IDs and names beginning `Can trajectory gradients retain reward?`,
with `native`/`pathwise`, `DDIM20` and `step 1920`, grouped as `Tau full trajectory`.
`trajectory/*` survives both critical and simplified logging. It includes reward
loss, endpoint/reward parity, and rank-local reward/reference gradient norms and
cosine. Any aggregation of those local scalars is not the norm/cosine of the
averaged distributed gradient. Native reference loss, total gradient/clipping and
actual `train/parameter_update_rms` remain logged. In pathwise runs,
`train/loss/dgpo=0`; the displaced surrogate is a separate monitor.

## Validation scope

CPU tests compare the differentiable reward to the original NumPy/frozen-trunk
path, including cross-attention, angle wrapping and pole reflections. They check
candidate derivatives, all twenty sampler-step derivatives against finite
 differences, checkpointed versus uncheckpointed backward, noise/dropout replay,
K8 accumulation with partial chunks, reference-gradient addition, source1920
preflight, prepare-only isolation, and cross-method comparison guards.

Additional local smoke checks exercised a small actual EveNet's frozen-trunk
input derivative/full-DDIM20 backward and both branches of the production trainer.
These are implementation checks, not real-case experiment results. Saved1920
GPU reward parity, sixteen-GPU memory/runtime and physics outcomes remain to be
verified when the user launches the real case.

## Equivalent engineering optimizations (2026-10-05)

The default remains DDIM20/K8/event-microbatch16. No loss coefficient, reward
head, reference update, stop-gradient, validation schedule, or parity threshold
is changed. These changes apply on the next launch; existing Python workers
retain already imported code.

- Reuse the classifier endpoint VJP in sampler backward. This eliminates the
  duplicate classifier/trunk backward while preserving the chain-rule gradient.
- Prepare observed reward inputs once per event chunk, reused across all eight
  candidates. Both scoring and differentiable scoring still forward one candidate
  at a time with the same event shape. The cache lasts only for one call/update;
  candidate-dependent features are never cached or detached.
- Accumulate diagnostic scalars on GPU and copy them together at the end. Keep
  reference-gradient snapshots on GPU instead of copying every parameter twice
  over PCIe. Snapshot storage costs approximately 63 MiB for 16,456,535 FP32
  trainable parameters, plus temporary diagnostic reductions.
- Retain per-DDIM-step checkpointing but disable nested PET block checkpointing
  during the replay/backward scope; restore flags even on exceptions. An actual
  small EveNet DDIM20 check reduced calls to the instrumented PET block from60
  to40 with identical endpoints and parameter-gradient parity. This saves
  recomputation but can increase peak memory for a single denoising step.
- Isolate and seed only the model/worker's CUDA generators, preserving CPU,
  Python and NumPy RNG restoration without initializing unrelated GPUs.
- Keep endpoint/reward/finite checks, per-eight-chunk progress, and synchronized
  phase timings. W&B retains the `trajectory/` namespace.

`config/tau_full_trajectory_1920_mb64.yaml` is an optional larger-chunk trial.
It reduces chunk count from256 to64 per update and retains the mathematical
objective, but changes dropout/noise assignment and floating-point batching.
Run both native and pathwise from1920 with this same configuration; do not pair
its results with microbatch16. It reuses the same complete training replay and
writes to a separate output series. Real-GPU memory, parity and speed are not
validated locally. No jobs were submitted.

Do not skip the frozen trunk backward, reference loss, parity checks or initial
no-update replay to obtain apparent speed gains. Caching actor visible features
across denoising steps requires a separate derivation because training-time
stochastic layers can change them. DDIM10 changes the sampler/distribution and
belongs to a separate experiment.

Validation: 88 focused tests and55 subtests passed, including exact cached versus
uncached reward values/input gradients, checkpointed versus full-chain gradients,
RNG restoration and both launcher protocols. The actual production train-step
smoke checks pass for both methods. A broader legacy reward suite additionally
has a pre-existing incomplete refit fixture (`train['source_ids']` absent) and
local distributed/torchvision environment failures; those checks are not claimed
as passed. Real16-GPU throughput still requires the user's launch.
