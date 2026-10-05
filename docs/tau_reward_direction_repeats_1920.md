# Repeated small-direction diagnosis at step 1920

Run inside the user's existing sixteen-GPU Shifter/Ray allocation:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_direction_repeats_1920.yaml repeat_diagnose
```

Validate and pin inputs without starting the trainer or connecting to Ray:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_direction_repeats_1920.yaml repeat_diagnose --prepare-only
```

The opt-in configuration creates a fresh directory under
`/pscratch/sd/y/yiren/Ztautau/tau_reward_direction_repeats_1920` and a fresh W&B
run named `Do local reward directions repeat? | frozen tau head | four draws | step 1920`.
Results are written to `mechanisms/local_direction_repeats.json`.
The original five-hypothesis launcher/configuration remains a separate protocol.
No allocation, job submission or production checkpoint mutation is performed.

## The one question

Do small perturbations in the proposed reward directions consistently increase
the **inherited frozen classifier's held-out all-K mean reward**, across
separately sampled native update draws? This is a reward-uptake measurement.
An independent judge and physics closure are separate questions; no new
classifier is fitted or installed in this test.

The actor, reference, AdamW moments and cosine scheduler come from the exact
step 1920 full checkpoint. The inherited head remains round 20/denominator 1880.
Production coefficient 1 velocity-MSE reference, architecture, raw weights,
pinned normalization, label 0 condition, K8, DDIM20, eight native timestep draws,
unscaled leave-one-out objective and `t in [0,0.7]` remain fixed. Training and
validation use the verified filtered 416,701/119,002-event populations. Evaluation
uses the same first 32,768 validation identities throughout.

The worker consumes four nonoverlapping chunks from the native shuffled
training iterator, with 512 events per GPU and fresh candidate generation for
each draw. Within a draw, two gradient sampling replicas use the same
candidates. These are separately sampled chunks, not IID whole-training-run
replications. Each direction starts from the same saved actor/optimizer state;
no update accumulates across draws.

Native batch uniqueness and train/validation disjointness use the complete
canonical identity `(source_sample_index, source_event_key, source_file_index,
source_event_index)`, matching the saved panels. The two-column prefix is not
globally unique, even within a file; prefix collisions are diagnostic metadata,
not duplicate events. Identity columns must agree across ranks, draws and the
validation panel. Actual full-key duplicates/overlap still fail; no rows are
dropped or deduplicated. This repairs the pre-update identity-check failure
without changing any candidate, objective, checkpoint or measurement setting.

For every draw, test three directions: `raw_reward`, `raw_total` and
`native_adamw`. All three directions use gradient replica 0, with the same
candidates/timestep/noise/dropout draws; the AdamW proposal also uses the
inherited optimizer state. Replica 1 is a fixed-candidate sampling-stability
check only, not an averaging treatment. This keeps gradient averaging separate
from the optimizer-geometry comparison; legacy `diagnose` is unchanged.
At radii 0.01/0.03/0.1 times that draw's native AdamW parameter RMS, measure both
signs using common random numbers. Equal parameter RMS is not equal
policy-distribution displacement and does not establish learning rate as the
main cause of failed uptake.

All evaluation points use two predeclared rollout seeds 202610041/202610042.
The actor, optimizer, reference, classifier and clocks are restored after each
temporary perturbation. The native `--max-steps 1921` cap enters the diagnostic
callback; it consumes four batches while native step remains 1920 and applies
**zero persistent optimizer updates**. There is no refit, reference recentering,
ensemble or fresh audit. The historical source RNG/data order was not saved;
this does not reproduce the original run bit for bit.

## Interpretation and uncertainty

Keep the three uncertainty sources distinct:

- Four native update draws test whether the proposed directions repeat across
  sampled training chunks/candidates.
- Two evaluation seeds measure rollout Monte Carlo sensitivity on the same
  held-out identities.
- Paired event-bootstrap 95% intervals resample identities, preserving all K8
  candidates within each event. They do not turn two seeds or four draws into
  hundreds of independent training repetitions.

Report per-draw/per-seed signed reward changes, their seed averages, and
odd/even response versus radius. The odd response is `(gain_plus-gain_minus)/2`;
the even response is `(gain_plus+gain_minus)/2`. A linear window should show
stable odd response per radius while even response shrinks faster. Four draws
are a screening experiment; failing to detect an effect is not proof of its
absence. Do not select the most favorable radius/draw and call it a replicated
effect.

Before interpreting a sign, require realized/requested displacement cosine
at least 0.95, realized/requested RMS ratio within [0.8,1.2], and nonzero generated
delta motion. These are predeclared engineering checks for finite-precision
resolution, not physics thresholds. An unresolved or quantized tiny
perturbation is **inconclusive**, not a failed direction. Retain these validity
flags alongside every reward measurement and odd/even comparison.

If raw reward helps locally but native AdamW harms, inspect the reward→reference/
optimizer interface; raw total distinguishes reference contribution from AdamW
geometry. If both signs remain harmful only at larger radius, finite-radius
effects are supported. If no sufficiently small direction gives reliable
uptake, examine the surrogate→generation interface and numerical resolution
before attributing the outcome to learning rate. Raw Cij and its component
errors can provide spin-closure context, but fixed-head improvement alone does
not establish distribution or entanglement closure.

Condition-specific/noise-specific direction efficacy, conditioning capacity,
classifier fitting noise and long-run accumulation are **not tested** here.
Retained condition/noise gradient labels are instrumentation and do not repair
the empty-stratum limitation of the earlier diagnostic.

## Workload and results

This diagnostic requires **144 perturbed K8 generations**:
3 directions × 3 radii × 2 signs × 2 evaluation seeds × 4 draws. Two baseline and two
no-update check replays bring the total to 148 evaluation calls at 32,768 events
and K8: 38,797,312 generated candidates with DDIM20. These are evaluation
generations, not optimizer updates; persistent updates remain 0. No wall-time
estimate is implied.

Read the recorded report inventory without launching training:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_direction_repeats_1920.yaml summarize
```

Local tests cover protocol isolation, fixed source/objective settings, robust
repeat/seed/radius validation, cap 1921 and prepare-only no-launch behavior.
The user's sixteen-GPU execution is required to establish numerical uptake;
no real-case result is claimed by implementation or preparation.
