# Step1920 reward uptake: five mechanism experiments

This series tests reference conflict, cancellation across observed conditions,
conflict across diffusion noise levels, insufficient update budget, and noisy
classifier preferences. The source is the full saved actor, reference, AdamW
moments and cosine scheduler at native step1920. The inherited classifier's
denominator is step1880, round20; it is not relabeled as a fresh step1920 head.

The user launches every stage inside the existing sixteen-GPU Shifter/Ray
allocation. This launcher never submits a job or requests an allocation.
Implementation and local tests do not constitute a completed real-case run.

## Preparation and execution

Validate and pin the source without connecting to Ray or starting training:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml prepare
```

Preparation can select a later runtime with `--target-stage trajectory
--updates 200`, or add `--prepare-only` to any execution command. Every
invocation creates a new directory under the declared `series_root`, writes
`source_metadata.json`, `invocation_manifest.json` and a resolved `runtime.yaml`,
and prints the full native command. The production output is untouched. Each
execution requests a fresh W&B ID and a readable display name; exact source,
arm and budget live in the configuration.

Fit the four matched fresh classifier heads and a separately initialized
common judge, with no actor update or reference recentering:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml ensemble
```

Use the `ensemble_directory` printed by that completed invocation in the
remaining commands. It contains `ensemble.pt` and a completed
`native_training/manifest.json` with sixteen native batch shards. The ensemble
stage records one complete416701-event native preprocessed pass after fitting.
The launcher rejects incomplete captures, missing shards and undertrained
heads before starting the GPU workers. The directory must belong to this series. The common
judge is excluded from the four-head reward ensemble and remains frozen for
all comparisons. It shares the members' fitting population and internal split,
so an independent initialization reduces self-scoring but cannot remove common
representation or dataset bias.

The following examples use a task-specific variable pointing to that exact
printed directory. When this series has exactly one completed ensemble,
`--ensemble-directory unique` selects it without editing a path. With zero or
multiple completed ensembles the launcher stops and requests an explicit
directory, so it cannot silently select a different fitted teacher.

```bash
TAU_MECHANISM_ENSEMBLE=/pscratch/sd/y/yiren/Ztautau/tau_reward_mechanisms_1920/ensemble-PRINTED-SUFFIX/ensemble
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml diagnose --arm inherited --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml diagnose --arm member0 --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml diagnose --arm ensemble --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
```

Each diagnosis computes the first native batch and rolls back all temporary
perturbations. It applies **zero persistent optimizer updates**. Its native
`--max-steps 1921` cap enters the callback; the mechanism driver returns before
the optimizer. An inherited diagnosis without an ensemble directory is also
allowed for gradient measurements, but it cannot establish improvement under
the shared independent judge used by the fresh-head arms.

Run the initial matched +50 trajectories:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml trajectory --arm inherited --updates 50 --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml trajectory --arm member0 --updates 50 --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml trajectory --arm ensemble --updates 50 --ensemble-directory "$TAU_MECHANISM_ENSEMBLE"
```

Use `--updates 200` or `--updates 500` for the longer budget arms. Each budget
starts independently from the exact step1920 source; +200 does not resume a
newly produced +50 checkpoint. Absolute native stop clocks are1970/2120/2420.
The original 1500-epoch cosine horizon is preserved. The same frozen artifact
directory across all arms is required for a common-judge comparison.

Collect recorded results without starting training:

```bash
shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py config/tau_reward_mechanisms_1920.yaml summarize
```

The summary distinguishes prepared configurations, partial reports and
completed measurements. It does not diagnose a cause from a gradient norm,
cosine or a W&B display name alone. Cross-arm paired results require a common
judge and matching completed native training replay metadata, identities,
weights, validation seed and relative update point.

## Fixed real-case contract

The launcher clones the production resolved runtime and verifies the actual
source checkpoint's state and clocks. It preserves raw actor weights, pinned
normalization, no-policy-Fourier architecture, classifier cross-attention,
label0 conditioning, coefficient1 velocity-MSE reference, native unscaled
leave-one-out advantage, K8, DDIM20, eight timestep draws and normalized
`t in [0,0.7]`. This reference penalty is a velocity-MSE surrogate; it is not
an exact distribution KL.

Policy updates use512 events/GPU. All classifier fits use1024 paired
events/GPU, the inherited250-epoch horizon, at least1000 optimizer updates and
minimum internal-validation BCE selection. Member0 is the predeclared
single-head control, not whichever member looks best afterward. The ensemble
is the arithmetic mean of the four training-time bounded log-ratios **before**
the native advantage and detached nonlinear gate.

All new fits reuse one current-step1920 K1 negative panel and the same frozen
backbone features and internal train/validation split. Different initializations
are the classifier intervention. External validation is not used to select
or refit a reward. Head selection in a diagnostic arm preserves the original
reference, AdamW and scheduler; no periodic head refit or reference recentering
occurs anywhere in the trajectory window.

The completed filter manifests must verify the416701-event training population
at `omnifold_attention_10pct_stic_filtered_test1/train` and the119002-event
validation population at `diffusion_val_20pct_seed42_stic_filtered_test1/val`.
There is no raw-data fallback. Trajectories evaluate the complete119002-event
validation population and allK8 candidates; local diagnosis defaults to32768
fixed validation identities. Generation uses the inherited fixed validation
seed for paired before/after comparisons. The source checkpoint did not save
its historical RNG/data iterator, so this is a counterfactual continuation,
not a bitwise replay of the original training run.

All arms using the same ensemble artifact consume **the same saved native
tensor batches in the same per-rank order**. The record contains all native
preprocessed fields, including masks, event weights and truth tensors; it is
not reconstructed from classifier feature panels. Each rank retains its
original partial tail and independently cycles its complete saved list.
There is no padding, truncation or synchronized reset that could discard
longer shards. The same update-index seed recipe couples candidate generation,
diffusion time/noise and dropout across arms. W&B metadata records whether
inputs came from this shared replay or an unmatched live Ray iterator. This
new deterministic sequence pairs the counterfactual experiments; it does not
recover the original source run's historical RNG/data order.

## Measurements and falsifiable hypotheses

The gradient diagnosis partitions the observed visible-pair opening cosine
into `[-1,-0.5)`, `[-0.5,0.5)`, `[0.5,1]`, and the original diffusion-time
range into three declared bands. These groups use observed inputs; momentum
opening cosine is a condition proxy, not a spin-correlation observable.

Every condition × noise cell uses the same candidates, timestep draws,
advantages and detached gate as the complete native call. Cell contributions
retain the native reduction weights; summing them must reconstruct the complete
reward gradient within the configured tolerance. Recomputing a gate after
selecting a band would define a different objective and is not this diagnosis.
Two independent diffusion/dropout sampling draws on the same fixed native
candidates and training batch quantify sampling reproducibility. The native
AdamW proposal uses replica0; signed component directions use the two-replica
mean, which is not labelled as an actual native optimizer step.
Empty or zero-gradient cells remain inconclusive.

Read-only replay includes the exact native one-step AdamW displacement and
signed small perturbations along reward/reference/condition/noise components,
with parameter RMS normalized to the native AdamW displacement. Model and
optimizer state are restored afterward. These
tests establish local direction response at the chosen radius. Equal parameter
RMS alone does not guarantee equal policy-distribution displacement, and a
successful signed probe does not establish a successful longer trajectory.

| Hypothesis | Supporting evidence required | Evidence against it as a sufficient cause |
| --- | --- | --- |
| Reference conflict | Reference removes a reliable reward component and the paired reference-free local replay improves common-judge response. | A reliable direction remains ineffective after removing the reference gradient locally. |
| Condition cancellation | Condition gradients are reproducible, oppose one another, and one group's useful update measurably harms another group. | Stratum directions agree, or apparent conflict disappears across independent gradient estimates. |
| Noise conflict | Noise-band directions reliably conflict and signed low-noise response is better on the same judge identities. | The full direction is as useful, or the apparent band ordering fails reproducibility. |
| More updates needed | +200/+500 retain and enlarge gains under the same frozen judge, instead of repeatedly gaining and losing them. | Longer matched trajectories erase early gains or stay indistinguishable from baseline. |
| Classifier noise | Members disagree on shared candidate ordering/advantages/directions, and their mean improves common-judge update response versus member0. | Reduced member variance without better independent-judge or spin closure. |

Classifier disagreement and condition cancellation are separate measurements:
the former changes the critic on the same candidates, the latter holds the
critic fixed and compares different observed conditions. Ensemble gains over
the inherited head alone cannot isolate variance reduction because the fresh
denominator also changes; **member0 versus the four-head ensemble** is the
matched classifier-noise comparison.

Trajectory measurements occur at relative0/1/5/10/20/35/50 and, when requested,
100/150/200/300/400/500. Fresh diagnostic audits occur at0/50/200/500 within the
selected budget. They are not installed as rewards. Retain fresh-audit fit
budgets and predeclared audit endpoints; undertrained near-chance AUC cannot
establish closure. The long-budget question is the common-judge all-K mean
reward change with paired event-bootstrap95% intervals and its persistence
across these points. Best-of-K and reward standard deviation are supporting
measurements.

Physics usefulness is assessed separately using polarimeter-based spin Cij
total, diagonal, offdiagonal and individual-component closure. The present
diagnostics use the existing fixed-energy reconstruction and TT2L conventions;
they do not supply full tau-energy unfolding or a direct entanglement estimate.
Angular marginals and opening/acoplanarity measurements provide context and
cannot replace spin closure. Improvement in the fitted reward is not alone
evidence of improved tau-pair entanglement inference.

## Validation scope

Local unit checks exercise source/config guards, inherited reference and
optimizer preservation, correct absolute stop caps, readable fresh W&B aliases,
ensemble artifact adequacy, and prepare-only no-launch behavior. The
sixteen-GPU production integration, numerical reconstruction tolerance and
physics response require the user's real-case execution. No such job is
submitted by this implementation.

Diagnosis is intentionally heavier than one ordinary update: it decomposes
multiple backwards and regenerates held-out K8 samples for signed local
directions. It applies no persistent updates. Fit the ensemble only after the
original-state diagnosis is worth pursuing; its five best-validation fits are
four reward members plus the separate evaluation judge, not five diffusion
training-seed repetitions.
