# Nonlinear and timestep-aware FiLM — authorized local comparison

User authorization (2026-09-24): "implementation starts and run for several
iterations". Bound this to three prespecified milestones: 1,000, 2,000 and
3,000 native DGPO updates per arm. No adaptive architecture, frequency, LR or
reward search; no further jobs automatically launched after this batch.

## Hypothesis and arms

The previous experiment found a small gain from linear multi-layer FiLM,
including cold-audit AUC 0.6414 -> 0.6376, but not the prespecified .01 mode-TV
improvement. The question is whether richer modulation strengthens that gain.

1. `linear`: exact previous three-layer linear FiLM, unchanged.
2. `mlp_c`: Fourier(c) -> Linear(8,32) -> SiLU -> Linear(32,32) -> SiLU ->
   parameter-free LayerNorm -> three independent Linear(32,256) heads.
3. `mlp_ct`: same network and identical common initialization; add Linear(5,32)
   of the existing five timestep features to the first encoder preactivation.

Each output head emits 128 scales and 128 shifts. Encoder weights initialize
normally; only final modulation heads initialize to zero. No second zero gate.
Normalization is **only within the new encoder**, keeping its feature scale
comparable to unit-RMS Fourier features. The pretrained backbone's operations
and normalization do not change. This normalization is part of the nonlinear
encoder package, not a separately isolated causal intervention.

Fourier features remain sqrt(2)*sin/cos(pi*c*[1,2,4,8]). Timestep features
remain [t, sin(pi*t), cos(pi*t), sin(2*pi*t), cos(2*pi*t)]. No new frequencies,
target bump locations, mode labels or truth functions enter the policy.

At hidden width 128, added parameters are 6,144 / 26,688 / 26,848. This is
not a parameter-matched architecture comparison. The c,t arm adds only 160
parameters relative to c-only but that difference is still recorded honestly.

## Fixed controls and continuation contract

- Original pretrained source is the uniform-cube reference from
  `nonperiodic_cube_classifier_extended_v1/reference.pt`, not a DGPO endpoint.
- Same frozen C reward/calibration from `reward_transport_abc_v1_retry1`.
  This uses analytic toy truth mode probabilities and estimated actual initial
  generator mode probabilities. It is NOT the learned H4 reward.
- AdamW LR1e-4, weight decay .001, global clip1, batch64, K8, M4, DDIM50,
  uniform t in [0,.7], rollout seed17 and existing endpoint/monitor seeds.
- Coefficient1 multiplies half velocity MSE, not an exact distribution KL.
  Native detached gate and unscaled LOO advantage unchanged. No EMA/refit.
- Initial velocities and DDIM samples must match exactly across all arms.
- At step1,000, linear weights must exactly replay the old `deep_film_last.pt`.
- At each boundary preserve policy, AdamW moments/step, training RNG and
  complete history. **Reference remains the original zero-modulation model**;
  never rebase it on the current policy. Unit tests compare staged and
  uninterrupted training bit-for-bit, including AdamW and RNG.

## Measurement and primary endpoint

At 1,000, 2,000 and 3,000, evaluate the same 128x1024 conditional endpoint
panel and cold-train full-truth H4 audits of baseline and all three policies.
Audit seed41, train/validation/test 32768/8192/16384, minimum2,000 updates,
validation patience20 checks x100 steps, max16,000; best validation BCE
selects weights. A fit that exhausts its budget without plateau is inconclusive.
Audits do not change or supply policy rewards.

**Primary endpoint: final step3,000 fresh H4 AUC gap**, compared with the
original source and the paired linear control. Earlier audits are descriptive
trajectory points, never selectors of the best endpoint. Report paired BCE
contrasts as well. AUC/BCE intervals are evaluation-sample uncertainty given
one training seed, not replication uncertainty or multiplicity-adjusted tests.

Preserve the mechanism screen: mode-TV decrease >= .01 vs source, corner
fraction drop <= .02, and zero missing endpoint mode cells. A nonlinear
package passes the combined rule only if it also has negative upper95 bounds
on both fresh AUC-gap contrasts (vs source and vs linear). Time conditioning
gets a separate direct mlp_ct-minus-mlp_c audit comparison. A result below
these margins may still be a measured partial effect; never lower the margins
after seeing it or call partial improvement full closure.

If every arm fails, report a negative finite-budget result rather than expand
the search until a positive arm appears. Exceptions/numerical failure stop the
batch and retain checkpoints; no silent fallback or restart with new settings.

## Diagnostics and runtime

Log every100 policy updates: fixed C reward, mode TV/corner retention,
native main/reference gradient norms and cosine/projection, within-K ESS,
layer residual/hidden RMS and scale RMS. Split gradient norms into backbone,
encoder, time encoder and modulation heads. Log coefficient differences under
t -> .7-t at fixed c; c-only must be identically zero. Zero encoder gradients
on the very first update are expected from zero heads; they must become live
after the heads move. Policy, milestone and cold-audit fit clocks are separate.

Inherited DGPO loss samples t<=.7 while the full DDIM trajectory starts at1.
The new time-aware branch therefore extrapolates above the optimized range;
record modulation RMS at t=0,.35,.7,1. This is a known limitation of the fixed
training setup, not authorization to change its time-weighted objective.

Save all three milestone checkpoints, full optimizer/RNG states and audit
panels/scores. Offline W&B and local JSON are default. Prior 1k three-arm
comparison took179 seconds locally; this run is larger and actual wall time
will be recorded. Files remain excluded from NERSC synchronization.

```bash
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.film_conditioning \
  --output artifacts/dgpo_toy/film_conditioning_rounds_v1 \
  --milestones 1000 2000 3000 --width 32 --wandb-mode offline
```

Add `--preflight` to check source/config/functions without starting training.

## Execution record — 2026-09-24

Completed all three arms through steps 1,000 / 2,000 / 3,000, followed by
full-truth cold audits at each milestone. Exit code 0; wall time 2,854.17 s.
No further job was launched. The predeclared rules above were not changed.

Final audit AUC: source 0.641407, linear 0.605399, c-only 0.590935,
c+t 0.590678. Both nonlinear packages pass the combined endpoint rule;
the time-specific advantage rule does not pass. Fresh AUC stops improving
after 2,000 updates even though mode-TV continues improving. This supports
the nonlinear package, not a claim of full closure or a uniquely identified
conditioning bottleneck. The use of idealized C rather than learned H4
reward remains the next transfer question.

All original reference/reward states were unchanged, initial matching was
exact, and linear step 1,000 exactly replayed the prior control. All nine
milestone checkpoint/optimizer/RNG consistency checks passed. The
validation suite passed 62 tests before launch and again after completion.
Offline W&B ID: m0gx2s6c.

Full results: `artifacts/dgpo_toy/film_conditioning_rounds_v1/RESULTS.md`;
raw evidence: `artifacts/dgpo_toy/film_conditioning_rounds_v1/report.json`.

## Separate learned-reward follow-up

`--reward-arm A` now selects the saved shape-matched H4 reward and the two-arm
linear/c-only comparison. Default `--reward-arm C` retains the original
three-arm protocol above. See `LEARNED_FILM_CONDITIONING_PROTOCOL.md` for the
new prespecified test, its limitations, and launch command. This option does
not imply that the learned-reward follow-up has run or alter this completed
C experiment's historical results.
