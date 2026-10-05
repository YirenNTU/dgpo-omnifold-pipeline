# Late fixed-reward conditioning screen — 50 updates

Status: implemented and CPU-tested; **not executed on NERSC by the assistant**.

## Question and evidence

The completed `6704e92e` probe started at **policy step1130 / reward round7**.
Its native all-sample held-out reward gain was +0.06615 at +5 and +0.00993 at
+20. The conditioning encoder and modulation gradients were nonzero, and its
LR remained near 9.87e-5. This is an early-gain-retention failure, not evidence
that the branch was disabled. Conditioning is a hypothesis, not an established
unique cause. The historical token-FiLM pilot did not show a clear advantage.

## Interventions

| Arm | Intervention |
|---|---|
| A | Original global-FiLM actor; new paired 50-update control |
| B | Add independent nonlinear velocity residual using individual noisy reconstructed angles |
| C | Same added parameters/initialization as B; replace angle basis with pair differences and sums |

This is NOT retraining a classifier or restarting from raw step1110.

B/C use the original normalized projected visible inputs, normalized current
`x_t`, diffusion time, masks, and visible-leg directions. The exact existing
invisible normalizer maps `x_t` back to a **noisy coordinate chart**, including
inverse-CDF normalization where configured. Adding visible angles gives two
noisy reconstructed directions. These are not clean estimates at high noise.

- B: theta_a, phi_a, theta_b, phi_b, each sin/cos k=1..4.
- C: theta_a-theta_b, phi_a-phi_b, theta_a+theta_b, phi_a+phi_b, same k=1..4.
- Both have 32 bounded angular channels, exactly identical parameters and
  initialization. No target labels or clean invisible coordinates are read.
- A two-linear SiLU token encoder is masked-mean-pooled. Its context, `x_t`,
  time bank, angular bank and masks enter a width128 three-hidden-layer SiLU
  MLP with LayerNorm and a **single zero-initialized four-coordinate output**.
- The resulting delta-v is added to the existing normalized velocity output;
  the original PET, global FiLM, heads and original token paths stay intact.

C-versus-B isolates this angular representation at matched capacity. It does
not isolate frequency, width or all possible conditioning mechanisms.
B-versus-A tests an added representation/capacity/output-route package, NOT
capacity alone. All branches carry inductive bias but no truth leakage.

## Fixed state and pairing

- Pin one raw full-state checkpoint and its exact runtime configuration.
- Restore installed H4 stack, original reference and paired round reference;
  **do not recenter references**, reset AdamW, or refit classifiers.
- All old Adam moments, LR groups and cosine clocks are retained. Only new
  parameters receive fresh moments in the existing conditioning group.
- Raw/no EMA; coefficient1 velocity-MSE; exact native DGPO objective, K8,
  DDIM20, batch512/GPU, 16 one-GPU workers. No extra training seeds.
- A caches its 50 actual update batches on shared storage; B/C replay them.
  Per-update RNG is paired across arms/ranks. Cache storage is additional to
  three endpoint checkpoints and saved validation panels.
- Same 32,768 validation identities, K8, noise, batch partition and reward.
  Held-out means held out from policy updates, not necessarily unseen during
  the historical classifier selection process.
- Initial sample/reward equality against A must pass at 1e-6 before B/C update.
  Every update checks identity disjointness from the validation panel.
- The initial checkpoint lacks historical iterator/RNG state. This pairs the
  **new** three-arm screen, not a bitwise replay of old run6704e92e.

## Budget, measurements and decisions

Each arm applies exactly **50 additional native updates**. Live paired
measurements at 0/1/5/20/35/50; exact gradient reconstruction at the five
nonzero milestones. No periodic refits, cold audits, or large physics plots.
Schedules keep their old absolute clock; they do not decay over 50 steps.

Primary: C-minus-B difference in held-out mean reward **gain at +50**.
Support requires point estimate >0.01, simultaneous 95% lower bound >0, and
positive C gain versus its start. Report B-minus-A, C-minus-A and each arm's
+5-to+50 change. Do not pick a best checkpoint after observing the curve.

The comparison report uses an event-cluster bootstrap with simultaneous bands
across nine predeclared contrasts. Candidate draws are not treated as IID
events. These are fixed-model/event uncertainty, not training-seed uncertainty.
An explicit-relations win that does not beat A is not a successful rescue.
Only higher early reward is insufficient to claim improved retention.

If C wins, extend the promising trajectory and independently train a matched
fresh H4 before claiming distribution improvement. A negative 50-step result
does not exclude longer training or all other conditioning representations.

## User launch (inside an existing 16-GPU Shifter/Ray allocation)

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_dgpo_conditioning.py
```

This sequentially runs A, B, C on the SAME 16 GPUs (not 48 GPUs) and uploads a
fourth W&B **comparison-only** run. Three training runs log live in
`ytchou97-university-of-washington/nu2flow-RL`, group
`Late reward conditioning screen`. Display names identify Original,
Individual angle residual, and Relative angle residual.

Default source:
`/pscratch/sd/y/yiren/Ztautau/dgpo_checkpoint_transfer/probe-20260928T163114-ph2iomf6/source.ckpt`

Exact source runtime: `runtime.yaml` in that same directory.
Outputs: unique `screen-*` directory under
`/pscratch/sd/y/yiren/Ztautau/dgpo_conditioning_late_50/`.
Old source/run outputs are not overwritten. Final per-arm checkpoint is under
`A/checkpoints/last.ckpt`, `B/checkpoints/last.ckpt`, `C/checkpoints/last.ckpt`.

Use `--prepare-only` to pin/validate configuration without Ray/W&B. The
printed directory can later be executed with `--prepared /absolute/screen-dir`.
Completed arms are skipped; partial arms are never silently restarted into an
existing W&B history. After fixing an interrupted B/C, explicitly restart that
arm against the completed A control, without rerunning A:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_dgpo_conditioning.py \
  --prepared /pscratch/sd/y/yiren/Ztautau/dgpo_conditioning_late_50/screen-20260929T020609-asqm6z14 \
  --retry-arm B
```

The interrupted B directory is renamed to `B.interrupted-<old-id>-<new-id>`;
its logs and measurements remain recoverable. The replacement B starts from
the SAME pinned step1130 checkpoint in a new W&B run, with the SAME A update
cache and held-out panel. The launcher skips A, executes B, then executes C
(or skips C if already complete). This is not a continuation from partial B
weights. `--retry-arm C` works the same way for an interrupted C. Do not run
this against an arm that is still active. An incomplete A requires a fresh
screen because its cache may not contain all 50 batches.

2026-09-29 compatibility fix: the training-only conditioning tensor allowlist
now retains the six observed `lead_{a,b}_visible_{px,py,pz}` fields in rollout
and gradient-bearing candidate/time expansion. These were already present in
the full evaluation and saved input batches. No truth auxiliaries are added;
the fixed reward, loss, architecture and initial model function are unchanged.
Regression tests cover both bases through native DDIM, input pruning,
microbatch/time expansion and two AdamW updates with nonzero branch gradients.

If only the final comparison/upload failed, rerun:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_dgpo_conditioning.py \
  --prepared /absolute/screen-directory --report-only
```

Main W&B plots: `checkpoint_transfer/heldout/delta_mean` and CI bounds versus
relative update; `train/visible_conditioning/probe_residual_rms` and
`reward_probe_grad_norm_post_clip`; unchanged reference/gradient diagnostics.
The comparison run adds `reward/A_gain`, `reward/B_gain`, `reward/C_gain`,
paired contrast summaries, `comparison.json` and `plan.json` as an artifact.
