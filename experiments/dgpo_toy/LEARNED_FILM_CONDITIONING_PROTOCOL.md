# Does nonlinear FiLM transfer the saved learned H4 reward?

Status (2026-09-24 completion check): **COMPLETED locally**, after the user's explicit
authorization, "啟動". This authorization covers only the two arms and three
milestones specified below. No production edits, remote jobs, uploads,
automatic follow-on jobs, or changes to old outputs.

## One question and evidence

Does the nonlinear c-only FiLM package retain its benefit when the policy
uses the saved learned H4 reward rather than idealized C?

All of the following start from the same
`artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1/reference.pt`.
It was ordinarily velocity-pretrained on the uniform noisy cube reference,
**not full truth**, and is not any DGPO endpoint.

- `reward_transport_abc_v1_retry1`: at 1,000 updates, learned A, mode-averaged
  B, and idealized C with first-layer-only Fourier conditioning all failed
  the mode-transport screen and fresh-audit improvement rule. Full-truth
  audit AUC was 0.641407 at source, 0.640970 for A, and 0.641492 for C.
- `conditioning_transport_v1`: keeping C fixed, multi-layer linear FiLM
  improved 1,000-step audit AUC to 0.637568, but mode-TV improvement was
  below the prespecified 0.01 screen.
- `film_conditioning_rounds_v1`: same C and reference. Final 3,000-step
  audit AUC was 0.605399 for linear, 0.590935 for MLP(c), and 0.590678 for
  MLP(c,t). MLP(c) mode-TV was 0.051765 versus source 0.162124. Timestep
  conditioning showed no resolved extra advantage; all audits plateaued.
  Neither MLP arm's fresh AUC improved further from 2,000 to 3,000.

These are evidence for conditioning under C, not learned-reward closure.
The new round changes the reward source back to A while retaining a paired
linear-versus-nonlinear comparison. Completed C outputs are historical
reference measurements, **not policy initialization or extra training arms**.

## Fixed reward and its limitation

Use exactly `artifacts/dgpo_toy/reward_transport_abc_v1_retry1/classifier.pt`.
The loader validates it against
`artifacts/dgpo_toy/shape_matched_dgpo_v1/classifier.pt`, selected at update
3,100 by validation BCE. It remains in eval mode with gradients disabled;
no reward fitting, recalibration, normalization, rescaling, clipping,
temperature, or refit is introduced.

**A is a shape-matched diagnostic H4, not the original full-truth H4.**
Its training positives use truth mode probabilities but initial-generator
within-mode shapes. Constructing that target used known toy modes. Its held-out
test AUC was 0.610158, BCE 0.669747, ESS fraction 0.842834. This is not the
original extreme-low-ESS setting and is not proof of a physics-prior-free
production method. The classifier architecture uses raw c/y plus coordinate
Fourier k=1..4; neither policy inputs nor classifier inputs receive truth mode
labels or the target bump function.

During policy optimization the reward is **only the frozen learned A logit**.
The original native DGPO detached gate and unscaled leave-one-out advantage
remain unchanged. C/B tables and analytic truth are used only in diagnostic
evaluation; their logged presence must not be confused with a training reward.
Fresh H4 audits train against the original full truth and never become the
policy reward.

## Arms and matched settings

| Arm | Conditioning package | Added parameters | Initialization |
|---|---|---:|---|
| `linear` | Existing linear scale/shift at all three hidden layers | 6,144 | All modulation heads zero |
| `mlp_c` | Fourier(c) -> 8/32/32 SiLU encoder -> encoder-only parameter-free LayerNorm -> three scale/shift heads | 26,688 | Encoder normal, only final heads zero |

The timestep branch is deliberately omitted. Identical functions at step 0,
not identical parameter counts: nonlinear capacity and encoder normalization
remain a package comparison. Both architectures and common initial weights
are exactly those used in the preceding C round. Fixed condition frequencies
are sqrt(2)*sin/cos(pi*c*[1,2,4,8]). No new condition features are added.

- Width 32; hidden width 128; all policy parameters trainable.
- AdamW LR 1e-4, weight decay .001, global gradient clip 1.
- Batch 64, K=8, M=4, DDIM 50, t uniform in [0,.7].
- Policy seed 17, native monitor seed 284017, encoder init seed 451017;
  unchanged endpoint and audit sample seeds.
- Reference coefficient **1 times half velocity MSE**, not exact distribution
  KL. Reference stays the original zero-modulation policy throughout.
- Raw model weights; no EMA, optimizer reset at milestone boundaries, reference
  recentering, classifier refit, warm audit, or oracle-warmstarted generator.

Budget: cumulative 1,000 / 2,000 / 3,000 policy updates per arm. Start with a
fresh policy optimizer; preserve full weights, optimizer state, RNG and history
between milestones. This is one seed trajectory per arm, not three independent
replicates. Do not adapt the architecture or select a favorable earlier endpoint.

## Primary endpoint and stopping rule

At each milestone cold-train full-truth H4 audits of the original source,
linear, and MLP(c). Same seed 41, paired conditions/truth/noise, and independent
train/validation/test pools 32768/8192/16384. AdamW 3e-4, WD .001, balanced batch
512, clip 1. Minimum 2,000 fit updates; validate every 100; patience 20 checks;
min delta 1e-4; maximum 16,000. Select each audit's lowest validation BCE,
not its test result. A budget-exhausted audit is inconclusive, never closure.

Primary endpoint: **final 3,000-step held-out fresh audit gap abs(AUC-.5)**.
Prespecified primary pass requires BOTH paired-bootstrap upper 95% bounds:

1. MLP(c) minus original-source audit gap < 0.
2. MLP(c) minus linear audit gap < 0.

BCE is reported alongside AUC. Mode-TV, corner retention and learned-reward
gain/decomposition are secondary, not a silent veto of the primary endpoint.
The earlier combined screen (mode-TV decrease >=.01, corner-fraction drop
<=.02, zero missing mode cells, plus both audit contrasts) is retained as
`combined_pass`, distinct from this round's `primary_pass`.

- Primary pass: supports nonlinear learned-H4 transfer at this budget, not
  complete distribution equality or a uniquely isolated conditioning mechanism.
- MLP improves versus source but not linear: learned-reward transfer observed,
  nonlinear advantage unresolved.
- Adequate audit with no resolved source improvement: this package is not
  sufficient at this budget; do not raise LR, change reward, or extend training
  automatically to find a positive result.
- Reward improves without fresh-audit improvement: no classifier-closure claim;
  inspect within-mode versus mode-probability reward decomposition.
- Numerical failure stops the run with saved progress, not a silent restart.

Use 300 paired-context bootstrap resamples. These intervals are conditional on
the fitted models, not training-seed uncertainty or multiplicity-adjusted tests.
Earlier milestones describe the trajectory only. Historical C comparisons
must report their own audit fit lengths, since joint audit stopping can differ
between a three-policy batch and this two-policy batch.

## W&B: a few useful curves, distinct clocks

Name: `Does FiLM transfer learned reward? | frozen H4 | linear vs nonlinear | V MSE 1`

Project `dgpo-toy`; group `Conditional reward transport`; default offline.
Run IDs remain separate from the display name. Use these principal plots:

1. `policy/{arm}/monitor_gain` versus `policy/{arm}/step`: frozen A reward
   gain on the fixed independent native-monitor panel.
2. `validation/{arm}/auc_gap` and `/bce` versus `validation/{arm}/step`:
   cold audit endpoints on the **policy** step axis. Check `/valid`,
   `/fit_steps`, `/selected_step`, and `/plateau` before interpreting.
3. `policy/{arm}/within_k_ess_fraction`, main/reference gradient norms,
   cosine and projection: candidate concentration and reward/reference conflict.
4. `structure/{arm}/mode_tv` and `/corner_fraction`: conditional transport and
   geometry diagnostics. These do not replace fresh-audit results.

Detailed audit learning curves use
`audit/round{milestone}/{arm}/step`, which is a **classifier fit** clock.
Each audit round has a separate namespace. Final audit metrics and decision
flags are also written to W&B summary. Learned-reward endpoint decomposition
is logged under `endpoint/{arm}/learned_reward_decomposition/*`:
mode-probability contribution versus residual within-mode shape/interactions.
This decomposition is diagnostic, not a causal KL attribution.

Existing conditioning telemetry is retained: per-layer residual/hidden RMS,
scale RMS, backbone/encoder/modulation gradient norms, and fixed-probe velocity
MSE. There are no `mlp_ct` training or audit plots in this round.

## Local launch

Read-only preflight (no W&B initialization, policy updates or output directory):

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.film_conditioning \
  --reward-arm A \
  --output artifacts/dgpo_toy/learned_film_conditioning_v1 \
  --milestones 1000 2000 3000 --width 32 --preflight
```

Training command (launched after explicit user authorization):

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.film_conditioning \
  --reward-arm A \
  --output artifacts/dgpo_toy/learned_film_conditioning_v1 \
  --milestones 1000 2000 3000 \
  --width 32 \
  --audit-max-steps 16000 \
  --wandb-mode offline
```

Use `--wandb-mode online` instead only when online syncing is desired. No cloud
run exists until launched and synced. Prior artifacts remain excluded from
NERSC synchronization. The existing C command remains compatible: its default
reward is C and it retains all three original arms.

Outputs: `report.json`, `progress.jsonl`, a copied `frozen_reward.pt`,
`{linear,mlp_c}_last.pt` and all milestone states/endpoints, paired audit panels,
selected audit checkpoints and test scores. Each policy checkpoint records
`reward_arm=A`, route, source, coefficient, optimizer/RNG/history and step.
No previous trained policy is loaded just because a historical control path
appears in provenance: the A run reads that control's settings only.

## Prelaunch implementation validation record

Actual-source A preflight passed: frozen classifier selected step 3,100;
initial velocities and DDIM samples bitwise match for both arms. Parameter
counts are 40,835 / 61,379 including the backbone. No training was started.
Actual-source C preflight also passed for all three original arms. The full
regression suite passed **68 tests in 22.34 seconds** outside the sandbox,
with W&B Git metadata disabled for the tests. Both preflight output paths
were verified absent afterward.
Tests cover native learned-reward objective without oracle access, frozen
critic state, exact staged/continuous optimizer and RNG equivalence, preserved
original reference, A/C routing, no C-trained policy initialization, audit
versus policy clock separation, primary-versus-secondary decisions, and
backward-compatible C behavior. W&B name validator passed.

The first real-source preflight hit a sandbox OpenMP shared-memory error;
the same read-only check passed outside the sandbox. This was an environment
restriction, not a model/checkpoint failure. No experimental output was created.

## Authorized launch record

Started the exact A-reward command above. Offline W&B ID `jkoadr76`; directory
`artifacts/dgpo_toy/learned_film_conditioning_v1/wandb/offline-run-20260924_225720-jkoadr76`.
Local execution session: 61220. Initial live check confirms `reward_arm=A`,
classifier selected update 3,100, coefficient 1, exact initial velocity/sample
matching, and an unchanged cached baseline. Linear policy reached update 200
with a saved checkpoint and finite logged losses/gradients; MLP(c) follows
sequentially. No fresh audit result is available at this launch check.

## Completion record

The authorized batch finished with exit code 0 in 1,055.42 seconds (17.59
minutes). Both arms completed 3,000 updates; all three milestone cold audits
met the declared patience condition (7,500 / 7,500 / 9,900 fit updates).
Nonlinear fresh audit AUC: 0.604301 -> 0.591300 -> 0.587874. Linear:
0.638486 -> 0.624186 -> 0.607771. Source: 0.641407.

At the prespecified final endpoint, nonlinear-minus-linear AUC gap is
-0.019897, paired 95% interval [-0.023938, -0.014769]. Nonlinear-minus-source
is -0.053533, interval [-0.059549, -0.045808]. Primary and combined criteria
both pass. Final nonlinear mode-TV is 0.066270 versus source 0.162124;
corner retention declines by 1.096 percentage points, within the 2-point
screen. All six policy milestone checkpoints passed integrity checks;
source/reference/reward states stayed unchanged. No further job was launched.

Full record: `artifacts/dgpo_toy/learned_film_conditioning_v1/RESULTS.md`.

Outcome: nonlinear learned-H4 transfer improves the valid final fresh audit.
Decision: supports at this single-seed finite-budget shape-matched-H4 setting,
not full distribution equality or proof for original full-truth/real-case H4.
Deleted: extra timestep arm and new reward/classifier fitting.
Learning index: 17.59 experiment wall-minutes / one prespecified hypothesis
resolved positively; implementation time excluded.
Next limiting factor: transfer from shape-matched diagnostic H4 to the original
full-truth H4. Do not infer that different learned rewards are interchangeable.
