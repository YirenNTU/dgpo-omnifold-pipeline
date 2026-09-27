# Conditioning route screen

Execution status: completed on 2026-09-24 after explicit user authorization.
All three 1,000-update arms and four cold audits finished. Results are in
`artifacts/dgpo_toy/conditioning_transport_v1/RESULTS.md`; the predeclared
protocol below is unchanged. Offline W&B ID: `yt7euuv0`.

## Question and evidence

Can changing how the denoiser uses the **same condition features** turn a
verified mode-probability reward into useful conditional distribution updates?

The completed `reward_transport_abc_v1_retry1` result is the source, not a
new calibration/classifier task. Its C reward (analytic truth numerator,
estimated actual-generator denominator) improves independent fixed-candidate
mode TV from 0.160747 to 0.035703 under reweighting. But 1,000 native DGPO
updates with C changed endpoint TV only 0.162124 -> 0.161349; all A/B/C fresh
cold-audit AUC changes were inconclusive. This motivates studying transport,
**not proof that conditioning is the unique cause**.

## Three arms

| Arm | Integration before each SiLU | Added parameters at hidden=128 |
| --- | --- | ---: |
| `first_add` | First hidden layer only: `a + W f(c)`; existing implementation | 1,024 |
| `deep_add` | All three hidden layers: `a_l + W_l f(c)` | 3,072 |
| `deep_film` | All three: `(1 + G_l f(c)) * a_l + W_l f(c)` | 6,144 |

`f(c)` is unchanged: sqrt(2) times sin/cos of pi*c*[1,2,4,8]. Raw c and
the existing time features remain in the pretrained input. No new frequency,
MLP encoder, normalization, target bump locations, truth function, or mode
label enters the policy. FiLM is affine modulation, **not adaLN**. Heads read
c only; their scale acts on hidden activations that already depend on x_t,t,c.

All new projections start at zero; they have immediate trainable gradients.
The pretrained path stays trainable and unchanged at initialization. Exact
velocity and DDIM sample equality is checked before updates. The first arm
must exactly replay the prior C endpoint weights at the matching 1,000-step
budget. Do not interpret differences if that control fails.

This is a **route/capacity package screen**, not an equal-parameter causal
comparison. Parameter counts are explicit; a winning package needs a later
capacity-matched confirmation before attributing its advantage only to depth
or multiplicative interaction.

## Fixed conditions and acceleration

- Same original uniform-cube reference checkpoint, not a prior DGPO endpoint.
- Reuse frozen `calibration.pt`, with no classifier fit or recalibration.
- Same C reward for every arm. Toy truth enters the reward/evaluator only.
- Native detached-gate DGPO, unscaled leave-one-out advantage, AdamW 1e-4,
  weight decay .001, clip norm 1, batch 64, K=8, M=4, DDIM50, t uniform[0,.7].
- Velocity coefficient 1 multiplies **half velocity MSE**. This is a reference
  surrogate, not an exact distribution KL. Reference fixed, no EMA/refit.
- 1,000 updates per arm, common rollout seed 17 and identical initial function.
- Monitor every 100; source/endpoint 128 continuous-condition grid points x
  1,024 samples, common random noise. Fixed monitors 64 x 256 are not selectors.
- No early best-checkpoint selection: compare the predeclared final step.
- Default: paired fresh, full-truth H4 audits of baseline and all endpoints;
  at least 2,000 fit updates, validation early stopping with 20 checks of 100
  steps, max 16,000. Best validation BCE selects weights, never test metrics.
- `--skip-audit` is an optional screen-only shortcut. Never label it closure.
  No automated expansion, remote upload, job submission or seed sweep.

The previous full A/B/C experiment took ~149 seconds locally. This package
adds per-layer operations and may take longer; report actual per-arm wall time.

## Predeclared decision

Fast mechanism screen: source-minus-final mode TV >= .01, corner fraction
degradation <= .02 and zero missing endpoint mode cells. To select a new
route it must additionally beat `first_add` TV by >= .005 and have a negative
upper paired-context 95% contrast interval. Intervals are conditional on one
training seed and a fixed grid, not replication uncertainty or a correction
for adaptive architecture search. Below 1,000 updates: inconclusive screen.

Full primary endpoint remains fresh, adequately trained held-out H4 AUC gap
to .5; BCE is also reported. AUC improvement is not complete closure. Only
interpret audits marked `completed`, not fits exhausted before plateau.
Compare both source and first-add control, and retain all negative arms.

- New route transports modes and improves fresh H4: supports an actionable
  conditioning-package limitation; replicate, then test frozen learned reward.
- Transports modes but fresh H4 unchanged/worse: partial mode correction only;
  inspect shape and other remaining discrepancies, not declare closure.
- All fail: these routes alone are insufficient at this budget. Do not claim
  conditioning ruled out or keep changing frequencies until a lucky run passes.

## Diagnostics: four useful views

1. Conditional mode TV and corner retention vs policy step.
2. Fixed C reward gain, with held-out common-noise endpoint comparison.
3. Main/reference gradient norm and cosine, total-on-main projection, velocity
   MSE, informative-group fraction and within-K ESS (existing native telemetry).
4. Per-layer condition residual/hidden RMS and FiLM scale RMS; condition vs
   backbone post-clip gradient norms, plus fixed-reference-probe velocity MSE.

Audits have a separate fit-step clock; no policy/audit axis mixing. All fixed
probes use private RNG streams. Offline W&B plus local JSON/checkpoints are
default. New W&B display name separates the question from the run ID.

## Local commands

From `/Users/yirenwu/Ztautau/ml_pipeline`, validate without training:

```bash
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.conditioning_transport \
  --output artifacts/dgpo_toy/conditioning_transport_v1 --preflight
```

User-started full three-arm experiment, including final cold audits:

```bash
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.conditioning_transport \
  --output artifacts/dgpo_toy/conditioning_transport_v1 \
  --steps 1000 --wandb-mode offline
```

Use a new output name for another run; no previous experiment is overwritten.
All files remain excluded from NERSC uploads by `NERSC/upload-excludes.txt`.

## Preparation record (2026-09-24)

- Outcome: runner, tests and protocol prepared; new experiment not started.
- Evidence: 52 conditioning/native/reward/audit regression tests passed in
  5.90 seconds. Actual-checkpoint preflight verified exact initial velocity
  and DDIM samples for all three arms. Read-only replay also reproduced the
  saved baseline mode probabilities and all frozen reward scores exactly.
- Decision: conditioning hypothesis **unresolved**, ready for the matched screen.
- Cost: zero new research-training updates; preparation changed no scientific
  hypothesis state, so a research learning-index improvement is not yet claimed.
- Deleted work: repeated reward fitting/calibration, simultaneous frequency/LR
  sweeps, full production runs, and automatic expansion of the search.
- Next limiting factor: measured mode transport with these routes and the
  unchanged coefficient-1 native objective.
- Better next round: select from this one preregistered comparison, then verify
  the winner rather than adding untracked architecture changes.
