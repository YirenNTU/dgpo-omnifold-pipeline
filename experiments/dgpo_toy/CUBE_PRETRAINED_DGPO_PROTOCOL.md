# Can DGPO refine a Fourier truth-pretrained diffusion?

Status: COMPLETED, 2026-09-22; offline W&B `g878cw0z`. All three3000step
continuations pass the primary reward rule, joint diagnostics and corner guards.
Results: `artifacts/dgpo_toy/cube_pretrained_dgpo_v1/RESULTS.md` (503.8seconds).
The user explicitly
authorized this local DGPO launch after implementation. Local-only follow-on to
`cube_truth_fourier_v1` (offline `mr15pebb`); no NERSC job or cloud sync.

## One question and scope

Can native DGPO absorb the remaining fixed mode-level correction after Fourier
pretraining? Keep the learned architecture, source, fixed reward, reference,
AdamW, and objective intact. No new classifier, truth pretraining, EMA,
reference refresh, reward scaling/tempering, sample selection, or alternative loss.

Use the selected RAW Fourier checkpoints from all three previous seed pairs:
seed17 epoch42/step10752; seed23 epoch62/step15872; seed41 epoch48/step12288.
Load the complete adapter state, never zero it again. The raw-condition models
are not another arm in this round. Each pretrained checkpoint uses its same
integer seed for policy randomness: three full pipelines, not repeated RL seeds
on a single source. No direct comparison of raw reward gains across different
reward functions as if they had the same available improvement.

## The fixed reward must refer to the ACTUAL source

The old `ModeReward` used truth/configured-reference probabilities. Here the
pretraining distribution IS truth, so that choice would give log(1)=0, or using
the historical biased reference would push an already repaired difference.

Instead define the following categorical approximation, for condition bin b
and sign mode m of y:

```
r(y,c) = log p_truth(m | b(c)) - log q_hat_initial(m | b(c))
```

- 256 equal-width c bins; eight sign modes. The truth probability is integrated
  analytically within each bin from `.5+.4*sin(8*pi*c)`, uniform within parity.
- Independently sample two panels from the actual initial model: each has
  8192 midpoint conditions x64 independent DDIM50 candidates =524288 samples.
  The two panels total 1,048,576 samples: 4,096 per condition bin.
- Normalize pooled counts with symmetric pseudocount .5 per mode. This is a
  stated finite-sample smoothing choice; save both original count tables.
- Freeze the table throughout RL. The numerator is oracle-known, but the
  denominator is estimated and binned. Do NOT call this an exact neural density
  ratio or an oracle for radial/within-mode discrepancies.
- The dense fixed condition grid approximates each source-bin average; a small
  bin/grid approximation and Monte Carlo uncertainty remain. No calibrated
  truth samples are fed to training, and no classifier is fitted.
- Calibration, training, native monitors, structure monitors, and primary
  endpoint use separate random streams. No calibration samples enter endpoint
  measurements. The endpoint noise is fixed before training for paired changes.

Before policy updates, evaluate an independent full-bin reweighting diagnostic:
ratio normalization, split-table centered-reward agreement, conditional mode TV
before/after weighting, and available reward gain. This is NOT K=8 resampling,
a trainable policy result, or a claim that the reweighted distribution is exactly
representable. Direction/headroom is reported, not used to secretly change the
reward or select another checkpoint. Weak headroom makes a negative DGPO result
inconclusive about policy transfer. Only numerical/provenance failures abort.

## Unchanged DGPO configuration

- Three fits, each3000 updates. Fresh AdamW lr1e-4,weight_decay.001,clip1.
- Batch64 conditions, K8 detached candidates, DDIM50.
- M4 shared time/noise draws per condition; t~Uniform[0,.7].
- Native unscaled leave-one-out advantage and native detached nonlinear gate.
- Fixed initial checkpoint is the reference; velocity-MSE surrogate coefficient1
  with the existing native half-MSE normalization. Not exact endpoint KL.
- All denoiser parameters, including the trained Fourier adapter, can update.
- Save full model/optimizer/RNG/history state at1,every100,and milestones.
  Same-output `--resume` restores this state; no extra updates after a completed
  final state, even if final evaluation was interrupted.

## Measurements and decisions

Primary endpoint is paired fixed-reward gain at3000, evaluated on8192 held-out
midpoint conditions x32 independent candidates. Context is the independent unit
for a paired normal-approximation95% interval; K candidates are averaged within
context, not counted as separate independent conditions. Reward absorption is
supported if the lower bound is positive for ALL three completed sources.
This measures reward absorption, not full truth closure.

Also save this panel at300 and1000; no checkpoint selection or early stopping
uses these diagnostic endpoints. Report separately:

- conditional eight-mode TV and parity RMSE;
- conditional moment error, truth moment .4;
- near-corner fraction (shape guardrail: no drop greater than .02);
- marginal mean/variance, off-diagonal covariance, corner width;
- reward ESS, mean ratio, tails, within-K ESS, advantage magnitude and gates;
- main/reference gradient norms, cosine and total-on-main projection every100;
- gain divided by initial independent fixed-support reward headroom (diagnostic,
  not a success threshold or bound; overoptimization can exceed it).

Because earlier TV was near its sampling floor, add an independent-half squared
joint-error diagnostic. In each bin, split K32 candidates into16+16 independent
halves covering the same contexts, and compute
`sum_m[(qA_m-p_m)*(qB_m-p_m)]`, averaged over bins. Conditional on the context
grid, its expectation is the squared mode-probability error without the usual
plug-in noise-floor term. Finite estimates can be negative: never clamp or take
a square root. This is a secondary point estimator, not a confidence interval,
and does not change the original pretraining report/decision.

Reward success, joint success, and shape preservation are separate report flags.
No inference that reward improvement necessarily means physics improvement.
Nothing in this toy establishes a production EveNet/H4 solution or an ESS cause.

## Logs and launch

Offline W&B by default, project `dgpo-toy`, group
`Conditional cube post-pretraining DGPO`. Display name:
`Can DGPO refine pretraining? | Fourier cube | frozen mode ratio | V MSE 1`.
Separate seed/step axes and calibration-half clocks. No cloud sync or NERSC upload.

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.cube_pretrained_dgpo \
  --source artifacts/dgpo_toy/cube_truth_fourier_v1 \
  --output artifacts/dgpo_toy/cube_pretrained_dgpo_v1 \
  --seeds 17 23 41 --steps 3000 --eval-every 100 \
  --threads 2 --wandb-mode offline
```

Append `--resume` after interruption. Do not launch a second process against an
active output. Models, frozen count tables, initial evaluations, policy state,
and primary panels are saved under each seed. Source artifacts remain unchanged.

## Prelaunch implementation record (historical)

- Outcome: one bounded continuation prepared; no scientific result yet.
- Evidence: 55 unit/smoke/resume/regression tests passed; all three real source
  loaders preserve every selected checkpoint tensor exactly.
- Decision: unresolved until an authorized run produces primary endpoints.
- Learning index: implementation cost /0 scientific decisions, infinite so far.
- Deleted: additional classifier, retraining, arbitrary reward rescaling and extra arms.
- Next limiting factor: residual mode reward transfer with the fixed reference.
- Better next round: distinguish actual-source ratio from the old configured
  mixture ratio and report reward gain separately from distributional change.
