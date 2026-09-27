# Velocity-MSE coefficient 1, 10000 updates — completed 2026-09-21

**Primary finding: no-KL PASSES the reward-transfer test.** Its purpose is to
establish that the fixed classifier signal can enter diffusion, NOT to recover
truth without a distribution constraint. Reward and selected high-order moments
improve after delayed onset. Unconstrained marginal drift does not invalidate
that success. The new velocity-MSE1 arm, in contrast, suppresses reward AND
selected high-order learning at this budget; it has not achieved truth alignment.

## Matched intervention and validity

One new step0->10000 DGPO arm, same frozen joint3 reward and pretrained conditional
neural DDIM20 source `artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt`.
Only additive loss changed: production legacy `.5*mean((v-v_step0)^2)`, coefficient1.
AdamW1e-4,weight-decay.001,batch64,K8,M4,t uniform[0,.7],seed17 unchanged.
Reference/critic fixed; no refit, EMA, architecture change or pathwise rerun.
Reused completed no-KL model/history from fixed_reward_joint3_resume10k_v1.

58 tests pass, including production formula/stop-gradient, exact first update,
gradient-diagnostic noninterference, comparison provenance, and truth-moment
instrument validation. Real first update matches ALL old no-KL telemetry exactly;
comparator reproduces its old monitor exactly. Frozen source checks pass.
New arm runtime299.86s on one CPU thread, final full optimizer/RNG snapshot retained.

## Independent paired endpoint (stream94017;4096 contexts x8)

Both arms re-evaluated on this NEW panel; intervals cluster the eight candidates
within context. Differences from older no-KL endpoint values reflect new samples,
not changed weights. No checkpoint selected by endpoint or maximum reward.

| Measurement | Initial | No KL,10000 | Velocity coefficient1,10000 |
|---|---:|---:|---:|
| Frozen reward mean | -2.181755 | -.704602 | -2.181558 |
| Reward gain | — | +1.477152 | +.000196 |
| Reward gain95%CI | — | [1.431883,1.522422] | [-.000282,.000674] |
| Mean triple-phase cosine;truth=.841712 | -.003736 | .458798 | -.003812 |
| Triple harmonic RMSE to truth | .439443 | .334499 | .439477 |
| Cross-triple RMSE to truth | .557242 | .402078 | .557272 |
| Combined higher-order moment RMSE | .493384 | .364997 | .493416 |
| Max marginal residual mean magnitude | .010149 | **8.049632** | .012160 |
| Max marginal variance error;truth variance1 | .014159 | **116.469510** | .015743 |
| Max pair covariance magnitude;truth0 | .014562 | **76.319869** | .014104 |

Higher-order evaluation uses56 sin/cos moments: first4 harmonics of4
context-corrected triple phases, and sum/difference harmonics between triples
(six-coordinate correlations). These features NEVER enter policy loss.
Truth targets are computed from the declared mixture and von-Mises moments,
including shared mixture membership for cross-triple expectations.

Velocity-minus-no-KL reward=-1.476956,95%CI[-1.522228,-1.431685]: declared result
`reward_suppressed_by_velocity`. The velocity arm also fails the original+.10
reward-gain threshold; no-KL passes. Selected structural MSE difference is
+.110237,paired-bootstrap95%CI[.107117,.113453],worse for velocity.
Versus initial, no-KL structural MSE improves-.110205,CI[-.113195,-.106488];
velocity change+.000032,CI[-.000028,+.000086] is unresolved.500 bootstrap draws.
These are evaluation-panel intervals, not training-seed uncertainty.

## Expected unconstrained drift: diagnostic, NOT a veto of no-KL transfer

User clarified the interpretation after completion: no-KL is the transfer
control; truth alignment is the purpose of the KL-regularized arm. Do not
silently change the no-KL success criterion to distribution closure. The
following marginal findings contextualize the unconstrained objective only.

The no-KL mean reward improvement is real. Its selected joint moments also get
closer to truth, but the model does NOT preserve lower-order distributions.
On94017 its coordinate variances span1.922–117.470 rather than approximately1;
34.54% of generated coordinates have |x-mu(c)|>5, and89.44% of generated vectors
have at least one such coordinate. Initial and velocity models have zero such
coordinates on this panel. This is not just a single rare event.

Read-only exploratory recheck on additional seed95017 confirms the anomaly:
no-KL max mean offset8.1550,max variance119.1413,89.47% vectors with any |z|>5.
Initial and velocity variances remain near1. The saved no-KL3000 checkpoint,
evaluated without further updates, has max mean offset.3202 and max variance
1.0864 on94017; the10000 distortion is far more severe.

Pure fixed-reward maximization need not recover truth. These results are
consistent with optimizing selected reward features while departing from the
reference's marginal support. Large residuals also approach saturation of the
Gaussian-CDF Fourier inputs; whether learned-critic extrapolation specifically
causes the drift is NOT isolated here. Do not call all no-KL gains fake, and do
not call a scalar cosine improvement full high-order-distribution recovery.

## Gradient and trajectory interpretation

No-KL has a long plateau, then learning, a sharp dip, and renewed growth;
8500->10000 monitor gain1.186349->1.458546. Not a sustained decline.
Velocity arm stays near its start throughout10000. Last1000 no-KL clips999
updates versus0 for velocity; median gate saturation.2461 versus0. Stability
alone is not success when there is no learning.

Velocity arm last1000 monitored pre-Adam gradient medians: main norm.010569,
penalty norm.001984,per-batch norm ratio.16724,cosine-.04091,total-on-main
projection.98950. Thus telemetry does NOT show near-complete per-minibatch
gradient cancellation. The matched intervention demonstrates trajectory
suppression, not a proven microscopic cancellation mechanism. Norms/cosines
of stochastic batches do not establish the mean-gradient or AdamW mechanism.

## Round closure

- Outcome: completed matched10k arm plus reward/high-order/marginal evaluation.
- Evidence: independent paired endpoint, structural bootstrap, saved checkpoints.
- Decision: **supports** effective no-KL reward transfer and coefficient1 legacy
  velocity regularization suppressing that accumulation in this toy. No-KL passes
  its intended control, despite expected unconstrained drift. The regularized
  arm has not improved truth moments. Production cause and low-ESS causality remain
  **unresolved**; coefficient1 does not make this surrogate exact endpoint KL.
- Learning index:299.86 CPU-seconds /1 new regularization hypothesis update ~=300s/update
  (training/evaluation runtime; excludes implementation and exploratory recheck).
- Deleted: no classifier refit, pathwise rerun, new feature bundle or production run.
- Next limiting factor: the KL formulation/relative gradient scale must permit
  demonstrated reward transfer while making the regularized objective align to truth.
- Better next round: keep the two success criteria separate—no-KL tests transfer,
  regularized training tests truth alignment; retain joint/marginal diagnostics.

Artifacts: `artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1/` contains report.json,
progress.jsonl,dgpo.pt,dgpo_state.pt,trajectory.png. See VELOCITY_KL_PROTOCOL.md
and README.md for the exact command. No W&B or production files changed.
