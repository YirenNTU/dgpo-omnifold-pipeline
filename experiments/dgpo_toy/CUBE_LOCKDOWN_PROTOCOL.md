# Function-matched k8 mechanism study — 2026-09-22

User authorizes autonomous local execution. Bound: six arms x three policy RNG
seeds x3000 updates =54000 updates, no automatic extra experiments/retraining,
no remote jobs or uploads. Offline W&B. Stop with a report even if unresolved.

## Question and controls

Can a condition-feature intervention rescue reward transfer starting from the
EXACT SAME k8 generator, and is the reference penalty sufficient for the plateau?
Source cube_frequency_v1/k8/pretrain/best_pretrain.pt, raw selectedstep17408.
Do not pretrain/refit any arm. Freeze source reference and categorical oracle.
Same AdamW1e-4/WD.001/clip1, batch64,K8,M4,DDIM50,tUniform[0,.7], native DGPO.

Three feature bases x coefficient0/1, crossed, not bundled interventions:
- raw: original denoiser with eight zero features (dummy zero-gradient adapter).
- polynomial: orthonormal LegendreP1..P8 on raw condition.
- Fourier: sin/cos at frequencies1,2,4,8, multiplied sqrt2.

Each adds the SAME bias-free8x128 projection to the first preactivation,
zero-initialized. Base parameters identical and all trainable. Analytic feature
means0/variances1; no evaluation fitting. Polynomial is the equal-size/unit-scale
control, not an equivalent high-frequency basis. It cannot eliminate every
geometric difference; geometry is the intervention, not pure capacity proof.
Frequency8 is deliberately included, a toy mechanistic intervention informed
by known structure, NOT a physics-prior-free production recommendation.

Verify exact initial velocity and full DDIM samples on a heldout probe, identical
parameter counts, and equal raw-network gradients for raw versus native. Abort
before running arms if exact initial function matching fails. This locks initial
coverage, ESS, reward scale, reference and source fidelity across all arms.
Same RNG stream per policy seed17/23/41 gives matched c/noise/t/eps; after policies
diverge candidates appropriately differ. Fresh AdamW in every arm; no reset midway.

## Endpoints and decisions

Primary independent paired oracle-reward gain at3000, plus fixed300 checkpoint
for the horizon question; never select checkpoint/arm using evaluation data.
8192 midpoint-grid conditions xK8, seed880017; native monitor seed860017 and
512contexts for inexpensive updates every300. Save all states every300 and all
primary reward/hit/moment arrays at300/3000. No evaluation feedback into training.
Report strict pair contrasts Fourier-raw V1, Fourier-polynomial V1,
polynomial-raw V1, rawV0-rawV1 for each policy seed and both horizons.

Use context-level SE with z3.5 (conservative individual intervals) and require
the relevant result in ALL3 rollout seeds; do not pool candidate observations or
pretend 3 seeds represent pretrain-seed uncertainty. Material margin .05 raw
oracle logreward fixed before runs, a diagnostic threshold not physics tolerance.
Small gain means whole interval within[-.05,.05], not merely p>0.05.

- Delay supported if rawV1 has upper bound<.05 at300 and lower>.05 at3000
  in all seeds. No claim that further delays are impossible otherwise.
- Penalty sufficient for raw plateau at this budget only if rawV0 gains>.05,
  rawV1 lies within small band, and rawV0-V1 gains>.05 in all seeds.
- Frequency-basis rescue supported if FourierV1 gains>.05 and beats BOTH rawV1
  and polynomialV1 by>.05 in all seeds. Still an intervention on representation
  and optimization geometry, not proof of a unique universal bottleneck.
- Polynomial rescue reports generic feature-basis benefit separately.
- Otherwise preserve unresolved conclusions; no adaptive new architecture sweep.

Secondary: preferred mass, bin-free E[sin(8pi*c)*parity] truth.4, modeTV over256
bins, near-corner fraction/ESS, native advantage, clipping/gates and full-network
main/reference gradient diagnostics. Shape cannot silently veto reward transfer,
but reward-only success cannot be called distribution closure.

Oracle uses ideal categorical reference, not actual neural ratio; identical in
all arms. Do NOT use pathwise gradients of this sign-based reward: almost-everywhere
zero derivative would manufacture an invalid control. This study isolates basis,
penalty and tested horizon; it does not by itself compare alternate DGPO objectives.

## Costs and closure

Single immutable source,18 arms, no new pretraining, no classifier fit. Record
elapsed seconds/update counts and produce machine-readable decisions. Cost per
mechanism changed is reported only after results, never assume a decisive result.
Do not extend beyond3000 without a new scientific question and authorization.
