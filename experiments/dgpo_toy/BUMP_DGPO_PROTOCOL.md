# Nonperiodic conditioning and DGPO reward absorption

Local exploratory pilot, authorized 2026-09-24. No production changes/uploads.

Question: does no-Fourier DGPO fail to absorb a fixed useful reward on the
nonperiodic bump, and does a zero-output Fourier branch rescue it?

Use conditioning_placement's bump source readiness and identical train/val data.
Train a useful imperfect no-Fourier diffusion, then fork None/Early. Identical
initial DDIM samples are asserted. Same optimizer reset, random contexts/noise,
native conditional.dgpo_objective, K8, four t samples in [0,.7], DDIM20,
AdamW1e-4, weight decay.001, clip1, velocity surrogate coefficient1.
This is NOT exact endpoint KL. No differentiable reward backpropagation.

Reward is a frozen binned log truth/source probability ratio: 32 c bins,
64 y bins including infinite overflow tails; truth probabilities use analytic
Gaussian CDF averaged over4096 c grid points. Source counts use8192 independent
random contexts x64 samples and pseudocount.5. This coarse estimated reward
does not equal the exact continuous density ratio. No classifier is fitted.

Pilot budget1000 updates; monitor every100 on2048 fixed independent contexts,
then evaluate a fresh4096-context panel. Both arms have equal budget.
The300-step point is interim only; do not call permanent failure from it.

Primary: final paired context-level mean reward gain. Practical margin is
5% of initial K32 candidate exponential-tilt reward headroom. Baseline stall
requires its upper95% gain bound below this margin; rescue requires Fourier
gain AND Fourier-minus-None lower95% bounds above the margin. Negative gain
can pass the stall test but means deterioration, not a plateau. Intervals are
pointwise conditional on trained models, not training-seed uncertainty.

If None improves, the failure is not reproduced. Do not tune solely until
Fourier wins. If both fail, inspect reward support and gradient/reference
conflict before changing target or architecture. Any next intervention must
be labeled exploratory and preserve this result. A discovered rescue requires
fresh training seeds before making a general claim.

Artifacts: source.pt, fixed reward.pt, per-arm model/optimizer/RNG snapshots,
progress.jsonl and report.json. Current CLI does not implement resume.

Validation before launch: exact wrapper/model forward equivalence, arbitrary
batch shapes through native DDIM/DGPO, finite loss and nonzero adapter gradient.
