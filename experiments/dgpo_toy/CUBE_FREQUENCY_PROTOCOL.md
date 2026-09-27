# Conditional-frequency difficulty sweep

Prepared, not launched, 2026-09-22. Local only; no NERSC upload.

Objective: find whether conditional complexity suppresses oracle-reward transfer.
One knob: k=1,2,4,8, g=sin(k*pi*c), c uniform[-1,1]. Truth positive parity
p=.5+.4g, reference q=sigmoid(4*logit(.5-.4g)). Each parity has four equally
weighted corners, Gaussian width.15; all single/pair coordinate marginals fixed.
Integer frequencies have identical ideal distributions of g, ratio, coverage
and ESS. This sinusoidal k1 is a NEW baseline, not the old linear-c control.
No oracle-derived input features: diffusion receives raw c only.

For each k, use same model128, initialization17,131072 train/8192val/8192test,
dataset RNG seeds, AdamW, velocity loss and validation selection. Fresh reference
data reflects k; stop after20 stale epochs with cumulative min_delta1e-4, no
fixed epoch cap. Best raw checkpoint, no EMA. Fit length may differ, so this is
not compute matched. Record selected/final updates and heldout velocity MSE.

From selected reference run300 native DGPO steps: fixed categorical oracle,
fixed source velocity reference, coefficient1, fresh AdamW1e-4, K8,M4,DDIM50,
same policy seed17. No classifier, no clipping/tempering/ranking changes.
Do not gate RL on perfect reference fit. Oracle is not actual neural ratio.

Primary: paired heldout oracle reward gain at300, context-level95%CI, seed760017,
8192 midpoint-grid conditions x8 independent noises. Positive lower CI supports
absorption; CI spanning zero is unresolved, not proof of zero effect. Compare
gain magnitude/interval across neighboring k and initial reward SD/ESS/coverage.
No automatic claim of the first failure or stopping at a noisy negative point.
Single seed: intervals quantify evaluation noise, not training-seed uncertainty.

Monitor every50 plusstep1 with seed770017, native gradient panel780017. Log
reward/advantage, gradient norms and conflict, gate saturation, actual weight
ESS, preferred mode mass and K8 hits, conditional structure and corner shape.
Full8mode probabilities/TV use32*k condition bins, not fixed8. All frequencies
have8192 evaluation contexts; higher k therefore has noisier per-bin estimates.
Also report bin-free E[g(c)*parity], whose truth value is.4, to avoid interpreting
histogram sampling noise as worsening fit. Save all per-bin counts/targets and
paired endpoint reward/hit arrays. Preferred truth mass=.5+.8/pi, NOT70%.

Interpretation: baseline reference-TV/structure/coverage mismatch at high k
exposes pretraining/representation difficulty and confounds an RL-specific
claim. Reasonably comparable baselines plus deteriorating reward absorption
support conditional update difficulty, not ESS causality. Better reward alone
does not establish target closure; shape and mode errors remain separate.
No claim that lambda1 is exact endpoint KL or comparable effective regularization
across representations. Offline W&B, separate readable pretrain/policy runs.

Deliverable: four-arm report and a bracket for follow-up, not an automatic
architecture change. Compute cost recorded as pretrain updates plus300 per arm;
no wall-time promise. User launches the sweep explicitly.
