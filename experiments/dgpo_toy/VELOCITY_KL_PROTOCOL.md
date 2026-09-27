# Velocity-MSE coefficient 1 — declared before execution, 2026-09-21

Question: does legacy velocity reference regularization stabilize fixed-reward
learning or suppress its accumulation in the same low-ESS conditional toy?
The no-KL curve has a long plateau followed by volatile but recovering growth;
a peak-to-end decrease alone does not establish sustained deterioration.

One new arm, from the ORIGINAL pretrained step-0 neural generator and frozen
joint3 Fourier classifier in low_ess_joint3_v1/reward.pt. Train 10000 updates,
seed17, AdamW1e-4/weight-decay.001, batch64, K8, M4, DDIM20, t uniform[0,.7].
All other settings match fixed_reward_joint3_resume10k_v1. Reuse its completed
no-KL DGPO checkpoint/history, no new classifier or pathwise training.

Only change: L_total = L_DGPO + 1 * .5 * mean((v_current-v_step0)^2).
Execute production build_reference_trust_loss(objective='velocity_mse').
Use the same detached on-policy candidates, t and eps as the main objective;
all dimensions active, reference fixed at step0, gradient only through current
velocity. This is the legacy velocity-MSE surrogate, NOT exact endpoint KL,
NOT VP-path KL, NOT the removed beta_kl denoising loss. Coefficient1 does not
imply gradient balance or a truth-distribution optimum.

Primary comparison: at10000, mean fixed-reward difference velocity minus no-KL
on NEW paired endpoint stream94017,4096 contexts x8 candidates. Evaluate both
saved policies with identical contexts/initial noise. Context-cluster95%CI:
above0 supports improved reward, below0 supports reward suppression at this
budget, overlaps0 unresolved. Also retain original operational success rule
for each arm: reward gain from step0 >=.10 and lower95% bound>0.
No best-checkpoint selection, no adaptive changes, no extension after results.

Monitor stream90017 every25 updates, train stream3017; neither endpoint nor
monitor affects optimization. Log same reward/std, gate, clipping and ESS.
Added full-network component-gradient norms/cosine/cancellation at monitor
cadence; these are pre-Adam/pre-clipping and are not actual update alignment.
Existing head-gradient ESS still refers to DGPO MAIN term only. Known joint
signal is secondary, not full distribution closure. No audit classifier fit.
User amendment before execution: high-order structure is ALSO an explicit
outcome, not reward alone. Evaluate 1st–4th harmonic sin/cos moments of each
context-corrected triple phase, plus sum/difference sin/cos between triples
(six-coordinate correlations). Compare moment RMSE to the known toy truth;
paired context bootstrap500 of velocity-minus-no-KL squared moment error,
negative upper95% bound supports closer structure, positive lower bound worse.
Report both policies versus initial too. Check marginal residual means,
variances and pair covariances for collateral damage. These are evaluation
features only, not a new training reward or truth-guided policy constraint.
Closer selected moments does not prove full joint-distribution closure.

Late stability can be described using all matched monitor points in last1000
steps, not chosen peaks; it cannot override the primary reward comparison.

Save full optimizer/RNG state every1000, verify frozen source and exact first
update against no-KL (penalty gradient initially zero). Zero coefficient must
preserve legacy trajectory. Unit tests cover production formula/detachment,
component gradients, diagnostic noninterference and comparison provenance.
Stop at10000 or nonfinite/error. Budget ~15–25 minutes local including tests,
no NERSC/W&B/production writes. One seed gives a toy conditional comparison,
not production attribution or a causal test of low ESS itself.
