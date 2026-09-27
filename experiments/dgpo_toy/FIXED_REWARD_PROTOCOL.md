# Fixed learned reward transfer — declared before policy execution, 2026-09-21

The user explicitly requests fixing the learned joint3 classifier and starting
DGPO. This is a NEW question/contract, not a retroactive pass for the previous
ratio-fidelity experiment. Preserve its `inconclusive_setup` report unchanged.

Source: `artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt`, classifier selected at
step20000 by validation BCE; initial neural DDIM copied from that same file.
Held-out AUC .947441, BCE .206318, raw ratio ESS/N .001590, top1% mass .930791.
Known limitations: log mean ratio .771115; truth logit RMS error1.166700.
These failed the earlier .50/1.0 fidelity thresholds. They are retained as
diagnostics, NOT vetoes for improving the user's chosen approximate reward.
No centering, calibration, temperature, clipping or retraining of the reward.

## Question, controlled conditions and endpoint

Can DGPO improve this exact frozen classifier logit reward on independent
contexts/noise? Same pretrained conditional12D neural DDIM20, K8, M4,
batch64, beta_dgpo1, production LOO-unscaled with detached gate, shared
time/noise within groups, t uniform[0,.7]. No additive KL/reference trust,
refit, scheduler or EMA. A frozen source still enters the DGPO gate, which
is part of the original surrogate and is NOT an additive regularizer.

300 updates; fresh AdamW LR1e-4, decay.001, clipping1. Keep the predeclared
pathwise actionability control: identical source/critic, rollout streams,
K, optimizer and update budget; gradient flows through DDIM and the frozen
critic. This is a different estimator and is NOT compute matched.

Source report must pass all original setup checks except `ratio_normalization`
and `truth_logit_rms`, explicitly waived for THIS approximate-reward question.
Do not edit old reports or let a general force flag bypass other failed checks.
The critic and source generator weights/buffers must remain exactly unchanged.
Numerical nonfiniteness stops the run; no endpoint-based early stopping.

Primary: paired final-minus-initial mean reward on4096 NEW contexts timesK8,
normal95% interval clustered over contexts. Seed streams: training seed+3000
(same across arms), monitor seed+90000, endpoint seed+91000, distinct from
classifier fitting/validation/density/confirmation streams. Monitor every25
updates and first update; step100 secondary. No best-policy selection.

Meaningful improvement remains gain>=.10 logits AND lower95% bound>0. DGPO
passes: `reward_improves`. Only pathwise passes: `dgpo_transfer_deficit` at this
budget. Neither passes: `inconclusive_actionability`. Numerical/setup failure
is separate. One seed is a pilot, NOT proof of production cause or of low ESS
being causal; no matched high-ESS intervention is included.

Reward/std, ratio ESS, within-K ESS, advantage, gate, clipping, update norms,
head-only per-context gradient concentration/cancellation and known joint
signal are diagnostics. Post-update ESS uses the FROZEN critic: it is NOT
a current-policy fitted density ratio. Joint improvement is not required for
the reward endpoint, but helps flag reward exploitation. No fresh classifier
audit, classifier closure, truth convergence or physics claim follows.

Local CPU only. Write a new output report/history/checkpoints; no production
edits, W&B writes or NERSC job. Smoke2 updates exercises code only.
