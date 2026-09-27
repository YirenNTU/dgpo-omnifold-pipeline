# Direct endpoint-density net reward — predeclared 2026-09-21

Completion update (does not change the protocol): completed10000 updates with
no validity stop; see [final results](DIRECT_ENDPOINT_KL_RESULTS.md). The
launch notes below retain their original pre-completion status as history.

Supersedes the unexecuted learned-KL-critic draft at the user's request. NO
learned KL estimator, online BCE fit, or shadow critic. H4 remains approximate.

Question: can direct conditional endpoint log(q_current/q_step0), subtracted
BEFORE unchanged leave-one-out advantage and detached DGPO gate, preserve
reward transfer and improve high-order truth alignment?

Same source low_ess_joint3_v1/reward.pt, step0 pretrained neural DDIM20, frozen
H4 joint3, seed17, AdamW1e-4/decay.001, batch64/K8/M4, t uniform[0,.7], clip1.
Budget10000, coefficient1. No EMA/refit/tempering/pathwise KL/velocity penalty.
Detach candidates and density scores. Reuse completed no-KL and velocity1
controls; no-KL SUCCEEDED at reward transfer, not required to align truth.
This comparison changes both regularization distance and its routing.

Density: continuous DDIM defined by actual weights/schedule in float64,
evaluated at the SAME actual float32 rollout points. Reverse each Ax+Bv step
by contraction; require global sufficient |B|Lip(v)<A, margin>1e-8, from
spectral norms and conservative SiLU derivative bound1.1. Sum analytic step
log|det J| and Gaussian latent log density. This is numerical validation, NOT
interval arithmetic or a continuous density of rounded floating-point samples.

Inverse max128 iterations/step, a-posteriori step error bound<=1e-11, full
forward residual<=1e-8, full-map sampled minimum singular value>=1e-8.
Require finite scores and positive step determinants. Global certificate both
before AND after every actor update; independent full-chain autodiff Jacobian
check on8 fixed samples initially/step1/every250/final, maxerror<=1e-8.

Tests before run: affine Gaussian ground truth, analytic/autodiff Jacobian,
independent existing whole-map inverse, ratio orientation, exact zero at
source, detachment/RNG isolation, full resume and default-off equality,
controlled numerical failure and last-valid preservation.

Stop on failed density validity/nonfinite, not plateau. Preserve last-valid
model/AdamW/RNG/history; save failed post-update attempt separately. No weight
projection, tolerance relaxation, uncertified inverse branches, learned
fallback, or continued optimization. Certificate failure is NOT proof of
noninvertibility. Early stop=unresolved at10k, actual steps reported and
comparisons against10k controls explicitly budget-unmatched.

Primary:56 high-order Fourier moment errors, independent96017 panel,
4096contexts x8, paired context bootstrap500. Improvement requires upper95%
MSE-change bound<0 AND >=10% RMSE reduction. Regularized-arm lower-order
requirements: max residual mean<=.10, variance error<=.15, pair covariance<=.08.
Raw H4 transfer: gain>=.10 with lower95%>0. These are diagnostic thresholds,
not proof of distribution closure. Report continuous values regardless.
All pass=joint_alignment_progress; transfer only=reward_transfer_without_alignment;
otherwise=no_decisive_alignment. Numerical stop=unresolved_density_validity_stop.

Raw monitor90017 every25; structure every250. Direct KL every250 on a separate
190017 panel (512contexts x8), full final endpoint panel. Raw/net advantage
size/cosine/sign flips, head-gradient concentration, clipping/gate saturation,
density residuals/conditioning/certificate margin. No monitor selects a model.
Full state every1000/stop and in-memory last-valid snapshot every update.

Exact reward ratios give r_net=log(p/q_current), constant at truth and zero
centered reward-driven DGPO gradient. NOT proof of surrogate convergence;
H4 approximation and AdamW decay remain. No production/NERSC/W&B changes.
Benchmark CPU cost before full run; initial engineering/run allowance45min,
reassess only runtime, not scientific budget, after the benchmark.

## Implementation verification / launch record (not a completed result)

-68 tests passed (entire toy suite), including analytic Gaussian density,
 independent inverse/autodiff, score orientation, net-score cancellation at an
 exactly known target, zero-KL update equality, detachment, full resume,
 pre-update failure and post-update failure recovery.
-Actual saved12D source: independent full-chain Jacobian max discrepancy
 <1e-14; inverse reconstruction ~1e-14; identical current/reference score0.
-Equivalent derivative-matrix layout reduced forward-Jacobian512-row time
 from~.12s to~.08s. No mathematical approximation was introduced. Estimated
 complete10000-update CPU runtime45–55min, dependent on inverse iterations.
-Launched direct_endpoint_kl1_joint3_10k_v1. First actual policy update matches
 completed no-KL history exactly. Progress contains early plateau points;
 this is NOT the independent final endpoint and NOT evidence of failure.

Implementation-round closeout: delivered tested direct-density mechanism;
measurement correctness supported by independent instruments. Training outcome
UNRESOLVED until completion or validity stop. Removed the unexecuted learned
KL-critic/shadow-fit design. Learning index: one local engineering round and
~3s final regression run resolved instrument/implementation readiness, not
the causal learning question; full experiment cost remains pending. Next
limiting factor: whether this score accumulates joint improvement over the
predeclared10000 updates. Process correction: validate the density instrument
before interpreting KL-regularized training.
