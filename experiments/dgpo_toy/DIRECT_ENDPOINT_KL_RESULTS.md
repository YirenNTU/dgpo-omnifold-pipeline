# Direct endpoint-KL net reward: completed 10,000 updates

2026-09-21; local CPU, seed17; no W&B or production changes. Completed in
3442.74s (57.38min), no validity stop. Output:
`artifacts/dgpo_toy/direct_endpoint_kl1_joint3_10k_v1/`.
Protocol: [NET_REWARD_KL_PROTOCOL.md](NET_REWARD_KL_PROTOCOL.md).

## Intervention and provenance

Same `low_ess_joint3_v1/reward.pt` pretrained neural DDIM20 and frozen learned
H4 joint3 reward; step0 start; AdamW1e-4/decay.001, batch64/K8/M4, seed17.
Subtract coefficient1 * direct log(q_current/q_step0) before the unchanged
DGPO leave-one-out advantage/gate. No learned KL estimator, pathwise KL
gradient, velocity penalty, feature change, refit, or intermediate tuning.
The learned-KL-critic draft was never executed and was replaced.

Independent final panel96017:4096 conditions x8 candidates, common random
numbers across saved policies. Reward intervals are paired by condition;
56-moment MSE changes use500 paired-condition bootstrap replicates. No
monitor-based checkpoint selection. These intervals measure evaluation
sampling uncertainty, NOT variation across training seeds.

## Final measurements

| Policy | Reward gain vs initial (95% CI) | 56-moment RMSE | Max residual mean | Max variance error | Max pair covariance |
|---|---|---:|---:|---:|---:|
| Initial | 0 | .491982753 | .009114 | .014201 | .010784 |
| No-KL10k | +1.454135 [1.408399,1.499872] | .367544276 | 8.252569 | 118.544558 | 77.654631 |
| Velocity-MSE1,10k | +.000224 [-.000266,.000715] | .491950121 | .012414 | .013975 | .013777 |
| Direct endpoint net reward1,10k | +.000418 [-.000641,.001477] | .491908483 | .012239 | .015479 | .014381 |

Direct arm higher-order MSE change=-.0000730731,
95%CI[-.000187795,+.0000484361]; RMSE reduction only.01510% (declared threshold10%).
Joint cosine change=-.000109499,95%CI[-.000364751,+.000145753].
Raw reward std .782744->.788163, not a decrease. Raw ratio ESS/N remains low:
.00170536->.00175903. Reference KL estimate=.000777239,
95%CI[.000384512,.001169966], with no learned estimator (expectation is still
Monte Carlo). Policy stays close to reference by this measurement.

Direct-minus-velocity reward difference=.000193255,
95%CI[-.000752650,.001139160]; structural MSE difference=-.0000409657,
95%CI[-.000142928,.0000643870]. No resolved advantage over velocity1.

## Instrument and checkpoint verification

All10k steps: maximum current inverse residual3.55e-14, minimum sufficient
step-injectivity margin.277954, zero gradient-clipped steps, zero gate
saturation. Final margin.290319; final independent autodiff logdet error5.58e-15.
No numerical validity condition failed. These are checked float64 numerical
calculations, not interval-arithmetic proofs.

First update exactly matches no-KL. Frozen source/reference and H4 checks pass.
Verified10000 sequential history entries, all AdamW clocks10000, density
clock10000, final weights identical between `dgpo.pt` and `dgpo_state.pt`,
and exact checkpoint/report history match. Full state supports later resume.
Implementation regression suite previously passed68 tests.

## Interpretation and bounded decision

Predeclared result `no_decisive_alignment`: reward transfer criterion false,
high-order improvement criterion false, lower-order preservation true.
This rules out this specific direct-density/routing/coefficient1 intervention
as sufficient for the declared meaningful improvement WITHIN10k in this seed.
It does not prove permanent failure, absence of any microscopic change,
or that more training cannot help. Logged mean monitor gains drift weakly from
.000063 in steps1–1000 to.000833 in9001–10000; correlated reused monitor points
are not independent replicates, and the fresh final interval still includes0.

No-KL remains a successful reward-transfer control. Its unconstrained marginal
drift does NOT invalidate that success; matching its raw-reward speed was not
the regularized arm's endpoint. KL may change learning timescales. The present
evidence cannot distinguish slower progress from a nearby equilibrium.
The intervention changes both KL formulation and routing compared with the
velocity arm. No isolated microscopic cancellation, low-ESS causal result,
or production diagnosis is established. H4 remains an approximate ratio and
the DGPO surrogate is not an exact truth-KL gradient.

Round closeout: completed, valid negative endpoint at the declared budget;
one57.38 CPU-minute run resolves sufficiency at10k, not eventual convergence.
No new critic fit or architecture sweep was added. Next unresolved link:
slow accumulation versus an equilibrium near the reference. No continuation
beyond10k was launched. Any longer run must preserve optimizer/RNG state and
be labelled a newly authorized budget extension, not a completed result here.
