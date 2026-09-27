# Oracle-assisted partial joint warm-start, then learned H4 DGPO

Source old low_ess_joint3_v1/reward.pt raw Gaussian-teacher initial. Same denoiser.
Warm-only pathwise loss: squared error of56moment means to .3*truth moments,
+5*(residualmean squared sum + covariance-minus-I squared sum),
+10*univariate sorted-normal squared error, +.1*ordinary truth velocity MSE.
Warm AdamW3e-4/WD.001, batch512,clip1,DDIM20; max3000 updates.
Known moments/mean/marginals are deliberate oracle assistance, not production.

Selection every100 on4096 new-seed410017 monitor: jointmean(.1,.5), regionhit>2x
baseline, mean<.1,varerror<.15,paircov<.08. First passing checkpoint only.
Independent confirmation16384 seed420018 and K128 coverage4096 seed420017:
same gates plus paired anyK8 improvement lower95CI>0. No classifier or DGPO if
gate fails; no post-hoc relaxation. This tests one warm-start recipe, not all oracle
initializations. Region all4 phaseerrors<1rad; not full support coverage.

After passing, remove oracle loss entirely. Retrain exact old strong joint3 with
existing streaming fit budget/config/seed17, validationBCE checkpoint selection.
Data from warm model vs complete truth. Freeze selected classifier; run10000
ordinary DGPO updates,seed17/train3017/monitor90017,AdamW1e-4,K8,M4,velocityMSE.1.
Reference is warm-start model, never refit. Endpoint440017 held-out, moment
bootstrap440018. Primary ownreward absorption AND reliable momentMSE improvement
vs warm without low-order deterioration; report all errors and CIs, no ESS-only
causal claim. Source unchanged check, full optimizer/RNG states every1000.

No oracle scoring or sample injection in DGPO loss. True structure appears only
in evaluation callbacks. No new architecture or postprocessing. Raw weights only.

Run: python -u -m experiments.dgpo_toy.oracle_warmstart --output artifacts/dgpo_toy/oracle_partial_joint_v1
