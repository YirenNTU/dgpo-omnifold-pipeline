# Oracle marginal repair diagnostic — 2026-09-21

Question: does pure-strong high-order improvement survive pooled residual
marginal repair? No training, distillation, classifier fitting, or production changes.
Sources: old low_ess_joint3_v1/reward.pt initial and saved
fixed_reward_joint3_long_v1/dgpo_state.pt (verified3000),
fixed_reward_joint3_resume10k_v1/dgpo_state.pt (verified10000);
new truth_diffusion_earlystop_v1/best_model.pt and
truth_three_arm_v1/strong_no_kl/state.pt (10000).
The new runner retained only its final state, so no new early endpoint is claimed.

Fit coordinate ECDF maps on32768 fresh conditions/noises seed240017.
Subtract known conditional mean, map pooled residual ranks to standard-normal
quantiles, add known mean back. Evaluate16384 independent paired conditions/noises
seed250017; same inputs for baseline and policies. Midrank knots, linear CDF
interpolation, bounded tails; ties and out-of-calibration fractions reported.
Finite ECDF tails/ties mean exact strict-monotone copula preservation is not guaranteed.
No evaluation data enters map fitting.500 paired event bootstraps; intervals are
conditional on the calibration maps, not calibration/training-seed uncertainty.

Predeclared endpoint: repaired moment MSE lower than identically repaired baseline
with upper95CI<0, plus pooled KS<.03, maxmean<.05, maxvarianceerror<.10.
The supports_two_stage field means this narrow endpoint ONLY; not covariance or
conditional-distribution closure. Conditional mean bins and covariance are secondary
failure checks; pooled marginal repair cannot guarantee either.

Completed results (111 tests passed):
- Old3k: repaired RMSE.49162 vs repairedbaseline.49194; unresolved.
- Old10k: rawRMSE.36452 -> repaired.49072; vsbaseline MSE CI
  [-.003259,+.000698], unresolved. Marginals pass but covariance max.9858,
  conditional-bin mean max.9239 remain severely wrong.
- New10k: rawRMSE.34005 -> repaired.47374 vs repairedbaseline.49190;
  MSE-change CI[-.019863,-.015322], narrow endpoint passes. Covariance max1.0042,
  conditional-bin mean max.5017 remain severely wrong.

Conclusion: supports partial surviving moments for new endpoint, rules out pooled
marginal repair alone as sufficient distribution repair. Old apparent high-order
gain mostly disappears under this intervention, not proof all dependence was absent.
Cost: one local evaluation round, no training; two hypothesis updates. Next unresolved
link: condition-dependent/covariance repair without destroying surviving moments.
Do not deploy or distill this output as a successfully repaired distribution.

Run: python -u -m experiments.dgpo_toy.marginal_repair --output artifacts/dgpo_toy/marginal_repair_v1
Full report: artifacts/dgpo_toy/marginal_repair_v1/report.json
