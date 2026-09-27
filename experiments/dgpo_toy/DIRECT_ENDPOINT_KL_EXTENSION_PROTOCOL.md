# Direct endpoint-KL:10k->20k full-state continuation

Declared2026-09-21 before extension. User requested another10k, not a restart
or coefficient sweep. Question: does more training alone reveal accumulation?

Resume `direct_endpoint_kl1_joint3_10k_v1/dgpo_state.pt`, step10000, preserving
model, AdamW moments/clocks, RNG, complete history and density-controller clock.
Apply exactly10000 NEW updates, ending at20000. Initial source/reference/H4
remain `low_ess_joint3_v1/reward.pt` step0; the reference is NOT recentered at
the resume checkpoint. All settings, including coefficient1 and LR1e-4, stay
unchanged. Existing10k artifacts are preserved.

Output `artifacts/dgpo_toy/direct_endpoint_kl1_joint3_resume20k_v1`.
No-KL and velocity-MSE controls are reused at10000, not rerun. Their step
budgets are explicitly recorded and20k-vs10k contrasts marked unmatched.
Primary comparison is the SAME direct-density arm at20000 versus10000.

New final panel97017:4096contexts x8, common contexts/noises at0/10k/20k and
historical controls. Paired-condition reward intervals and500 paired-bootstrap
high-order MSE intervals. Keep monitor90017 and density-monitor190017 unchanged;
no intermediate stopping/selection based on reward. New endpoint never used
for training. Report:

- Original meaningful-alignment criteria versus step0 (unchanged): reward
  gain>=.10 with positive lower95% bound;56-moment RMSE reduction>=10% and
  negative upper95% bound for MSE change; lower-order health preserved.
- Extension-specific detectable accumulation: reward gain since10k lower95%
  bound>0 AND high-order MSE change since10k upper95% bound<0 AND lower-order
  health preserved. All pass=supports_accumulation; otherwise report the
  individual outcomes and unresolved_at_extended_budget. Detectable does NOT
  automatically mean practically meaningful; report continuous effects.

Unchanged lower-order thresholds: max residual mean<=.10, variance error<=.15,
pair covariance<=.08. A negative result at20k is budget-bounded, NOT proof of
permanent stagnation; no single-seed production/low-ESS causal conclusion.

Unchanged numerical stops: global sufficient inverse certificate before and
after every update, checked float64 inverses/Jacobians, independent autodiff
spot checks. Failed validity stops and preserves last valid model/AdamW/RNG;
no projection or critic fallback. Continue despite early plateau.

Verification before/at launch: source10000 clocks/history; exact resumed
boundary reward/std/joint monitor; tests of uninterrupted vs resumed weights,
optimizer/RNG; first new optimizer and density clock10001; unchanged history
prefix. Persist first new full-state update in addition to every1000/final.
Record requested/completed NEW updates separately from total step.

Estimated additional runtime~1h from prior57.38min; no production/W&B writes.
No extra classifier fits, other policy arms, coefficient changes or LR changes.
The implementation work changes comparison/reporting only, not the update.

Launch check:70 regression tests pass. Actual boundary monitor reproduced;
resume_verified at10000 and resume_first_update_verified at10001 emitted.
New policy progress is beyond10000; the run is active, not completed. This
launch round delivered full-state continuation, not a scientific endpoint.
No extra arms were run. Learning about the20k outcome remains pending; the
next measurement is the predeclared independent endpoint, not an early peak.

Completion addendum2026-09-21: the above describes launch-time status. All
10000 new updates have now completed (total20000). Paired reward gain versus
10k is small and positive; high-order improvement is unresolved. See
[DIRECT_ENDPOINT_KL_EXTENSION_RESULTS.md](DIRECT_ENDPOINT_KL_EXTENSION_RESULTS.md)
for the predeclared endpoint and numerical checks. No additional run launched.

```bash
python experiments/dgpo_toy/net_reward_kl.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --comparator artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1 \
  --velocity-comparator artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1 \
  --comparator-steps 10000 \
  --resume-from artifacts/dgpo_toy/direct_endpoint_kl1_joint3_10k_v1 \
  --output artifacts/dgpo_toy/direct_endpoint_kl1_joint3_resume20k_v1 \
  --policy-steps 20000 --coefficient 1 --endpoint-seed 97017
```
