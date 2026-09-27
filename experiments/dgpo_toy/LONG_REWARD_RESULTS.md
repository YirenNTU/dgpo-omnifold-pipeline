# 3000-update fixed-reward extension — completed, 2026-09-21

Subsequent full-state continuation to10000:
[RESUME_10K_RESULTS.md](RESUME_10K_RESULTS.md). DGPO eventually passes the fixed
reward endpoint. Preserve the3000 observations below, but do not interpret
their plateau as permanent inability or low-ESS causality.

## Outcome

DGPO does not achieve a resolved positive reward gain after10x the original
budget. The pathwise control DOES begin improving late: independent reward
gain+.077022 with a strictly positive confidence interval, and known joint
signal gain+.061977. Thus the same low-ESS learned reward is not completely
unlearnable by this neural diffusion. The prior300-update result was too early
to see this pathwise improvement.

Both gains remain below the predeclared+.10 magnitude threshold. Keep formal
decision **`inconclusive_actionability`** unchanged; do NOT interpret that label
as no detectable improvement in either arm. Statistical evidence for positive
pathwise improvement and the stricter operational pass are different claims.

## Exact trajectory extension

Same saved `low_ess_joint3_v1/reward.pt`, classifier selected at20000; no
classifier fit, temperature or reward recalibration. Same neural source,
seed17, AdamW1e-4, K8/M4, DDIM20, batch64, no additive KL/refit/EMA. Only the
training horizon changes300->3000. Because old files lack AdamW state, each
arm replays from step0. Both step300 weights AND all300 history rows match
the previous run EXACTLY. The same optimizer/RNG instance then continues;
no reset at the boundary. Frozen source/critic weights and buffers are unchanged.

Protocol: [LONG_REWARD_PROTOCOL.md](LONG_REWARD_PROTOCOL.md).

## Independent primary endpoint

New endpoint stream92017,4096 contexts x8 candidates, paired initial/final
noises and contexts. Initial mean reward=-2.172220, std=.789986.

| Arm | Final mean reward | Paired gain | Context-cluster95% CI | Final std |
|---|---:|---:|---|---:|
| DGPO | -2.165436 | +.006784 | [-.000957,+.014526] | .808601 |
| Pathwise | -2.095199 | +.077022 | [+.063746,+.090298] | .950750 |

Joint-signal paired changes: DGPO+.001229,CI[-.000708,+.003165]; pathwise
+.061977,CI[+.056716,+.067238]. This is the toy's known joint statistic,
not a physics/Cij metric or a proof of full distribution closure.

The intervals describe evaluation sampling uncertainty conditional on one
training seed. No independent training-seed replication ran. The final panel
differs from the earlier300-update endpoint, so use the SAME monitoring panel
below for longitudinal numerical comparisons.

## Fixed independent monitoring panel (descriptive, not policy selection)

| Updates | DGPO reward gain | Pathwise reward gain |
|---:|---:|---:|
| 300 | -.001280 | -.002411 |
| 1000 | -.001478 | -.000683 |
| 2000 | -.002852 | +.004224 |
| 3000 | -.000371 | +.076879 |

The pathwise gain grows markedly near the end (monitor at2950/2975/3000 is
+.046708/+.060091/+.076879). This supports delayed learning, not a converged
plateau for pathwise. Do not select the best intermediate point or continue
beyond the declared3000 based on seeing this favorable trend.

## Diagnostic scope

- DGPO clipping0/3000; gate saturation0 throughout; gate mean .498173–.501498.
- DGPO median within-K ESS/N=.807569 and head-gradient-norm ESS/N=.274958.
  Informative-group fraction remains1. Global ratio concentration must not be
  equated to either quantity.
- DGPO median gradient norm=.012326; median actual update L2=.002342.
- Pathwise clipping232/3000, median gradient norm=.729098 and update L2=.002374.
- Frozen-weight ESS on the new endpoint: baseline .001922, DGPO .001410,
  pathwise .002652. Post-update values are not fitted current-policy ratios.

Supports: fixed reward/joint structure can start improving despite very low
ratio ESS;300 updates are insufficient to reveal all learning behavior.
Rules out as sufficient in this tested configuration: increasing DGPO budget
to3000, without another change, as a way to reach the declared+.10 gain.
Unresolved: low ESS as a causal slowdown, DGPO direction/variance versus policy
geometry, later-time behavior and production generalization. Pathwise and DGPO
are different estimators, not compute-matched methods or a high-vs-low ESS test.

Next limiting factor: why DGPO does not show the actionability emerging in
pathwise under the shared fixed reward. A gradient-direction/held-out response
probe can now use these frozen sources/endpoints rather than fitting another
classifier. Do not claim surrogate impossibility from one LR/budget/seed.

## Artifacts and verification

Run directory: `artifacts/dgpo_toy/fixed_reward_joint3_long_v1`.
`report.json` contains all6000 update records and replay checks;
`progress.jsonl` contains live monitoring. `dgpo.pt`/`pathwise.pt` are endpoint
weights; `dgpo_state.pt`/`pathwise_state.pt` additionally contain AdamW, training
RNG and history at the latest saved step. Full-state files are overwritten
only within this named output; earlier runs are preserved.

48 tests passed. Full run185.80s (DGPO83.32s,pathwise101.96s plus baseline).
No production edits, W&B writes, NERSC jobs or classifier fits.
Elon-workflow learning index: about186 local CPU wall seconds to distinguish
persistent DGPO plateau from delayed pathwise improvement. Better next round:
use the saved full states and retain an independent held-out primary endpoint;
do not reset AdamW or change thresholds when extending a trajectory.
