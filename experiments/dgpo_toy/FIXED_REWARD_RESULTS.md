# Fixed learned reward transfer — completed local pilot, 2026-09-21

Subsequent authorized3000-update extension: [LONG_REWARD_RESULTS.md](LONG_REWARD_RESULTS.md).
The original300-update observations below remain unchanged; pathwise starts
improving late in the longer run, so its early plateau is not permanent.

## Outcome

The observable plateau is reproduced in this toy: the fixed low-ESS classifier
generalizes, but300 policy updates give no detectable independent mean-reward
increase. The pathwise control also fails, so the formal decision is
**`inconclusive_actionability`**, not a DGPO-specific transfer deficit or proof
that low ESS caused the plateau.

Source `artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt`, selected classifier
step20000. No new classifier fitting. Both arms start from the exact saved
pretrained neural conditional DDIM20. Classifier/source weights and buffers
remain exactly unchanged. Full configuration and seed streams are in the report.

Protocol: [FIXED_REWARD_PROTOCOL.md](FIXED_REWARD_PROTOCOL.md).
User-authorized question is improving this approximate learned reward. The
earlier ratio-fidelity failures remain recorded; they are not retroactively
declared passed. No calibration, scaling, clipping or temperature is applied.

## Primary endpoint: NEW4096 contexts x8 candidates

Shared initial mean reward=-2.172394; std=.791304. Initial empirical ratio
ESS/N=.001318 on this panel. The larger classifier confirmation panel had
ESS/N=.001590; sample-size/panel differences matter for heavy tails.

| Arm | Final mean reward | Paired gain | Context-cluster95% CI | Reward std |
|---|---:|---:|---|---:|
| DGPO | -2.173334 | -.000940 | [-.003317,+.001436] | .787021 |
| Pathwise control | -2.174084 | -.001691 | [-.005585,+.002204] | .785285 |

Neither reaches the declared meaningful improvement of+.10 with a positive
lower confidence bound. The intervals span zero; do not call the tiny negative
point estimates reliable harm. These intervals describe endpoint sampling
uncertainty conditional on this one trained trajectory, not training-seed
variability. No best checkpoint was selected.

Independent monitoring at step100: DGPO gain=-.000535, pathwise=-.000966,
both intervals span zero. Step300 monitor gains=-.001280/-.002411, again
consistent with no improvement. Thus the early100-step plateau persists to300
under this tested optimizer/LR/budget; this does not prove permanent stagnation.

Known joint-signal endpoint changes are also unresolved: DGPO+.000078,
CI[-.000471,+.000628]; pathwise+.000248, CI[-.000739,+.001236]. No physics or
fresh-classifier closure claim follows.

## What the diagnostics say (all300 DGPO updates)

- Ratio weights are extremely concentrated globally, but median within-K
  ESS/N=.8074. Global ratio ESS is NOT a per-context candidate ESS.
- Median last-layer per-context gradient-norm ESS/N=.2750, not~.0013.
  This head-only measure is NOT a whole-network effective sample size.
- Median gradient-norm mass of the top ceil(1%*64)=1 context is .1254.
  Median head gradient cancellation ratio is .2362. Neither establishes a
  cause without a matched reference or gradient reproducibility measurement.
- Every batch has100% groups with reward range>1e-3; advantage is not zero.
- Gate mean stays .499737–.500334; saturated fraction is always zero.
- DGPO clipping occurs0/300 times. Median gradient norm=.011628 and actual
  AdamW update L2=.002215; parameters really update.
- Pathwise median gradient norm=.695439, median update L2=.002376;
  clipping occurs7/300 times. Larger raw gradient norm does not imply larger
  effective AdamW steps or better reward transfer.

Supports: an executable conditional neural toy with a generalizing, concentrated
learned reward and a300-update pure-reward plateau. Rules out gate saturation,
DGPO clipping, zero within-context reward and accidental frozen policy as
explanations of THIS trajectory. Does not isolate low ESS or DGPO surrogate
as the cause: both estimators fail at the chosen LR/model/budget, and no matched
high-ESS intervention or training-seed replication ran.

Next limiting factor: determine whether the shared neural parameterization has
a reproducible held-out reward-improving direction at the initial checkpoint.
A split-gradient plus/minus held-out probe would distinguish noisy estimation
from an actionable direction with an ineffective finite update. Do not start
another classifier fit or claim an LR/representation cause from these data.

## Reproduction and verification

```bash
python experiments/dgpo_toy/fixed_reward.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --output artifacts/dgpo_toy/fixed_reward_joint3_v1 --seed 17
```

Output: report.json, progress.jsonl, dgpo.pt and pathwise.pt. Policy checkpoints
contain endpoint model weights/config/source, not resumable optimizer states.
The named output files are overwritten on rerun; the classifier source is
protected from same-directory overwrite. `--smoke` exercises2 updates only.

45 tests pass. Full run21.16 seconds (DGPO9.67s, pathwise11.08s plus baseline).
No production files, W&B runs, NERSC jobs or datasets changed. One pilot seed.
Learning index: about21 CPU wall seconds for one outcome-state change—plateau
observed, estimator-specific attribution unresolved. Avoided another classifier
fit; next round should measure direction actionability before adding architecture.
