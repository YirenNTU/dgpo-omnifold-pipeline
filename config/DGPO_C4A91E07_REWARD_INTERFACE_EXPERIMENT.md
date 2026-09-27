# c4a91e07 H4 reward-interface diagnostic

This is the first one-question experiment after the W&B history and literature
review.  It asks:

> At the fixed 10% `c4a91e07/last.ckpt` policy, does H4 fail to give DGPO a
> stable improving direction because of classifier ordering, or because of the
> scale/transformation applied between classifier logits and DGPO advantages?

It does not continue the old optimizer.  The checkpoint is loaded in
`weights_only` mode and remains the common policy anchor.  The experiment makes
no policy optimizer step, installs no reward, and has no hard trust region.

## Controlled comparison

The checked-in configuration is
[`dgpo_10pct_c4a91e07_h4_reward_interface.yaml`](dgpo_10pct_c4a91e07_h4_reward_interface.yaml).
It pins:

- source W&B run `ytchou97-university-of-washington/nu2flow-RL/c4a91e07`;
- the old 10% run's exact `last.ckpt` path and expected global step 1110;
- nonlinear periodic phi H4, late fusion, no direct Fourier shortcut;
- one K=1 residual ratio increment;
- two cold top-level seeds and two identity-stable folds per seed;
- a separate H4 judge trained on disjoint fit identities;
- one fixed K=8 candidate panel and shared policy-evaluation randomness;
- raw `0.75 × logit`, held-out temperature-calibrated LOO, per-event z-score,
  and centered candidate rank.

Event identities are assigned to five non-overlapping populations:

| Population | Fraction | Use |
|---|---:|---|
| reward fit | 50% | two-fold reward members |
| early stop | 15% | reward-member plateau selection |
| calibration | 10% | scalar temperature; judge early stop |
| judge fit | 15% | independent H4 judge |
| final audit | 10% | K=8 gradients and signed probes only |

Repeated event identities always remain in one partition.  Each classifier is
rebuilt from the pretrained backbone and a new PEFT bank, and each fit creates a
fresh AdamW.  Training runs for at least 1000 updates and at most 3000, with
held-out validation every 100 updates and patience 10.  Checkpoints are selected
by held-out balanced accuracy because this experiment tests candidate ordering;
the disjoint calibration population handles logit scale afterward.

## Run

From the repository root on the configured NERSC Ray allocation:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h4_reward_interface.yaml \
  --check-only
```

The preflight validates the configured `last.ckpt` path, global step, H4
contract, and weights-only load mode.  Each Ray worker loads that declared path
and directly compares every loaded policy tensor with the checkpoint tensor.

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h4_reward_interface.yaml
```

Use `--output-dir` to give a new path for a rerun.  Existing output directories
are never overwritten.  W&B logging creates a separate diagnostic run named
`c4a91e07_h4_reward_interface_fixed_policy_v5`; it never appends to the source
run.  It streams each member's training metrics every 10 updates and held-out
validation metrics every 100 updates, then uploads the final report artifact.

## Produced evidence

`report.json` contains:

- fit, OOF, early-stop, calibration and untouched final-audit classifier scores;
- the held-out temperature and before/after calibration BCE;
- member rank, winner and advantage-sign agreement;
- reward tails and top-1%/top-10% event advantage energy;
- raw policy-gradient norm, split-half cosine, block concentration, pairwise arm
  cosine and the gradient cosine between the two classifier seeds;
- the exact-anchor raw/calibrated scale-equivalence check and a second gradient
  linearization after the normalized `+epsilon` displacement, where the
  nonlinear DGPO gate can first make global reward scale affect direction;
- exact per-event gradient-energy concentration on a 32-event audit subset;
- normalized `-epsilon`, `0`, `+epsilon` common-noise policy probes for every
  reward mapping;
- independent-judge AUC/BCE, marginal and topology JSD, target residuals, and
  truth-normalized response matrices for all signed probes;
- event-paired bootstrap intervals for changes in response-bin displacement.

`fixed_k1_pool.pt`, `fixed_k8_panel.pt` (including all four members' logits), the
four frozen reward-member checkpoints, the independent judge checkpoint, the
merged runtime YAML, source metadata and the manifest make the result replayable.

## Predeclared interpretation

The `+epsilon` direction is gradient descent on the corresponding DGPO loss;
`-epsilon` is the symmetry control.  Lower independent-judge AUC gap and lower
truth-normalized response displacement are treated as local improvement.

| Observation | Diagnosis | Next experiment |
|---|---|---|
| raw/calibrated differ at the exact normalized step | numerical/contract failure, because the anchor directions must be collinear | inspect scale-equivalence and finite-step relinearization |
| raw/calibrated separate only after finite-step relinearization | global logit scale affects the nonlinear gate | a later short scale-specific pilot, after repeatability checks |
| z-score or rank passes while calibrated LOO fails | event-dependent scale or reward tails | 50-step pilot with the winning invariant map |
| seed/fold ordering or gradient direction disagrees | classifier estimator variance | separate 1×2 versus 3×2 ensemble experiment |
| all mappings agree but `-epsilon` wins | shared representation/physics bias | H4/input/condition ablations |
| `+epsilon` fails despite stable reward members | DGPO loss/projection mismatch | inspect loss geometry before any policy training |
| one mapping passes judge and response checks | usable local reward interface | only then run a 50-step frozen-reward pilot |

The signed test establishes a local direction at one checkpoint.  It does not
establish long-run DGPO convergence or final unfolding closure.  The response
matrices here use the four generated tau delta coordinates available to the
policy; the full downstream ROOT response matrix must still be produced after a
short policy pilot with the standard extraction pipeline.
