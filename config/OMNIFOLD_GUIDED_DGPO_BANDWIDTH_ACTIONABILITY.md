# Critic-bandwidth actionability test

## Question

At the completed old-classifier DGPO endpoint, is a saturated H2 reward still
actionable at radii where the previous saturated-H4 evidence is already weak
or nonlocal, when evaluated by a saturated H4 judge?

This is the first direct test of the proposed coarse-to-fine critic path. It is
a fixed-policy diagnostic: neither arm installs a reward or commits a policy
update.

## Source and new arm

Both arms load weights only from `c4a91e07` policy step 1110:

```text
/pscratch/sd/y/yiren/Ztautau/
dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/
checkpoints/last.ckpt
```

The new arm uses an H2 reward (`topology_max_harmonic: 2`) and an independent
saturated H4 judge. It uses one repeat, two folds, K=8 candidates, the exact raw
LOO objective, five normalized RMS radii, and eight common-noise rollout seeds.
Classifier fits have a 1,000-update minimum and a 3,000-update budget. Its W&B
ID is `h2band01`.

No new H4 arm is needed. The historical saturated-H4 references at the same
step-1110 source are:

- `aqbszk1r`: the fully trained H4 direction was slightly harmful at RMS
  `1e-5` (`plus - zero = +0.000066`) on its original signed panel;
- `qlhcslov`: saturated-H4 improvement was strong at `1e-6`, weaker and
  nonlinear by `1e-5`, where the gradient cosine fell to `0.1825`;
- `h4grad01`: replicated the correct saturated-H4 local sign at RMS `1e-6`;
- `h4b55lr1`: BA55 H4, rather than saturated H4, remained reliable through
  RMS `3e-5` and reversed at `1e-4`.

## Primary endpoint

For each radius, define a reliable H4-gap improvement when:

1. the mean paired `plus - zero` H4 AUC-gap change is negative;
2. its 90% paired bootstrap upper bound is below zero;
3. plus beats zero in at least 6/8 rollout seeds; and
4. plus beats minus in 8/8 rollout seeds.

The H2 arm must first reach a held-out balanced-accuracy 95% lower bound of
0.55; otherwise it does not expose a usable intermediate residual. The
bandwidth hypothesis is supported when saturated H2 is reliable at RMS
`3e-5`. That radius is beyond the established saturated-H4 local regime and is
the largest reliable BA55-H4 radius. Reliability only at `1e-6` is consistent
with H4 and does not license a curriculum. Failure by `1e-5` rejects H2 as the
missing bridge. A reliable `1e-4` result would motivate extending the grid
before closed-loop training.

Reward AUC is secondary evidence. H2 should normally expose no more global
discrimination than H4, but AUC alone does not establish actionability.

## Commands

Run preflight in the Perlmutter shifter environment:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h2_bandwidth_actionability.yaml --check-only
```

Run the single H2 arm:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_h2_bandwidth_actionability.yaml
```
