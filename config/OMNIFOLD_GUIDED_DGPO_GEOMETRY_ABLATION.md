# OmniFold-guided DGPO block-natural-gradient geometry audit

Status: completed on 2026-09-15 as W&B run `h4natg01`.

## Single research question

> At the same measured cosine-VP path distance, does a target-preserving
> block-subspace natural-gradient direction convert the established H4 DGPO
> gradient into a larger fixed-judge improvement than the ordinary Euclidean
> direction?

The H4 reward, calibrated-LOO advantage, DGPO loss, source policy, and gradient
are unchanged. This experiment changes only the metric used to convert the
already-computed gradient into a parameter-space direction. It performs no
optimizer step and fits no classifier.

## Evidence selecting this experiment

- `h4cfw001` showed a fixed-candidate H4 reweighting gap change of `-0.12080`.
- `h4grad01` showed that the exact production gradient is reproducible: mean
  pairwise cosine `0.95524`, minimum `0.91110`, half-mean cosine `0.97506`.
- Every `h4grad01` plus probe improved the fixed judge and every minus probe
  worsened it, but the four-gradient mean changed the gap by only `-0.00517`.

Therefore additional copies of the same gradient estimator are not the next
test. The unresolved link is parameter geometry or weak local policy
sensitivity.

## Immutable sources

- Reward-interface directory:
  `/pscratch/sd/y/yiren/Ztautau/c4a91e07_h4_reward_interface_v5`.
- Production-gradient directory:
  `/pscratch/sd/y/yiren/Ztautau/c4a91e07_h4_production_gradient_reproducibility_v1`.
- Policy: c4a91e07 weights-only checkpoint at global step 1110.
- Gradient: saved `gbar` from `h4grad01`; all four component gradients must be
  present and reproduce the stored mean.
- Fixed independent judge: the completed H4 judge from the reward-interface
  diagnostic.

Validate paths, schemas, policy step, parameter names/shapes, W&B provenance,
and file metadata. Do not introduce a SHA requirement.

## Identity split

Reconstruct and exclude every event used by the four `h4grad01` gradients and
its signed-probe panel. From the remaining `final_audit` identities, draw:

- 2,048 curvature events for the VP metric;
- 2,048 disjoint evaluation events for the fixed-judge probes.

Both totals divide equally over 16 GPU workers. Curvature events never select
the direction by judge performance, and evaluation events never estimate the
metric.

## Preconditioner

Partition `gbar` by the saved trainable parameter layout. Drop only blocks with
zero gradient. For each remaining block, define a basis vector containing that
block's gradient and normalize it to unit full-vector parameter RMS.

On the curvature panel, generate one fixed K=8 candidate set. Draw eight
importance-sampled cosine-VP path times and common diffusion noise. Estimate
each basis response by a central `1e-6` parameter-RMS finite difference. Their
masked velocity-response Gram matrix is the empirical Gauss–Newton/Fisher
metric `F` in this block subspace.

Let `q = B^T gbar`. Define the natural direction by:

```text
c = (F + lambda I)^(-1) q
d_natural = B c
```

Choose `lambda` without judge feedback: use the smallest nonnegative ridge
that makes the regularized metric condition number at most 100, with an
additional floor of `1e-6 * max_eigenvalue`.

## Matched functional distance

The vanilla arm is the Euclidean `gbar` descent direction at parameter RMS
`1e-6`. Measure its symmetric mean plus/minus cosine-VP path distance on the
curvature panel.

Scale `d_natural` until its symmetric measured VP distance matches the vanilla
distance within 5%. Scaling changes step length only; the preconditioned
direction is fixed before any judge evaluation. Abort interpretation if the
match is outside tolerance.

## Evaluation and primary endpoint

Evaluate common-noise zero, plus, and minus rollouts for:

- vanilla `gbar`;
- block natural gradient.

Use the disjoint fixed independent H4 judge. The primary endpoint is:

```text
efficiency_gain =
  (zero_gap - natural_plus_gap) / (zero_gap - vanilla_plus_gap)
```

Because measured VP distance is matched, this is the fixed-judge improvement
per equal local policy movement.

Predeclared decision:

| Result | Conclusion |
| --- | --- |
| Distance match passes, natural plus beats its minus, and efficiency gain >= 2 | Blockwise policy conditioning is a material bottleneck; advance this geometry to a short update trajectory. |
| Natural beats vanilla but gain is 1–2 | Block geometry helps but is not yet sufficient; refine the metric within dominant blocks before closed-loop training. |
| Natural does not improve over zero or does not beat vanilla | Blockwise conditioning is not the missing mechanism; test within-block curvature or an intrinsically weak DGPO surrogate. |
| Distance match fails or natural plus loses to minus | Diagnostic invalid or locally nonlinear; do not start a training run. |

This local experiment cannot claim fresh-classifier closure. A passing result
only authorizes a short update trajectory followed by a cold H4 audit trained
for at least 1,000 optimizer steps.

## Result

The run finished with a VP-distance ratio of `0.98271`, so the 5% matching gate
passed. Vanilla reduced the fixed-judge gap by `0.0012716`; block natural
reduced it by `0.0015148`. Both plus directions improved and both minus
directions worsened. The resulting efficiency gain was `1.1912`, below the
predeclared decisive threshold of 2, and the emitted decision was
`block_geometry_helps_but_is_not_decisive`.

The five-block metric condition number was only `18.51`. Its natural direction
rotated substantially from `gbar` (cosine `0.62518`) but produced only a modest
additional judge improvement. Coarse blockwise conditioning contributes, but
does not explain the main classifier-to-policy efficiency loss. No long
closed-loop run is authorized by this result; the next clean test is a finer
within-block empirical metric under the same matched-distance protocol.

## W&B contract

```text
project: ytchou97-university-of-washington/nu2flow-RL
id: h4natg01
name: c4a91e07_h4_block_natural_gradient_v1
group: c4a91e07_h4_projection_diagnostics
```

Required summary keys:

```text
geometry/primary/vanilla_plus_gap_delta
geometry/primary/natural_plus_gap_delta
geometry/primary/efficiency_gain
geometry/primary/vp_distance_ratio
geometry/primary/direction_cosine
geometry/primary/decision
```

## Execution

Inside the initialized 16-GPU Ray allocation:

```bash
shifter python3 scripts/diagnose_block_natural_gradient.py \
  config/dgpo_10pct_c4a91e07_h4_block_natural_gradient.yaml
```

Implementation:

- runner: `scripts/diagnose_block_natural_gradient.py`;
- config: `config/dgpo_10pct_c4a91e07_h4_block_natural_gradient.yaml`;
- contract tests: `scripts/test_diagnose_block_natural_gradient.py`;
- local verification: 14 geometry and source-gradient tests pass.
