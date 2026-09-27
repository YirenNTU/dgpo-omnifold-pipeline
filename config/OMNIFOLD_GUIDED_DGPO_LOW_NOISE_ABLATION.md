# H4 DGPO low-noise time-stratum alignment audit

Status: completed on 2026-09-15 as W&B run `h4time01`.

## Single research question

> Is the low-noise quarter of the unchanged DGPO diffusion-time expectation
> substantially more effective at reducing the fixed H4 judge gap than the
> complete time-averaged production direction at the same measured VP policy
> distance?

This is a read-only decomposition. The H4 reward, calibrated-LOO advantage,
K=8 candidate contract, DGPO loss, reference term, source policy, and full
time expectation remain unchanged. No time-reweighted optimizer step is made.

## Motivation

- `h4cfw001` established a large useful H4 ordering on fixed K=8 support.
- `h4grad01` established a reproducible and correctly signed complete
  production gradient, while its small rank-local time-stratum diagnostic had
  cosine only about `0.36–0.45`.
- `h4natg01` rotated the direction substantially at matched VP distance but
  improved fixed-judge efficiency only `1.191x`.

The remaining high-leverage hypothesis is deterministic cancellation across
diffusion noise levels. Fine conditional or periodic H4 residuals may be
actionable mainly when the noisy state retains enough candidate detail.

## Exact time decomposition

The source production interval is `t in [0, 0.7]`. Split it into four equal
uniform bands:

| Direction | Interval | Interpretation |
| --- | --- | --- |
| `low` | `[0.000, 0.175]` | lowest-noise quarter |
| `mid_low` | `[0.175, 0.350]` | second quarter |
| `mid_high` | `[0.350, 0.525]` | third quarter |
| `high` | `[0.525, 0.700]` | highest-noise quarter used by production |

Each band uses the exact calibrated-LOO DGPO loss, 8,192 events, K=8 fixed
anchor-policy candidates, and eight time/noise samples per microbatch. Split
the event panel into two disjoint 4,096-event halves and compute each band on
both halves. Their mean is the band direction; their cosine is the
reproducibility control.

Because the bands have equal width,

```text
g_full = (g_low + g_mid_low + g_mid_high + g_high) / 4
```

is an estimator of the unchanged production time expectation. Also form
`g_rest = mean(g_mid_low, g_mid_high, g_high)`. Save and report the cosine of
`g_full` to the independent saved `h4grad01` `gbar`.

## Identity contract

- Direction estimation deliberately reuses the first `h4grad01` production
  event panel, but uses new candidate and diffusion seeds. Every time band sees
  the same identities and anchor candidates, preventing an event/candidate
  confound.
- VP-distance matching reuses the `h4natg01` curvature identities. It never
  sees judge performance.
- The 2,048 primary judge identities are new: exclude all `h4grad01` gradient
  and probe identities plus both `h4natg01` curvature and judge panels before
  drawing them.

## Functional-distance matching and judge probes

Use `g_full` at parameter RMS `1e-6` to define the cosine-VP path distance.
Independently rescale `low`, `rest`, `mid_low`, `mid_high`, and `high` until
their measured symmetric plus/minus VP distance matches `g_full` within 5%.

On the new judge panel, use identical rollout noise for zero and every signed
arm. The primary endpoint is

```text
low_noise_efficiency_gain =
  (zero_gap - low_plus_gap) / (zero_gap - full_plus_gap)
```

Predeclared gate:

| Result | Conclusion |
| --- | --- |
| Low/full split controls pass, VP match passes, low plus beats minus and rest, and gain >= 2 | Low-noise credit is materially diluted by the complete time expectation. Advance to a separately authorized time-weighted training ablation. |
| Low improves more than full but gain is below 2 | Low noise helps modestly; time cancellation is not yet the dominant explanation. |
| Low does not beat full | The low-noise hypothesis is not supported; do not change production time weighting. |
| Split cosine below 0.8, full control has the wrong sign, or distance match fails | Diagnostic invalid or underpowered; do not interpret the comparison. |

This experiment cannot establish cold-classifier closure. It performs zero
policy updates and zero classifier fits.

## Result

All validity controls passed. Band split cosines were `0.916–0.966`; the full
split cosine was `0.9515`; the reconstructed full direction had cosine
`0.9812` to the saved independent `h4grad01` mean. Every VP-distance ratio was
within 0.5% of one.

The matched plus AUC-gap changes from low to high noise were `-0.0003243`,
`-0.0001292`, `-0.0000405`, and `+0.0002083`. The full change was
`-0.0002359`, giving low/full efficiency `1.3744`; rest changed by
`-0.0000987`. Thus low noise was about `3.28x` more effective than rest and the
highest-noise direction was harmful, but the predeclared primary `2x` low/full
gate did not pass. The official decision is
`low_noise_helps_but_is_not_decisive`.

The result supports a graded noise-dependent credit effect but does not show
that vector cancellation dominates: the cancellation ratio was `0.9154`.
Because the absolute gap changes are small and no paired uncertainty interval
was saved, confirm the signed ordering over additional rollout seeds/panels
before starting a time-weighted training objective.

## W&B contract

```text
project: ytchou97-university-of-washington/nu2flow-RL
id: h4time01
name: c4a91e07_h4_low_noise_time_alignment_v1
group: c4a91e07_h4_projection_diagnostics
```

## Execution

Inside the initialized 16-GPU Ray allocation:

```bash
shifter python3 scripts/diagnose_diffusion_time_alignment.py \
  config/dgpo_10pct_c4a91e07_h4_time_stratum_alignment.yaml
```

Implementation:

- runner: `scripts/diagnose_diffusion_time_alignment.py`;
- config: `config/dgpo_10pct_c4a91e07_h4_time_stratum_alignment.yaml`;
- contract tests: `scripts/test_diagnose_diffusion_time_alignment.py`;
- local verification: 19 time-alignment, geometry, and source-gradient tests
  pass.
