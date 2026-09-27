# H4 low-noise saved-direction replication

Status: completed on 2026-09-15 as W&B run `h4trep01`.

## Single research question

> Does the `h4time01` ordering—low noise beats the complete direction while
> high noise harms the fixed H4 judge—replicate across new rollout noise on
> evaluation identities that did not estimate the time directions?

This is a read-only confirmation. It loads the saved `low`, `full`, `rest`, and
`high` directions and their already matched parameter RMS values from
`h4time01`. It performs no gradient computation, classifier fit, optimizer
step, or change to the DGPO objective.

## Confirmatory panel

Use the union of:

- the 2,048 `h4grad01` signed-probe identities;
- the 2,048 `h4natg01` fixed-judge identities.

Both panels are disjoint from the 8,192 identities used to estimate the
`h4time01` time directions. Exclude the original `h4time01` judge panel from
the primary test because its result selected this confirmation.

Generate K=8 candidates for the 4,096-event union with eight new rollout seeds.
Within each seed, zero and every plus/minus arm use common random numbers.
Report union and constituent-panel metrics; the union is primary.

## Primary statistics

For each seed, define `delta_arm = plus_gap_arm - zero_gap`. Lower is better.
The paired low-versus-full value is `delta_low - delta_full`.

Across the eight seeds report:

- mean low, full, rest, and high deltas;
- fraction of seeds where low beats full;
- fraction where high is harmful (`delta_high > 0`);
- aggregate low/full improvement ratio;
- deterministic 90% percentile bootstrap intervals over the eight paired seed
  values for low-minus-full and high delta.

Predeclared decision:

| Result | Conclusion |
| --- | --- |
| Low improvement is positive, low/full gain >= 1.25, low beats full in >= 75% of seeds, high harms in >= 75%, low-minus-full CI upper < 0, and high CI lower > 0 | Noise-dependent credit ordering replicates; authorize a short explicitly time-weighted training ablation. |
| Mean signs and consistency fractions pass but either CI crosses zero or gain < 1.25 | Pattern persists but effect remains uncertain; do not start a long training run. |
| Mean ordering or consistency fails | `h4time01` was panel/noise-specific; do not change time weighting. |

This experiment still cannot establish cold-H4 closure.

## W&B contract

```text
project: ytchou97-university-of-washington/nu2flow-RL
id: h4trep01
name: c4a91e07_h4_low_noise_replay_v1
group: c4a91e07_h4_projection_diagnostics
```

## Execution

```bash
shifter python3 scripts/diagnose_low_noise_replay.py \
  config/dgpo_10pct_c4a91e07_h4_low_noise_replay.yaml
```

The runner logs one W&B history row after every completed rollout seed and
uploads the final manifest and report as a diagnostic artifact. Local unit
tests cover exact settings, event-panel disjointness, saved-direction/RMS
provenance, deterministic bootstrap intervals, and all three decision paths.

## Result

The predeclared decision is `h4time01_ordering_not_replicated`. Full improved
the primary union gap by `0.0029247` on average versus `0.0025349` for low;
low/full gain was `0.8667`, and low beat full in 2/8 seeds. The paired 90%
interval for low-minus-full was `[+0.0001670,+0.0005976]`. High improved in all
8/8 seeds, with mean delta `-0.0024297` and 90% interval
`[-0.0027780,-0.0020818]`. The run does not authorize time-weighted training.
