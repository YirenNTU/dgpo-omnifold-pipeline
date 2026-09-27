---
name: omnifold-guided-dgpo
description: Research memory and workflow for the Ztautau OmniFold-guided DGPO study (nu2flow-RL W&B project). Use when checking a nu2flow-RL run, interpreting H4 audit AUC/gap or ESS trajectories, deciding whether an experiment was already done, designing the next one-question ablation, launching on NERSC, or updating the registry/diagnosis. Covers the old-classifier success c4a91e07, the H2/H4 transfer failure, and the gradient/optimizer/reference diagnostics through the BA55 series.
---

# OmniFold-guided DGPO research skill

Discuss in Traditional Chinese; keep run IDs, metrics, config keys, code, and
math in English. The repo `CLAUDE.md` states the goal and contract; this skill
tells you what to read and how the user actually runs experiments.

## Opening the workflow (mandatory)

1. Read `musk-workflow.md` in this directory. Its Part 2 rules and Part 3
   checklists are binding for the session; fill the session-opening block
   (3.1) before any non-reading tool call and the session-close block (3.4)
   before ending.
2. Then read the references below.

## Read the right reference first

`references/` is a symlink to the shared knowledge base also used by Codex.
Edit files there, never copies.

| Before you... | Read |
| --- | --- |
| compare runs or propose an experiment | `references/experiment-registry.md` |
| explain the mechanism or pick the next test | `references/current-diagnosis.md` |
| open a live W&B run or extract histories | `references/wandb-analysis-protocol.md` |
| connect to literature or a general method | `references/method-map.md` |
| cite a number | `references/evidence-sources.md` |
| write a protocol, launch, or record a result | `workflow.md` in this directory |
| decide what to delete, run first, or stop | `musk-workflow.md` (rules R1–R14, gates 3.2–3.3) |

## The research question

Can repeated classifier-guided DGPO updates make a cold, saturated H4 audit
progressively unable to separate policy samples from truth?

```text
gap(t) = abs(held_out_AUC(t) - 0.5),  audit >= 1,000 optimizer updates
```

The causal chain is evaluated stage by stage; never infer a later stage from an
earlier one:

```text
global discrimination -> usable ratio -> within-event candidate ordering
  -> centered advantage and gradient -> optimizer displacement
  -> policy-distribution change -> fresh best-response H4 audit
```

Established so far: H4 sees a real residual (`b8e2f04h`); H4 ordering helps a
fixed judge on fixed candidates (`h4cfw001`, gap -49%); the production gradient
is reproducible and correctly signed (`h4grad01`); block geometry and low-noise
time weighting are not the fix (`h4natg01`, `h4trep01`); the coefficient-1
reference gradient opposes H4 and removes most of its projection late in a
trajectory (`h4xfer01/02`). Current bottleneck: persistent realization of an
attenuated but correctly signed H4 direction across many updates. See
`current-diagnosis.md` for the hypothesis table and `CLAUDE.md` for the
registry backlog verified on 2026-09-17.

## Evidence rules

1. Priority: immutable downloaded report > exact W&B history aligned by
   `global_step` and `staleness/reward_round_id` > config and logs > W&B
   summary > local config of an unexecuted plan.
2. For each audit report `staleness/raw_auc`, `staleness/raw_auc_gap`, and
   `staleness/raw_audit_training_steps` together. Below 1,000 updates say
   "undertrained" before saying anything else.
3. Distinguish installed reward classifier, warm-started monitor, cold fresh
   audit, and fixed independent judge.
4. `crashed` means interrupted; evidence emitted before the crash still counts.
   An arm without a W&B run or output artifact did not run.
5. Labels: **supports**, **rules out as a sufficient explanation**,
   **unresolved**. Always state the limitation.
6. Default source: `c4a91e07` policy step 1110, weights-only, fresh state.
7. No SHA bookkeeping.

## Designing the next experiment

1. Search the registry for the same intervention and its matched control.
2. Name the single unresolved link.
3. Freeze everything unrelated to it.
4. Predeclare the primary endpoint, minimum audit budget, result table, and
   decision rule.
4b. Append the Musk gate (`musk-workflow.md` 3.2): requirement owners,
   deletions, fastest falsifier, time to first valid point, kill criterion,
   expected idiot index. No gate, no launch.
5. Require live W&B with enough keys to separate reward, monitor, and policy
   clocks.
6. Preserve the DGPO objective unless the user authorizes a new one. Softmax
   tilt, z-score, rank, clipping, time weighting, or projection surgery are new
   objectives, not diagnostics.

Do not re-propose as untested: training H4 longer, holding one H4 reward
longer, ESS control alone, 4 vs 6 members without a matched control, AdamW
reset/keep alone, removing staleness alone, BA70 early stopping, or reading a
356-update near-0.5 audit as closure.

## Checking a run quickly

```python
import wandb
api = wandb.Api(timeout=60)
r = api.run("ytchou97-university-of-washington/nu2flow-RL/<id>")
keys = ["global_step", "staleness/raw_auc", "staleness/raw_auc_gap",
        "staleness/raw_audit_training_steps"]
df = r.history(keys=keys, pandas=True, samples=5000)  # scan_history fails on these runs
print(r.state, r.name); print(df.dropna(subset=[keys[2]]).to_string(index=False))
```

Credentials come from `~/.netrc`; never print them. For a durable export use
`scripts/wandb_run_snapshot.py` (symlinked here) and save under
`artifacts/c4a91e07_review/`.

## Recording a result

After a run has enough valid evidence, in this order (the RESULT note must
include the result gate from `musk-workflow.md` 3.3):

1. Snapshot JSON into `artifacts/c4a91e07_review/`.
2. `artifacts/c4a91e07_review/<ID>_RESULT_<YYYYMMDD>.md`: question, arms,
   valid points, decision, limitation, next decision.
3. Registry entry: date, W&B link, state, display name, source checkpoint and
   step, single question, intervention, valid primary measurements with audit
   updates, conclusion, limitation, which hypothesis changed status.
4. Update the hypothesis table in `current-diagnosis.md` and the "highest
   leverage unresolved test".
5. Add `## Result` to the protocol md in `config/`.
6. Add the run to `evidence-sources.md` if it produced an immutable artifact.

Keep planned, running, crashed-before-endpoint, and completed runs distinct.
Dated corrections, never deletions.

## Keeping this skill current

This skill was derived from the repo trail and W&B on 2026-09-17. When you
observe the user doing something the lifecycle in `workflow.md` does not
describe (a new naming rule, a new launch path, a new audit convention), update
`workflow.md` in the same session and note it in the project memory.
