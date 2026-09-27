# Ztautau ml_pipeline — OmniFold-guided DGPO research project

Discuss results in Traditional Chinese unless asked otherwise. Keep run IDs,
metric names, config keys, code, and mathematical terms in English.

Every reply in this repo follows `.claude/skills/plain-language/SKILL.md`:
plain, conclusion-first prose, no default headers/tables/bold-bullet report
formatting unless the content genuinely needs it. This applies to every
answer, not just research discussion.

Load the skill `/omnifold-guided-dgpo` before analyzing a run, proposing an
experiment, or updating the research record. It carries the canonical
references and the experiment lifecycle. Opening the workflow also binds the
session to `.claude/skills/omnifold-guided-dgpo/musk-workflow.md`: one
limiting factor per session, question and delete before adding, fastest
falsifier first, kill criteria, idiot index per run, register results the
same day. The user chose this operating system on 2026-09-17.

## Research goal

Make DGPO reliably turn a strong classifier's signal into a policy-distribution
improvement, the way it did with the old classifier.

Problem statement (user, 2026-09-17):

- The old classifier learned a real signal and, used as an OmniFold-guided DGPO
  reward, corrected low-order marginals and correlations until it could no
  longer separate generated from truth (`c4a91e07`, policy step 1110).
- A stronger classifier (H4) still sees a clear residual at that endpoint
  (raw AUC about 0.89). That residual is the actual target.
- DGPO does not convert the H4 signal into a stable distribution improvement
  the way it converted the old-classifier signal. That transfer failure is the
  bottleneck.

Primary endpoint: a cold, adequately trained H4 audit at policy step `t`,

```text
gap(t) = abs(held_out_AUC(t) - 0.5)
```

An audit with fewer than 1,000 classifier optimizer updates is undertrained and
never counts as closure. Since 2026-09-17, 1,000 updates is necessary but not
sufficient: at a fixed policy AUC keeps rising until about 2,500 updates, so
only audits with matched (or fixed, >= 3,000) realized budgets are comparable.
See `artifacts/c4a91e07_review/PROBLEM_DEFINITION_20260917.md`. Physics, correlation, topology, and response metrics
are secondary unless the user changes the objective.

## Single source of truth

Read these before reasoning about the project; do not duplicate them.

| Purpose | Location |
| --- | --- |
| Chronological experiment registry | `artifacts/codex_skills/omnifold-guided-learning/references/experiment-registry.md` |
| Hypothesis matrix and next test | `artifacts/codex_skills/omnifold-guided-learning/references/current-diagnosis.md` |
| W&B reading rules | `artifacts/codex_skills/omnifold-guided-learning/references/wandb-analysis-protocol.md` |
| Method and literature map | `artifacts/codex_skills/omnifold-guided-learning/references/method-map.md` |
| Provenance of every conclusion | `artifacts/codex_skills/omnifold-guided-learning/references/evidence-sources.md` |
| Predeclared protocols, one per experiment | `config/OMNIFOLD_GUIDED_*.md`, `config/DGPO_*.md`, `config/*_ABLATION.md` |
| Execution configs | `config/dgpo_10pct_c4a91e07_*.yaml`, `config/dgpo_omnifold_ztautau_10pct_*.yaml` |
| Runners and their contract tests | `scripts/train_dgpo_*.py`, `scripts/diagnose_*.py`, `scripts/test_*.py` |
| Immutable W&B snapshots and result notes | `artifacts/c4a91e07_review/` |
| Live evidence | W&B project `ytchou97-university-of-washington/nu2flow-RL` |

The same reference files are exposed to Codex through
`artifacts/codex_skills/omnifold-guided-learning/` and to Claude Code through
`.claude/skills/omnifold-guided-dgpo/` (symlinked). Edit them in one place.

## Research contract

1. One experiment answers one question. Search the registry first; state the
   unresolved causal link; freeze everything else; predeclare the primary
   endpoint, the result table, and the decision rule before launch.
2. Default source snapshot is `c4a91e07` policy step 1110, loaded weights-only
   with fresh experiment state, unless the protocol says otherwise.
3. Preserve the DGPO mathematical objective. Reward transformations, time
   weighting, projection surgery, or a different advantage are new objectives
   and need explicit authorization; they are not diagnostics of the old one.
4. Live W&B logging is required. Never claim an arm ran without a W&B run or an
   output artifact. A `crashed` state means interrupted, not numerically failed.
5. Compare cold audits with cold audits. Label installed reward, warm monitor,
   cold audit, and fixed independent judge separately.
6. Use the labels **supports**, **rules out as a sufficient explanation**, and
   **unresolved** precisely. Report intervention, matched control, source
   checkpoint, valid primary points, outcome, limitation.
7. No SHA bookkeeping. Use checkpoint path, policy step, config validation, and
   W&B provenance.
8. When new evidence contradicts an older interpretation, keep the old
   observation and add a dated correction. Never rewrite history.

## Experiment lifecycle

Each experiment leaves this trail; missing pieces are backlog, not optional.

1. `config/<PROTOCOL>.md` predeclared (question, frozen contract, measurements,
   result table, decision rule, W&B IDs, run command).
2. `config/<experiment>.yaml` with `wandb.id`, display name of the form
   `Question? | arm | endpoint`, and group.
3. `scripts/<train|diagnose>_<name>.py` plus `scripts/test_<name>.py`
   contract tests; `python -m pytest scripts/test_<name>.py` passes locally.
4. `rsync` to NERSC (command in `README.md`), `--check-only` preflight inside
   the 16-GPU Ray allocation, then `shifter python3 ...` launch.
5. Monitor live W&B by `global_step`; check every audit's
   `staleness/raw_audit_training_steps` before reading its AUC.
6. On completion: export a snapshot to `artifacts/c4a91e07_review/`
   (`wandb_run_snapshot.py`), write `<ID>_RESULT_<date>.md`, append a registry
   entry, update the hypothesis matrix in `current-diagnosis.md`, add a
   `## Result` section to the protocol md.
7. Commit with the protocol, config, script, test, and record together.

## Registry backlog (verified against live W&B on 2026-09-17)

These runs exist in W&B but have no registry entry yet. Numbers below were read
from the live history; treat them as pending, not curated.

| Run | State | Verified live reading | Action |
| --- | --- | --- | --- |
| `h4xfer02` | finished, step 50 | cold gaps 30/40/50 = `0.395049 / 0.394925 / 0.279060` (2848/2396/1056 updates); `mean - gap0 = -0.005944` vs `0.362289`; passes the predeclared late-window endpoint only through step 50, whose audit ran 1056 updates (budget artifact band) | registry corrected 2026-09-17; status unresolved pending fixed-budget re-audit |
| `h4rep01` | finished, step 50 | gaps 0/10/20/30/40/50 = `0.371191 / 0.408326 / 0.401674 / 0.404966 / 0.401273 / 0.349291`; `mean(30,40,50) - gap0 = +0.013986`; replication fails; step-50 audit ran 1116 updates | registry corrected 2026-09-17; full-budget audits read 0.90 at every step |
| `h4trust01` | finished | 4 accepted updates, first rejected boundary at proposal 5; endpoint cold gap `0.389377` (1712 updates) vs baseline `0.3711905`; `function_trust/primary/passed = 0` | registry + diagnosis: hard reference constraint at coefficient 0 did not close within the calibrated radius |
| `h4opt01` | finished (v3) | native-minus-raw judge gap `+0.000138`, 95% CI `[+0.000063, +0.000212]`; but `raw_gradient` VP match failed (`all_vp_matches_passed = 0`) | registry with the matching caveat; do not cite as clean |
| `h4b55lr1` | finished | `larger_ba55_radius_supported`, selected RMS `3e-5` (plus `0.1776` vs zero `0.2233`) | registry |
| `h4b55phys1` | finished | decision `base_ba55_radius_preferred`; protocol names a corrected replay `h4b55phys2` that does not exist yet | registry as superseded; run or drop the corrected replay |
| `ba55nat1` | crashed | step-0 audit gap `0.401677` (3536 updates), then crashed before step 1 | relaunch; `ba55rms1`, `ba55lr10` never started |
| `h2band01`, `oldbase01`, `oldr2h201`, `oldr4h201`, `h2r2direct` | crashed | see `config/OMNIFOLD_GUIDED_NESTED_RESIDUAL_SERIES.md` for what each emitted before stopping | series relaunch planned as `oldbase02`, `h2r2direct`, `h2r4direct` |

## Useful commands

```bash
# contract tests for one runner
python -m pytest scripts/test_<name>.py -q

# authenticated compact W&B export (uses ~/.netrc, prints no key)
python artifacts/codex_skills/omnifold-guided-learning/scripts/wandb_run_snapshot.py \
  --run <id> --output artifacts/c4a91e07_review/<id>_snapshot_<YYYYMMDD>.json

# search display names
python artifacts/codex_skills/omnifold-guided-learning/scripts/wandb_run_snapshot.py --search 'BA55' --limit 30
```

NERSC paths, the rsync command, and the Ray/shifter launch pattern are in
`README.md`. The NERSC SSH certificate expires; run `sshproxy` before asking
Claude to read pscratch outputs.
