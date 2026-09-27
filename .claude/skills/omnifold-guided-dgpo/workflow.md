# Observed experiment workflow

Reconstructed on 2026-09-17 from `config/*.md`, `config/*.yaml`,
`scripts/`, `artifacts/c4a91e07_review/`, git history, and the W&B project.
Update this file whenever the user's practice changes.

## Naming conventions

- **W&B run ID**: short, stable, hand-chosen, lowercase, 7–10 characters,
  encoding lineage + question + ordinal: `h4xfer01`, `h4xfer02` (continuation),
  `h4rep01` (replication), `h4trust01`, `h4opt01`, `h4b55lr1`, `ba55nat1`,
  `ba55rms1`, `h2band01`, `oldbase02`, `h2r2direct`. Never reuse an ID for a
  relaunch after a crash unless the protocol says the phases append to the same
  run; otherwise bump the ordinal or add a suffix (`_v2`, `phys2`).
- **Display name** (current style): `Question? | arm | endpoint`, for example
  `Can a larger policy step escape the plateau? | BA55 two-fold H4 | checkpoint replay`.
  Older runs use snake_case `c4a91e07_h4_<intervention>_<budget>_v<n>`; both
  are searchable with `wandb_run_snapshot.py --search`.
- **Group**: one per causal stage, e.g. `H4 policy projection`.
- **Protocol md**: `config/OMNIFOLD_GUIDED_DGPO_<TOPIC>.md`,
  `config/OMNIFOLD_GUIDED_<TOPIC>.md`, `config/DGPO_<TOPIC>_ABLATION.md`.
- **Config yaml**: `config/dgpo_10pct_c4a91e07_<topic>.yaml` for c4a91e07
  forks, `config/dgpo_omnifold_ztautau_10pct_<topic>.yaml` for trainer runs.
- **Runner**: `scripts/train_dgpo_<topic>.py` (commits policy updates) or
  `scripts/diagnose_<topic>.py` (read-only replay / probe), always paired with
  `scripts/test_<same>.py`.
- **Result note**: `artifacts/c4a91e07_review/<ID>_RESULT_<YYYYMMDD>.md`;
  snapshots `<id>_summary_<date>.json`, `<id>_history_<date>.json`,
  `<id>_snapshot_<date>.json`.
- **Output dir on NERSC**: `/pscratch/sd/y/yiren/Ztautau/c4a91e07_<topic>_v<n>`.

## Protocol md template

Every protocol has these sections, in this order. A run without a protocol md
is not a registered experiment.

```markdown
# <Title> (`<wandb id>`)
## Question            one sentence; what link it tests; what it does NOT change
## Frozen contract     source ckpt + step, reward, K, reference, optimizer, seeds,
                       panels, identity splits, audit budget
## Measurements        every logged key that the decision uses
## Required result table   empty table with the rows to fill
## Decision rule       pass/fail statistic, threshold, what happens on each branch
## W&B                 id, display name, group, x-axis key
## Run                 exact shifter command, --check-only preflight
## Musk gate           see musk-workflow.md 3.2; required before launch
## Result              appended after the run; dated; includes result gate 3.3
```

Read-only diagnostics say so explicitly: "zero classifier fits, zero policy
optimizer steps, installs zero rewards, checkpoint unchanged".

## Lifecycle

0. **Open.** Fill the session-opening block in `musk-workflow.md` 3.1: limiting
   factor, first-principles claim, deliverable, time box, deletions, registry
   debt. One limiting factor per session (R8).
1. **Question.** Search `references/experiment-registry.md` and
   `current-diagnosis.md`. If the intervention was tried, cite the run and
   state what was missing (control, budget, seed) before proposing again.
2. **Protocol.** Write the md above. Predeclare the endpoint and decision
   before touching code. Pass the Musk gate (3.2): name every requirement's
   owner, list deletions, choose the fastest falsifier, set a kill criterion.
3. **Config.** Copy the nearest existing yaml; change only the frozen-contract
   keys; set `wandb.id`, name, group; set the output dir.
4. **Code.** New runner + contract tests. Tests cover: settings/schema, identity
   split disjointness, provenance of loaded artifacts, endpoint statistic,
   decision gate. `python -m pytest scripts/test_<name>.py -q` must pass on the
   Mac before upload.
5. **Upload.** `rsync` per `README.md` (no `--delete`).
6. **Preflight.** Inside the 16-GPU Ray allocation (16 workers, 1 GPU each):
   `shifter python3 scripts/<runner>.py --config <yaml> --check-only`
   (fail-closed). Expect "Ignored normalizer parameter" and unmatched FAMO
   task-weight warnings when 553 model keys load weights-only; those are fine.
7. **Launch.** Same command without `--check-only`. Multi-phase runners
   (`--phase 20`, `--phase 50`) resume from `last.ckpt` after preemption.
   Export `RUN_DIR` before `ls $RUN_DIR`.
8. **Monitor.** W&B live. Align by `global_step`; for audits require
   `staleness/raw_audit_training_steps >= 1000` and saturation. Interim
   findings can be appended to the registry "Active diagnostic" section with
   the date, marked running.
9. **Close.** Snapshot → RESULT md (with result gate 3.3: label, rows
   changed, GPU-hours, idiot index) → registry → diagnosis → protocol
   `## Result` → evidence-sources → commit. Same day the run ends (R13).
10. **Decide.** The protocol's decision rule names the next experiment; write
    it into `current-diagnosis.md` "Highest-leverage unresolved test".
11. **Session close.** Fill block 3.4: done better, do better next, registry
    debt count.

## Evidence hygiene the user enforces

- Audits < 1,000 updates are undertrained; near-0.5 AUC there is invalid.
- Compare cold with cold; warm monitors are labeled separately.
- Configured ≠ executed. `h4consg1` never existed; `h4ranka1` emitted no
  rank-audit metric.
- A `crashed` run's pre-crash history is evidence; its missing endpoint is not.
- Numbers come with their audit budget and panel; different panels are scale
  comparisons, not exact efficiencies.
- Keep the old observation when a later run changes the interpretation.

## Known pitfalls

- `run.scan_history()` raises "Step column '_step' not found" on these runs;
  use `run.history(keys=..., pandas=True, samples=...)`.
- W&B summary values can come from different times; read the history.
- A W&B payload-normalization bug once dropped exact `gradient_transfer/*`
  values (`h4xfer01`); confirm a new key actually appears in history before
  relying on it.
- The trainer does not expose the process-level rollout RNG as a YAML seed;
  fresh Ray workers give an independent rollout stream.

## Tooling the user has

- Codex skill at `artifacts/codex_skills/omnifold-guided-learning/`
  (same references, `agents/openai.yaml`). Claude and Codex share the files.
- W&B CLI/API authenticated via `~/.netrc`.
- NERSC access via `~/.ssh/nersc` + certificate (expires; `sshproxy`).
- Local pytest for contract tests; no GPU locally.
