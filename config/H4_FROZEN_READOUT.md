# Frozen early-representation readout test

## BCE-primary diagnostic follow-up (h4bceq1)

Interpretation correction after h4readb1: its full-batch holdout BCE improved
0.691143 (update 50) -> 0.690710 (200) -> 0.690789 (300), despite falling
AUC. The previous AUC gate diagnoses readability, **not** density-ratio
quality. Do not call this classifier collapse solely from AUC. The next round
preserves the exact standardized linear-head optimization and adds logging
only; no new architecture or policy update. Primary endpoint is held-out
balanced BCE at 200, with 50/100/300 and the null BCE log(2) as context. Do
not select the best holdout checkpoint or use ridge AUC as a success gate.

```bash
shifter python3 -u scripts/diagnose_h4_readout_bce.py
```

CPU only, reuses the step-50 panel. Three identical minibatch controls and one
full-batch fit run 300 updates. W&B h4bceq1 / output readout_bce300 are separate
and exclusive. Diagnostics at every 25 updates include:

- fit/holdout balanced BCE, class-specific BCE, gap and auxiliary AUC;
- per-sample loss improvement from initialization, gross gain versus gross
  deterioration, balanced fraction improved, and top-1% share of positive gain;
- feature-norm cohorts using a p99 threshold fitted on fit rows only;
- generated-only log-ratio quantiles, untempered ESS/N, top-1% weight mass and
  log mean ratio (ideal population E_policy[ratio]=1, but finite-panel
  agreement is not proof of calibration).

Saved per-row logits/labels/masks at 0/50/100/200/300 allow matched loss
attribution over 0->50, 50->100, 50->200 and 200->300. Concentration is computed
on gross positive gains, not unstable fractions of near-zero net improvement.
This is descriptive; no event IDs were saved, so no independent-event CI or
statistical significance is claimed. ESS is only a concentration diagnostic,
not a validity gate. These validation-panel readouts cannot establish final
ratio consistency, useful candidate ranking, or DGPO closure.

## Matched full-batch follow-up to h4readq1

h4readq1 finished the 300-update screen. At capture 50 / head update 200,
unstandardized encoder heads stayed around AUC 0.491–0.492. Standardized
seeds 17/29/43 reached 0.829964/0.882074/0.693570; at update 300 they were
0.876570/0.869995/0.574730. The all-seed primary gate failed. Seed43's fit
and holdout AUC both fell while BCE improved; this is not typical train-only
overfitting. Scaling matters but did not yield reliable ordering.

The next comparison reuses **only step-50 encoder features** and reruns all
three standardized minibatch seeds alongside one deterministic full-batch
head. All heads start at zero, use fit-only standardization, AdamW LR 2e-4,
weight decay 1e-3 and 300 updates. Full-batch loss is exactly
`0.5*mean(positive BCE) + 0.5*mean(negative BCE)`, preserving the expected
minibatch objective even with unequal class counts. No clipping or scheduler.
Do not treat identical deterministic full-batch runs as independent seeds.

Primary update stays 200. Prespecified stability window is 200/225/250/275/300:
all points must have AUC >= max(ridge AUC -0.02, original head AUC +0.03),
BCE <log(2), and AUC range <=0.05. Ridge must be >=0.65. Full-batch passes
while at least one matched minibatch seed fails supports minibatch variability
as a contributor **on this panel**. Both pass means instability did not
replicate. Full-batch failure means removing sampling variability was not
sufficient; it does not uniquely prove a conditioning or objective cause.
This is an equal-update comparison, not equal examples or compute;
`rows_processed` is logged to expose the difference. No final-audit claim.

```bash
shifter python3 -u scripts/replay_h4_frozen_readout.py --batch-ablation
```

Run once on CPU, no capture/GPU rerun. W&B **h4readb1**, resume=never;
output `h4_frozen_readout_capture/readout_batch300/`. Historical results
are recorded in the new run's config, but the causal control is the rerun
minibatch arm on the same saved tensors. Existing reports are not overwritten.

Status: implemented, not executed on NERSC. No DGPO policy update, reward
installation, architecture change, or independent closure claim.

## Quick early-plateau mode (300 updates)

Use `--quick` for an early-plateau screen. Capture stops at exactly 300
updates (steps=min_steps=300; no saturation requirement), exports the same
0/50/100 panels, and otherwise preserves the original protocol. Replay caps
each head at 300 updates. Primary readout update 200 and all decision thresholds
stay unchanged. No conclusion about final generalization or saturation follows.

The existing h4frzcap1 has already reported all three exports. Reuse those
panels without rerunning capture or waiting for its full classifier fit:

```bash
shifter python3 -u scripts/replay_h4_frozen_readout.py --quick
```

This writes W&B h4readq1 and `readout_replay_quick300`, leaving the original
3000-update replay and capture unchanged. Run once in a separate shell if the
original command chain is still active; do not also launch a duplicate replay.
Updating code does not shorten an already-running capture.

For a future clean quick capture on 16 GPUs:

```bash
shifter python3 scripts/train_h4_frozen_readout.py --quick --validate-only
shifter python3 scripts/train_h4_frozen_readout.py --quick
shifter python3 -u scripts/replay_h4_frozen_readout.py --quick \
  --panels /pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_quick300/panels
```

Quick capture uses separate W&B h4frzq1 and output directory. The replay ID
h4readq1 is single-use regardless of panel source; do not run both examples
into the same W&B run. Choose one source, not both.

## Evidence motivating this experiment (verified 2026-09-20)

| Run | Observation | Constraint on the next experiment |
|---|---|---|
| [h4clf02](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/h4clf02) | Raw H4 linear shortcut plus nonlinear fusion; first validation AUC >=0.85 at update 1280; completed 3000 updates; final test AUC 0.907521 | Failed the historical 960-update acceleration screen. Do not repeat a raw direct shortcut as a new idea. This was not a simultaneous matched control. |
| [h4clflb1](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/h4clflb1) | Last-block nonlinear late fusion; 3000 updates; final test AUC 0.906931 | Similar endpoint is descriptive, not equivalence: trainable modules and regularization differ. |
| [h4clfpath1](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/h4clfpath1) | At update 50, encoder nested-ridge holdout AUC about 0.809 versus fusion about 0.495 | A standardized ridge solution can read early encoder features. It does not prove minibatch BCE can read them quickly or that fusion destroys information. |
| [oldr2h201](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/oldr2h201) | At 1700 old-base updates train BA 0.7664 versus val BA 0.5094 | Simply training the legacy model longer is not the next question. |

Other provenance: bg5av8mr mixes direct/staged training and short audits;
h4clfdir entered reward bootstrap; h4clff01 was not found in W&B. Neither is a
valid direct-vs-fusion control. H2 low-rank direct is a different bilinear
architecture. No claim is made that every Fourier variant works.

## One question

Can a fresh linear head trained with balanced BCE and AdamW quickly extract
the signal seen by nested ridge in the *same frozen early encoder outputs*?

Capture uses the existing h4clfpath1 pipeline: fixed c4a91e07 step-1110 policy,
weights-only, clean validation (119002 events), same classifier initialization,
constant group LRs, regularization, sample generation, and 16 GPU workers.
The original full classifier fit runs unchanged (minimum 1000 updates and
existing patience); only detached exports at updates 0/50/100 are added.
This preserves the live baseline and avoids changing stopping or model paths.

Rank zero saves gathered CPU features, logits, target, exact context-derived
outer/inner partition masks, and fit settings with exclusive file creation.
Export has no RNG use and no backward pass. Existing mode/buffer/RNG restoration
and distributed broadcast remain intact. Reusing populated output is rejected.

The panel has at most 128 rows per class per rank: 4096 at 16 ranks. Its outer
fit and holdout both come from early-stop validation, never final test. This
is a bounded diagnostic, **not** a replacement classifier training population.
Representations may indirectly depend on prior validation selection; do not
report this holdout as an independent generalization/closure audit.

## Replay arms and controls

At every capture, compare encoder output and raw Fourier (negative/control
branch) on identical rows, labels, outer split and inner CV split:

1. Existing nested ridge: fit-only standardization; lambda selected only by
   two-fold inner CV from [1e-4, 1e-2, 1, 100]. No outer-holdout selection.
2. Linear BCE/AdamW on unstandardized frozen features.
3. Same linear BCE/AdamW with fit-only mean/std standardization.

Heads start at zero, constant LR 2e-4, weight decay 1e-3, 3000 updates,
balanced sampling with replacement (512 per class), no dropout/scheduler or
gradient clipping. Each uses seeds 17/29/43 and matching minibatch indices.
Only head tensors train. Mean/std, weights, bias, complete curves and ridge
results are persisted. Mini-head batch size is intentionally smaller than the
production classifier batch: update counts are comparable **within replay**,
not a claim of production wall-time/sample efficiency.

## Predeclared decision (before looking at new replay)

Primary: capture update 50, encoder output, readout update 200. Captures 0/100,
raw Fourier and later readout updates are secondary. No best-holdout selection.
Thresholds below are operational screening rules, not statistical significance.

- If matched ridge holdout AUC <0.65: early signal did not replicate; inconclusive.
- A readout arm passes only if all three seeds at update 200 have AUC >=
  ridge AUC minus 0.02, AUC >= original-head AUC plus 0.03, and BCE <log(2).
- Unstandardized passes: supports accessible early signal; a subsequent
  matched end-to-end readout-path intervention is justified, not yet proven.
- Only standardized passes: supports scaling/conditioning sensitivity;
  a shortcut alone is not established as the remedy.
- Neither passes: do not add attention or shortcut on this evidence. Inspect
  BCE-versus-ridge objective, optimization and conditioning. Later recovery
  distinguishes slow learning from lack of recovery within the tested budget.

Three seeds share one panel and representation; they measure minibatch
variability, not independent dataset/model replication. Ridge and BCE use
different objectives/regularization; differences cannot uniquely identify
AdamW as the cause. Ridge scores are not calibrated logits; no ridge BCE is
reported. This test cannot establish useful DGPO candidate ordering.

## Run

After syncing changed Python and YAML files, use the existing 16-GPU Ray
allocation for capture:

```bash
shifter python3 scripts/train_h4_frozen_readout.py --validate-only
shifter python3 scripts/train_h4_frozen_readout.py
```

After capture completes, run replay **once**, not through 16 tasks. It only
needs CPU and reads already gathered features; it does not reload the policy.

```bash
shifter python3 -u scripts/replay_h4_frozen_readout.py
```

W&B capture: h4frzcap1. W&B replay: h4readout1; both resume=never.
Replay logs each arm on its own head-update axis, with capture update in the
metric prefix. Historical numbers are configuration provenance, not aligned
points on new curves. `--offline` disables W&B for local diagnostic replay.

Outputs:
`/pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_capture/panels/` and
`/pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_capture/readout_replay/report.json`.
Existing output directories are not overwritten. Inspect `decision`, the
step-200 paired curves, BCE generalization gaps, gradients and update norms.
