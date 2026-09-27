# Fixed-policy residual-weight investigation

This is a standalone diagnostic, **not a DGPO training job**. It uses the current
10% fullfit classifier/inputs, the filtered full 10% data pool and source step-320
live policy from the fullfit YAML. It never installs a reward, updates a policy,
resumes an existing W&B run, or writes into a production checkpoint directory.
The diagnostic YAML now enables a **new online W&B run** in
`ytchou97-university-of-washington/nu2flow-RL`.

## Why this test

The experimental history already includes fresh initialization, full inheritance,
zero-output inheritance, partial freezing, staged unfreezing and longer training.
Later residual iterations still develop train/validation divergence. Repeating
those architecture/optimizer changes is not the first diagnostic here.

This test asks whether the divergence is associated with concentrated weights,
fold-specific weighted targets, a weak/unstable fitter, or discrimination of the
weighted **event context** rather than candidate differences. It does not promise
one test can establish a unique cause or that all generalization errors are bugs.

## Protocol

1. Read-only preflight checks source step 320, hashes its complete `state_dict`
   and source files, and refuses any existing diagnostic output directory.
   `--check-only` creates no files and launches no Ray/GPU work.
2. Load live policy tensors (not EMA), verify the loaded tensors, generate K=1
   once for every event in the filtered 10% pool, verify the policy is unchanged,
   then discard it. Save this pool and exact split indices for further replay.
3. Use production identity splits: fixed 80/20 seed 42 and crossfit seed 20260906.
   Training seed changes do not change data, noise, split or classifier backbone.
4. Reproduce up to iterations 1--3 with the **production residual fitter**, its
   initialization, 1000-update cold floor, warm 10-fold-epoch floor, validation
   early stopping, best restoration, dropout, batching, LR and tempering 0.75.
   Stop earlier if production closes. The iteration cap is a diagnostic boundary,
   not a successful closure claim. Unexpected errors fail visibly.
5. Save each fold's initial state, restored-best state, actual fit config, initial
   log weights, train and validation logits. Evaluate the restored-best classifier
   on its complete fit population and OOF population in **eval mode**. These
   must not be confused with minibatch/dropout training loss.
6. Replay the following arms for every captured iteration-2/3 fold. Within a fold
   the arms have identical initial parameters, fresh AdamW, sampling/dropout seed,
   global batch and 400-update budget. No early stopping or best restoration in
   these interventions; evaluate at 0,40,...,400. Repeat the stack and controls
   for two training seeds (20260906, 20260907).

| Arm | Samples and weights | Question |
| --- | --- | --- |
| original | Original residual weights | Fixed-budget reference curve |
| unit_raw | Same truth/generated samples, all weights one | Can this initialized classifier still solve the raw task? |
| capped | Original samples; cap mean-one weights at training p99.5, renormalize | Is the behavior sensitive to the weight tail? |
| truth_null | Identical truth candidates on both sides, unit weights | Does the scoring/training path behave sensibly on a zero-signal problem? |
| condition_only_weighted | Identical fixed candidate angles on both sides, original weights | Can weighted visible/event context alone support discrimination? |

The cap is estimated only from training weights. It changes the learning target
and is **not a proposed production fix**. `truth_null` AUC is 0.5 by construction
for a deterministic eval-mode classifier; its BCE also checks whether the fitter
can undo a confident inherited head. Poor null BCE after the finite budget is not
by itself a software bug. `condition_only_weighted` is intentionally NOT a null
test: weighting can change the event-context distribution. This arm can expose
that mechanism, but cannot prove it is the sole cause of residual failure.

## Diagnostics and interpretation

- ESS, ESS/N, max/p99 mean-one weight, and top 1% / 0.1% weight mass, before and
  after each proposed increment. "If applied" does not mean the gate accepted it.
- Fraction of negative BCE contributed by the largest-weight 1% of events.
- Fold-logit RMS, p99 absolute difference and sign disagreement on the **same
  outer-validation events**, avoiding in-sample ensemble contamination.
- Each restored-best fold is evaluated twice on that same validation population:
  with production ensemble-propagated weights, and with weights propagated by
  the opposite fold's earlier models. For TWO folds, that opposite model is the
  one that supplied OOF weights on the current classifier's training rows.
  This changes evaluation weights only, not fitting or model selection. A large
  change identifies weight-path sensitivity, not proof of improved truth matching.
- Full-fit eval BCE vs OOF/outer-validation BCE distinguishes a real generalization
  gap from merely comparing dropout training minibatches against eval-mode loss.
- Per-fit production loss curves, and matched-budget control curves.

Compare the same arm and objective across seeds. AUC/BCE from unit-raw, weighted
residual and condition-only problems are NOT interchangeable performance scores.
Good ranking does not certify calibrated density ratios, and chance AUC alone
does not prove closure. Validation is used for earlier model selection, so these
are diagnostics, not an untouched-test statistical claim. Two seeds are a minimum
robustness check, not a significance test.

Recursive dependence across crossfit iterations is **not removed** by this test.
If weight-path sensitivity persists, use the saved artifacts to design a separate
independent-weight-estimation test; do not label crossfitting intrinsically wrong.
The fresh Ray pool need not match historical row/noise ordering bitwise.

## NERSC commands

Sync the new script/config and the modified `evenet_ratio.py` observer hook first.
Use the existing designated Shifter image and an allocated **16-GPU Ray cluster**.
This is substantial classifier work (up to 40 control fits × 400 updates, plus
two production stack fits). There is no diffusion backward pass or repeated
generation. No automatic retry/resume is enabled; completed artifacts survive
failures. Existing output directories are deliberately refused.

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/diagnose_residual_weights.py \
  config/dgpo_10pct_residual_weight_diagnostic.yaml --check-only

shifter python3 scripts/diagnose_residual_weights.py \
  config/dgpo_10pct_residual_weight_diagnostic.yaml
```

Outputs:
`/pscratch/sd/y/yiren/Ztautau/residual_weight_diagnostic_step320_v1/`

- `manifest.json`, `runtime.yaml`, `pool.pt`: pinned sources, resolved settings,
  fixed samples and actual train/validation indices.
- `seed_*/production_fit.jsonl`: actual production fit curves.
- `seed_*/iteration_*.json`: weight and fold diagnostics.
- `seed_*/i*_f*_restored_best.json`: full fit/OOF/outer-val comparisons.
- `seed_*/control_*.json` / `*.pt`: control curves, logits and classifier states.
- `summary.json`: compact comparison; `report.json`: detailed results/limitations.

Use `--output-dir /pscratch/sd/y/yiren/Ztautau/residual_weight_diagnostic_step320_v2`
for another independent invocation. Local CPU tests do not verify remote paths,
the installed NERSC image, available GPU memory, or multi-node execution.

## W&B diagnostic logging

Only rank zero calls `wandb.init`, with a generated unique ID and `resume="never"`.
The name starts with `residual_weight_diagnostic_10pct_step320`, job type/group
is `residual-weight-diagnostic`. The terminal prints its URL, also saved to
`wandb_status.json`. Use the existing NERSC W&B login/credentials; no API key is
stored in this YAML. `--check-only` still never connects to W&B.

- `production/sSEED/iITER/fFOLD/*`: training BCE, freshly evaluated validation
  BCE/AUC/balanced accuracy, gradient norm and clipping. Each fit owns its own
  `fit_step` x-axis. Cached validation values are not re-logged as fresh points.
- `restored/sSEED/fFOLD/*`: complete fit, OOF and validation metrics from the
  restored-best state, plotted against iteration, separate from live curves.
- `stack/sSEED/*`: weight ESS/tails, fold differences and ensemble metrics vs
  iteration. AUC-gate acceptance is a diagnostic observation, not an endorsement.
- `control/sSEED/iITER/fFOLD/ARM/*`: training loss/gradients every 10 updates and
  full validation metrics at the existing scoring cadence, each with its own
  `fit_step`. Five different objectives remain explicitly separated.

W&B's transport step is never assigned the policy or classifier step, so a new
fit cannot rewind it. Only small `summary.json` and `report.json` are additionally
uploaded; datasets, pools, logits, checkpoint weights and full config files stay
local. No extra model evaluation is introduced by logging.

Initialization failure stops all ranks before expensive work. Logging failures
after initialization warn and preserve local computation/results, avoiding a
rank-zero logging exception leaving other ranks stuck in a collective. The SDK
keeps its normal local run files; final-sync errors are reported in
`wandb_status.json`. Network delivery cannot be guaranteed by local tests.

### Upload an existing diagnostic without rerunning it

```bash
shifter python3 scripts/diagnose_residual_weights.py \
  config/dgpo_10pct_residual_weight_diagnostic.yaml --upload-existing
```

No Ray/GPU allocation or model/checkpoint loading is required for this command.
It creates a **new** W&B run and replays existing JSON logs/reports only. Partial
diagnostics upload their available completed records and are marked as partial;
this is a one-time snapshot, not live tailing. Completed control reports contain
training points only at their saved evaluation cadence, unlike live logging.
The historical `manifest.json` supplies experiment provenance; the current YAML
supplies only the output directory and W&B destination/settings for upload.

Do not replace hashed source files while an older diagnostic job is still running:
its final source-integrity check would correctly reject that change. Wait for it
to finish, then sync this update and use `--upload-existing` (or run the uploader
from a separate checkout). Set `wandb.enabled: false` to retain local-only mode.
