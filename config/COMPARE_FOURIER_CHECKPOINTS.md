# Paired generator checkpoint evaluation

Prepared for user execution. This runner never submits a job or updates either
generator. It generates a common event/noise panel and fits independent audit
classifiers. Use the existing NERSC ml_pipeline checkout and Shifter.

## Launch

Edit `compare_fourier_checkpoints.yaml` to change paths/settings; no CLI
checkpoint arguments are required. Defaults compare these two saved endpoints:

- baseline: no-Fourier continuation mpvmx945, seed42/none/checkpoints/last.ckpt;
- candidate: multiscale readout snx80o90, seed42/readout_multiscale/checkpoints/last.ckpt.

`last.ckpt` is a mutable pointer, not a promise of epoch49. The training recipe
retains only three best checkpoints. The default `checkpoint_selection:
configured` resolves the two configured paths once and records their actual
epoch/global_step. With `expected_epoch: null`, different epochs are allowed:
this measures the two endpoints' generation quality, and the report explicitly
states that different update budgets prevent attributing a difference solely
to Fourier. It does not select a different checkpoint after seeing test scores.

For a strict equal-budget architecture comparison, set `checkpoint_selection:
latest_common`. This reads actual metadata from all saved `.ckpt` files in both
directories and selects the latest common epoch with matching update counts.
If no common epoch exists, it prints the available files and stops before
W&B, sampling or fits; it does not silently relax that requirement. Alternatively
pin two exact matched paths with `configured`. `expected_epoch` can optionally
enforce a particular epoch for both. Each model is built from its own saved
runtime and raw weights are loaded strictly.

Inside the existing four-node / 16-GPU Ray allocation, from ml_pipeline:

```bash
shifter python3 scripts/compare_fourier_checkpoints.py --check-only
shifter python3 scripts/compare_fourier_checkpoints.py --inspect-checkpoints
shifter python3 scripts/compare_fourier_checkpoints.py
```

The inspection command needs no Ray cluster or GPUs. The last command connects
to `RAY_ADDRESS` (or Ray's `auto` discovery), requires 16 free GPUs and 32 CPUs,
and uses one worker per GPU. Launch the driver ONCE, not through one process per
GPU. `--ray-address` can explicitly identify an already running cluster. It
does not start a cluster, request an allocation or submit a job.

Generation is sharded across all 16 GPUs, reusing the saved noise by event
identity. Each of the six H4 judges then trains across all 16 GPUs using the
production fitter's gradient averaging. Validation and final scoring are also
sharded and gathered in event order. The global batch remains 512 per class
(32 truth + 32 generated rows per GPU); learning rates and update budgets stay
fixed. Incomplete global training batches are dropped and reshuffled each
epoch so gradient averaging always combines equal-sized shards. This preserves
the paired protocol, not bitwise equivalence to a one-GPU run with dropout.
Shard files are checked for duplicates, missing events and nonfinite outputs.

The user starts the command. Each invocation writes a fresh epochs-and-timestamp
subdirectory under output_dir; failed earlier outputs are preserved. Only rank
0 writes W&B, audit checkpoints and reports. CPU panel setup and the final
bootstrap/report run once; the GPU phases use all 16 workers. There are no
automatic retries or resume. Three seeds per checkpoint still require up to
5,000 classifier updates each; distributed communication limits the speedup.

For repository-root uploads, keep `--exclude-from=NERSC/upload-excludes.txt`
and all standing exclusions. No toy code or artifacts are needed.

## Fixed comparison contract

1. Uniformly sample 32,768 rows without replacement across all sorted evaluation
   parquet files using a fixed panel seed. Save exact file/row identities. The
   default is the shared generator validation directory; this data was withheld
   from generator weight updates but previously used for model selection and
   inspected in earlier experiments. It is NOT an untouched final test set.
   Set evaluation_data_dir to an existing independently held-out compatible
   parquet directory for an untouched benchmark; no path is guessed.
2. Save one CPU-drawn initial-noise tensor and reuse it exactly for both models.
   Generate four unselected conditional candidates per event with stable-v DDIM,
   100 steps, the same batch size, raw weights and eval mode. No EMA, candidate
   ranking, calibration, clipping or resampling. This evaluates the specified
   common sampler; it is not automatically equivalent to legacy 20-step DGPO.
3. Hash only visible conditions into fixed disjoint 60/20/20 fit/early-stop/test
   folds. Identical conditions stay together. No generator target values enter
   sampling; only zero shape placeholders and valid-slot masks are supplied.
4. For each classifier seed, fit a NEW H4 judge per checkpoint using the first
   candidate only (K=1). Both start from the common original no-Fourier epoch190
   backbone, identical H4 architecture and paired initialization/minibatch seeds.
   H4 uses production internal adapters, candidate decoder and k4 pair features;
   grouped-sequential embedding and invisible projector train as in the existing
   H4 audit recipe. It does not inherit either generator's endpoint backbone or
   any reward classifier. These classifier trainability settings do not change
   the generator's earlier full-backbone continuation settings.
5. Train at least 1,000 optimizer updates, up to 5,000, with validation BCE checks
   every50 updates and patience15, minimum improvement1e-4. Restore the best BCE
   checkpoint. A fit that reaches the cap without plateau remains inspectable
   but makes the final comparison INCONCLUSIVE. Near-0.5 AUC alone is not proof
   of a ready classifier. Report updates, best step, plateau and test metrics.
6. Evaluate only the untouched classifier test fold for final metrics. Also score
   each fitted judge on BOTH generators, yielding a cross-judge matrix. This is
   a diagnostic for judge preference; the primary endpoint is the independently
   trained best-response H4 gap for each generator, not that matrix's minimum.

## Primary endpoint and interpretation

Primary statistic is mean over three paired classifier seeds of

```
abs(candidate_test_AUC - 0.5) - abs(baseline_test_AUC - 0.5)
```

Negative favors the candidate. Prespecified material AUC-gap margin: 0.005.
Use 500 paired event-cluster bootstrap replicates, resampling identical event
indices across both generators and all fitted judges, keeping duplicate event
conditions together. Declare candidate improvement only if all audits are
ready, every classifier seed improves, and the upper 95% interval is below
-0.005. Apply the symmetric deterioration rule. An interval entirely within
+/-0.005 is reported as no material gap difference ON THIS PANEL. Otherwise
report inconclusive. No decision is selected from the test after inspecting it.

The bootstrap is conditional on the six fitted judges, fixed generation noise,
and fixed generator checkpoints. It does not estimate generator-training seed
uncertainty, unseen-tail risk or full retraining uncertainty. Classifier seed
spread is shown separately. H4 plateau is a training diagnostic, not proof the
classifier family detects every distributional discrepancy.

## Secondary diagnostics

On the SAME test events, use all four draws to report marginal Wasserstein-1
errors for acoplanarity and acollinearity (radians), and invalid reconstructed
theta counts. All finite candidates remain in the sample.

Report a fixed conditional-joint moment discrepancy: visible-leg theta/phi
sin/cos features at harmonics1,2,4,8 (plus intercept), crossed with generated
tau-pair sin(delta_phi), cos(delta_phi), and opening-angle cosine. Compare
sample-average moments with truth and report their RMSE. This tests selected
conditional dependencies; it is not a complete conditional-distribution metric.
These physics diagnostics are descriptive, do not veto the primary endpoint,
and have no independently calibrated significance threshold in this runner.

## Outputs

Default output root: /pscratch/sd/y/yiren/Ztautau/fourier_checkpoint_comparison/none_vs_multiscale_16gpu

The printed attempt directory is `baseline_epoch<B>_candidate_epoch<C>_<UTC timestamp>/` below it.

- REPORT.md and report.json: decision, per-audit results, cross-judge metrics,
  secondary physics and limitations.
- manifest.json, spec.json and both runtime YAMLs: exact checkpoint epoch/path,
  sampler/settings and selected parquet rows.
- panel.pt: packed visible inputs, raw truth, split indices and initial noise.
- baseline_candidates.pt and candidate_candidates.pt: all four samples/event.
- baseline_rank*.pt and candidate_rank*.pt: generation shards with global row positions.
- judge_*_seed*.pt: fitted audit states; reproduce with the recorded common
  classifier foundation and H4 architecture, not the generator endpoint.
- judge_*_seed*_on_*.npz: paired test-row identities and logits for reanalysis.
- progress.jsonl, report.partial.json, FAILED.json on failure, COMPLETE on success.

A fresh W&B run in EveNet / Fourier checkpoint evaluation logs generation
progress, each audit's fit/validation trajectories and final results. It never
appends to or renames the two source runs. Reports/artifacts remain local to the
chosen NERSC output directory; sample tensors are not uploaded to W&B by this
script.
