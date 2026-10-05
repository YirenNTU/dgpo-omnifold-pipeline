# Real-case FiLM effective-strength diagnostic

Status: prepared, not executed on NERSC. No allocation or training submitted.

Question: Does the trained global visible FiLM materially affect hidden features and final predicted velocity, and where does that effect vary by layer, time and tau slot?

## Fixed protocol

- Raw supervised global-FiLM checkpoint, three generation blocks, no token readout. Exact epoch/global step recorded; raw tensors copied once into the new output directory before evaluation. Existing checkpoint weights are never edited. No EMA, optimizer, classifier fits or policy updates.
- Reuse every event from the saved joint-coverage validation panel. By default obtain its path from `diffusion_low_noise_lr_10pct_seed42/checkpoints/joint_coverage/epoch-0100.json`; `--panel` can select an explicit panel. Save event identities and a panel copy. This is a reused exploratory panel, not a new confirmation set.
- Normalize the true two-slot corrections using the loaded model's normalizer. Construct x_t = alpha*x0 + sigma*epsilon and target v = alpha*epsilon - sigma*x0 with the production cosine schedule. Time points: .02, .1, .3, .5, .7, .9, .98.
- One fixed CPU noise draw per event (seed 42017 + panel position), reused across time and every intervention. Batching and rank partition do not change noise. Workers take panel positions rank::16; rank 0 verifies unique complete positions and merges in original event order.
- Baseline, all-layer scale off, all-layer shift off, all FiLM off, and each of the three blocks' FiLM off separately. Keep PET Fourier, time embeddings, global conditions and all weights unchanged. Interventions affect both attention and MLP modulation within targeted blocks.
- Measure normalized-feature and FiLM-delta RMS, scale/shift RMS, and attention/MLP residual RMS before/after LayerScale, separately by slot. Record event RMS quantiles .1/.5/.9/.99 and all per-event mean squares. Report paired velocity change and paired velocity-MSE change by time/slot.
- Verify recording hooks preserve baseline output exactly and baseline is restored exactly after interventions. Nonfinite values stop the run. Hooks are removed on exceptions.

## User execution

Update the existing remote repository with these new files and existing dependencies. Repository-root rsync must include `--exclude-from=NERSC/upload-excludes.txt`; never include toy artifacts implicitly.

From `/global/homes/y/yiren/ml_pipeline`, validate source files without GPU, Ray or W&B:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/diagnose_film_strength.py \
  --checkpoint /pscratch/sd/y/yiren/Ztautau/diffusion_global_film_long_resume/checkpoints/last.ckpt \
  --output /pscratch/sd/y/yiren/Ztautau/film_strength_real_01 \
  --check-only
```

This explicit example uses the saved last checkpoint, not a claim that it is the best checkpoint. Supply a known epoch checkpoint instead if desired. The actual run pins the raw weights and records their stored epoch/step; check-only metadata is not itself a pinned snapshot.

Run once inside the user's existing Ray GPU allocation:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_film_strength.py \
  --checkpoint /pscratch/sd/y/yiren/Ztautau/diffusion_global_film_long_resume/checkpoints/last.ckpt \
  --output /pscratch/sd/y/yiren/Ztautau/film_strength_real_01
```

Uses exactly 16 GPU workers, two CPUs per worker, from the existing cluster; never starts a local cluster or submits a job. Output must be new. W&B project EveNet, group `FiLM effective strength`, name `Does FiLM reach velocity? | real events | layer and noise sensitivity`; fresh run ID stored in manifest. `--no-wandb` disables online logging. `--dry-run` resolves configuration without accessing source files or creating output.

Outputs: `manifest.json`, `runtime.yaml`, pinned `raw_policy.pt`, `panel.pt`, each `time-*.pt` (event identities and per-event squared metrics), incremental `report.json`, and `COMPLETE` only after every check passes.

## Interpretation

This probes forward-noised real truth states, not the model's DDIM trajectory. It does not measure coverage or prove that retraining without a branch would fail. Velocity-MSE change is in normalized training coordinates, not angular physics units. No arbitrary RMS threshold defines success.

- Tiny hidden FiLM delta and tiny velocity change: investigate weak modulation; small signal alone does not prove it is useless.
- Substantial hidden delta but tiny velocity change: downstream cancellation/insensitivity is a hypothesis; inspect layers and LayerScale before changing architecture.
- Clear velocity change and higher MSE when disabled: the branch contributes at these states; inspect relation representation next rather than assuming more gain is better.
- Lower MSE when disabled: investigate the checkpoint/time region and validate on another panel; do not automatically remove the branch.

Tests: actual GeneratorTransformerBlock intervention parity, baseline invariance, LayerScale measurement, exception cleanup, batching-independent noise and dry-run. Remote checkpoint/data compatibility and GPU runtime remain unverified until user execution.

Local validation on 2026-09-29: 19 relevant checks passed (9 new diagnostic checks plus 10 existing source/config checks). An attempted broader coverage suite had 9 failures because this local Python environment lacks `lightning`; those tests exercise the older coverage callback, not this diagnostic's inference hooks. No dependency installed. Actual distributed 16-GPU execution and full model construction still require the NERSC container.

## Scale calibration sweep after run 23522203

Add `--scale-sweep` to evaluate `(1 + lambda * gamma) * LN(h) + beta` for lambda = 0, .25, .5, 1. Both attention and MLP scale in all three blocks use the same lambda. Shift, PET Fourier, time pathways and saved parameters stay unchanged. This is an inference experiment, not a permanent model change. Lambda=1 must exactly reproduce the unhooked baseline; lambda=0 is the existing no-scale intervention.

To reproduce the exact epoch286 source from run 23522203, use its pinned raw snapshot rather than a possibly updated `last.ckpt`:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_film_strength.py --scale-sweep \
  --checkpoint /pscratch/sd/y/yiren/Ztautau/film_strength_real_01/raw_policy.pt \
  --panel /pscratch/sd/y/yiren/Ztautau/film_strength_real_01/panel.pt \
  --output /pscratch/sd/y/yiren/Ztautau/film_scale_sweep_epoch286_01
```

Append `--check-only` for file validation or `--dry-run` for configuration-only validation. Requires the existing 16-GPU Ray allocation for execution. The reused panel is explicitly exploratory. For independent validation, supply a separately saved held-out panel via `--panel`; do not relabel the old panel as independent.

Repeat the same command with `--checkpoint` set to an actual named better-validation checkpoint and a new output directory. Keep the identical panel, seed and time grid across checkpoints. Its filename must be confirmed from existing files; do not guess a best-checkpoint filename or automatically select latest. Each run records epoch/step and has its own W&B ID. Do not compare raw velocity changes across checkpoints without accounting for their different baseline velocity scales.

W&B name: `Does weaker scale improve velocity? | global FiLM | paired gain sweep`.

New outputs include baseline/arm MSE, absolute and relative MSE change, and pointwise 95% paired event-bootstrap intervals (2000 replicates). Both slots stay clustered per event. These intervals cover evaluation-event uncertainty, not training seeds/noise repeats, and are not corrected for selecting the best gain/time/checkpoint. No automatic winning lambda is installed. Inspect the time-dependent pattern; the seven unevenly spaced times do not define a uniformly weighted training-loss estimate.

The full report and per-event `time-*.pt` metrics are uploaded to the diagnostic's W&B files on successful completion, making future confidence-interval checks possible without reading remote scratch. The checkpoint and input panel are not uploaded to W&B. Status remains prepared; no new diagnostic submitted or executed by the assistant.
