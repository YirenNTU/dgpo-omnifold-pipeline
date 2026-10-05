# Paired precision fast check

Prepared, not scientifically executed. User launches compute; no allocation,
remote inference, training or synchronization performed by implementation.

One question: does internal FP32 matmul precision materially affect this saved
global-FiLM policy's velocity and generated coverage? This is not a test of
training precision or DGPO reward retention.

## Fixed protocol

- Reuse the pinned raw epoch286/step22386 snapshot and event panel from
  `film_strength_real_01` (23522203), rather than a mutable `last.ckpt`.
  This is the previously diagnosed late checkpoint, not claimed to be best.
  Check-only verifies its actual stored metadata; the run copies raw tensors
  once before distributing workers. EMA/optimizer state is not installed.
- Exactly 16 GPU workers in an existing Ray cluster. All saved panel events
  (historically 1024); K8; DDIM20; legacy inversion in every arm; sequential
  candidate chains; batch16/rank. No classifier fits or updates.
- Three explicit settings: `medium`, `high`, `highest`. Do not assume which
  was effective in an old run: both lower-precision settings are measured.
  Settings are applied after model construction, recorded on every worker,
  and restored after each arm. Autocast is disabled, model/input tensors are
  FP32, and cuDNN TF32 is disabled identically across arms. Kernel choice can
  make two nominal settings behave identically; backend flags aren't a kernel trace.
- Same events, checkpoint, batch shapes, initial noise, schedule, sampler,
  masks and target normalizer for all arms. CPU noise uses seed42017 + panel
  position, independent of worker assignment. Exact noise tensors and IDs
  are saved. No truth values are supplied to the sampling condition.
- Seven paired forward-noised truth probes: t=.02,.1,.3,.5,.7,.9,.98.
  Save per-event velocity and MSE; require exact same-mode velocity replay
  after the settings round trip. Forward probes and DDIM sampling are separate.
- Reconstruct unchanged sampled corrections with the same stable atan2
  geometry in FP32 and FP64. Cast before reconstruction. Verify FP64 against
  the existing canonical coverage implementation. Independently report the
  truth reconstruction difference. This isolates measurement arithmetic;
  it does not recover precision already lost in stored inputs or train in FP64.

## Execute

Update the existing remote `ml_pipeline` checkout with the new script and
launcher; existing helpers/model code must match this checkout. A repository
upload must use `--exclude-from=NERSC/upload-excludes.txt`, preserving toy and
classifier-checkpoint exclusions, with no implicit deletion. No separate remote
code copy is needed.

For the existing remote checkout used by the previous FiLM diagnostics, upload
only this new diagnostic, launcher and protocol from the local repository:

```bash
bash NERSC/sync_precision_check.sh
```

This preserves all remote files and transfers no toy code, artifacts or checkpoints.
It relies on the already deployed FiLM/coverage helpers and global-FiLM config.

From the remote repository, validate files without GPU, Ray or W&B:

```bash
bash NERSC/run_precision_check.sh --check-only
```

Then run once in your existing 16-GPU Ray allocation:

```bash
bash NERSC/run_precision_check.sh
```

`--dry-run` prints resolved configuration without reading remote source files.
`--no-wandb` disables logging. Otherwise results and live progress go to EveNet,
group `Numerical precision diagnostics`, with a fresh run ID and readable name.
`--checkpoint`, `--panel`, `--output`, `--K` and sampler arguments can override
launcher defaults. The output directory must not exist. The strict loader is
currently for supervised global-FiLM checkpoints with PET angular conditioning;
DGPO and additional token/relation architectures are not silently accepted.

## Read the result

`SUMMARY.md` and `report.json` compare highest-minus-medium and highest-minus-high:

- Primary diagnostic: change in absolute truth/generator probability gap for
  joint radius <1e-4 rad, with paired whole-event bootstrap intervals.
- Supporting: joint and marginal W1, thresholds1e-6/1e-5, velocity and target
  coordinate RMS/max/p99 differences, per-time velocity MSE contrasts.
- Measurement-only: FP64-minus-FP32 physics differences for identical samples,
  plus FP32/FP64 truth fractions. Canonical coverage already uses FP64.

If strict FP32 improves coverage meaningfully and the contrast interval is
below zero, validate on another named checkpoint and independent events.
If only measurement arithmetic changes, investigate diagnostics/input precision
before changing the network. Large paired coordinate divergence alone is not
evidence of improvement. Small/no observed coverage change gives no support
for an inference precision rescue at this checkpoint; it does not rule out
training effects or small effects under this sample budget. K8 is intentionally
fast, so sparse spike counts may remain inconclusive; zero-hit bootstrap bands
are not upper bounds on unseen probability. No automatic architecture or
production-precision change follows this screen.

Intervals are pointwise, on one reused event/noise panel, not simultaneous or
training-seed uncertainty. FP64 postprocessing isn't full FP64 network inference.
No reward classifier is evaluated, so there is no DGPO closure claim.

Outputs also include pinned raw weights, runtime, panel, manifest, all16 rank
files, merged `paired.pt`, worker precision/device/version records and `COMPLETE`
only after successful analysis. Source checkpoints and panels aren't uploaded
to W&B. No existing result is overwritten.

Local validation: 25 tests passed across the precision screen and existing
global-FiLM checkpoint/panel contracts; Python compile and shell syntax passed.
These include production sampler-interface replay and canonical FP64 geometry,
not a distributed GPU execution. Remote files and actual CUDA kernel behavior
remain unverified until user execution.

## Serialization startup fix (2026-09-29)

The first remote attempt passed source checks but failed before GPU workers
started: cloudpickle traversed the script's contextmanager and tried to pickle
PyTorch's CudnnModule. The Ray entry point now imports the diagnostic module
inside the worker, avoiding that backend-object graph. A serialization preflight
runs before output creation. This changes dispatch only, not scientific settings.

The failed attempt already created `precision_screen_epoch286_01`. Preserve it
and retry in a new directory after uploading the fix:

```bash
bash NERSC/run_precision_check.sh --output /pscratch/sd/y/yiren/Ztautau/precision_screen_epoch286_02 --check-only
bash NERSC/run_precision_check.sh --output /pscratch/sd/y/yiren/Ztautau/precision_screen_epoch286_02
```
