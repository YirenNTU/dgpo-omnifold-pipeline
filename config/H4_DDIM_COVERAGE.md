# Does DDIM limit coverage?

Status as of 2026-09-20 10:15 UTC: **running on NERSC**. Legacy20 and stable20
have completed; stable100 is sampling, and stable200 has no results yet. W&B ID `h4ddim1`;
display name `Does DDIM limit coverage? | raw step-1110 | stable inversion and 20/100/200 steps`.

Interim numerical-inversion contrast (history steps 32 and 66): joint <1e-4
draw counts are 12 versus 11 out of 131,072. Stable-minus-legacy fraction is
-7.62939453125e-6, paired-event bootstrap 95% CI
[-3.814697265625e-5, 2.288818359375e-5]. Topology and target-marginal W1 changes
are below 2e-7 rad in absolute value; invalid-direction counts are zero.
Stable inversion alone does not materially repair this panel's coverage gap.
This is not evidence about step-count convergence: 100/200 results are pending.

## One question, two isolated contrasts

Does the current sampler underresolve the frozen model's narrow joint structure?
The earlier `h4cov1`, `h4covpre1`, and `h4covfull1` all used DDIM20. They
establish a large checkpoint-plus-sampler coverage deficit, not which component
caused it. No classifier fitting, DGPO update, target projection, event cut,
candidate selection, calibration, reweighting, EMA or extra physics input here.

| Arm | x0 reconstruction | Steps | Contrast |
| --- | --- | ---: | --- |
| legacy20 | `(x - sigma * eps) / alpha` | 20 | Current production arithmetic |
| stable20 | `alpha * x - sigma * v` | 20 | Numerical inversion only |
| stable100 | `alpha * x - sigma * v` | 100 | Resolution, same stable formula |
| stable200 | `alpha * x - sigma * v` | 200 | Resolution/convergence check |

The existing cosine VP schedule, uniform-t spacing, finite logSNR endpoints
(-20,20), epsilon reconstruction, denormalizer and matmul precision are unchanged.
`eta=1` retains this repo's deterministic update: its historical `eta` is an
epsilon multiplier, **not** canonical stochastic-DDIM eta. Do not change it to
zero. Existing production sampler callers retain `x0_mode="legacy"` by default.

The stable inversion removes high-noise FP32 cancellation. It is algebraically
equivalent in exact VP arithmetic; the previous synthetic test is not evidence
of better physical coverage. This experiment measures the actual effect.

## Matched sampling

- Load the completed `h4_spike_coverage_1110` runtime, panel, checkpoint path
  and seed. Require its K128/DDIM20 protocol. Load `state_dict`
  directly and verify compatibility/global_step=1110; ignore all EMA state.
- The original `h4cov1` manifest predates the `arm` and `weights` labels
  (confirmed against its W&B config on 2026-09-20). Missing labels are accepted
  and recorded as unknown in `source_manifest_labels`; explicit incompatible
  labels are still rejected. This does not certify the historical replay's
  weight selection: all four new arms load raw state directly and regenerate
  their candidates. No source manifest is edited or falsely relabelled.
  Other mismatches now report the exact field and saved/expected values.
- Same 1,024 events, their original order and inputs, 16 GPU workers, batch16,
  K128 sequential chains; all four arms see the same conditions. Truth is
  used only for measurement, never passed as the sampler's invisible inputs.
- Generate initial Gaussian tensors once, with source seed+10000+rank and the
  original batch/chain draw shapes. Replay those tensors explicitly in every
  arm. Save the event-aligned tensors as `initial_noise.pt`.
- Keep dropout off and freeze every parameter. No optimizer or classifier
  exists in this runner. Checkpoint normalization buffers remain unchanged.
- Match workers/batch size to the old source, but do not claim bitwise
  historical reproduction across different hardware/software. The causal
  comparisons are the four newly generated arms within this run.
- 20+20+100+200 = 340 network evaluations per draw, approximately **17x one
  20-step coverage replay**. This is a convergence test, not a cost-matched
  sampler benchmark. W&B records synchronized sampling wall time on the
  slowest rank and summed GPU-rank sampling seconds; these include progress
  logging and do not include loading or CPU analysis.

## Endpoints and interpretation

Primary: fraction of **all K128 generated draws** with both acoplanarity and
acollinearity strictly below 1e-4 rad, relative to the same panel's truth
fraction. Fixed numerical contrast: stable20 minus legacy20. Fixed resolution
contrast: stable200 minus stable20; stable100 and 200-minus-100 characterize
convergence. Always report hit counts, absolute truth-gap change and magnitude,
not only statistical significance or a multiplicative improvement from a tiny
baseline. The historical truth fraction is approximately 47%, so a small
increase in a near-zero generated fraction is not closure.

Also report both individual angles and their intersection at 1e-6/1e-5/1e-4,
full angular CDFs (including `joint_radius=max(acoplanarity,acollinearity)`),
angular W1 and all four original target marginal W1/quantiles. The joint-radius
CDF measures these nested joint regions, not every aspect of the 2D joint PDF.
Any-hit is secondary candidate availability, **not** generated probability mass.
Invalid finite theta encodings are counted without cutting/projecting them;
nonfinite outputs stop the run. Improvement with invalid-coordinate increases
must not be read as a clean sampler win.

Paired uncertainty resamples whole events with all K draws and truth together
(1,000 bootstrap replicates). A zero empirical interval with no observed hits
does not establish zero support or certainty. Report event counts contributing
to each difference; the repeatedly inspected panel is exploratory. Use a new
panel to confirm any selected sampler before changing production.

If stable20 improves, high-noise numerical inversion contributes. If stable100
and stable200 recover substantial truth mass, DDIM20 discretization contributes.
If 100-to-200 still changes materially, convergence remains unresolved. If the
curves stabilize far from truth, simply increasing steps through 200 is not
sufficient; investigate denoiser training and target provenance next. That
negative result does not prove architecture incapacity or exclude other solvers.

## Run once on the existing 16-GPU Shifter/Ray allocation

```bash
shifter python3 -u scripts/diagnose_h4_ddim_coverage.py \
  /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110 \
  --output /pscratch/sd/y/yiren/Ztautau/h4_ddim_coverage_1110 \
  --workers 16 \
  --batch-size 16 \
  --run-id h4ddim1
```

Use `--ray-address "$RAY_ADDRESS"` if auto-discovery is unavailable. The script
requires the running cluster and connects before replacing prior outputs; it
does not fall back to a one-node Ray process. Run the command once, not with
one driver per GPU. `--no-wandb` is only for local/debug runs.

Reruns overwrite this diagnostic's named output files; source artifacts and
unrelated files are preserved. Rank shards are overwritten by the current
workers before aggregation; old Ray logs remain in attempt-specific folders.
The W&B ID allows resumption and logs an attempt identifier; use a new run ID
if a clean history is wanted. Never run two attempts into one output directory
concurrently. `COMPLETE` is removed on retry and written only after all arms.

W&B logs per-arm chain progress, all coverage fractions/counts, paired deltas
and event-cluster intervals, angular/target W1, invalid-coordinate counts,
CDF overlays, a final comparison table and downloadable report artifact.
The report is refreshed after each completed arm (with `complete=false` until
all four finish). The run contains no classifier AUC because no classifier is
trained or evaluated in this diagnostic.

Outputs: `report.json`, `cdf.png`, each arm's full `generated`/`angles` tensors
(`legacy20.pt`, `stable20.pt`, `stable100.pt`, `stable200.pt`), `initial_noise.pt`,
the source panel/runtime, manifest, rank shards and completion marker.

Local checks:

```bash
python -m pytest -q scripts/test_h4_ddim_coverage.py scripts/test_h4_spike_coverage.py
```
