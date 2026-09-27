# Fixed-policy classifier LR stability

One cold classifier fit, no control run, reward fit or policy updates. Pin the
h4lbnr01 step-50 DGPO-finetuned policy (not the initial diffusion checkpoint).
The launch checks that the source checkpoint exists before starting Ray.

Only PET adapter and complete classifier decoder learning rates change from
2e-4 to 5e-5. Fourier/output remains 2e-4; the last PET block and pretrained
position encoder remain 1e-5. AdamW, dropout, weight decay, Fourier mechanism,
data, and the original audit stopping rule are unchanged: at least 1000
updates, no maximum, ten classifier epochs of validation-BCE patience.

The original K=1 validation pool is regenerated using the inherited seed,
loader and 16-worker geometry. An empty cold-start cache preserves the source
audit's identity split without loading classifier weights. Exact sample
equality requires unchanged shards and Shifter/runtime; this is not a replay
of saved sample tensors. The launcher compares the complete audit settings
against the source config to reject accidental extra interventions.

Within an existing four-node, 16-GPU Shifter/Ray allocation:

```bash
shifter python3 scripts/train_h4_classifier_lr_stability.py --validate-only
shifter python3 scripts/train_h4_classifier_lr_stability.py
```

W&B ID: `h4clflr1`. Raw classifier loss histories are enabled. Compare full
validation BCE/AUC trajectories with the original step-50 audit, not only the
restored best AUC. Under `omnifold_live`, inspect module gradient norms,
clipping, logit statistics, `parameter_update_rms_*`,
`update_to_parameter_rms_ratio_*` and `optimizer_group_lr_*` against classifier
fit steps. Updates are measured after AdamW (including decay), sampled at the
existing progress cadence; they do not alter optimization. The decoder
diagnostic group also includes the pretrained position encoder, whose LR
remains unchanged. Relative updates of zero-initialized parameters can be
large; use absolute RMS alongside them.

Success means retaining learned validation separation without the prior
collapse toward chance, not merely suppressing gradient spikes. This tests
LR as a cause; it does not establish normalization or attention as the cause.
No normalization, initialization, attention, or regularization change is
included in this round. A 16-GPU end-to-end run still requires NERSC validation.

## Stability diagnostics

Enabled only with `--diagnostics`, using the separate overlay
`dgpo_h4_classifier_stability_diagnostics.yaml` and W&B ID `h4clfd1`.
The original `h4clflr1` config/run is unchanged. Launch in the existing
16-GPU allocation with:

```bash
shifter python3 scripts/train_h4_classifier_lr_stability.py --diagnostics --validate-only
shifter python3 scripts/train_h4_classifier_lr_stability.py --diagnostics
```

The
first update and every 10th update measure a fixed early-stop validation
subset (16 truth and 16 generated rows per rank; 256 of each on 16 ranks).
The final test subset is never used. Probe forwards use eval/no-grad and
restore every module's mode, buffers and Python/NumPy/Torch RNG, including the
unregistered shared EveNet backbone. They do not select checkpoints or change
the validation schedule, patience, gradients or optimizer updates.

W&B metrics live under `omnifold_live/raw_staleness_audit/stability/`:

- `probe/bce_before`, `bce_after`, `bce_delta`, `separation_before`,
  `separation_after`, `logit_change_rms`: same probe immediately before and
  after an update. Positive BCE delta is worse. These are small-probe
  diagnostics, not final AUC or evidence of distribution closure.
- `layer/...`: pre-LayerNorm variance, input/output activation and gradient
  RMS, adapter residual/input RMS ratio. Rank mean/max (and min for variance
  minima) identify rank-local outliers. Masked/padded tokens are included in
  general activation/LayerNorm statistics; low variance alone is not proof
  of a normalization bug.
- Attention `projected_q_rms`, `projected_k_rms`, `qk_absmax`,
  `attention_entropy`, `all_masked_fraction`: detached shadow computation on
  at most two examples and 16 queries, using the full key length and masks.
  It does not request weights or change the real attention backend. Entropy
  is measured before attention dropout; this is a probe, not a fused-kernel
  trace. Unsupported bias/extra-token attention is skipped.
- `update/.../rms`, `relative_rms`: actual post-AdamW per-layer displacement.
- `local_gradient_norm_rankmax`, `local_training_loss_rankmax`,
  `gradient_spike`: rank-local gradients before averaging/clipping, checked
  every step. They are distinct from the applied global gradient.

Probe/spike summary charts use classifier fit-step x axes. Detailed layer
scalars survive critical/simplified logging; select fit-step metadata as the
x axis when inspecting the underlying history.

Snapshots trigger on a local gradient norm >=1000, a >20x jump over the
previous 50-step median (after 10 observations), nonfinite training values,
or a sampled probe BCE increase >=0.02. These thresholds are diagnostic
heuristics, not training controls. A trigger is synchronized across ranks.
At most two snapshot attempts per rank per fit (one per step) are permitted.
Files contain event data and remain on NERSC, not uploaded to W&B:

`/pscratch/sd/y/yiren/Ztautau/h4_classifier_stability_diagnostics/diagnostic_snapshots/`

Each artifact stores the rank-local batch, microbatch size, pre-forward RNG
and buffers, pre-update parameters/AdamW state, full backbone state and
local statistics. For post-update triggers, `model_state` is a container;
the separate `pre_update_parameters` and `pre_forward_buffers` must be
overlaid to recover the pre-step state. Use `replay_snapshot(model, payload)`
from `stability_diagnostic.py` with an identically reconstructed, disposable
classifier. It applies those overrides. Distributed replay requires the
original world size and each rank's matching artifact, the same runtime and
attention backend. Do not load untrusted pickle snapshots. Replay mutates
the disposable model, RNG and precision settings; run it in a separate job.

Overhead: two small eval forwards per sampled step, hooks and per-layer
statistics; while snapshot quota remains, one extra GPU copy of trainable
parameters plus AdamW state every step. Full frozen-backbone state is copied
to CPU/disk only upon a trigger. After quota exhaustion, only sampled
parameter copies remain. Budget additional GPU memory and disk space before
running; two snapshots on all ranks can be substantial. Snapshot write
failures are logged and consume quota but do not stop training.
