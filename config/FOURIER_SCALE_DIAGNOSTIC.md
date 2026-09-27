# Fixed-checkpoint Fourier reliance diagnostic

Question: does the trained input_fast model use its Fourier branch to improve
velocity prediction? This is an evaluation, not a training arm or a comparison
of retrained architectures.

The settings in `fourier_scale_diagnostic.yaml` load the s93qlh9m runtime config
and raw `last.ckpt`. The symlink is resolved once and weights are loaded once;
the report records the actual file, epoch and global step. To target a different
saved epoch or output directory, edit the YAML before running. An existing
output directory is never overwritten. A running source run's last checkpoint
is an interim snapshot; wait for completion to diagnose its final checkpoint.

Use one allocated GPU, from the existing NERSC ml_pipeline checkout:

```bash
shifter python3 scripts/diagnose_fourier_scale.py --check-only
shifter python3 scripts/diagnose_fourier_scale.py
```

No Ray cluster is needed. The user obtains the allocation and runs the command.
The script does not submit jobs, train, or create a W&B run.

Protocol: first 8,192 events in sorted validation parquet order, batch size 256,
three fixed noise seeds, model.eval(), FP32 defaults. Each event/noise panel is
reused at branch scales 0, 0.5, 1, 2. The script restores RNG state per forward
and asserts identical target velocities, diffusion times and masks across
scales. A temporary hook multiplies only the Fourier residual; it removes the
hook after each forward. No parameters or checkpoint files are changed.

Primary diagnostic: paired masked velocity MSE change relative to scale 1;
negative means improvement. Report the absolute masked MSE and RMS velocity
output change as well. Mask out padded invisible features as production does.
This matched continuation uses no event or low-noise weights; the diagnostic
rejects those alternative objectives. The local panel differs from distributed
W&B validation, so compare scales within this report rather than its absolute
loss to W&B's val/loss.

Outputs in the YAML output_dir:

- report.json: checkpoint provenance, settings, pooled losses and paired deltas,
  including per-noise-seed losses and deltas.
- paired_event_statistics.npz: arrays [noise seed, event, statistic] per scale;
  statistics are squared-error sum, squared velocity-change sum, valid feature
  count. These preserve paired event-level information for further analysis.
- manifest.json: parquet files and row ranges, establishing the event panel.
- runtime.yaml: source architecture and training settings.

Interpretation: negligible output change at scale 0 supports weak functional
reliance on this panel. A loss increase at scale 0 supports checkpoint reliance;
it does not establish superiority to a separately trained no-Fourier model.
Output changes without a loss benefit indicate influence without demonstrated
prediction gain. If scale 0.5 or 2 helps, confirm on a separate panel before
choosing a scale; these are diagnostics, not a new tuned benchmark result.
