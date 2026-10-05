# Last-block diffusion attention continuation

Run in a user-provided 16-GPU allocation:

```bash
shifter python3 -u scripts/resume_tau_attention_dgpo.py config/dgpo_tau_diffusion_attention_last.yaml
```

First launch resolves `dgpo_tau_attention_1780/checkpoints/last.ckpt`, pins the
actual checkpoint and runtime under the separate `dgpo_tau_diffusion_attention_last`
folder. Subsequent launches use that experiment's own last.ckpt. No original
checkpoint is overwritten. `--prepare-only` pins/configures without training.

Intervention reuses the existing TokenSpecificConditioning implementation:
last-block hidden residual, width64, four heads. Queries are the invisible
noisy/time/slot representations. Memory is the existing unpooled visible encoder
output. Readout concatenates query and attended values through two SiLU layers
and a zero-initialized output projection. Add residual to the carried invisible
hidden after the last generation block, before output normalization/projection.
Global FiLM, Fourier flags and all observations remain unchanged. This is not
the first attention in EveNet; it is an additional task-adaptable readout path.

Inherit the latest trained attention classifier and its refit clock, without an
extra startup refit. Fresh training-panel refits remain every five completed
epochs and validation/audits every five epochs, aligned to the saved installation
clock. At each boundary, validate/audit first, then refit. Filtered data,
classifier batch1024/GPU and raw weights remain mandatory. Preserve velocity
MSE coefficient1. Inherit actor optimizer/scheduler; new readout parameters join
the existing visible_conditioning group with empty Adam moments and its existing
effective LR/decay, not a new reset schedule.

Exact shared raw weights are checked during migration. Saved reference weights
are not recentered: add a zero-output branch to their structure only, preserving
their initial function and allowing subsequent full-actor copies at refits.
This differs from the old short conditioning probes, which removed the new
branch from reference models. Migrated checkpoints store full actor/reward and
optimizer state; normal resumes no longer run optimizer migration.

Use actual unweighted Cij/nn as the endpoint, along with within-condition reward
probes, reference penalty, reward, branch gradients and residual/hidden RMS.
The current parent run is not an exactly paired control because its checkpoint
and future minibatches differ; improvements after continuation alone cannot
prove attention causality. A matched no-readout continuation from this pinned
source is needed for that stronger claim.

Local CPU tests exercise zero-output start, padding, gradients and preservation
of old Adam moments. Full 16-GPU execution has not been run by the assistant.
