# Direct and repeated visible conditioning: supervised screen

## Evidence and question

Toy round7/factorial supports keeping multiplicative FiLM. Round10 shows that
effective branch amplitude matters. Round12 supports repeated injections in
that toy, but does not isolate amplitude from repeated access. The early real
`h4tokfilm1` DGPO trajectory has not shown a resolved advantage over global FiLM.
None of these results proves direct residual attention will help EveNet.

Keep global FiLM and test whether an extra **direct** route makes supervised
velocity prediction easier. This is not a new DGPO objective, a classifier
experiment, or a claim that Fourier frequencies are insufficient.

## Architecture

Observed visible angles/kinematics and original projected visible tokens
-> existing Fourier features and nonlinear visible encoder
-> unpooled visible memory.

At selected generation blocks, the invisible state (including noisy input,
timestep and slot representation) queries that memory through four-head attention.
Concatenate query and attended context, apply two Linear/SiLU layers plus LayerNorm,
then a zero-initialized linear output. Add this output to the carried invisible
representation **after** the original block. Original global scale/shift FiLM in
every block, PET, FFNs and LayerScales are unchanged. No clean generated target is
passed into this conditioning branch. Supervised truth is only a training target
and the usual source of noisy diffusion inputs.

The direct route bypasses the existing attention/FFN/LayerScale output path, but
not the final output normalization and velocity projector. Only its final linear
projection is zero initialized. Initial velocity and iterative sampling updates
match the common source; gradients can open the branch.

| Arm | Extra routing beyond global FiLM | Injection sites | Role |
|---|---|---|---|
| `global` (existing) | None | None | Baseline |
| `last` token FiLM (existing) | Readout produces scale/shift | Last block | Optional routing comparison |
| `residual_last` | Readout produces direct hidden residual | Last block | Direct-route arm |
| `residual_all` | Same weight-tied direct readout | All three blocks | Repeated-access arm |

`residual_all` and `residual_last` have identical parameters and initialization.
Extra calls change compute and effective path strength: not equal-FLOPs or
amplitude-matched. Token-FiLM and direct-residual output widths differ, so that
comparison is not parameter-matched. No new frequencies or physics features.

## Frozen training protocol

Inherit `train_diffusion_token_film_global.yaml`: same raw checkpoint
`/pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt`,
weights only, fresh optimizer/scheduler, same10% train and cleaned validation,
ordered batches, paired timestep/noise draws,16 GPUs, FP32,2048 events/GPU.
No EMA, RL, classifier fits, velocity-reference penalty, joint-coverage callback
or low-noise reweighting. All backbone parameters remain trainable.

Body and PET Fourier LR2e-5; generation/projector/visible conditioning LR1e-4.
The new readout belongs to visible conditioning exactly once. Five-epoch warmup
then cosine over50 total epochs; early-stop patience51 cannot truncate the screen.
Do not resume an RL optimizer here or use the later changing old-DGPO `last.ckpt`.

## Measurements and decision

Primary: mean held-out `val/loss` over epochs45-49 at the matched50-epoch budget.
Inspect full train/validation velocity-MSE curves and elapsed time alongside it.
The existing logger exports, for every active block N:

- `train/visible_conditioning/block_N/token_residual_rms`
- `train/visible_conditioning/block_N/token_residual_to_hidden_rms`
- Matching `val/visible_conditioning/…` metrics.

The denominator is the valid invisible query's pre-block hidden RMS; padded or
empty-memory entries do not contribute. These measure branch amplitude, not
reward quality. Existing global scale/shift and Fourier diagnostics remain.

Compare `residual_all` to `global` for the practical package question, then to
`residual_last` to test repeated access. A small advantage needs independent
validation-noise confirmation, not best-epoch selection or training loss alone.
No improvement is a negative finite-budget screen, not proof conditioning cannot
help joint structure. A later fixed-reward DGPO comparison and matched fresh
classifiers are still needed to establish reward-transfer/distribution benefits.

## User launch only

On an existing16-GPU Ray allocation with the updated normal repository, run one
arm at a time unless using separate allocations. These commands do not submit jobs.

Existing global baseline (do not duplicate a matching completed supervised run):

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_token_film_global.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_token_film_global/ray_results
```

New repeated-residual candidate; use `ARM=last` for the one-injection ablation:

```bash
cd /global/u2/y/yiren/ml_pipeline
ARM=all
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config "config/train_diffusion_residual_${ARM}.yaml" \
  -- --ray_dir "/pscratch/sd/y/yiren/Ztautau/diffusion_residual_${ARM}/ray_results"
```

W&B project **EveNet**, family **Diffusion conditioning architecture**:
`difftokglobal1` / `diffreslast1` / `diffresall1`. Existing DGPO IDs, configs,
source checkpoints and jobs are unchanged. Prepared locally; remote checkpoint
loading and16-GPU runtime remain user-run validation.

## Local verification (2026-09-27)

Three isolated test invocations passed:26 new/legacy readout and supervised cases,
69 representation/classifier/optimizer regressions, and42 DGPO bootstrap/audit/
high-LR-resume contracts (some supervised cases are repeated between invocations).
Tests cover exact initial velocities and input gradients, iterative state updates,
zero-gate opening, shared initialization/parameter count, finite masked/empty
memory, query/visible dependence, unchanged visible slots, per-block metrics,
raw checkpoint loading, exact optimizer continuation and unique parameter ownership.
Readable W&B names pass the naming validator; no existing run was renamed.
No remote training, upload, allocation or scientific outcome is claimed.
