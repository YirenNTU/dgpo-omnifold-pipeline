# Shift-only and pair-attention physics screen

Two independent supervised jobs, each **16 GPUs (4 nodes x 4), 50 epochs**.
Both now use the existing filtered datasets and compare against fresh
`EveNet/relcontext02`; historical `relcontext01` used unfiltered train data. The assistant prepares
code/configuration; the user submits and starts both jobs.

| Arm | Overlay | W&B ID | Intervention |
|---|---|---|---|
| Shift-only | `train_diffusion_context_shift_only.yaml` | `ctxshift02` | Disable all generation FiLM gamma; keep beta and context adapter |
| Pair attention | `train_diffusion_pair_attention.yaml` | `pairattn02` | Keep full context FiLM; add observed pair bias to existing attention |

## Common protocol and interpretation

Both inherit the actual context-control configuration: fixed raw supervised
epoch286/global_step22386 from
`/pscratch/sd/y/yiren/Ztautau/film_strength_real_01/raw_policy.pt`, fresh optimizer
and scheduler, seed42, paired diffusion seed420024, ordered10% data,
2048 events/GPU, FP32, no EMA, no RL, same learning rates and five-epoch warmup.
The source is the checkpoint used by the diagnostic and matched adapter screen;
it is not claimed to be the best-validation checkpoint.

The context adapter is initialized exactly as in `relcontext01`, not loaded
from its trained endpoint. Full model weights remain trainable. A strict source
loader verifies every shared tensor's keys, shapes, dtypes and finiteness;
only zero-output new adapter/bias parameters may be missing. Auxiliary
`famo.w.*` entries are handled as in the completed control. Neither source nor
existing run outputs are overwritten. Each job has a separate runtime directory,
checkpoint directory, local log directory, Ray results directory and W&B ID.
`save_last: true` writes a real full-state last checkpoint, avoiding the previous
missing epoch49 problem. Both also retain the existing best-three checkpoints.

The shift-only arm starts with a deliberately different prediction function.
It removes gamma from both the pre-existing FiLM and the new context adapter,
in attention and MLP modulation, at all three blocks. LayerNorm, beta and
residual LayerScale stay intact. Weight tensor shapes are retained for strict
loading; gamma rows are inactive. `scale_rms` remains a raw-output diagnostic;
`effective_scale_rms` must be zero. Continue using the shift-only overlay when
loading these weights for generation or downstream DGPO: weights alone do not
encode this configuration switch.

The pair arm uses the same four bounded features as the previous relation
summary, **for each unordered visible pair**: cosine of delta-phi, opening
cosine, squared energy asymmetry, squared pT asymmetry. A shared32-wide MLP
maps them to a separate per-head logit bias for generation blocks0 and1.
Pair identity is retained; no mean/max event pooling is applied. The branch
uses no truth, tau assignment or geometry reconstructed from clean targets.
Diagonal/padded pairs have zero bias; the existing directed attention and
padding masks remain authoritative.

Only visible-visible logits receive the bias. Subsequent generation blocks
let invisible queries read the updated visible representations. There is no
block2 bias head: its visible update would have no later route to the invisible
outputs. Existing PET and global/context FiLM are retained. The new outputs are
zero-initialized, preserving the initial mathematical function and shared RNG;
a differentiable floating-point attention mask can change the SDPA kernel and
produce FP32 rounding differences. Tests check initial outputs/input gradients
within FP32 tolerance, rather than claim bitwise identity. With four generation
heads this branch adds1480 parameters; memory/compute are not matched to the
control. This tests the added branch as a package, not geometry separately from
additional capacity, and is not a full Pairformer or MMDiT migration.

## Evaluation and decision

Keep the same1024-event panel, K32, per-event CPU seed42017+position,
legacy DDIM20, evaluation batch16/GPU. Save samples and reports before training
and at completed epoch50, alongside ordinary validation MSE.

Primary for each new arm: **absolute epoch50 joint-radius W1 minus the context
control's epoch50 W1**. Negative favors the new arm. Do not compare the two
arms' change-from-initial-model numbers: shift-only changes the initial function.
The control's recorded endpoint W1 is0.008133006002108336 rad; comparison scripts
require its actual saved samples/report, not this scalar alone.

Report whole-event paired bootstrap intervals, narrow-region truth-gap change,
fine theta/phi marginal W1, existing quantiles/CDFs, and invalid-direction counts.
Keep all32 draws, with no best-of-K or projection. Lower MSE alone is not a win
on the primary endpoint. If W1 improves but narrow spike mass does not, label it
broader topology improvement, not recovery of the spike. Intervals are exploratory
and pointwise (two comparisons are not familywise adjusted); training-seed
uncertainty is not covered. Confirm a selected improvement on independent events
and another training seed. A DGPO closure claim still needs a fresh adequately
trained H4 audit; these supervised jobs do not train one.

## Sync and launch

Upload the complete changed repository, including the new Python modules and
the modified model/head/transformer. The older `sync_relation_film.sh` copies
only a small file list and is **insufficient** for these jobs. Use the existing
root upload from the Mac:

```bash
rsync -avz --progress \
  -e "ssh -i ~/.ssh/nersc" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude-from='/Users/yirenwu/Ztautau/ml_pipeline/NERSC/upload-excludes.txt' \
  /Users/yirenwu/Ztautau/ml_pipeline/ \
  yiren@perlmutter.nersc.gov:/global/homes/y/yiren/ml_pipeline/
```

Run the jobs on **two separate16-GPU allocations**,32 GPUs total. Use the
existing Ray setup separately in each allocation; its `RAY_ADDRESS` must point
to that allocation's head. Do not launch both against a single16-GPU cluster.
The shared source/data are read-only. Run each command once, from its own
allocation's shell in `/global/homes/y/yiren/ml_pipeline`.

Job1:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_context_shift_only.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_context_shift_only_02/ray_results
```

Job2:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_neutrino_backend.py --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/train_diffusion_pair_attention.yaml \
  -- --ray_dir /pscratch/sd/y/yiren/Ztautau/diffusion_pair_attention_02/ray_results
```

Do not reuse either run ID/output path for an independent rerun. A full-state
resume requires its own explicit checkpoint/protocol, rather than treating a
partially populated directory as a new experiment.

## Compare the saved endpoints

This is CPU statistical postprocessing of existing samples; it starts no model,
Ray worker, allocation or training. It verifies matched coverage protocols,
epoch50, event identities and reconstructed saved metrics before resampling
whole events. It writes only the requested new JSON file.

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/compare_conditioning_endpoints.py \
  --control /pscratch/sd/y/yiren/Ztautau/diffusion_relation_context_filtered_02/checkpoints/joint_coverage/epoch-0050.json \
  --candidate /pscratch/sd/y/yiren/Ztautau/diffusion_context_shift_only_02/checkpoints/joint_coverage/epoch-0050.json \
  --output /pscratch/sd/y/yiren/Ztautau/diffusion_context_shift_only_02/context_comparison.json

shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 scripts/compare_conditioning_endpoints.py \
  --control /pscratch/sd/y/yiren/Ztautau/diffusion_relation_context_filtered_02/checkpoints/joint_coverage/epoch-0050.json \
  --candidate /pscratch/sd/y/yiren/Ztautau/diffusion_pair_attention_02/checkpoints/joint_coverage/epoch-0050.json \
  --output /pscratch/sd/y/yiren/Ztautau/diffusion_pair_attention_02/context_comparison.json
```

## Validation limits

36 targeted tests passed. Synthetic CPU tests cover strict source migration, old RNG/parameter preservation,
scale removal without beta/LayerScale changes, zero pair initialization,
pair geometry/masks/permutation/periodicity, gradient flow through visible states
to invisible loss, optimizer parameter ownership, checkpoint/optimizer replay,
matched16-GPU configurations and paired endpoint statistics. The broader legacy
model suite cannot collect locally because Lightning is missing. The local checkout
also lacks the remotely generated event-info file required by the full runtime;
the network defaults and both overlay branches were resolved separately. Actual NERSC
checkpoint loading, CUDA kernels,16-GPU DDP and memory use require the real run;
no remote training or allocation was started by the assistant.


## Existing filtered data (2026-09-29 correction)

The common parent uses train `omnifold_attention_10pct_stic_filtered_test1/train`
(416701 events) and validation `diffusion_val_20pct_seed42_stic_filtered_test1/val`
(119002 events). Source normalization is preserved. The manifest and row-count
check rejects missing/incomplete inputs without falling back to raw data.
Fresh IDs/output roots are ctxshift02/pairattn02 and `_02`; existing jobs and
historical results are unchanged. Prepare the same-data control with
`config/train_diffusion_relation_context_filtered.yaml` (relcontext02). Upload
updated files before launching; all jobs remain user-submitted on16GPUs each.
