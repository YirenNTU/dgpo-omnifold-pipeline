# Step-260 raw-monitor initialization quick screen

This is a separate **classifier-only** test. It reuses the completed replay's
`pool.pt`, `scores.pt`, manifest, report, and archived `old_runtime.yaml`. It does
not instantiate/train a diffusion policy, generate candidates, unfold, install
rewards, resume W&B, or write to any production checkpoint directory.

## NERSC command

Sync the two new scripts (`ablate_raw_monitor_initialization.py` and its test)
and `config/dgpo_10pct_raw_monitor_init_ablation.yaml` with the current repository.
The existing `scripts/diagnose_raw_monitor_replay.py` is also required. Use the
usual designated Shifter image and an existing **16-GPU Ray allocation** with
`RAY_ADDRESS` set. The arms run sequentially on those same 16 GPUs, not 48 GPUs.

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/ablate_raw_monitor_initialization.py \
  config/dgpo_10pct_raw_monitor_init_ablation.yaml
```

Optional read-only preflight (no GPU, no output directory, safe before the command):

```bash
shifter python3 scripts/ablate_raw_monitor_initialization.py \
  config/dgpo_10pct_raw_monitor_init_ablation.yaml --check-only
```

Input: `/pscratch/sd/y/yiren/Ztautau/raw_monitor_replay_step260_old_vs_cold100_v1/`.
Output: `/pscratch/sd/y/yiren/Ztautau/raw_monitor_init_ablation_step260_v1/`.
Existing output directories are refused. For another attempt use
`--output-dir /pscratch/sd/y/yiren/Ztautau/raw_monitor_init_ablation_step260_v2`.
No automatic retry/resume or source-overwrite behavior is enabled.

## Controlled comparison

| Arm | Initialization | Training |
| --- | --- | --- |
| A_cold | Pretrained classifier body plus fresh classifier bank | 200 updates |
| B_old_features | A's exact initial state, replacing only old monitor GroupedSequentialEmbedding, InvisibleInputProjector and internal PET adapters | 200 updates |
| C_full_warm | Entire old monitor state, including bank and trainable body | 200 updates |

“Fresh classifier bank” includes decoder, slot-position encoder (its standard
pretrained initialization, not the old monitor's fine-tuned version), and final
readout. **All three feature modules remain trainable in every arm.** B does not
inherit the old decoder/position/readout. The frozen PET core is unchanged.

All arms share the old clean architecture, exact cached candidates, old training
identity fold (~200k events), common old/new validation intersection (~10k),
global batch 32768 (2048/rank), LR 0.0002, AdamW decay 0.0005, clip 5, seed,
sampling order and training dropout RNG. Every arm gets fresh AdamW state.
The remaining old validation identities are **not** reassigned to training.
There is no early stopping, LR scheduler, best-weight restoration, or 100-epoch
readiness floor in this fixed-budget diagnostic.

Evaluations occur at updates 0, 10, 25, 50, 100 and 200. Higher raw AUC means a
more capable judge of this **fixed** policy, not a worse policy update. All
classification weights are one. A single short seed is a quick screen, not a
convergence claim or a statistically independent generalization evaluation.

## Safety and outputs

- CPU preflight verifies the old step-260 checkpoint and monitor against replay
  hashes, pool policy identity, exact common validation indices, source backbone,
  and split protocol. Full protected-file SHA256 checks bracket the experiment.
- Before any training, the old monitor must reproduce the saved replay AUC
  within 0.005. Failure stops the test instead of interpreting an invalid control.
- Each arm has `initial_monitor.pt`, `final_monitor.pt`, `step_NNNN.json`,
  `scores_NNNN.pt`, and `report.json`. These are diagnostic classifier states,
  **not resumable DGPO checkpoints**. Partial evaluations remain if interrupted.
- The top-level `summary.json` gives a compact A/B/C table and matched-step AUC
  differences; `report.json` retains full curves and module-update diagnostics. `manifest.json` and
  `split.pt` preserve source fingerprints and exact training/validation indices.
- Each evaluated update reports train loss and module-wise gradient availability,
  post-all-reduce/post-clip L2 gradient norms, and actual single-update weight
  changes relative to pre-update weights. Changes include AdamW decay; a nonzero
  change alone does not prove a useful data gradient. Zero-initialized groups
  use a 1e-12 denominator floor, so absolute norms must also be examined.

If B learns much faster than A, inherited input/adaptation features are important
under this controlled setup. If only C works, the missing decoder/position/readout
or its co-adaptation with features needs a follow-up test. If no arm reproduces
the old score, first repair the control. A weak A after 200 updates does not prove
it cannot learn with a different initialization, training schedule or budget.
