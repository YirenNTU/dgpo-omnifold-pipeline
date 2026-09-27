# Fourier integration with an unfrozen backbone

**Current execution order:** start with matched trained-10pct no-Fourier and input-linear Fourier continuation in
[DIFFUSION_FOURIER_PRETRAIN_CONTINUATION.md](DIFFUSION_FOURIER_PRETRAIN_CONTINUATION.md).
The output-adapter comparison below is now an optional architecture follow-up.
The no-Fourier continuation control is restored and runs independently alongside input.

Prepared experiment; no jobs submitted. All PET/backbone and generation-head
parameters remain trainable in every arm. No stop-gradient, frozen teacher,
distillation loss, or DGPO update is introduced.

## Question and limits

Does a late, independently encoded angular residual improve supervised velocity
prediction during continuation of the same trained no-Fourier model?

| Arm | Injection | Projection | Added parameters at hidden_dim=256 |
| --- | --- | --- | ---: |
| `none` | None | None | 0 |
| `input` | After PET input feature embedding | 16 -> 256, bias-free, zero init | 4,096 |
| `output` | After PET blocks **and** outer skip addition, before generation head | 16 -> 64, GELU, LayerNorm(64), 64 -> 256 zero-init bias-free output | 17,600 |

Both Fourier arms use the existing raw visible eta -> theta and phi harmonics
k=1,2,3,4. Invisible and padded tokens receive zero direct residual. The
existing generation-head attention conveys visible information to neutrinos.
The output adapter is used identically in denoising training and sampling.

Late injection avoids rewriting PET's internal activations **within that
forward pass**. It does not protect PET weights from downstream gradients or
remove the fusion risk at the generation head. All three backbones adapt.
The production test explicitly checks that late-adapter gradients reach PET.

The `input` vs `output` comparison changes placement **and** projection
architecture/parameter count. It tests the proposed integration package, not
placement alone. If it helps, a later output-linear or input-MLP comparison can
separate these effects. Neither is silently included in this three-arm pilot.

## Relevant primary references

### Existing EveNet mechanisms (rechecked 2026-09-24)

The original visible-angle branch already has zero-initialized linear output
and additive residual fusion. PET already uses timestep-dependent scale/shift
(`encoded * (1 + scale) + shift`) and pre-normalized attention/MLP residuals
with learnable LayerScale. The generation head also has LayerNorm, LayerScale,
and a timestep/global-condition token pathway. LayerScale is initialized at
1e-5 in these configs, but its checkpoint values are learned and have not been
measured here; it is not a condition-dependent adaLN-Zero gate.

The repository also already contains a bottleneck residual `Adapter` with a
zero-initialized up projection. PET's optional `use_adapter` defaults to false;
it is not enabled in the inspected baseline/Fourier run configs or this pilot.
That adapter operates on hidden tokens, not on separate visible-angle features.

Thus this experiment does not introduce residuals, zero initialization,
normalization, or conditional modulation to EveNet for the first time. Its new
architecture is the late visible-angle MLP residual. Existing time modulation
does not explicitly consume angular Fourier features. Neither old nor new
angular branch implements condition-dependent scale/shift/gates or an
independent Fourier cross-attention path.

- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/html/2212.09748v2),
  especially the conditioning ablation and adaLN-Zero. The diffusion transformer
  is trained jointly; identity-initialized residual conditioning is a useful
  precedent. Its [official code](https://github.com/facebookresearch/DiT/blob/main/models.py)
  uses zero-initialized scale/shift/gate outputs. This supports careful
  initialization, not a claim that a late Fourier sum will outperform an early
  sum in a pretrained EveNet. DiT's global conditioning also differs from our
  per-particle angular features. Do not zero or replace EveNet's existing
  pretrained blocks when borrowing this principle.
- [ReZero](https://proceedings.mlr.press/v161/bachlechner21a.html): zero-initialized
  learnable residual gates support optimization of fully trainable deep
  networks. This adapter uses a zero output projection instead of a ReZero
  scalar gate; the shared idea is an initially inactive residual, not an exact
  reproduction. Do not zero both an additional gate and the output projection.
- [FiLM](https://arxiv.org/abs/1709.07871): feature-wise conditional scale/shift is
  an established fusion alternative. The paper includes an end-to-end trained
  CNN setting as well as a fixed pretrained feature-extractor setting; neither
  directly establishes the outcome of unfreezing our pretrained diffusion
  backbone. FiLM/adaLN remains a subsequent candidate, not an extra arm here.
- [Net2Net](https://arxiv.org/abs/1511.05641): function-preserving architecture
  transformations motivate checking identical initial predictions before
  further training. Our new residual starts at zero and preserves the source
  function; this is not an implementation of Net2WiderNet or Net2DeeperNet.

The old Fourier branch was already zero initialized. The literature therefore
does **not** establish that another zero initialization fixes the observed loss
gap. The new ingredients under test are late injection and a separate nonlinear
feature encoder; gains remain empirical.

## Source and common settings

Baseline: [EveNet/bvp5rn76](https://wandb.ai/ytchou97-university-of-washington/EveNet/runs/bvp5rn76).
The W&B save log was read on 2026-09-24 and records the exact best filename:

```text
/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/epoch=190_train=0.1381_val=0.1263.ckpt
```

The remote file's present availability has not been checked. The launcher
reads the checkpoint from the shared YAML (optional CLI override) and requires it to exist and preflight checks the actual model
schema. Load raw `state_dict` including normalization buffers, never EMA or old
optimizer state. Reject checkpoints already containing the angular branch and
reject missing/shared-shape-mismatched weights. Disabled task heads and FAMO
may remain in the source without being used.

- Same fixed 10% training and cleaned 20% validation population as the recent
  angular runs. Same normalization source, FP32, 16 GPUs, 2048 events/GPU.
- Fresh AdamW and cosine schedule for all arms: 50 epochs, 5-epoch warmup;
  patience 51 so early stopping cannot shorten a finite-loss pilot.
- Effective peak LR: PET/GlobalEmbedding/ObjectEncoder `2e-5`, generation/input
  projectors `1e-4`. Fourier has a separate optimizer at `2e-5` in both arms,
  matching PET rather than introducing a branch-LR intervention. No LR GPU
  scaling. The existing world-size weight-decay scaling remains unchanged.
- Same dropout/stochastic-depth configuration across arms; PET and generation
  retain the historical 0.1 settings. Fourier MLP has no dropout.
- All backbone parameters unfreeze; a runtime guard rejects frozen parameters.
- Ordinary velocity loss, no low-noise weighting, EMA, classifier, or DGPO.
- Fresh W&B IDs; common group `Fourier pretrained continuation` in `EveNet`.
  Each arm and seed has a distinct output directory. Existing runtime outputs
  are never silently reused; choose a new output root for a rerun.

This is a controlled continuation pilot, **not** a reproduction of the old
run's full optimization recipe or total training budget. At the previous data
size, 50 epochs is approximately 650 optimizer update rounds.

## Randomness and diagnostics

The experiment opts into ordered parquet reads, order-preserving Ray operators,
and no worker-local shuffle. Model-forward RNG is isolated from model
construction and unrelated generation diagnostics. Train draws change by
epoch/rank/batch; validation draws remain fixed by rank/batch across epochs.
This controls noise only when the ordered batches and worker sharding agree;
it is not event-identity seeding and does not guarantee bitwise reproducibility
of distributed GPU training. Keep worker count, data files and batch size fixed.

Monitor `train/loss`, `val/loss`, `lr-body`, `lr-generation`, `lr-fourier`, and
`train/fourier/{base_rms,residual_rms,residual_to_base_rms}` (also `val/fourier/*`).
RMS ratios are evaluated at each arm's injection site; their denominators are
different representations and should not be treated as identical units.
No-Fourier has no branch or Fourier optimizer. Use epochs / actual update rounds
for alignment: Lightning global_step may count multiple optimizers differently.

Primary endpoint: mean `val/loss` over epochs 45--49. Secondary: full learning
curve, best validation loss, train loss, branch RMS and finite gradients. First
screen seed 42, then confirm a promising arm vs control on two further seeds.
Lower output-arm loss vs input but not vs none only supports reduced damage;
it does not establish a useful Fourier contribution. No improvement rules out
this particular late-adapter recipe, not every fusion method.

## NERSC launch

Update the existing remote `ml_pipeline` checkout with these changes using the
usual synchronization workflow. Repository-root rsync uploads must include
`--exclude-from=NERSC/upload-excludes.txt`; preserve other existing exclusions.
Do not upload toy code/outputs or the excluded classifier/review checkpoints.
Do not create another remote code copy or implicitly delete remote files.

Run from the existing remote repository, with `config/generated_event_info.yaml`
already generated and the shared dataset/normalization files available:

```bash
# Checkpoint is read from train_diffusion_fourier_integration_common.yaml.

# Config only: no data access, Ray cluster, W&B initialization or training.
python3 scripts/train_fourier_integration.py --arm output --check-only

# CPU preflight on real checkpoint + two validation events, no training.
shifter python3 scripts/train_fourier_integration.py --arm output --preflight

# Optional architecture comparison, submitted personally (4 nodes / 16 GPUs each).
sbatch NERSC/submit-fourier-integration.sbatch none
sbatch NERSC/submit-fourier-integration.sbatch input
sbatch NERSC/submit-fourier-integration.sbatch output
```

The batch script repeats preflight before starting its Ray cluster and refuses
training unless all three arms give identical initial train-mode and eval-mode
velocity predictions with matched RNG. Full backbones are checked trainable.
Logs are in `launch_logs/<job id>/`, including the resolved config and preflight
report. Outputs default to
`/pscratch/sd/y/yiren/Ztautau/fourier_integration/seed42/{none,input,output}/`.
A third batch-script argument selects a different output root.

If already inside a correctly configured Ray allocation, run an arm directly:

```bash
shifter python3 scripts/train_fourier_integration.py --arm output
```

Do not pass a Lightning resume checkpoint or `--load_all`. The explicit launcher
accepts neither; these would invalidate fresh-optimizer or ordered-data controls.

## Local verification

```bash
PYTHONPATH=evenet_dgpo:scripts python -m pytest -q \
  evenet_dgpo/evenet/network/body/test_angular_conditioning.py \
  evenet_dgpo/evenet/network/body/test_fourier_integration.py \
  scripts/test_diffusion_angular_config.py scripts/test_fourier_integration_config.py
bash -n NERSC/submit-fourier-integration.sbatch
```

Tests use the actual PET and generation head on small CPU tensors: initial
prediction equivalence with dropout, nonzero new-branch gradients, continued
backbone gradients, visible-only injection, learned padding masking, two
optimizer updates, nonoverlapping optimizer ownership, strict source rejection,
RNG restoration, and common configuration. They do not replace the real NERSC
checkpoint/data preflight or a distributed run.
