# DGPO Neutrino RL

Standalone RL add-on that fine-tunes the EveNet neutrino diffusion model with **DGPO + linear CPO + frozen latent-SWD constraint**.

Foundation supervised training and inference are unchanged in `evenet/`. This folder only runs when `rl.enabled: true` in `config.yaml`. When `rl.enabled: false`, `dgpo_trainer.py` exits with:

> RL pipeline disabled. Use evenet/train.py for original foundation training.

## Method (single path)

When RL is enabled, all components are always active:

- **DGPO**: `K=8` candidates per event; `num_train_timesteps=8` inner policy-eval steps per batch (gradients accumulated into one AdamW step); pure DGPO loss; z-score advantages from component-normalized truth-distance reward. Candidate rollout uses **DDIM** (the only sampler).
- **CPO**: linear/Taylor post-Adam repair with AdamW-metric trust cap. Fires when the constraint `C_norm` exceeds `epsilon` and projects the AdamW step; `lambda = v/(b·p)` keeps the constraint gradient in the denominator (coordinate-invariant, phi-safe).
- **Latent-SWD constraint** (the only constraint provider): a **frozen, pre-trained object-token bottleneck autoencoder** (`latent_constraint/`) encodes truth and predicted neutrino configurations — conditioned on the frozen EveNet `event_token` + `object_token` columns — into the same latent space; the sliced-Wasserstein distance between the two latent clouds forms the ratio-normalized constraint `C_norm = (SWD_pred − SWD_tt)/(SWD_tt + eps)`. No on-policy retrain or finetune: the encoder is trained once, independently, then loaded frozen on every rank.

## Main files

| File | Role |
|------|------|
| `dgpo_trainer.py` | Ray Train entry point |
| `config.yaml` | Canonical config (`rl.enabled` is the only RL switch) |
| `projection_cpo.py` | Linear CPO repair |
| `latent_constraint/` | Frozen latent-SWD constraint model (training + DGPO provider) |
| `model_utils.py` | Model load/save, Lightning-compatible checkpoints |
| `rewards.py` | `ComponentNormalizedTruthDistanceReward` |
| `data_preprocess/` | Optional train/val subset tooling |

## Setup

1. **Train the latent constraint encoder first** (or reuse an existing checkpoint):

   ```bash
   python RL/DGPO_neutrino/latent_constraint/train_latent_constraint.py \
       RL/DGPO_neutrino/latent_constraint/config.yaml
   ```

   See `latent_constraint/README.md` for the model description and monitoring.

2. Edit `config.yaml` before running DGPO:

   - `platform.data_parquet_dir` / `data_parquet_val_dir` — must be the `*_evttok` augmented parquet carrying `event_token` **and** `object_token` (`preprocessing/augment_event_token.py --object-tokens`)
   - `options.Dataset.normalization_file`
   - `options.Training.model_checkpoint_load_path` (supervised or DGPO resume)
   - `options.Training.model_checkpoint_save_path`
   - `dgpo.projection_constraint.latent_swd.checkpoint_file` (frozen encoder `best.ckpt`; must be readable from every node)
   - `dgpo.projection_constraint.latent_swd.normalization_file` (same stats as the policy dataset)

3. Optional pre-flight check (CPU login node is enough):

   ```bash
   python3 RL/DGPO_neutrino/latent_constraint/verify_latent_swd_checkpoint.py \
       RL/DGPO_neutrino/config.yaml
   ```

## Train

```bash
ray start --head
python RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml
```

`platform.number_of_workers` sets Ray/DDP worker count. To skip W&B, set `WANDB_DISABLED=true` in the environment.

## Resume

Point `options.Training.model_checkpoint_load_path` at a DGPO `last.ckpt`. Restored payload includes model, save EMA, rollout EMA, optimizer, reference policy, global step/epoch, and the latent-SWD constraint provenance (`constraint_type`, encoder checkpoint path). Resume rejects checkpoints whose saved constraint type is not `latent_swd`.

Loading a supervised-only Lightning checkpoint starts DGPO from epoch 0 with a fresh optimizer; the frozen constraint encoder is always loaded from `latent_swd.checkpoint_file`.

## Predict

Use the saved DGPO checkpoint in `evenet/predict.py` or `reasoning_module/predict_TT2L.yaml`. Keep network config, `event_info`, and normalization aligned with training.

DGPO checkpoints include default FAMO keys for strict Lightning loading (compatibility shim only).

## Optional dataset subset

```bash
python RL/DGPO_neutrino/data_preprocess/make_dataset_subset.py \
  RL/DGPO_neutrino/data_preprocess/subset_dataset_object.yaml
```

Then point `platform.data_parquet_dir` and `data_parquet_val_dir` at the generated folders.

## Tests

On **NERSC** (login node is enough — no GPU / no `salloc`):

```bash
cd ~/EveNet-private
./NERSC/smoke_dgpo_tests.sh
```

Uses the same Shifter image as `NERSC/salloc_1node_4gpu.sh` (`evenet:1.3`), so `python3` has PyTorch. Do **not** use login-node `python` / `(base)` conda — those often lack `torch`.

On **Mac / local** with the `MyEve` conda env:

```bash
conda activate MyEve
cd ~/EveNet-private
export PYTHONPATH=$PWD
python -m unittest RL.DGPO_neutrino.test_normalizer_denormalize_grad_parity -v
python -m unittest RL.DGPO_neutrino.latent_constraint.test_dgpo_constraint -v
python -m unittest RL.DGPO_neutrino.latent_constraint.test_object_token_ae -v
```

Or any env with `torch` from `requirements.txt` (do **not** use `(base)` unless it has PyTorch).

```bash
pytest RL/DGPO_neutrino
```
