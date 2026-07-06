# Latent-space constraint model for DGPO neutrino reconstruction

A compact, **independently trainable** autoencoder that learns an event/neutrino
latent space. After training, its frozen encoder maps both **truth** and
**DGPO-predicted** neutrino configurations into the *same* latent space, where
their distributions are compared with **sliced Wasserstein distance (SWD)** to
produce the CPO constraint source.

This is **not** a replacement for EveNet or DGPO. It is a lightweight constraint
model that should never become the DGPO training bottleneck.

## Contents

| File | Purpose |
|------|---------|
| `object_token_ae.py` | `ObjectTokenBottleneckAutoencoder` (the single supported model) + `encode_latent` / `decode` / `reconstruction_loss`, and `save_checkpoint` / `load_checkpoint`. |
| `normalizers.py` | `load_normalizers_from_pt(...)`: visible/conditions/invisible `Normalizer` triple from `normalization.pt`. |
| `sliced_wasserstein.py` | `sliced_wasserstein_distance(z_pred, z_truth, ...)` and `random_projections(...)`. |
| `dgpo_constraint.py` | **DGPO constraint provider**: `compute_latent_swd_constraint(...)` + `LatentSWDState` / `LatentSWDConfig`, plus the shared mask/kinematics helpers and `sync_projection_constraint_C_across_ranks`. |
| `train_latent_constraint.py` | Independent Ray Train (DDP) training script, mirroring the EveNet / DGPO style. |
| `config.yaml` | Repo-style config; the `latent_constraint:` block exposes model-size knobs. |
| `plots.py`, `mass_diagnostics.py` | W&B monitoring plots + W/top-mass physics diagnostics. |
| `test_object_token_ae.py` | Model sanity tests (shapes, masking, gradients, checkpoint roundtrip) — no data/GPU needed. |
| `test_dgpo_constraint.py` | Constraint tests: SWD-ratio behavior, CRN seeding, encode-from-kinematics differentiability, frozen-encoder grad isolation. |
| `verify_latent_swd_checkpoint.py` | Pre-flight checkpoint/config check before launching DGPO. |

## Model — object-token bottleneck autoencoder

Inputs are the frozen EveNet event representation — the pooled **event CLS
token** (`event_token`) plus the **full set of per-object ObjectEncoder tokens**
(`object_token`: `(B, P, D)`, both from `preprocessing/augment_event_token.py
--object-tokens`) — together with the two-neutrino kinematics. The target is
the **original pretrain-model event token** and the **neutrino kinematics**.

```
nu_tok     = nu_mlp(normalize(nu))            # separate MLPs per neutrino slot
antinu_tok = antinu_mlp(normalize(antinu))
evt_tok    = token_embed(standardize(event_token))
obj_tok    = token_embed(standardize(object_token))     # (B, P, d), padded objects masked
seq        = [z-cls, evt_tok, obj_tok..., nu_tok, antinu_tok]   # + type embeddings
out        = TransformerEncoder(seq, key_padding_mask)
z          = latent_head(out[:, z-cls])                 # (B, latent_dim) BOTTLENECK
recon      = decoder(z) -> [nu_pair_normalized (2*3) ; event_token_standardized (D)]
loss       = MSE(nu_recon, normalize(nu_truth)) + MSE(tok_recon, standardize(event_token))
```

- **Rich input conditioning**: the neutrinos share one self-attention stack with
  the event CLS token and all object tokens, so the ν↔object (e.g. ν↔lepton)
  correlation is available *before* the bottleneck pools `z` — no assignment /
  no physics label: attention picks the relevant object itself. Objects enter as
  input only; they are never a reconstruction target.
- **Hyperparameter-free loss**: both targets are standardized to ~unit variance
  and each MSE is mean-reduced, so the loss is their plain SUM (implicit weight
  1 — principled, not tuned).
- Because `z` must reconstruct BOTH the neutrinos and the event token, it
  necessarily entangles neutrino kinematics with event information, so
  `SWD(z_pred, z_truth)` reacts to event-inconsistent neutrino predictions.
- Neutrino inputs are **physical** `(log1p(pt), eta, phi)` and are normalized
  *inside* the model with a **differentiable** mirror of `Normalizer`
  (inv-CDF phi), so the gradient survives from `z_pred` back to the DGPO
  prediction path even with a frozen encoder. `encode_latent` is deliberately
  NOT `@torch.no_grad()`.
- Token standardization stats (`token_mean/std`, `obj_token_mean/std`) are
  computed once from training data and persist in the checkpoint, so the frozen
  DGPO encoder standardizes bit-identically to training.
- ⚠️ Build the encoder from the **same `normalization.pt`** the policy uses, or
  the latent spaces diverge.

## 1. Train the model

The script reuses the EveNet / DGPO data pipeline (Ray Data parquet shards) and
distributed conventions (Ray Train + DDP, rank-0 checkpoints, all-reduced
validation). The parquet must be the `*_evttok` augmented mirror carrying BOTH
`event_token` and `object_token` columns. From the repo root:

```bash
python RL/DGPO_neutrino/latent_constraint/train_latent_constraint.py \
    RL/DGPO_neutrino/latent_constraint/config.yaml
```

Standalone AE training can run on 1 node (default `number_of_workers: 4` in
`latent_constraint/config.yaml`) or any Ray cluster size. **DGPO integration does
not depend on how the encoder was trained** — only that `checkpoint_file` is
visible on every node and built from the same `normalization.pt` as the policy.

- Single-GPU / CPU debugging: uncomment the `LOCAL TESTING` blocks in
  `config.yaml` (`number_of_workers: 1`, `use_gpu: false`) and optionally pass
  `--max-steps N` for a smoke run.

Checkpoints are written by **rank 0** to
`options.Training.model_checkpoint_save_path` as `last.ckpt` and `best.ckpt`
(monitoring validation reconstruction MSE). Training runs up to `epochs` but can
stop earlier via `latent_constraint.train.early_stopping`.

### What gets logged to W&B (rank 0 only)

| Metric | Meaning | What to look for |
|--------|---------|------------------|
| `train/loss`, `val/loss` | Reconstruction MSE: neutrinos + event token | Decreasing, no large train/val gap. |
| `val/recon_nu_mse`, `val/recon_token_mse` | The two loss components | Both decreasing. |
| `residual/res_mse/{pt,eta,phi}` | Per-component PHYSICAL neutrino reconstruction MSE | Decreasing — "is the error actually shrinking?" |
| `val/latent_rms` | RMS of the latent on validation | **Stays > 0** — collapse toward 0 means a degenerate latent. |
| `val/swd_null` | SWD between two bootstrap resamples of truth latents | Stays **~1× noise floor** (same distribution). |
| `val/swd_shuffled` | SWD truth vs wrong ν↔event pairing | **Grows** with training. |
| `val/swd_separation` | `swd_shuffled / (swd_null + eps)` | **> 1 and rising** — the headroom DGPO's constraint exploits; the single most important curve. |
| `val/swd_pt_shift_{N}GeV` | SWD truth vs raw-pt-shifted truth (default 5, 10 GeV) | **Rising** — physical sensitivity check; larger shifts give larger SWD. |
| `val_mass/*` | W/top-mass JSD diagnostics (truth vs AE-recon vs shuffled) | Recon ≈ truth; shuffled clearly separated. |
| `generation-invisible/*` | Truth-vs-recon histograms + latent-z (every `plot_every_n_epochs`) | Visual companion to the scalars. |

## 2. Enabling it in DGPO (wired into `dgpo_trainer.py`)

The latent-SWD encoder is the **only** projection-constraint provider. In the
DGPO config (`RL/DGPO_neutrino/config.yaml`), under `dgpo.projection_constraint`:

```yaml
dgpo:
  projection_constraint:
    type: latent_swd
    epsilon: 0.1                       # CPO activation margin (defaults to latent_swd.margin)
    latent_swd:
      checkpoint_file: ".../best.ckpt"   # REQUIRED: frozen encoder checkpoint
      normalization_file: ""             # optional; defaults to the ckpt's stored path
      margin: 0.1
      eps: 1.0e-6
      min_samples: 8
      apply_to: all_candidates           # all_candidates | best_candidate
      num_projections: 1024
```

What the wiring does:

1. `resolve_projection_constraint_config` parses the `latent_swd` block;
   `epsilon` defaults to `latent_swd.margin`.
2. At startup **each Ray Train worker** builds a frozen `LatentSWDState` via
   `init_latent_swd_state`, then `broadcast_latent_swd_state` copies rank-0
   weights so all ranks are bit-identical before CPO repair.
3. Every repair step routes through `compute_latent_swd_constraint`; the CPO
   repair, scalar all-reduce, and DDP-averaged `∇C` follow the standard path.
4. Diagnostics are logged under **`swd/*`** (dedicated W&B panel) and
   `projection/*` CPO repair scalars (e.g. `swd/C_norm`, `swd/ratio`,
   `projection/lambda`, `projection/multi_sample/C_mean`).

**Data requirement**: the DGPO parquet must carry `event_token` AND
`object_token` (the `*_evttok` augmented mirror);
`compute_latent_swd_constraint` aborts with a clear `KeyError` if either column
is missing. `repeat_batch_for_candidates` tiles both columns over the K
candidates automatically.

**Constraint** (ratio-normalized null excess):

```
C_norm = (SWD(z_pred, z_truth) − SWD_tt) / (SWD_tt + eps)
```

`SWD_tt` is the same-distribution null floor for `z_truth`, computed from **two
size-`n` bootstrap resamples** (with replacement) of *all* `n` truth latents.
One random projection set is sampled per call (`num_projections`, averaged) and
**reused** for both distances so the ratio is self-consistent. Gradients flow
through `z_pred` only.

**Enforcement** is post-AdamW **CPO trust-region projection repair** (Achiam
2017 style): the constraint never enters the backward; CPO fires when
`C_norm > epsilon` and projects the AdamW step. `lambda = v/(b·p)` keeps the
constraint gradient in the denominator, so the repair is invariant to
coordinate reparametrization (e.g. the inv-CDF phi channel).

### Common random numbers (why CPO can *see* the violation)

SWD is a **stochastic** estimator: each call samples random projection directions and a
random bootstrap null. CPO measures the constraint several times per step — at
``theta_old`` (for ``C`` and ``b = grad C``) and at ``theta_adam`` (the post-AdamW proxy).
The CPO repair therefore passes a **per-(global_step, multi-sample) seed**
(`compute_latent_swd_constraint(..., seed=...)`) so the ``theta_old`` evaluation and its
matching ``theta_adam`` proxy use the **same** projections and the **same** null split.
Only the parameters differ, so the measured ``Delta C`` reflects the step (common random
numbers / variance reduction). The seed still advances across ``global_step``, so the
projection space is covered over training.

### Distributed training (multi-GPU / multi-node)

| Stage | Behavior |
|-------|----------|
| **Startup** | Every rank calls ``init_latent_swd_state`` (loads the frozen ``.ckpt`` onto its local GPU), then ``broadcast_latent_swd_state`` copies rank-0 weights so all ranks are bit-identical before training. |
| **Per step** | Each rank runs ``compute_latent_swd_constraint`` on its own Ray-Data shard (local ``KB`` rows). ``C_norm`` is differentiable w.r.t. the local policy; DDP averages policy gradients during ``backward()``. |
| **CPO repair** | ``sync_projection_constraint_C_across_ranks`` all-reduces the scalar ``C`` so every rank computes the same ``lambda`` and identical CPO repair. |
| **Skipped** | No on-policy retrain, finetune, or weight broadcast during DGPO (the encoder is frozen). |

Requirements for a healthy multi-node run:

1. ``latent_swd.checkpoint_file`` must be readable from **every** node (shared filesystem).
2. Use the **same** ``normalization.pt`` as the policy (trainer warns if paths differ).
3. Set ``latent_constraint.model.dropout: 0.0`` before the final AE train if you want a
   fully deterministic frozen encoder (optional; encoder stays in ``eval()`` during DGPO
   either way).

DGPO checkpoints record ``constraint_type: latent_swd`` plus encoder paths so resume
rejects mismatched constraint backends.

### Pre-flight: verify checkpoint before DGPO

On NERSC (login node, CPU is fine):

```bash
python3 RL/DGPO_neutrino/latent_constraint/verify_latent_swd_checkpoint.py \
    RL/DGPO_neutrino/config.yaml
```

Checks: config `type: latent_swd`, checkpoint file exists, `load_checkpoint` succeeds,
encoder is frozen/eval, synthetic encode + SWD returns finite scalars. Exit code 0 → safe to launch DGPO.

At DGPO startup (rank 0) you should also see:

```
[latent_swd] loaded encoder: ckpt=... latent_dim=16 ... val_loss=... requires_grad=False
[DGPO] DGPO + CPO + latent-SWD (frozen): checkpoint=... latent_dim=16 ... world_size=16 ...
```

## 3. Manual use (lower-level building blocks)

```python
from RL.DGPO_neutrino.latent_constraint.object_token_ae import load_checkpoint
from RL.DGPO_neutrino.latent_constraint.sliced_wasserstein import sliced_wasserstein_distance

constraint_model, cfg = load_checkpoint(path, device)
constraint_model.eval()
for p in constraint_model.parameters():
    p.requires_grad_(False)               # freeze the constraint encoder

# batch dicts need: x_mask, event_token, object_token, and a neutrino
# representation ('nu_kin' (B,2,3) physical, or 'x_invisible').
z_pred = constraint_model.encode_latent(pred_batch)        # differentiable wrt prediction
with torch.no_grad():
    z_truth = constraint_model.encode_latent(truth_batch, detach_neutrinos=True)

C = sliced_wasserstein_distance(z_pred, z_truth, num_projections=64, seed=0)
```

Encode the **predicted** batch normally (no `no_grad`); encode the **truth**
batch under `torch.no_grad()` and/or `detach_neutrinos=True` so the constraint
stays differentiable w.r.t. the prediction only.

## 4. Sanity test / debug

```bash
python -m unittest RL.DGPO_neutrino.latent_constraint.test_object_token_ae
python -m unittest RL.DGPO_neutrino.latent_constraint.test_dgpo_constraint
python -m unittest RL.DGPO_neutrino.latent_constraint.test_mass_diagnostics
```

The tests synthesize a tiny `normalization.pt` in a temp dir and verify shapes,
gradient flow (including SWD differentiability wrt predicted neutrinos with a
frozen encoder), deterministic projections, and checkpoint roundtrip — no
dataset or GPU required.

## Notes

- Neutrino target convention follows the repo: `(log1p(pt), eta, phi)` per slot,
  matching `x_invisible` in the nu2flow parquet and `TruthGeneration.cartesian:
  false`. For 7-feature inputs, kinematics are read from indices `1:4`.
- **Cartesian mode** (`TruthGeneration.cartesian: true`): the target is `(px, py,
  pz)` read from `x_invisible_cartesian`, normalized with
  `invisible_cartesian_mean/std` from `normalization.pt`. There is no uniform
  azimuthal channel, so `phi_index` is forced to `null` and the pt-shift
  diagnostic shifts `pt = sqrt(px^2 + py^2)`. Train the encoder in the **same
  coordinate system as the DGPO policy**. Verify the cartesian stats exist
  first: `python reasoning_module/check_norm_cartesian.py <normalization.pt>`.
- **phi normalization matches EveNet**: the invisible normalizer Gaussianizes the
  uniform phi channel via its inverse-CDF transform (set from `model.phi_index`,
  default `2`). The model takes **physical** neutrinos and normalizes internally,
  so its latent space is consistent with the space DGPO produces neutrinos in.
  `Normal.icdf` is differentiable, so the `z_pred` gradient still reaches the
  DGPO policy.
- For deterministic DGPO use, set `latent_constraint.model.dropout: 0.0`.
