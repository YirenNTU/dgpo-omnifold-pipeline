# DGPO neutrino — Weights & Biases metrics

Logged from `RL/DGPO_neutrino/dgpo_trainer.py` for the frozen method: **DGPO + linear CPO + frozen latent-SWD constraint**.

## Meta

| Key | Description |
|-----|-------------|
| `epoch` | Float epoch index. Training scalars use W&B `step = global_step`. `val/*` and `train_dist/*` use `epoch` as x-axis. |

## `reward/dist/` — overlapped reward histograms

| Key | Description |
|-----|-------------|
| `reward/dist/overlap` | Best / worst / median reward per valid event among K candidates (density-normalized histogram). Logged every `dgpo.log_reward_dist_every` steps. |

## `reward/monitor/` — scalar reward summaries

Logged every optimizer step: `reward/monitor/best_of_k`, `median`, `mean_gap`, `last_place`, `p10`, `p30`, `p70`, `p90`, `advantage_pos_neg_gap`.

## `train/loss/` — training losses

| Key | Description |
|-----|-------------|
| `train/loss/total` | Scalar passed to `backward()`: pure DGPO main term (`beta_kl=0`). CPO repair runs after AdamW. |
| `train/loss/dgpo` | DGPO main: detached gate × advantage × `L_cur`. |
| `train/loss/kl` | Always 0 in backward (velocity KL disabled). |
| `train/loss/L_cur` / `L_ref` / `delta` | Current vs reference velocity MSE diagnostics. |

## `train/grad/`

| Key | Description |
|-----|-------------|
| `train/grad/global_norm_pre_clip` | L2 norm before `clip_grad_norm_` (max over inner timesteps when accumulating). |
| `train/grad/clip_active` | 1 if clipping fired. |

## `projection/*` — linear CPO repair (W&B panel)

Logged every optimizer step; x-axis `global_step`. W&B scalars:

| Key | Description |
|-----|-------------|
| `projection/v_linear` | `C_adam_pred − ε` — linear violation after AdamW; drives λ when positive. |
| `projection/C_adam_pred` | Taylor estimate `C_old + bᵀδ₀` at `θ_adam`. |
| `projection/lambda` | CPO multiplier `λ★ = [v / (bᵀp + damping)]₊`. |
| `projection/final_update_norm` | ‖θ_final − θ_old‖ after projection (incl. final-update cap). |
| `projection/summary/C_projected_minus_old` | `C_projected − C_old`; negative ⇒ projection reduced C vs pre-step. |
| `projection/multi_sample/C_mean` | Mean normalized constraint `C_norm` over multi-sample draws at `θ_old`; per-batch trace for sawtooth / oscillation diagnostics. |

Other projection diagnostics are still computed internally for the repair step but are not logged to W&B.

## `swd/*` — frozen latent-SWD constraint (W&B panel)

Logged every optimizer step; x-axis `global_step`.
Create a dedicated W&B panel with these keys (separate from the CPO repair `projection/*` panel):

| Key | Description |
|-----|-------------|
| `swd/active` | 1 when SWD was computed; 0 when batch skipped (`min_samples`). |
| `swd/pred_truth` | SWD(z_pred, z_truth) in the frozen encoder latent space. |
| `swd/truth_truth` | Null floor: truth/truth split SWD within the batch. |
| `swd/ratio` | `swd_pred_truth / (swd_truth_truth + eps)`. |
| `swd/C_norm` | Normalized constraint `(pred - null) / (null + eps)`; CPO fires when > `margin`. |
| `swd/mask_count` | Valid rows encoded this step. |
| `swd/skipped_small_mask` | 1 when too few valid rows. |

Also logged: `projection/multi_sample/C_mean` (mean C_norm over multi-sample draws).

## `train_dist/*` — training kinematics (epoch end)

`train_dist/{pt,eta,phi}`: best-of-K reward argmax vs truth, accumulated over the epoch.

## `val/*` and `val_neutrino/*`

End-of-epoch DDIM validation (`validation_K` candidates). `val/reward/mean` drives top-K checkpoint selection. `val_neutrino/*` overlays truth / current policy / frozen reference for neutrino kinematics (pT, η, φ, and p_x/p_y/p_z in GeV).

## `val_mass/*`

Same validation pass: W and top mass reconstructed from ground-truth `assignments-indices` (b + lepton from point cloud + neutrino). **Truth** histogram uses target neutrinos; **Pred** / **Ref** use DDIM neutrinos (best-of-`validation_K` vs frozen reference, same candidate rule as `val_neutrino/pt`).

## Config knobs

See `rl.enabled` and numeric fields in `RL/DGPO_neutrino/config.yaml`. Notable:

- `dgpo.K`, `dgpo.num_train_timesteps`, `dgpo.beta`, `dgpo.adv_clip_max`
- `dgpo.projection_constraint.epsilon`, `multi_sample.samples`, `trust_region_ratio`
- `dgpo.projection_constraint.latent_swd.checkpoint_file`, `margin`, `num_projections`, `min_samples`, `apply_to`
