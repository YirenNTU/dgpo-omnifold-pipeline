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
| `train/loss/total` | Scalar passed to `backward()`: DGPO main term plus the configured round-reference trust, optional supervised anchor, and auxiliary regularizers. CPO repair runs after AdamW. |
| `train/loss/dgpo` | DGPO main: detached gate × advantage × `L_cur`. |
| `train/loss/kl` | Legacy supervised diffusion anchor. It is zero in the active OmniFold overlay (`beta_kl: 0`). |
| `train/loss/L_cur` / `L_ref` / `delta` | Current vs reference velocity MSE diagnostics. |

## `reference_trust/` — paired round-reference anchor

| Key | Description |
|-----|-------------|
| `reference_trust/loss` | `0.5 * masked_mean((v_policy-v_round_ref)^2)` on the same `(t, eps)` draw as the DGPO loss. |
| `reference_trust/velocity_mse_ratio` | Trust velocity MSE divided by the round reference's own velocity loss. |
| `reference_trust/coefficient` | Configured trust multiplier; `1.0` in the active Ztautau overlay. |

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
When `dgpo.train_dist_enabled: false`, this path is fully cold: no per-step truth/pred
arrays, histogram buffers, cross-rank gathers, figures, or W&B uploads are produced.

## `diagnostics/ztautau_back_to_back/*`

Ztautau-only scalar topology diagnostics from the reconstructed tau directions when `feature_names: [theta, phi]`.
Logged per train step for both `all/*` rollout candidates and reward-selected `best/*` candidates:
`cos_opening`, `delta_phi_to_pi`, `back_to_back_loss`.

## `val/*` and `val_neutrino/*`

End-of-epoch DDIM validation (`validation_K` candidates). `val/reward/mean` drives top-K checkpoint selection. `val_neutrino/*` overlays truth / current policy / frozen reference for neutrino kinematics (pT, η, φ, and p_x/p_y/p_z in GeV).

Also logged as scalars:

- `val_neutrino/jsd/current/{pt,eta,phi,px,py,pz}`: JSD between truth and current-policy validation histograms.
- `val_neutrino/jsd/ref/{pt,eta,phi,px,py,pz}`: JSD between truth and frozen-reference validation histograms.
- `val_neutrino/all_metrics/{feature}/{count,mae,rmse,bias,pearson_r,slope,intercept}`: pooled truth-vs-pred response summaries.
- `val_neutrino/by_process/{process}/metrics/{feature}/*`: the same response summaries split by EVENT process; the corresponding `*_truth_vs_pred` keys are 2D response panels.

`val/response/*` compares the fixed pre-DGPO validation baseline with the current
policy. It includes pooled reward and event-mean pT-delta panels/metrics plus
`val/response/by_process/{process}/*` reward response panels and metrics.

## `val_mass/*`

Same validation pass: W and top mass reconstructed from ground-truth `assignments-indices` (b + lepton from point cloud + neutrino). **Truth** histogram uses target neutrinos; **Pred** / **Ref** use DDIM neutrinos (best-of-`validation_K` vs frozen reference, same candidate rule as `val_neutrino/pt`).

Also logged as scalars:
- `val_mass/jsd/current/{w_mass,top_mass}`: JSD between truth and current-policy mass histograms.
- `val_mass/jsd/ref/{w_mass,top_mass}`: JSD between truth and frozen-reference mass histograms.

## `val_ztautau/*` — targeted Ztautau physics

Enabled by `ztautau_domain.enabled` with `feature_names: [theta, phi]`. These
truth/current/reference 1D density overlays always use candidate zero for the
current policy and candidate zero for the frozen reference. They never use the
reward-best member of the validation group.

- `val_ztautau/target/*`: the four diffusion targets
  (`tau_{a,b}_delta_{theta,phi}`).
- `val_ztautau/reco/*`: reconstructed tau-a/tau-b theta and phi after the
  shared direction reconstruction.
- `val_ztautau/topology/*`: `cos_opening`, `delta_phi_to_pi`,
  `back_to_back_loss`, and the shared physics-calibration direction changes
  `calibration_deltaR_{a,b,sum}`.
- `val_ztautau/jsd/{current,ref}/*`: histogram Jensen-Shannon distance to
  truth; lower is better.
- `val_ztautau/residual/{current,ref}/*/{mean,abs_mean}`: paired candidate-zero
  residual summaries. Phi residuals are periodic.

## `val_tarp/*` — posterior calibration

TARP uses all `dgpo.validation_K` candidates for each event; it is therefore
separate from the candidate-zero 1D panels. The conditional decision panel
bins events by visible tau-pair acoplanarity, which is observed input and does
not use target truth or generated candidates.

- `val_tarp/tarp_binned_min_holm_pvalue`: family-wise Holm-adjusted minimum
  p-value across acoplanarity bins and configured joint arms. Values below
  `dgpo.tarp.alpha` reject calibration.
- `val_tarp/bin*/{full,rank_copula}_{pvalue,holm_pvalue,max_gap}`: per-bin
  diagnostics and coverage gaps.
- `val_tarp/geometry/{events,candidates,holm_power_floor}`: effective test
  geometry and the attainable family-wise p-value floor.
- `val_tarp/coverage`: binned coverage curves. `val_tarp/pooled_*` is an
  orientation-only pooled panel; use the binned Holm value for decisions.

The OmniFold population fit and adaptive refits are independent K=1 draws.
`dgpo.validation_K: 16` does not change the OmniFold fitting population or
`dgpo.K` used by training.

## Live OmniFold fitting

The active Ztautau config publishes periodic rank-0 classifier progress.
`omnifold_live/meta/{fit_step,iteration,dgpo_epoch,global_step}` identifies the
fit position. Every classifier epoch traverses its complete fit split exactly
once. The phase-specific namespaces are:

- `omnifold_live/residual_reward/*`
- `omnifold_live/acceptance_audit/*`
- `omnifold_live/staleness_audit/*`

Each phase reports training loss and balanced accuracy. Once a validation has
run, it also reports validation loss, validation balanced accuracy, validation
AUC/AUC gap when available, best validation loss, threshold-crossing state, and
saturation state. The W&B chart step is
`omnifold_live/log_index`; the physical classifier step remains
`omnifold_live/meta/fit_step`.

## Config knobs

See `rl.enabled` and numeric fields in `RL/DGPO_neutrino/config.yaml`. Notable:

- `dgpo.K`, `dgpo.num_train_timesteps`, `dgpo.beta`, `dgpo.adv_clip_max`
- `dgpo.reference_trust.enabled`, `dgpo.reference_trust.coefficient`
- `dgpo.validation_K`, `dgpo.ztautau_metrics.*`, `dgpo.tarp.*`
- `dgpo.adaptive_omnifold.recalibration.fit.progress_every_n_steps`
- `dgpo.projection_constraint.epsilon`, `multi_sample.samples`, `trust_region_ratio`
- `dgpo.projection_constraint.latent_swd.checkpoint_file`, `margin`, `num_projections`, `min_samples`, `apply_to`
