# DGPO neutrino — Weights & Biases metrics

Logged from `RL/DGPO_neutrino/dgpo_trainer.py`. The latest OmniFold scaling
overlays use the `critical` profile; legacy CPO/SWD metric definitions below
remain available for other configurations.

## Meta

| Key | Description |
|-----|-------------|
| `epoch` | Zero-based logical epoch. The true pre-DGPO baseline is -1. Validation plots use this explicit x-axis. |
| `global_step` | Completed DGPO training steps, matching checkpoint `global_step`. Baseline is 0; the first training result is 1. A trust-rejected proposal still consumes a training step. |
| W&B `_step` | Transport/event-row index only. Classifier fits advance this counter without advancing the DGPO clock. Never use it to compare training progress. |
| `omnifold_live/log_index` | Classifier progress-record index within the W&B run. Local fit step/fold/iteration remain separate hidden metadata. |

## Critical profile and resume clocks

`logger.wandb.profile: critical` keeps an explicit allowlist of approximately
40 decision/physics charts. Core groups are raw AUC/accuracy and plateau/rollback,
VP trust distance/radius/update scale, classifier-trust confidence bounds,
DGPO loss/gradient/memory, OmniFold initial/closure AUC, six physics JSDs, two
topology overlays, and binned TARP coverage. Dormant ablation switches, repeated
reference diagnostics, generic response matrices, and per-process plots are
not emitted. Fit loss/AUC/provenance remain in history but are hidden from
automatic charts, since independently initialized folds are not one learning
curve. Controller calculations, saved state and console logging are unchanged.

Every logging event has its own `commit=True` row and explicit `global_step`.
Validation also carries `epoch`; staleness carries its evaluation epoch.
Axes use `step_sync=False` so W&B cannot borrow a previous event's epoch/step.
Training, trust and staleness charts use `global_step`; physics charts use
`epoch`. This follows W&B's [custom-axis protocol](https://docs.wandb.ai/models/track/log/customize-logging-axes).

Cold pretrained loading starts at `(epoch=0, global_step=0)`, regardless of the
supervised checkpoint's counters. Full DGPO resume uses `dgpo_next_epoch`,
`global_step`, and `dgpo_epoch_step`, and consumes only the remaining logical
epoch budget. An epoch-0/step-5 resume does **not** regenerate a baseline at
epoch -1. Rollback changes policy parameters, not the training clock; the
selected source step is logged separately. New W&B runs preserve checkpoint
training counters while starting a fresh transport-row counter. Startup
provenance is recorded under Summary `resume/*`.

Legacy runs use the old zero-based **pre-increment** training log step, so
their final training row is one below the corresponding checkpoint count.
They may also contain merged baseline/train rows. Do not overlay their raw
`_step` axis against new runs or interpret this display correction as a model
change. Existing remote history is deliberately not rewritten.

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
| `reference_trust/loss` | Configured soft anchor. `velocity_mse` uses the legacy shared-DGPO-time proxy; `vp_path_kl` uses the separate full-time cosine-VP reverse-path estimator below. |
| `reference_trust/vp_path_kl` | `0.5 * Z * mean(sum_active_dims((v_policy-v_round_ref)^2))`, with time importance-sampled from `rho(t) ∝ beta(t) alpha(t)^2 / sigma(t)^2` and `Z ≈ 20`. |
| `reference_trust/objective_vp_path_kl` | 1 when the VP path-KL estimator, rather than legacy dimension-mean velocity MSE, contributes to backward. |
| `reference_trust/sequential_backward` | 1 when DGPO and VP path-KL are backpropagated as two consecutive graphs before the same AdamW update, reducing activation-memory peak without changing their summed gradient. |
| `reference_trust/velocity_mse_ratio` | Trust velocity MSE divided by the round reference's own velocity loss. |
| `reference_trust/coefficient` | Configured trust multiplier; `1.0` in the active Ztautau overlay. |
| `reference_trust/control_ratio_global` | Accepted post-step fixed-probe distance enforced by strict backtracking: trust-MSE/reference-loss or VP path-KL according to the configured distance. |
| `reference_trust/delta` | Active functional radius. It stays constant for `radius_mode: fixed` and is raw-AUC-scaled only for `radius_mode: auc_scaled`. |
| `reference_trust/warning_ratio` | Pre-step distance where the initial candidate LR begins to be damped. |
| `reference_trust/nominal_interior_limit` | v2 backtracking target, normally `interior_fraction * delta`. |
| `reference_trust/acceptance_limit` | Effective step limit; equals the interior target except when a legacy resume must first hold its existing legal distance. |
| `reference_trust/fixed_probe_per_reward_round` | The exact event/timestep/noise/reference rows are reused and checkpointed for the complete reward round. |
| `reference_trust/probe_reused` | The current step reused the persisted reward-round probe rather than capturing it. |
| `reference_trust/pre_step_distance` | Distance measured on frozen shared-noise rows before AdamW. |
| `reference_trust/candidate_distance` | Distance after the initial candidate AdamW update, before backtracking. |
| `reference_trust/candidate_distance_over_delta` | Direct boundary test statistic. Values above `1` exceed the current trust region. |
| `reference_trust/candidate_excess` | `candidate_distance - delta`; positive values exceed the current trust region. |
| `reference_trust/post_step_distance` | Distance after the accepted update; strict mode requires it to remain at or below `acceptance_limit`. |
| `reference_trust/post_step_distance_over_delta` | Accepted distance divided by `delta`; strict mode requires this to remain at or below `1`. |
| `reference_trust/update_scale` | Accepted AdamW displacement scale after warning damping and backtracking. |
| `reference_trust/accepted_step_scale` | Alias exposing the final accepted absolute step scale. |
| `reference_trust/backtrack_steps` | Number of multiplicative backtracking trials used for the accepted step. |
| `reference_trust/boundary_hit` | Initial candidate crossed the boundary, or the pre-step probe was already outside. It does not force reward refit. |
| `reference_trust/preexisting_violation` | Pre-step probe was outside `delta`, so optimizer, scheduler, and EMA did not advance. |
| `reference_trust/interior_saturated` | Probe is inside hard `delta` but the interior update budget is exhausted, so policy waits for an adaptive audit. |
| `reference_trust/step_accepted` | A feasible AdamW update was committed. |
| `reference_trust/optimizer_state_advanced` | AdamW state incorporated the gradient. This may be 1 while `step_accepted=0` when all positive backtracking scales fail; the optional convergence controller then clears first moments, while EMA and scheduler do not advance. |
| `reference_trust/zero_step_adam_first_moments_reset` | Number of AdamW `exp_avg` buffers cleared after an `alpha=0` trust rejection. Variance estimates and step counters remain intact. |
| `reference_trust/policy_lr/scale` | DGPO-only LR multiplier `sqrt(delta_r / delta_0)`, clipped by the configured floor. OmniFold classifier LRs are unchanged. |
| `reference_trust/policy_lr/effective_step_scale` | Product of the round-level policy-LR multiplier and the within-round warning/backtracking scale. |
| `reference_trust/cross_round/cap_active` | A new reward round proposed a larger radius and was capped at the previous round's radius. |
| `reference_trust/probe_velocity_mse` | Fixed-probe policy-versus-round-reference velocity MSE. |
| `reference_trust/probe_reference_loss` | Frozen round-reference denoising loss normalizing the probe distance. |
| `reference_trust/raw_auc` | First-iteration raw cross-fit AUC used to calibrate the installed round radius. |
| `reference_trust/statistically_closed` | Raw AUC gap is inside the configured closure/uncertainty band. |
| `reference_trust/adam_first_moments_reset` | AdamW `exp_avg` buffers cleared after an accepted reward/reference swap. |
| `reference_trust/empirical/distance_*` | Robust median/MAD interval from the latest accepted real DGPO updates before an audit. |
| `reference_trust/empirical/audit_auc_gap_{lcb,ucb}` | Confidence interval used to classify a fresh audit as safe, ambiguous, or unsafe relative to the installed baseline threshold. |
| `reference_trust/empirical/delta_cap` | Conservative upper cap learned from a statistically unsafe distance and applied to later reward rounds. |
| `reference_trust/empirical/acceptance_rate` | Fraction of recent optimizer attempts that committed a feasible step. |
| `reference_trust/empirical/mean_update_scale` | Mean accepted AdamW displacement scale over the attempt window, with rejected attempts counted as zero. |
| `reference_trust/empirical/throughput_starved` | Acceptance rate or mean update scale is below the configured target. |
| `reference_trust/empirical/radius_action` | `-1` contracts the future-round cap, `0` holds, and `+1` expands the current safe radius. |
| `reference_trust/empirical/delta_{before,after}` | Active radius immediately before and after the audit controller decision. |
| `reference_trust/exhaustion/mean_update_scale` | Mean accepted AdamW scale over the dedicated recent exhaustion window (10 steps in the round-guard ablation). |
| `reference_trust/round_acceptance/improvement` | Previous installed raw `|AUC-0.5|` minus the candidate value; positive means classifier distinguishability decreased. |
| `reference_trust/round_acceptance/required_improvement` | Statistical dead band, `z * sqrt(SE_previous^2 + SE_candidate^2)`. |
| `reference_trust/round_acceptance/action` | `+1` significant improvement, `0` plateau, `-1` regression, and `+2` bootstrap/protocol-change bypass. |
| `reference_trust/round_acceptance/rollback_required` | Restores the live policy to the incumbent `round_ref`, clears AdamW state, and synchronizes EMA before another step. |
| `reference_trust/round_acceptance/stop_requested` | Two consecutive plateaus (configurable) have been confirmed; save the rolled-back checkpoint and stop cleanly. |
| `reference_trust/signed_probe/decision_code` | Diagnostic result for `theta_ref +/- alpha*(theta-theta_ref)`: `+1` only the trained sign improves raw AUC, `-1` only the reverse sign improves, `+2` both improve, and `0` neither improves beyond the uncertainty band. |
| `reference_trust/signed_probe/best_overall_scale` | Signed `alpha` with the lowest raw `|AUC-0.5|` on the fixed 100k panel. The probe never installs this point. |
| `reference_trust/signed_probe/reward_raw_misaligned` | The reward-maximizing signed point increased installed reward relative to the anchor but significantly worsened raw AUC. |
| `reference_trust/signed_probe/smaller_positive_step_improves` | A positive scale below `1` significantly improves raw AUC while the trained `+1` endpoint does not, identifying step-size overshoot rather than a wrong sign. |
| `reference_trust/signed_probe/all_raw_audits_saturated` | Anchor and every signed candidate judge reached validation-loss saturation; require this before interpreting the direction diagnosis. |
| `reference_trust/signed_probe/candidate/{plus,minus}_*/raw_auc_gap` | Raw unweighted truth-vs-policy `|AUC-0.5|` for one signed scale; lower is better. |
| `reference_trust/signed_probe/recovery_triggered` | A fully saturated reverse-only result rejected the local direction, restored `round_ref`, cleared optimizer state, and requested a fresh reward fit. |
| `reference_trust/signed_probe/recovery_attempt` | Consecutive independently trained reverse-only directions since the last genuinely improved policy round. |
| `reference_trust/signed_probe/recovery_reward_installed` | The fresh reward at the restored incumbent passed residual closure, acceptance, and topology gates. |
| `reference_trust/signed_probe/recovery_stop_requested` | Reverse-only recoveries reached `failed_direction_patience`; save and stop at the incumbent instead of looping. |
| `reference_trust/extragradient/triggered` | Trust exhaustion launched the opt-in block extragradient treatment. The selected trajectory is a virtual predictor, not a committed point. |
| `reference_trust/extragradient/lookahead_scale` | Fraction of the selected incumbent-to-trajectory displacement used to train the transient response classifier. |
| `reference_trust/extragradient/lookahead_reward_installed` | The response classifier passed residual, acceptance, and topology gates at the virtual look-ahead. This alone does not update the final reference point. |
| `reference_trust/extragradient/rebased_optimizer_params` | Number of trainable tensors restored to the incumbent after corrector backward at the look-ahead and before `optimizer.step()`. |
| `reference_trust/extragradient/corrector_distance` | Functional fixed-probe distance from the original incumbent after the rebased corrector step. |
| `reference_trust/extragradient/corrector_scale` | Backtracked fraction of the corrector AdamW displacement that satisfies the original incumbent trust region. |
| `reference_trust/extragradient/final_reward_installed` | The corrected policy passed a second full classifier stack and the paired round-AUC gate against the original incumbent, so its policy/reward/reference pair was committed. |
| `reference_trust/extragradient/final_optimizer_states_cleared` | Number of transient corrector AdamW states discarded after the corrected policy and final reward/reference pair were committed. |
| `reference_trust/extragradient/rejected` | A failed look-ahead/final classifier gate, non-finite/zero corrector, or failed trust backtrack restored the complete incumbent pair and consumed one failed-direction attempt. |

## `train/grad/`

| Key | Description |
|-----|-------------|
| `train/grad/global_norm_pre_clip` | L2 norm before `clip_grad_norm_` (max over inner timesteps when accumulating). |
| `train/gradient_sync/manual_parameter_tensors` | Number of globally active parameter tensors manually averaged across ranks after sequential DGPO/VP accumulation. Zero selects the ordinary DDP reducer path. |
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

When `dgpo.validation_compute_winrate: false`, no `val/winrate` scalar is
computed or logged. The configured adaptive cadence reports a fresh raw
truth-vs-unweighted-policy classifier under `staleness/raw_*`. With
`adaptive_omnifold.monitor_mode: raw_only`, this is the only routine classifier:
reward scoring, the weighted staleness controller, and automatic refits are
skipped while the installed reward/reference pair remains fixed.

With `adaptive_omnifold.monitor_mode: raw_plateau_refit`, the saturated raw
`|AUC-0.5|` is compared with the best value in the current reward round. Two
consecutive non-improving logical epochs (as configured by
`trigger.required_consecutive_epochs`) launch a forward OmniFold refit. The
current policy becomes the next denominator/reference; no earlier policy is
restored. `classifier_trust/*` is a separate fresh current-vs-round-reference
classifier. Its balanced-accuracy ceiling can request the same forward
recenter, while the per-step VP path-KL boundary remains the hard update gate.

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
`omnifold_live/meta/{fit_step,iteration,repeat,dgpo_epoch,global_step}` identifies
the fit position. Every classifier epoch traverses its complete fit split
exactly once. The phase-specific namespaces are:

- `omnifold_live/residual_reward/*`
- `omnifold_live/acceptance_audit/*`
- `omnifold_live/topology_acceptance_audit/*`
- `omnifold_live/staleness_audit/*`
- `omnifold_live/raw_staleness_audit/*`

Topology repeats share the registered canonical namespace and are distinguished
by `omnifold_live/meta/repeat`. They use the exact same
EveNet+adapter+Fourier model builder and trainable scope as OmniFold; only
initialization and event splits are independent.

Each phase reports training loss and balanced accuracy. Once a validation has
run, it also reports validation loss, validation balanced accuracy, validation
AUC/AUC gap when available, best validation loss, threshold-crossing state, and
saturation state. The W&B chart step is
`omnifold_live/log_index`; the physical classifier step remains
`omnifold_live/meta/fit_step`.

## Config knobs

`options.Training.weight_decay` and the per-component `weight_decay` values
control DGPO's decoupled AdamW decay. The active v18 overlay explicitly sets
`0.001` for all trainable policy groups. On resume, optimizer moments and the
scheduler are restored while weight decay comes from the current configuration.
The complete AdamW parameter proposal, including decay, is checked by the
existing trust-region backtracking.

The active v18 overlay uses `reference_trust.adaptive_boundary.radius_mode:
round_decay`, `delta_max: 0.10`, `delta_floor: 0.02`, and
`round_decay_factor: 0.9`. The live radius is `max(0.02, 0.10 * 0.9**r)`:
`r=0` at the initial installed reference, then it advances once per successful
reward/reference installation. Failed fits, monitor epochs, and policy-only
best-checkpoint rollback do not advance or rewind it. Each round keeps a fixed
radius; shrinking occurs only after recentering, when the new probe distance is
zero. The soft VP-weighted trust coefficient remains 1.0, independent of decay.

The v18 cheap-trust overlay starts in a new output directory. With no local
`last.ckpt`, `dgpo.auto_resume_best_source_checkpoint_dir` selects the lowest
finite saturated `raw_auc_gap` from v17's saved `probe_history` (ties prefer
the earlier epoch). This uses the recorded history, retained for up to 256
monitor entries, rather than claiming a best score outside that history.
It requires the matching complete snapshot. With
`best_source_start_new_experiment: true`, only the first parent-source load
starts a new experiment: policy and paired reference/OmniFold are kept, while
epoch/step, AdamW/scheduler, EMA, plateau history and trust-decay age reset.
The initial radius is 0.10 regardless of the parent's decay age. The reference
is **not** silently recentered to the best policy: retaining the saved reference
preserves its reward pairing, so initial distance need not be zero.
Missing snapshots or inconsistent metadata fail closed; there is no fallback
to a runner-up or pretrain. Rank zero selects once and broadcasts the path.
Subsequent restarts prefer v18's own `last.ckpt`. W&B starts a new run, and the
parent directory is not written to. The existing v17 one-shot-refit completion
marker is retained to avoid repeating a completed initial fit. A new 400k-event
raw baseline is fitted before policy updates and saved at epoch=-1, next_ep=0;
the initial point can therefore be recovered by best-point rollback.

`trigger.warm_start_classifier: true` fine-tunes the previous raw staleness
classifier and checkpoints its CPU weights as `raw_monitor_state`. Old
checkpoints without this cache require one cold monitor fit. Each fit uses
new current-policy samples, a fresh optimizer and reset early stopping;
only classifier weights transfer, never OmniFold reward weights. Approximately
320k/80k of the 400k event pool are training/validation, determined by a stable
visible-condition hash: reordering or growing the pool cannot turn an earlier
training identity into validation. The current/reference trust classifier
remains fresh and capped at 50k. `staleness/raw_classifier_warm_started` and the
fit log's `warm_started` indicate reuse. Policy-only rollback retains the latest
monitor; a full restart restores the monitor saved with its checkpoint.
Warm-start AUC reflects both policy change and accumulated classifier training,
so larger validation samples alone do not establish unbiased sensitivity or
prove physics improvement. Reusing validation for early stopping and repeated
monitoring is not an untouched confirmation test.

An old checkpoint without this schedule initializes `r=0` on resume; the next
successful refit uses `r=1`. Cold bootstrap itself uses `r=0`. New checkpoints
store the schedule index, associated reward round, and its initial/floor/factor
protocol. Restart preserves them and the fixed probe. Changing the schedule
protocol on resume fails closed rather than silently shrinking an active
round. W&B reports `reference_trust/delta` and `reference_trust/round_decay/step`.
This is a hard-radius annealing heuristic, not an exact output-KL or convergence
guarantee; independent current/reference drift monitoring remains enabled every epoch.

`dgpo.adaptive_omnifold.recalibration.warm_start_iterations: [1, 2]` initializes
each of the first two residual iterations from its two previous-round fold
classifiers. Later iterations and trust monitors start fresh; raw monitors use
their own separate warm-start cache when enabled. This is
weight initialization only: all fits optimize against current generated data,
reset their optimizer, and recompute cumulative weights from zero.

Warm starts require the same condition-hash fold protocol, which keeps event
identities in the same fold even when Ray changes row order. The fitted weights
and protocol are checkpointed in a CPU cache; a closure-only classifier can be
cached without contributing a reward increment. A legacy checkpoint without
this protocol resumes its installed reward normally but fits fresh at its first
new refit. Subsequent rounds can reuse the cache.

`omnifold/fit/iter01/warm_started_folds` and `iter02/warm_started_folds` should be
`2` when both folds were reused; `0` indicates fresh initialization. Per-fold
live logs expose `omnifold_live/residual_reward/warm_started`.

See `rl.enabled` and numeric fields in `RL/DGPO_neutrino/config.yaml`. Notable:

- `dgpo.K`, `dgpo.num_train_timesteps`, `dgpo.beta`, `dgpo.adv_clip_max`
- `dgpo.reference_trust.enabled`, `dgpo.reference_trust.coefficient`
- `dgpo.reference_trust.objective`, `dgpo.reference_trust.vp_path_kl.*`
- `dgpo.validation_K`, `dgpo.ztautau_metrics.*`, `dgpo.tarp.*`
- `dgpo.adaptive_omnifold.recalibration.fit.progress_every_n_steps`
- `dgpo.adaptive_omnifold.monitor_mode`, `dgpo.adaptive_omnifold.fixed_audit_panel`
- `dgpo.projection_constraint.epsilon`, `multi_sample.samples`, `trust_region_ratio`
- `dgpo.projection_constraint.latent_swd.checkpoint_file`, `margin`, `num_projections`, `min_samples`, `apply_to`
