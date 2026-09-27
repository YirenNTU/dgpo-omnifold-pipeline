# Angular conditioning + theta H4: prepared, not run

Configuration: `config/dgpo_h4_angular_conditioning_1110.yaml`.
W&B ID: `h4angc1`. This is a new experiment, not a resume of `h4epkl1on`.

## Changes

1. Load **raw state_dict** from the explicit old-policy step-1110 checkpoint,
   weights only. Do not restore old optimizer, classifier, reference lifecycle,
   EMA rollout, or run counters. The new angular projection starts at zero;
   the initial policy function is unchanged on finite valid inputs.
2. Policy PET receives 16 observed-only features: sin/cos(k theta),
   sin/cos(k phi), k=1..4, from raw visible `Part_eta`/`Part_phi`.
   PET's existing trainable optimizer group includes the new projection.
3. Fresh H4 OmniFold and cold audits retain the latest standardized late-fusion
   classifier setup. Keep the nine old channels (eight delta-phi harmonics plus
   opening cosine); append **16 theta channels**, sin/cos of both k(theta_a-theta_b)
   and k(theta_a+theta_b), k=1..4. The old optional linear theta channels remain off.
   Total pair input dimension: 25. This is pair geometry, not Fourier of raw
   detector theta; each tau direction is visible-leg angle plus that sample's
   candidate delta. Generated samples never borrow truth deltas.
4. `topology_theta_fourier` defaults false and is serialized in classifier
   payloads, propagated through bootstrap, fold fits, audit overrides, and loaders.
   Old classifier checkpoints keep their original feature dimensions.

Preserved: 10% filtered data, cleaned validation, 16 GPUs, two folds/one repeat/
one residual iteration, frozen installed reward, constant grouped classifier LRs,
minimum 300 / maximum 3000 classifier updates, BCE delta 0.001 and ten-epoch
patience. Velocity-MSE reference penalty is enabled at coefficient 1; endpoint KL
is disabled. This is a KL surrogate, not an exact endpoint-distribution KL.
The adaptive hard boundary remains disabled.
Cold audits every five DGPO epochs (50 updates); policy validation every ten
DGPO epochs (100 updates).
No automatic total policy-step limit; stop manually or at job walltime.

Classifier PET trainability remains the latest last-block/adapters setup; it does
not additionally unfreeze the zero-initialized observed-angle projection. The
policy projection is trainable. The classifier intervention is its candidate
theta Fourier branch, not an additional classifier-body unfreeze experiment.

## Monitor

- `train/angular_conditioning/trainable`, `gradient_present`,
  `grad_norm_post_clip`, `weight_norm`: verify the attached projection learns.
- Existing fixed-classifier held-out metrics, reward mean/std, ESS/tails, and
  gradient/conflict diagnostics: evaluate reward transfer, not std alone.
- Separate cold-audit BCE/AUC and fit duration from fixed-classifier metrics.
  Early stop or a 3000-update cap does not itself establish saturation.

Compare to this run's own step-zero baseline. Both the policy representation and
judge representation change, so historical H4 AUC is not a matched causal control.
Any improvement supports the **combined** intervention, not conditioning alone.

## Launch (user submits; existing 16-GPU Ray allocation)

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_angular_conditioning_1110.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_angular_conditioning_1110/ray_results
```

Run from the updated existing NERSC `ml_pipeline` checkout. Do not use the old
`train_dgpo_endpoint_kl.py` launcher: it selects its own old experiment config.
No remote job or upload was performed while preparing this experiment.

### Continue after the classifier has already fitted

Use `config/dgpo_h4_angular_conditioning_resume_epoch0.yaml` instead of the
fresh-start overlay in the command above. It explicitly loads this run's
`dgpo-epoch=-1-next_ep=0-step=0.ckpt`, uses full-state resume, preserves the
installed classifier and reference, and disables bootstrap/startup refitting.
The saved `next_ep=0` starts policy training at epoch zero. It retains velocity
MSE coefficient 1 and the same W&B ID. This does not eliminate later scheduled
cold audits. It requires the new theta-Fourier classifier's actual completed
bootstrap checkpoint; its presence on NERSC has not been verified locally.
Do not use it to rewind a run already logged past step zero in the same W&B ID;
such a branch needs a separate output/run ID to preserve both histories.

## Local verification

Theta Fourier, OmniFold and endpoint-KL regression selection: 110 passed,
one skipped, six subtests passed. Visible angular-conditioning tests: 11 passed.
Syntax compilation and `git diff --check` pass. Actual 16-GPU training and the
remote checkpoint's availability have not been tested in this implementation turn.

The broader suites are not all green: one best-decay trust-resume test and two
rest-frame tests (dropout expectation and materialization call count) fail.
The same failures were reproduced with this turn's theta wiring removed in
memory. Their unrelated implementations and assertions were left unchanged.
