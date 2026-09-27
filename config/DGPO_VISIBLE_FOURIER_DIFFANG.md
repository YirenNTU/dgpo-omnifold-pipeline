# DGPO from trained angular diffusion

To resume after interruption, use overlay
`config/dgpo_h4_visible_fourier_diffang_resume.yaml` instead of the fresh-start
overlay. It requires this run's `h4_visible_fourier_diffang/checkpoints/last.ckpt`
and restores complete DGPO/OmniFold state. Missing/incomplete recovery files
stop startup rather than silently loading supervised diffusion weights.
The actual saved epoch/step must be checked on NERSC; no remote file inspection
has been performed locally.

Source: raw `diffusion_angular_10pct_lr5e4/checkpoints/last.ckpt`, not the
subsequent historical-optimizer run. The remote alias still requires checking
before launch. Weights only: fresh DGPO optimizer, clocks and classifier fits.

Both OmniFold and cold audit use two complementary existing paths:
- Every valid observed particle: theta from raw Part_eta and raw Part_phi,
  sin/cos k=1..4, 16 channels, learned diffusion projection added to that
  particle's PET token. Padding masked; no truth angles enter this branch.
- Existing H4 late fusion: candidate reconstructed tau-pair delta-phi Fourier,
  opening cosine, plus theta difference/sum Fourier (25 features).

No new duplicate Fourier implementation is needed: the ratio forward already
passes raw visible x into PET. The model builder preserves the configured
angular architecture and checkpoint weights. Classifier fitting retains the
last-PET-block/adapters/head training setup; the angular projection stays
frozen as part of the pretrained input representation. Policy training retains
the inherited DGPO optimizer configuration, not the diffusion 8e-4 setup.

Bootstrap a new 2-iteration, 2-fold/1-repeat reward for this generator. Refit
every 20 policy epochs (after epochs 19, 39, 59, ... from epoch-zero start).
Disable log-only mode and the one-reward-round cap. Each refit starts fresh;
policy AdamW state is retained on accepted installs. Two useful increments
are installed without requiring residual closure; fit/signal checks remain active. A rejected scheduled refresh
stops rather than silently continuing with stale reward. No classifier
from step1110 or the old epoch-zero checkpoint is reused. OmniFold and audit
retain standardized H4 and constant group LRs, with no minimum or maximum update count,
BCE early stopping delta .001 / patience 25 validation epochs from the start.
The extra gradient-conflict monitor is disabled because it is incompatible with
fixed-schedule audit skipping; ordinary gradient-norm logging and velocity-MSE
training are unchanged. Cold audit every 5 policy epochs; physics
validation every 10. Velocity-MSE reference coefficient 1, endpoint KL off,
raw/no EMA, 16 workers, unbounded DGPO trajectory. This combined change is not
a one-variable Fourier ablation. No job submitted by the assistant.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_visible_fourier_diffang.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_visible_fourier_diffang/ray_results
```

W&B ID: `h4visang1`. Do not supply a resume override.

Classifier research probes are disabled in both recalibration.fit and audit_fit:
stability before/after probes, representation/readout/path diagnostics and
parameter-update snapshots. This does not disable loss/AUC validation,
early stopping, gradient clipping, Fourier standardization, cold audits or
DGPO policy-gradient diagnostics. Existing running processes keep their loaded
runtime settings; syncing this overlay does not hot-reconfigure a running fit.

W&B `classifier_loss_curves: false` disables the repeat-1/fold-1-only
cumulative `plot.line_series` table rebuild on every validation callback.
`classifier_loss_curves_raw: true` instead logs scalar train/validation BCE,
validation AUC and LR with a separate local-step axis for each fold/iteration/
refit. Normal gradient scalars remain in omnifold_live. Validation and early
stopping are unchanged. No additional forward/backward passes or W&B tables
are created by the scalar loss tracker.
# Fixed two-iteration installation

OmniFold and audit have no minimum update count and no fixed step cap. Validate
every classifier epoch; early stopping counts from the start and stops after
25 epochs without a BCE improvement of `0.001`. Restore the absolute lowest
validation BCE checkpoint (its selection does not require the `0.001` margin).
These are classifier epochs, not DGPO policy epochs.

This overlay now uses `fixed_iteration_budget: true`: after two useful cross-fit
increments, install the cumulative reward and start DGPO without requiring a
third residual classifier or residual closure. Fit/signal and finite-weight
checks remain active. Other overlays retain the default fail-closed behavior.
W&B records `omnifold/closure_evaluated=0` and
`omnifold/last_residual_auc_before_update`; the last residual AUC is not closure
of the final stack. The same rule applies to the 20-epoch refits.

Both OmniFold and cold audit explicitly use `restore_best: true` and
`checkpoint_selection_metric: loss`. Select the lowest observed validation BCE,
including checkpoints before the minimum training step count. Early-stop
`min_delta` controls patience only; ESS-aware alternative selection is disabled.
