# Step260: repeated OmniFold inheritance versus restart

## Current selected run: step260, iteration-1 inheritance, patience12, no TARP

Use `dgpo_omnifold_ztautau_10pct_step260_i1inherit_p12.yaml` for the user's confirmed
step260 starting point. The corrected `pinned_classifier_restart` mode loads the live
policy AND installed OmniFold/reference pair, and preserves the raw monitor weights.
It resets the experiment/optimizer clocks to zero and clears historical best scores.
The inherited raw monitor is recertified with the full cold budget under matching
identity splits before a new baseline is established. The initial OmniFold stack is
NOT retrained. Saved iteration-1 fold caches are mandatory; absence fails explicitly.
Only iteration1 inherits across later rounds; later iterations are cold. Global-best
rollback is enabled with patience12 at10-step cadence. TARP is off, regular physics
validation remains every3 epochs. It does not load the bootstrap of run30d9dc0f.

```bash
shifter python3 scripts/train_dgpo_restart_ablation.py \
  --config config/dgpo_omnifold_ztautau_10pct_step260_i1inherit_p12.yaml
```

Add `--check-only` to run preflight without training. Output is separately named
`dgpo_10pct_step260_i1inherit_latercold_p12_notarp_loadstack_v2_seed42` under the usual scratch root.
The older paired arms and full-resume branches below are preserved as alternatives.

## Earlier paired controls

Both arms load only the live policy from the same pinned filename:
`dgpo-epoch=25-next_ep=26-step=260.ckpt` in the original v26 resume3 checkpoint directory.
Both reset the DGPO epoch/step to zero, initialize a new optimizer/reference,
and train the initial reward stack and raw monitor from scratch. No old global-best
or reward/monitor state is imported. This is a new experiment, not a full-state resume.

The sole training-setting difference is `warm_start_iterations`:

- `inherit`: only iteration 1 reuses compatible same-fold weights from preceding
  rounds. Iterations 2 onward start cold every round. `warm_start_from_iteration_one`
  stays false: later iterations do NOT clone this round's iteration-1 weights.
- `restart`: no cross-round reward weights are reused. Trainable classifier banks
  are reset at every fit; the fixed pretrained backbone is still shared.

Both use the old 128-dimensional, one-decoder-layer classifier without Fourier/rest-frame
features, dropout 0.15, tempering 1, identical fit/validation budgets and fixed identity
folds. Classifier optimizers and residual cumulative weights reset each refit in both arms.
Cold fits now require **at least 1000 optimizer updates per fold** (after cross-fit
splitting, not 1000 shared between folds). Matching inherited folds instead require
at least 10 actual fold-data epochs. Both arms use the same conditional budget:
initial cold fits get 1000; all restart refits get 1000; compatible inherit refits
use the 10-epoch minimum. Saturation/early stopping remains required and can run
longer; 1000 is not a fixed stop. Validation-best checkpoint restoration is unchanged,
so the selected weights may come from before update 1000. This change takes effect
on a new launch, not in already running processes.
Raw staleness monitors retain the same warm-start protocol in BOTH arms: this test changes
reward initialization, not the evaluation protocol. Staleness cadence is 10 DGPO steps,
patience is 8 checks, and global-best rollback remains enabled. DGPO uses cosine LR,
velocity-MSE reference regularization, adaptive trust boundary and weight decay as in the
old-classifier configuration. Policy validation remains every 3 epochs.

The two fresh-start arms now also use certified staleness training: cold minimum
250 actual data epochs, warm minimum 5 epochs, with saturation required. For the
current approximately 200k fit events and global batch 32768 with drop-last, this
is 1500 cold updates and 30 warm updates minimum. Training can continue longer.
Only a matching certified monitor cache qualifies for the warm budget. Readiness
and saturation are needed before monitor results can affect best/refit decisions.
This uses the existing readiness mechanism; no classifier capacity or input changes.

The separate `dgpo_omnifold_ztautau_10pct_restart_best_resume.yaml` deliberately
keeps the historical monitor protocol for full-state resume of run30d9dc0f.
Changing its monitor budget requires re-evaluation/reset of historical global-best
comparisons, not simply overwriting an audit hash. These fresh-start commands do
not upgrade an already running or resumed experiment. The iteration-one configs
already have the cold250/warm5 monitor protocol.

## Launch on NERSC

From `/global/u2/y/yiren/ml_pipeline`, after starting Ray in the designated image:

```bash
shifter python3 scripts/train_dgpo_restart_ablation.py --arm inherit --check-only
shifter python3 scripts/train_dgpo_restart_ablation.py --arm restart --check-only
```

Then run each arm in its own allocation/Ray cluster (each config requests 16 GPUs):

```bash
shifter python3 scripts/train_dgpo_restart_ablation.py --arm inherit
shifter python3 scripts/train_dgpo_restart_ablation.py --arm restart
```

Each invocation validates the paired configs, source epoch/step, finite live policy,
required data/backbone/config dependencies and an empty output checkpoint directory.
The printed live-policy SHA256 must match between arms. It is computed from the current
source rather than certified against a historical digest. The source file must remain
unchanged between launches. No automatic fallback to last.ckpt or existing branch occurs.

Independent outputs under `/pscratch/sd/y/yiren/Ztautau/`:

- `dgpo_10pct_step260_i1inherit_latercold_p8_stale10_seed42/`
- `dgpo_10pct_step260_restart_p8_stale10_seed42/`

New W&B runs use those names. Existing classifier live logs report `warm_started`
and its source. The inherit arm should be cold during initial bootstrap and warm on
compatible later rounds; restart must remain cold for reward fits. Raw monitors may
warm-start in either arm, by design.

## Interpretation

Compare matched DGPO update counts AND wall time, since saturation can cost different
amounts in each arm. Check actual warm-start logs, held-out classifier loss/AUC,
physics observables, ESS when available, and number of refits. The same monitor protocol
does not imply identical trained judges once policy histories diverge. Confirm final
policy ranking with common fixed evaluation data and independent judges; do not declare
a winner solely because its own monitor has a smaller AUC. A single paired seed is
exploratory, not a statistical demonstration of general superiority.

The additional `dgpo_omnifold_ztautau_10pct_i1inherit_best_resume.yaml` branch uses
old 128x1 architecture, iteration-1-only inheritance, later iterations cold,
global-best rollback, patience12 at 10-step cadence, and TARP disabled. Ordinary
physics validation is retained. It preserves the legacy monitor protocol on full
resume. A parent trained with no selected warm-start iterations has an empty warm
cache; the first refit then falls back to cold fitting, and subsequent iteration-1
fits can inherit. It does not synthesize fold provenance from unlabelled weights.
