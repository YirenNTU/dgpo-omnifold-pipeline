# Selected context256 classifier -> native DGPO

Status: implemented, local CPU regression tests passed including real two-rank
DDP. NERSC 16-GPU end-to-end execution is not yet verified; the user launches it.

## 2026-10-01 process-label consistency fix

`bae1ac1b` exposed an initial training/validation reward offset (+2.10 versus
-3.56), before policy learning could explain it. Code inspection found that
native training retained parquet `classification` while the historical cached
classifier negatives, refit and validation omitted it (EveNet defaults to0).
The generation head really uses that label embedding. Nonzero parquet labels
therefore select a different generator conditioning from the ratio denominator.
The remote label values and magnitude of this causal effect still need the
new paired startup probe; population/overfitting effects are not ruled out.

The fix explicitly uses process label0 for this experiment's live rollout,
gradient-bearing policy/reference evaluation, validation/refit generation and
frozen feature extraction. This preserves the selected saved classifier's
denominator, not a claim that0 is the physically correct process identity.
Observed visible features and decay `event_category` are unchanged. A future
switch to the actual process label requires new negatives and a newly fitted
ratio, not silently reusing the label0 head.
This process-label contract does NOT alter BCE class labels: truth remains1,
generated remains0. It is separate from the visible decay category.

The first live training batch performs a paired32-event/rank probe with identical
noise: original parquet label, explicit0 and rebuilt validation-panel path.
W&B logs `tau/startup/incoming_label_nonzero_fraction`,
`paired_original_label_reward`, `paired_fixed0_reward`, `paired_panel_reward`,
and `live_panel_max_sample_error` (all under `tau/startup/`). Training stops if
the explicit0 live and panel draws differ beyond tolerance. RNG/mode are restored;
the probe never performs an optimizer update. Every resumed launch rechecks it.

Use a NEW `_label0` output, starting raw1110 and reusing original `zrv2yfgt`,
not `bae1ac1b/last.ckpt`. Checkpoints record the conditioning contract and reject
pre-fix DGPO state. Old files/runs are preserved. Fourier/FiLM, bound30, dataset,
16 GPUs and coefficient1 velocity-MSE are unchanged. On the corrected run the
training reward may begin negative; that is not evidence of lost learning.
Fix verification:97 selected local CPU tests passed, including the paired
label-effect probe, actual native train-step entry, old-reward noninterference,
pre-fix resume rejection and two-rank DDP. The two unrelated historical
orchestration failures described below remain excluded, not silently fixed.

## Question and frozen choice

Can native DGPO absorb the saved best conditional-tau ratio and improve the
**unweighted generated tau spin-correlation matrix**, including off-diagonal
entries? Reweighting improvement alone is not evidence of generator improvement.

Reuse `zrv2yfgt/best.pt`: condition256, candidate64, three nonlinear residual
FiLM blocks, training-time smooth ratio bound30, selected by minimum validation
BCE. The completed K64 reweighting comparison had Cij errors 0.565440 unweighted,
0.373213 context64, 0.256479 context256. K1 did not show the same benefit. These
are previous reweighting results, not results of this new DGPO experiment.

The classifier retains its independent **frozen raw step1110 EveNet trunk**.
Each candidate uses its clean-t0 pre-velocity representation, the existing
six relative-angle features and fifteen tau-direction features; the observed
condition follows the exact saved masked preprocessing. No new input feature,
adapter, moment penalty, MMD, cascade, or classifier backbone fine-tuning.

## Policy and reward lifecycle

- Initial actor: raw step1110 **weights-only**, new DGPO step/epoch0, with the
  existing zero-output angular/momentum Fourier and depth3 FiLM conditioning
  additions. No EMA. Startup checks matched-noise initial samples against the
  frozen raw1110 denominator before reusing the classifier.
- Reward: the saved head's bounded **log ratio**, not its probability or latent
  unbounded logit. No extra ratio normalization or rescaling. This deliberately
  bounded estimate does not guarantee exact truth closure.
- Native DGPO: K8, DDIM20, leave-one-out unscaled advantage, beta1, AdamW.
  Soft velocity-MSE reference coefficient1; no endpoint KL. Velocity MSE is a
  surrogate constraint, not an exact distribution KL or a guarantee of closure.
- Every10 policy epochs: evaluate the current actor, then generate new K1
  negatives on the training population and train a fresh head of the same
  architecture/bound. Restore the absolute minimum-val-BCE head, **replace**
  the old reward (do not multiply ratios), and freeze the new actor reference
  matching that denominator. Actor AdamW and cosine clock are not reset.
- Initial saved reward is reused without fitting a new reward. A distinct
  fresh audit is trained at step0 for a comparable monitoring baseline.
- Ten optimizer steps per policy epoch; every10 epochs is every100 updates.
  1500 policy epochs, no additional command-line step limit. Final epoch does
  not refit a reward that will never be used.

Actor batch512/GPU, classifier fit1024 paired conditions/GPU, candidate generation
1024/GPU with feature extraction microbatch256. All execute on the same16-GPU
Ray worker group; classifier fitting does not request another GPU allocation.
Float32 with highest matmul precision matches the saved feature/sampler path.

Actor body/generation/projector LR5e-5; angular/FiLM branches LR1e-4. All five
groups cosine-decay over1500 epochs to10% of their base rates. Classifier fits:
AdamW2e-4, weight decay.001, inherited dropout.05, cosine250 epochs to1e-5,
early-stop patience25/min_delta1e-4/min_steps1000. The minimum applies to
classifier optimization only, not a DGPO minimum. Best BCE is restored even
when its improvement is smaller than the patience-reset threshold.

## Data and monitoring

Use only the verified filtered416701 training and119002 validation events.
Training head split354488/62213 and source preprocessing are inherited exactly.
Completed filter manifests, identities, condition tensors, weights, and source
checkpoint are checked. Source normalization is never recomputed.

At step0 and every10 policy epochs:

1. Generate K8 **unselected** draws for every external validation condition.
   `tau/cij/*` logs all nine generated/truth entries plus total, diagonal and
   off-diagonal error. Classifier ratios are NOT used as physics weights;
   existing base event weights remain. Every event contributes the K-draw mean.
2. Train a fresh **unbounded** audit using the same features/context256 FiLM
   architecture on candidate0, with an independent fixed60/20/20 event split
   of that external pool. Restore best validation BCE; report BCE/AUC on the
   audit's disjoint test subset (`tau/fresh_audit/*`). This is not a complete
   architecture-independent two-sample test.
3. Record deployed reward mean/best-of-K/worst-of-K/within-condition std and ESS
   (`tau/reward/*`). Compare rewards within a reward round; refits change the
   scoring function. Within-condition K8 ESS is not the earlier global K64 ESS.
4. Preserve native policy loss, velocity-reference penalty, gradient and LR
   logging. Separate `tau/audit_fit/*` and `tau/refit_fit/*` clocks prevent the
   classifier epochs from advancing the DGPO update counter.

The first success criterion is generated Cij improvement against this run's
step0, without hiding cross-component deterioration. Lower fresh-audit AUC
toward0.5 and higher test BCE toward log2 support alignment but are not required
to be monotonic. Reward increase alone is only evidence of reward absorption.
Reports are monitoring point estimates; aligned per-event measurements are
saved for paired uncertainty analysis. The repeatedly inspected external
population is exploratory, not an untouched final test.

Cij uses the exact existing `tau_from_deltas` fixed-energy reconstruction,
TT2L basis, and stored analyzing powers. Its truth comparator is the same
reconstructed-truth convention, not full tau-energy unfolding. This integration
does not newly certify the polarimeter/acceptance convention.

## User launch / resume

Update the existing NERSC checkout, then run on the user's16-GPU Ray allocation:

```bash
shifter python3 -u scripts/train_dgpo_tau_ratio.py \
  config/dgpo_tau_ratio_1110.yaml
```

Append `prepare` for source/parquet preparation only. GPU feature/logit replay
and initial actor-denominator checks run at training startup. No remote job is
submitted by implementation.

Original integration verification:25 focused reward/refit/resume and residual-head tests passed,
including two real two-rank CPU DDP tests;92 tests pass in the broader selected
orchestration suite (overlapping coverage). The unfiltered broader run also
exposes two unrelated existing tests: an old10% raw-dataset path assertion and
PyTorch2.2's rejection of a `Path` with `mmap=True` in the old rollback helper.
Those old paths were not changed for this experiment. These tests do not replace
actual NERSC checkpoint/feature replay or16-GPU execution.

Selected classifier:
`/pscratch/sd/y/yiren/Ztautau/conditional_tau_condition_width_1110/condition256-40e1f8b288/best.pt`

Output:
`/pscratch/sd/y/yiren/Ztautau/dgpo_tau_context256_1110_label0`

On first launch there must be no checkpoint in this new output. Subsequent
launches with the **same command** fully resume its own `checkpoints/last.ckpt`:
actor, reference, installed reward, reward-round markers, AdamW and scheduler
clock. Completed audit/refit phases are not repeated. Each launch creates a
**new W&B run**, retaining the true absolute DGPO step. W&B run name:
`Can DGPO absorb tau alignment? | context256 FiLM | bound30 | matched label0`.

Interrupted, uncheckpointed generation or classifier fits restart from the last
completed DGPO checkpoint; they do not pretend to have resumed head optimizer
state. Source classifier artifacts and original raw1110 checkpoints are never
overwritten. If uploading the repository root, retain the exclusions in
`NERSC/upload-excludes.txt`; do not upload toy outputs or saved classifier caches.
# Fourier-off policy ablation (2026-10-01)

Matched Fourier-on launch:
`shifter python3 -u scripts/train_dgpo_tau_ratio.py config/dgpo_tau_ratio_1110_fourier_on.yaml`.
This uses a separate `dgpo_tau_context256_1110_label0_fourier_on` output root,
starting raw1110 on first launch, then resuming only its own checkpoints.
Each arm requires 16 GPUs; concurrent execution needs separate appropriately
provisioned allocations/clusters, not two jobs sharing one occupied 16-GPU
cluster. The resolved overlay equality test allows only the two Fourier flags
and output/logging metadata to differ. Compare step100 BEFORE first refit for
the same fixed reward; later fresh-refit rewards differ between arms, so use
matched Cij and fresh audit endpoints rather than raw cross-round reward.

User launch: `shifter python3 -u scripts/train_dgpo_tau_ratio.py config/dgpo_tau_ratio_1110_no_fourier.yaml`.
Compare against the corrected label0 Fourier-on configuration, not the old
label-mismatched run. Both start weights-only from raw1110, reuse the identical
context256 best-val reward, preserve velocity-MSE coefficient1, LR/schedule,
16 GPUs, filtered populations, refit/audit settings and normalization.
The new arm uses an independent `dgpo_tau_context256_1110_label0_no_fourier`
output root and a fresh W&B run. Subsequent launches resume only that arm.

Only the added visible Fourier channels are zeroed: PET angular input and
FiLM angular/energy/pT Fourier input. Parameter shapes and optimizer groups
are retained, as are normalized energy/pT scalars, original PET features,
three-block nonlinear FiLM, and timestep embeddings. This is a feature
ablation, not removal of the whole conditioning network. The independent
classifier is unchanged. Startup sampler parity and paired label checks
still run. Compare matched validation steps for generated Cij (including
off-diagonals), fresh-audit BCE/AUC and within-round held-out reward; reward
after refit is not the same scoring function as before refit.
Local CPU tests verify zero channels, retained FiLM gradients, unchanged
parameter shapes, config isolation and classifier-spec isolation. NERSC
16-GPU execution remains user-submitted and unverified locally.

### Resumed-run 1D monitoring (2026-10-01)

The original c4a91e07 W&B summary includes target delta-theta/delta-phi
JSD and cos-opening / delta-phi-to-pi panels, absent from the tau-cycle runs.
Every existing tau validation now also writes `tau/marginal/plots/{target,reco,topology}`:
four target offsets, four reconstructed theta/phi distributions, and two
pair-topology distributions. Definitions reuse the original Ztautau domain
helpers. Each plot overlays truth, initial step-zero samples and current
samples. Only candidate zero is used (not reward-best); original event weights
are retained, with no classifier ratio reweighting. No extra generation,
classifier training, or GPUs are needed. Cij continues to use all K samples.

The baseline comes from this output root's
`tau_diagnostics/eval-step-00000000/measurements.npz`, verified against panel
IDs, and does not change on reward refits or resumes. Missing baseline is
explicitly logged as `tau/marginal/baseline_available=0`; it is never replaced
with a recentered velocity reference. Keep the original diagnostics directory.

`tau/marginal/{tv,jsd}/{current,initial,change}/...` reports histogram distances
to truth; negative change means improvement versus step zero. JSD here is
natural-log Jensen-Shannon divergence, not its square root. Bins remain fixed
across epochs, with explicit under/overflow in both distances; plots show
outside-range mass in their legends. These new event-weighted, tail-inclusive
metrics should not be numerically equated to historical differently binned
JSD metrics. Raw histograms and metrics are saved in each `marginals.json`.
Monitoring uses the repeatedly inspected validation panel, not a final blind
test, and supplies no statistical-significance claim. Tau energy is fixed in
this reconstruction; these plots do not certify full energy/mass unfolding.

After updating code on NERSC, the existing no-Fourier launch command resumes
its own last checkpoint and logs these plots at the next scheduled validation
to its new W&B run. No model, reward, optimizer, schedule or checkpoint state
is changed by these diagnostics. Local tests: 19 passed including candidate-zero
selection, tail accounting, fixed bins, periodicity and existing resume/config
regressions; distributed execution must be verified on NERSC by the user.

## Step1920 reward-uptake mechanism series (2026-10-04, prepared only)

The new [mechanism protocol](tau_reward_mechanisms_1920.md) keeps the original
step1920 actor, inherited step1880 reward, velocity reference, AdamW and cosine
clock for diagnosis. It separates reference/condition/noise gradients using
the exact native gates and accumulation weights, then tests transactional local
directions on independent validation conditions. These are measurements, not
evidence yet that a particular cause is established.

A separate matched fresh member0 versus four-head ensemble study shares current
policy negatives, saved native training batches and update-index sampling. A
fifth frozen classifier is the common evaluation judge, not an ensemble member.
Fixed-teacher trajectories permit declared 50/200/500 update budgets; fresh
audits and unweighted Cij remain separate checks. No production architecture,
reward or reference lifecycle changes when this opt-in protocol is disabled.
No NERSC job has been submitted or run by the assistant for this series.
