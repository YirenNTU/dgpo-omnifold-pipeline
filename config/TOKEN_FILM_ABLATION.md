# Token-specific conditioning: two separate questions

## 1. DGPO reward uptake, with velocity MSE unchanged

Control retains PET angular Fourier and the existing three-block global nonlinear
FiLM. Treatment adds a width-64, four-head visible-token readout **only in the last
generation block**. Invisible queries carry the current noisy state/time/slot;
memory reuses unpooled shared visible encoder outputs. A nonlinear readout produces
per-query scale/shift for attention and FFN, added to existing global FiLM.
Only its final projection is zero initialized. No clean truth is an input.

Both inherit this immutable completed classifier bootstrap:

`/pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt`

Raw step1110 policy before new RL updates; installed classifier payloads and
original/round references stay unchanged. The reference keeps its original
architecture and exact hash. Both rebuild **empty step-zero** optimizer groups;
nonempty Adam moments or nonzero policy progress are rejected. This is a new A/B,
not a late-checkpoint resume. Source files are not modified.

- Velocity-MSE coefficient **1** in both arms; no loss/KL ablation.
- Body/head/projector LR5e-5; Fourier/visible-conditioning LR1e-4. Token readout
  belongs to the existing visible-conditioning optimizer group, exactly once.
- Cosine1500 logical epochs, 10 updates/epoch, ending at10% of base LR.
- **First screen: stop after1000 policy updates** (100 logical epochs). The
  launcher forwards `experiment.policy_update_budget: 1000` as `--max-steps`;
  it does not shorten the1500-epoch cosine schedule. Save/review this endpoint
  before authorizing an extension. No automatic next iteration or job submission.
- Same 10% data, seed and16 GPUs; no EMA, reward refit or periodic classifier audit.
- Existing validation every10 epochs, K16/DDIM20/max4 batches remains.
- Primary: `val/reward/all_sample_mean`, versus step-zero baseline at equal
  policy steps. Legacy `val/reward/mean` is **best-of-K**, now also explicitly
  named `val/reward/best_of_k_mean`; checkpoint ranking is unchanged.
- Also inspect `reference_trust/velocity_mse_ratio`, raw reward, actual LR,
  `train/visible_conditioning/token_scale_rms`, `token_shift_rms` and
  `token_readout_grad_norm_post_clip`. Compare elapsed time as well as steps.

User launches on an existing16-GPU Ray allocation, after updating the normal
remote repository. These commands do not submit a Slurm job; run one at a time
unless separate resources have already been allocated:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_dgpo_token_film.py --arm control
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_dgpo_token_film.py --arm last
```

`--dry-run` prints the resolved config without accessing NERSC files or launching.
Each arm has its own checkpoint folder and W&B ID (`h4tokctrl1`, `h4tokfilm1`).
Do not rerun these start-from-zero commands to resume a partially trained arm;
full resume must disable `step_zero_architecture_bootstrap`, load that arm's saved
checkpoint, and keep its optimizer/scheduler/reward/reference state.

### Independent fresh H4 audit

Use the preselected1000-step endpoints, not each arm's best-reward winner. Both
standalone cold audits now reuse the **matched-fold** protocol from`9592bbca`:

- Fit on repeat1/fold1 of the OmniFold training identities: historically208,355
  events, rather than about71,400 from the old probe split.
- Split the cleaned external validation pool by the same identity rule into
  early-stop59,465 / final-test59,527 events. These counts are reference values,
  not new quotas; actual counts are logged. This pool has previously been used
  for generator validation and is not a newly untouched generator test set.
- Generate fresh K1/DDIM20 raw samples for each policy, with matching panel,
  generation, fold and classifier seeds. Same event identities do not mean
  reusing samples from the older generator.
- Train the same existing three-block H4 classifier in both arms. The new
  token-specific readout belongs only to diffusion, not to the classifier.
  Classifier backbone initialization, trainable scope and optimizer stay matched
  to`9592bbca`; no saved reward classifier is warm-started.
- Best validation-BCE checkpoint, min_delta1e-3, patience25 classifier epochs,
  minimum0 / no maximum fit steps. The1000-update policy cap does **not** cap
  classifier training. Inspect convergence before interpreting near-chance AUC;
  an early-stopped undertrained judge is not evidence of closure.
- Exactly one fresh fit, zero policy updates and zero reward refits. No audit
  result is installed as a new training reward.

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_dgpo_token_film.py --arm control \
  --audit-checkpoint /pscratch/sd/y/yiren/Ztautau/h4_token_film_control/checkpoints/last.ckpt \
  --expected-step 1000
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_dgpo_token_film.py --arm last \
  --audit-checkpoint /pscratch/sd/y/yiren/Ztautau/h4_token_film_last/checkpoints/last.ckpt \
  --expected-step 1000
```

Run each audit only after its corresponding training job has stopped at1000.
The launcher pins`last.ckpt` to its resolved file and checks the stored step and
architecture before starting. Source policy step1000 is logged separately from
the measurement job's step0 clock. The two audit IDs are
`h4tok-control-matched-1000` and`h4tok-last-matched-1000`; outputs are under
`h4_token_film_{control,last}_matched_fold_audit_step1000`.
These are new measurement identities, separate from old probe-split audits;
existing training IDs remain unchanged. No existing W&B run is renamed.

Optional shared fresh-judge baseline (one fit, not one per arm):

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_dgpo_token_film.py --arm control \
  --audit-checkpoint /pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt \
  --expected-step 0
```

### Screen result and decision

Record each arm's all-sample held-out reward trajectory, velocity-MSE drift,
elapsed time, and the1000-step fresh judge's test AUC/gap, best-validation BCE,
fit steps and actual fit/early-stop/test counts. Classifier optimization minimizes
BCE; for equally well-trained judges, a smaller test AUC gap and test BCE moving
toward`log(2)` indicate reduced distinguishability. Do not compare raw logits
across independently trained judges.

- Better reward uptake **and** smaller fresh-H4 gap, with adequate fits: supports
  extending both arms from their saved states to5000 and potentially15000.
- Better fixed reward alone: do not declare distribution improvement; investigate
  fixed-reward versus fresh-judge alignment before adding more capacity.
- No resolved advantage at1000: this screen is inconclusive about eventual
  convergence, not proof that conditioning cannot help. Review curves/time cost
  before deciding on another intervention. No extra seed sweep is scheduled.

`9592bbca` at step2780 and`227e4975` are population/provenance references. They
are not substitutes for this experiment's equal-step global-FiLM control.

## 2. Supervised diffusion architecture screen: no RL or reference penalty

All arms load the same raw step-zero policy **weights only**, use the same10%
training population and cleaned validation data, paired noise/timestep draws,
ordered batches, FP32,16 GPUs,2048 events/GPU, fully trainable backbone.

| Arm | PET angular Fourier | Three generation blocks | Generation conditioning |
|---|---|---|---|
| `pet_only` | unchanged | unchanged | none |
| `global` | unchanged | unchanged | existing global nonlinear FiLM |
| `last` | unchanged | unchanged | global FiLM + last-block token FiLM |

This is not a no-Fourier feature ablation: every arm keeps the PET Fourier path.
The strict loader allows removing the global branch only if its saved modulation
outputs are zero, ensuring the common step-zero function. It rejects other shared
weight omissions/shape mismatches and never loads EMA.

Reuse the earlier supervised matched screen: **50 epochs**, five-epoch warmup +
cosine, body/PET Fourier LR2e-5, head/projector/visible-FiLM LR1e-4. Early-stop
patience51 cannot shorten this screen. No low-noise weighting, classifier, reward,
reference penalty or joint-coverage callback. These are screening settings, not
an assertion that50 epochs establish final convergence.

Primary: mean `val/loss` over the final five epochs at the same budget, plus the
complete train/validation velocity-loss curves and elapsed time. Validation uses
fixed noise across epochs; confirm promising results on independent noise before
claiming a small gain. Lower velocity MSE alone does not prove joint-physics gains.

Run the following once for each `ARM` value (`pet_only`, `global`, `last`):

```bash
ARM=global
shifter python3 -u scripts/train_neutrino_backend.py \
  --backend pure-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config "config/train_diffusion_token_film_${ARM}.yaml" \
  -- --ray_dir "/pscratch/sd/y/yiren/Ztautau/diffusion_token_film_${ARM}/ray_results"
```

Separate W&B runs: `difftokpet1`, `difftokglobal1`, `difftoklast1` in EveNet.
No jobs have been submitted. Local CPU tests cover function equivalence, masks,
gradients, checkpoint ownership and optimizer/config contracts; real NERSC
checkpoint loading and16-GPU execution still require the user's run.

Earlier local verification:102 architecture/config/optimizer tests passed;77 trainer and
validation regressions plus38 subtests passed in a separate process. The legacy
trainer test installs a process-wide debug-module stub, so combining that suite
with Lightning-engine import tests produces a test-isolation import failure;
neither production imports nor the isolated suites have that failure.

Matched-audit update (2026-09-27):112 targeted tests plus12 subtests pass across
three isolated invocations. Coverage includes both launcher modes,1000-step
forwarding without a classifier cap, matched original fold/seeds/fit settings,
the actual classifier-only terminal branch for both policy architectures,
raw checkpoint step/architecture checks, data identity/disjoint-test behavior,
zero-start outputs, masking/gradients, optimizer groups and supervised-screen
regressions. Training and audit dry runs do not load remote checkpoints or launch
work. W&B display names pass the naming validator. No NERSC runtime verification
or new experiment result is claimed by these local tests.

Interpretation: supervised improvement suggests easier velocity learning; DGPO
improvement with unchanged velocity MSE suggests easier reward transfer. Neither
alone proves velocity regularization is irrelevant. This tests a routing/capacity
package, not a parameter-matched proof that global pooling was the bottleneck.
