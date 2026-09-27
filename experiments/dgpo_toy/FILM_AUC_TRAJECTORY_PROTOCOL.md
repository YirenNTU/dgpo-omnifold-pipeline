# Is fresh H4 AUC monotonic during successful toy DGPO?

Status: **launched locally on 2026-09-25**, after the user's explicit
"start training" authorization for this specific experiment.
Toy-only: no production config changes, NERSC upload, reward refit, or remote job.

## One question

Does the previously successful learned-H4 + nonlinear FiLM policy have a
monotonically decreasing fresh-audit AUC, or can early deterioration/plateau
precede later improvement? Policy updates and audit-fit updates are separate
clocks. This is a measurement-cadence experiment, not a new conditioning arm.

Previous `learned_film_conditioning_v1` sampled only policy steps 0, 1000,
2000, 3000: AUC 0.641407, 0.604301, 0.591300, 0.587874. It cannot answer
what happened between 0 and 1000. The earlier oracle-C experiment had a small
2k-to-3k rebound; neither observation establishes a universal trajectory.

## Frozen training setup

- Same raw pretrained source: `nonperiodic_cube_classifier_extended_v1/reference.pt`.
  This diffusion learned the uniform noisy cube reference, not full truth.
- Same frozen learned reward A: `reward_transport_abc_v1_retry1/classifier.pt`,
  selected at classifier update 3100. No new classifier fit supplies reward.
- **A is shape-matched diagnostic H4**, not the original low-ESS full-truth H4.
  Its positive training distribution used known toy mode probabilities.
  This limitation prevents a physics-prior-free production claim.
- Same c-only Fourier `[1,2,4,8]`, nonlinear 8→32→32 SiLU encoder and three
  zero-initialized FiLM output heads. No linear or time-conditioning policy arm.
- AdamW LR 1e-4, weight decay .001, clip 1; batch64, K8, M4, DDIM50.
- Native detached DGPO gate and unscaled LOO estimator unchanged. Coefficient1
  multiplies **half velocity MSE**, a surrogate, not exact distribution KL.
- Original reference remains fixed. Preserve policy optimizer/RNG/history
  through every measurement boundary. No EMA, refit, LR schedule or reset.
- Same policy seed17, native monitor seed284017, encoder seed451017.

The runner loads existing helpers instead of redefining the objective. Its
read-only preflight checks exact initial velocities and DDIM samples. Regression
tests compare dense staged updates to continuous updates including Adam moments
and RNG, with audit activity inserted between stages.

## Evaluation grid and adequacy

Policy steps **0, 100, 250, 500, 750, 1000, 2000, 3000**. Step0 is a real
fresh fit, not a copied historical AUC. The fixed terminal endpoint is 3000;
do not stop/extend based on a favorable intermediate test value.

At each point independently cold-initialize the same Fourier H4, using original
full truth, seed41, and paired train/validation/test panels 32768/8192/16384.
Condition/truth identities and rollout-noise seeds remain paired across policy
checkpoints. The test pool is disjoint from fitting and checkpoint selection.

Audit settings: AdamW3e-4, WD.001, balanced batch512, clip1; minimum2000 updates,
validation every100, patience20 checks, min_delta1e-4, maximum16000; restore the
lowest validation BCE weights. A budget-exhausted audit is inconclusive.

Measurement clarification: the old multi-arm experiment stopped audit arms
jointly. This single-arm trajectory uses the same patience/minimum rules
independently at every checkpoint; exact historical audit fit lengths or
selected scores need not match. Policy training itself is unchanged.

## Predetermined interpretation

1. **Point-estimate monotonicity:** every measured adjacent AUC difference ≤0
   (floating tolerance1e-12). Report AUC-gap monotonicity separately.
2. **Resolved rebound:** an adjacent AUC increase with its simultaneous95%
   interval strictly above0. No resolved increase is *not* proof of monotonicity.
3. **Early worsening followed by recovery:** at least one preterminal AUC-gap
   contrast versus step0 has lower95%>0, while the final-versus-step0 contrast
   has upper95%<0. The test does not select a favorable intermediate endpoint.
4. **Endpoint improvement:** report final-versus-step0 AUC gap and BCE even if
   the path is nonmonotonic. Higher optimal audit BCE toward log2 is favorable
   to the generator, not permission to undertrain the audit.
5. If *any* audit is invalid, full-grid monotonicity is inconclusive; do not
   silently skip it and connect the remaining points.

Use 2000 paired-context bootstrap replicates with fixed seed491017. For each
metric separately, a bootstrap maximum centered-error band covers all adjacent
and versus-start contrasts simultaneously. Also save ordinary pointwise
intervals, explicitly labeled. Primary discrimination uses AUC/gap; BCE and
mode-TV are complementary diagnostics, not silent replacements.

These intervals condition on the fitted models, not policy/audit training-seed
uncertainty. Nothing proves monotonicity between measured points, eventual real
case convergence, or that a higher production LR will work. A pure plateau
followed by improvement may be visible descriptively without passing the
stricter early-*worsening* test.

## Outputs and W&B

Display name: `Is fresh AUC monotonic? | learned H4 | nonlinear FiLM | dense early audits`.
Project `dgpo-toy`, group `Conditional reward transport`; online by default.
Do not reuse a historical run ID. `--wandb-mode offline` is available explicitly.

Main plots:

- `validation/auc`, `/auc_gap`, `/bce` against `validation/step` (**policy** steps).
- `validation/valid`, `/fit_steps`, `/selected_step`, `/plateau` for adequacy.
- `policy/monitor_gain` and reward/reference gradient diagnostics on policy steps.
- `audit/policy{checkpoint}/validation_auc` and `/validation_bce` against each
  audit's own fit-step clock. Their upward AUC curve is not policy deterioration.
- After the final point, `adjacent/auc/*` and `versus_start/auc_gap/*` contain
  deltas and pointwise/simultaneous intervals on their own policy-step axes.

Each completed audit refreshes `auc_trajectory.csv` and `auc_trajectory.png`.
No smoothing; invalid points are marked and lines are broken. `monotonicity.json`
holds final contrasts and decisions; `report.json` and `progress.jsonl` expose
live status. Each `round_*` contains a policy snapshot, cold-audit states and
paired panels/test scores. `mlp_c_last.pt` includes optimizer/RNG/history.
Existing result directories are never overwritten. This runner does not
automatically resume an interrupted experiment or submit a follow-on job.

## Local commands

Read-only validation (does not start training, W&B, or create output):

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.film_auc_trajectory --preflight
```

Launch personally:

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/miniconda3/envs/MyEve/bin/python -u \
  -m experiments.dgpo_toy.film_auc_trajectory \
  --output artifacts/dgpo_toy/film_auc_trajectory_v1 \
  --milestones 0 100 250 500 750 1000 2000 3000 \
  --audit-max-steps 16000 \
  --wandb-mode online
```

The audit cap is a compute bound, not an assertion of convergence. Inconclusive
fits remain explicitly inconclusive; there is no automatic experimental extension.

## Implementation verification

Actual-source read-only preflight passed: initial velocities and full DDIM samples
match exactly; learned reward is frozen. No experiment output or W&B run was
created. The focused trajectory/FiLM/transport regression suite passed 54 tests,
including exact staged-versus-continuous policy/Adam/RNG equivalence, paired
bootstrap AUC with score ties, invalid audit handling, and separate W&B clocks.

## Authorized launch record

Started the exact 0/100/250/500/750/1000/2000/3000 command above locally,
with two compute threads and online W&B. No other training arm was started.
Run ID: `mn9ols7y`; project/entity resolved from the signed-in account:
https://wandb.ai/b11202011-national-taiwan-university/dgpo-toy/runs/mn9ols7y

Execution session: 72865. Output: `artifacts/dgpo_toy/film_auc_trajectory_v1`.
Initial diagnostics match the source (mode-TV 0.1621239044, all FiLM modulation
and fixed-probe velocity deviation zero). The step-0 cold audit completed at
7500 fit updates, selecting update5500: test AUC0.6414070912, BCE0.6553682060,
matching the historical baseline. The policy has started updates and the
step-100 cold audit is running at this launch check. This record confirms
launch, not completion or a monotonicity result. Live state is in `report.json`/W&B.
