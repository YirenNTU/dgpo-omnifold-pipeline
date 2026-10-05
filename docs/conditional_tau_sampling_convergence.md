# Frozen FiLM ratio: conditional sampling convergence

## Current default: extend the completed K8 panel to K64

The YAML now pins `extend_from` to `sampling-ee3acf9a12`, analyzed in
`ppoem0g9`, and compares K=1,2,4,8,16,32,64. Same simple launch command below.
This extension is implemented, not yet executed on NERSC by the assistant.

- Reuse all952016 old samples. Copy the128 completed candidate shards byte-for-
  byte to a fresh attempt directory; do not mutate the source or resample K<=8.
- Generate only candidate indices8–63 (human numbering9–64):6664112 NEW draws,
  total7616128. Same16 GPUs, batch1024, sequential independent candidate streams.
- Preflight checks source completion/run ID, unchanged model/runtime paths,
  sampler/seed/sharding/batches, full inputs and all inherited shard identities.
  On mismatch, stop rather than silently regenerate a supposedly matched panel.
- Recompute prefixes using the same event-bootstrap seed and verify old K<=8
  matrices, intervals, errors and ESS reproduce the previous report.
- New W&B run includes reused/new sample counts and prefix-verification status;
  the nine-component figure now uses a base2 log K axis. All raw ratios,
  conventions, fixed-model limitations and diagnostic definitions below persist.

The original K8 protocol and executed results below remain historical records;
the extension changes sample count only, not classifier or generator training.

Status: implemented;51 local CPU tests passed (including K32/K64 extension,
byte-preserving shard reuse and unchanged inherited-prefix reports; inference/estimator/report
tests plus relative-input, normalization and Cij regression tests). Data-reader
boundaries are mocked in the preparation unit test; real NERSC checkpoints,
parquet and CUDA execution have not been exercised locally. No NERSC job
submitted by the assistant.

Runtime recovery (2026-09-30): run39atiwne completed all16 workers and952016
samples, then failed before analysis at W&B `Summary.update(phase=...)`.
Both keyword-style summary updates now pass a dictionary. The endpoint test
uses the real SDK Summary class (no online run), replacing the permissive dict
mock that missed this incompatibility. Reuse `sampling-ee3acf9a12` with
`--analyze-only`; do not regenerate candidates for this reporting failure.

## Fixed protocol

Question: does more generation on the SAME conditions reduce reweighted Cij
failure, or stabilize the result away from matched truth?

- Frozen `pzq0nl1i` three-block FiLM classifier, selected epoch250.
- Generator AND candidate-feature trunk: its original raw old-DGPO step1110
  checkpoint. Never EMA or the newer global-FiLM generator.
- Full filtered validation:119002 conditions, one existing truth/condition.
- Eight independent DDIM20 draws/condition:952016 fresh candidates. Nested
  prefixes K=1,2,4,8; no fitting, DGPO updates, cuts, or training-seed reruns.
- 16 one-GPU Ray workers, generation batch1024 conditions/GPU. Candidates run
  sequentially; native hidden extraction microbatch256. No eightfold GPU batch.
- Same condition normalization, clean-candidate t=0 pre-velocity hidden tokens,
  six fit-scaled relative angles and fifteen tau direction moments as the fit.
- Seed streams: seed+1000003*candidate_index+rank. Fixed worker/batch layout;
  changing these would change random-number assignment.
- Generation retains the historical test runtime's float32 matmul setting;
  feature extraction uses `highest`, matching the original cache worker.

Rebuild/identity-align validation parquet against saved test arrays. Verify
normalized conditions, matched truth map, categories, base weights, analyzing
powers and raw checkpoint provenance. Each worker replays64 historical generated
samples through the full feature/head path, requiring score agreement
(atol1e-3,rtol1e-4). Analysis reproduces the historical Cij endpoint too.
No physics labels or truth values enter generator/classifier inputs.

## Estimator and uncertainty

For per-candidate observable f and raw r=exp(logit), use

    C_R(K) = sum_i w_i (1/K) sum_j r_ij f_ij
             / sum_i w_i (1/K) sum_j r_ij

Unweighted uses r=1; truth counts each condition ONCE. Stable common exponent
shifts are numerical only. No cap, tempering, per-event normalization, best-of-K,
or physics penalty. Signed per-event analyzing powers and the matched tau
reconstruction map are unchanged; this is not new polarimetry certification.

Primary: all nine signed Cij residuals and weighted-minus-unweighted Frobenius
error at every K. Do not select whichever K happens to give the best Cij.

- Shared500-replicate EVENT-cluster bootstrap across K/arms/truth. Candidates
  within a condition are not treated as independent events. These are pointwise
  intervals conditional on saved models/draws, not classifier-refit uncertainty.
- Separate within-condition delta-method Monte Carlo SE for U,R,R-U at K>=2.
  Holds conditions/truth fixed; K1 is unavailable. Heavy tails can invalidate
  this small-sample approximation. It is not the event-bootstrap interval.
- Disjoint panels: eight K1, four K2, two K4, one K8 matrix/error estimate.
  One K8 estimate alone does not prove convergence.
- Candidate AND aggregated-event ESS, max masses, top1% candidate mass, raw
  log mean ratio, per-Cij top1% absolute event influence.
- Same fit-only category x visible-pT groups: tau moment errors and mass drift.

Interpretation: stabilization nearer truth supports a generation-variance
contribution; stable worsening argues against generation noise alone. Continued
tail-driven jumps leave convergence unresolved. More draws do not repair a
support hole. The reused external validation population is exploratory.

Historical K1 is a separate realization, never substituted for the fresh K1
prefix; the latter need not reproduce the original point error1.42241.

## User launch on existing NERSC allocation

```bash
shifter python3 -u scripts/run_tau_sampling_convergence.py \
  config/conditional_tau_sampling_convergence.yaml
```

Append `--prepare-only` for a data/CPU preflight without Ray or W&B.
Ray uses `$RAY_ADDRESS` or auto and requires16 GPUs. Each normal launch creates
a fresh scratch subdirectory and a NEW online W&B run, not the source run.
No automatic job/allocation submission occurs.

If sampling completed but reporting failed, append
`--analyze-only /pscratch/.../sampling-<attempt>`: reuse exact shards and that
output directory, create a new reporting run, no generation or GPU allocation.

W&B receives progress, K-indexed Cij/weight curves, full JSON and nine-panel Cij
figure. JSON also includes group tau moments, conditional MC SE and disjoint
panels. `samples_and_scores.npz` stays on scratch (no large automatic upload).

Root repository uploads must use `--exclude-from=NERSC/upload-excludes.txt`;
exclude toy directories and update the existing remote checkout only.

## Executed recovery report — 2026-09-30

[ppoem0g9](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/ppoem0g9),
`Does more sampling stabilize Cij? | frozen FiLM | K=1,2,4,8 | raw 1110`,
finished analysis of952016 draws from39atiwne. Full uploaded JSON, K-indexed
history and runtime config inspected. No new fits or policy updates.
Historical score replay max error0.00011301; historical Cij errors reproduce
0.5213076/1.4224107 (unweighted/reweighted). Fresh K1 is a different draw.

| K | Unweighted Cij error | Reweighted Cij error | R-U paired95% interval | Candidate ESS |
| ---: | ---: | ---: | --- | ---: |
|1|0.554058|2.735160|[1.52430,4.82193]|530.45|
|2|0.578122|2.096654|[1.05193,3.02882]|1227.89|
|4|0.579230|1.605180|[0.68000,1.95846]|3054.22|
|8|0.578324|1.185320|[0.24162,1.54193]|4233.52|

With fixed models, observed weighted error falls56.66% across nested prefixes.
Group tau error falls0.29236→0.22308→0.16992→0.11553, but K8 unweighted
group tau error is0.0029004. Every disjoint panel remains worse than its
unweighted estimate. Eight disjoint K1 weighted errors span2.01665–2.73516;
the two K4 panels give1.60518 and1.24781. K8 conditional MC SE per component
is0.22747–0.37443 (vector norm0.89477): approximate, not a bias subtraction.
An asymptotically stable estimate is not established.

K8 candidate ESS fraction0.44469%, event ESS3932.34 (3.30443%), top1%
candidate mass57.1084%, maximum candidate mass0.56656%. Max candidate mass
increases from K4 to K8; log mean ratio moves0.12476→0.16054. Continued tail
sensitivity remains. More draws increase absolute ESS, not necessarily ESS
fraction. K8 category mass TV is0.08042.

Supports: generation/importance-sampling variability is material. Rules out
K8 raw reweighting as successful matched-Cij correction on this population.
Unresolved: ratio bias versus unresolved tails/support. Falling error does not
certify exact ratios or guarantee eventual closure. This is not improved
diffusion: parameters never changed. K8 Ckk=-0.87262 vs truth-0.34090
(unweighted-0.53360), with a pointwise residual interval excluding zero; most
component intervals are broad and these are not simultaneous tests. Any next
sampling extension should reuse the first eight draws and fixed models, not
refit the classifier.

## Executed K64 extension — 2026-09-30

[pt4n8f9t](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/pt4n8f9t),
`Does Cij converge with more draws? | frozen FiLM | K=8 to 64 | raw 1110`,
finished. Authenticated config, exact K-indexed history and uploaded JSON were
inspected. All16 workers completed7616128 samples (952016 reused,6664112 new).
Inherited K<=8 reports reproduce; zero fits and zero policy updates.

| K | Unweighted error | Reweighted error | R-U paired95% interval | Candidate ESS | Largest candidate mass |
| ---: | ---: | ---: | --- | ---: | ---: |
|8|0.578324|1.185320|[0.24162,1.54193]|4233.52|0.5666%|
|16|0.577080|0.957707|[0.10520,1.02254]|7366.08|0.4168%|
|32|0.570930|2.837997|[0.00809,6.57470]|478.44|4.5076%|
|64|0.573871|1.647877|[-0.06271,3.44567]|1767.30|2.3243%|

Disjoint panels localize the large excursion to human-numbered draws25–32:
that K8 panel has weighted error9.00462 versus unweighted0.55570. The separate
draws33–64 panel has weighted error0.641127 versus unweighted0.577253. All
eight disjoint K8 panels have worse weighted point errors. These are
descriptive comparisons, not selection rules for discarding unfavorable draws.

K64 candidate ESS fraction0.02320%, event ESS1691.23 (1.42117%), top1%
candidate mass57.2282%. Conditional MC SE vector norm rises from0.67363 at
K16 to2.32138 atK32, then1.23890 atK64; the heavy-tail delta approximation
and event bootstrap cannot certify convergence or include unseen-tail/model
uncertainty. The K64 R-U interval crosses zero: no significant improvement,
and not a conclusive worsening under this particular interval either.

Supports: substantial raw-ratio tail sensitivity; the earlier K1–K8 downward
trend was not evidence of stable convergence. Rules out K64 sampling alone as
a demonstrated successful correction. Unresolved: legitimate rare support
versus classifier ratio overestimation, and asymptotic residual bias. Next
read-only diagnostic should trace the largest weight/influence candidates in
the saved panel (especially draws25–32), their conditions, logits and per-Cij
contributions; any leave-out/capping analysis is sensitivity only, not an
unbiased replacement estimator. No additional training was launched.

## Next prepared experiment: saved-panel tail attribution

Implementation (not a NERSC result): `scripts/diagnose_tau_sampling_tail.py`
with `config/conditional_tau_tail_attribution.yaml`. The source is pinned to
completed `pt4n8f9t` / `sampling-3c003b208f`. No checkpoint loading, classifier
fit, generation, DGPO update, or change to the saved panel. Uses the entire
119002-condition population, K32 and K64, with the original raw/unweighted
Cij matrices, errors and ESS verified against the completed report.

Question: which candidates/regions drive the raw-ratio excursion, and does
the rest of the observed signal remain useful when their leverage is reduced?

1. Rank by candidate mass, Cij influence-vector norm, and each component's
   absolute influence. Save the union of top50 per ranking with source ID,
   human-numbered draw, category, visible-pT, signed kappas, logit, tau
   directions, deltas, visible four-vectors, truth Cij, weighted contributions,
   and exact leave-one matrix changes. Report eight-draw block contributions
   so draws25–32 can be localized without selectively hiding them.
2. Compare truth/proposal/reweighted bin masses in fixed tau-direction
   summaries: 4x4 joint cos(theta), 4x4 joint phi (periodic), eight opening-cos
   bins. Repeat globally and in decay-category x saved fit-only pT quartiles.
   Report both conditional probabilities and global masses, including the
   condition-mass drift. Required coarse ratio is truth/proposal bin mass;
   applied coarse ratio is reweighted/proposal bin mass. Zero proposal support
   gives null ratios, never fabricated finite values. Top candidates link to
   their coarse-region report. This is a projection diagnostic, not proof of
   a correct high-dimensional ratio or a complete tau four-vector closure.
3. Keep raw as primary reference. Compare absolute raw-ratio caps10/30/100/
   300/1000 and logit multipliers0.9/0.75/0.5; unweighted is a separate baseline.
   These caps depend on the fixed classifier's absolute logit calibration.
   Also remove top1/10/100/1000 generated candidates by mass/influence as a
   diagnostic only, keeping the original full truth target. None is selected,
   written back to a model/config, or called an unbiased correction.

Uncertainty: 500 common event-cluster bootstrap resamples for all Cij arms,
paired truth/candidates and identical resamples across K. Histograms use
pointwise paired event-cluster delta-method intervals. Mark bins sparse when
truth count or weighted bin-level event ESS is below20; report candidate
counts separately from supporting events. These intervals omit unseen tails,
classifier-fit uncertainty and multiplicity, and cannot validate a test-set-
selected cap. No automatic pass/fail or best hyperparameter is chosen.

Interpretation: large leave-out shifts establish observed leverage, not bad
events. Reliable coarse overshoot supports investigating ratio overestimation
in that projection; sparse/undersupplied bins support a sampling/coverage
branch but do not establish the full cause. Robust transformed-arm improvement
would justify a separately trained regularization ablation and independent
validation, not adoption of the apparent best cap on this reused population.

Run on the existing NERSC environment:

```bash
shifter python3 -u scripts/diagnose_tau_sampling_tail.py \
  config/conditional_tau_tail_attribution.yaml
```

This CPU-only analysis reuses the completed16-GPU panel; no Ray or idle GPUs
are required. Allow several GB of host RAM for the full arrays. Native numeric
thread pools are capped at16. W&B creates a new run with per-arm curves, raw
coarse-region / top-candidate / leave-out tables, a compact sensitivity figure,
and full JSON reports. The original source remains unchanged; each launch
gets a fresh output directory. Partial K32 JSON is saved before K64 completes.
`--no-wandb` is available for offline local tests. Real NERSC files were not
executed locally; synthetic fixtures test endpoint replay, clustered
uncertainty, tail transforms, source identity and source non-mutation.

## Executed tail attribution — 2026-09-30

[12nni684](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/12nni684),
`Why do tails move Cij? | frozen FiLM | K32 and K64 | saved samples`, finished.
Authenticated config, per-arm history and full uploaded JSON inspected.
Original endpoints verified; no fits, policy updates or regenerated samples.
Output: `conditional_tau_tail_attribution/tail-40a858e4bb` on NERSC scratch.

| Arm | K32 Cij error | K64 Cij error | K64 error minus unweighted, pointwise paired95% CI |
| --- | ---: | ---: | --- |
|unweighted|0.570930|0.573871|reference|
|raw|2.837997|1.647877|[-0.06271,3.44567]|
|cap10|0.375997|0.350466|[-0.35198,-0.06828]|
|cap30|0.325115|0.311894|[-0.36660,-0.11182]|
|cap100|0.488227|0.457971|[-0.24034,0.04075]|
|alpha0.75|0.609144|0.524511|[-0.19158,0.16885]|
|alpha0.5|0.571623|0.575860|[-0.09558,0.09643]|

Cap10 and cap30 also have negative K32 paired intervals. Cap30 lowers the
observed K64 Cij error45.65% versus unweighted; event ESS rises1691→63236.
This supports useful observed Cij correction after limiting tail leverage,
not only reduced damage relative to raw. Alpha0.5 has even larger event ESS
(84760) but no Cij improvement: ESS alone is not the objective or explanation.

Top candidate is source `1:9175524:0:4439909`, human draw31, category22,
kappas(0.41,0.41), logit12.248879. It has4.5076% total mass atK32 and2.3243%
atK64. Exact deletion sensitivity changes Cij error2.838→0.694 and1.648→0.625,
respectively. The largest excursion is therefore directly attributable to
this observed candidate's leverage, but deleting it alone does not beat
unweighted. Removal is not an unbiased estimator or a reason to discard the
underlying event.

Its category22/pT-bin1 coarse joint-costheta bin needs an empirical global
mass ratio0.9995 atK64; observed raw weighting applies4.734. Joint-phi ratio
is1.0109 needed versus5.314 applied. Both bins have weighted event ESS<2,
so their broad pointwise intervals cannot establish high-dimensional ratio
miscalibration or exclude genuine rare substructure inside the coarse bins.
They show observed coarse overconcentration, not a proof of its population cause.

Important tradeoff: mean base-stratum-weighted tau-histogram TV atK64 is
0.005538 unweighted,0.025379 raw,0.015835 cap30. All tested transformed arms
remain worse than unweighted on this coarse diagnostic (point estimates).
Cap30 improves the Cij norm, not every matrix component or complete conditional
tau alignment. K32/K64 are nested, not independent replications; cap30 was
identified among multiple tested arms on an already inspected population.

Decision direction: do not deploy the apparent best cap or claim closure.
Freeze a candidate cap protocol and confirm on conditions not used for its
selection before changing training. The evidence prioritizes tail leverage /
ratio reliability over another architecture change or blindly increasing K;
underlying density-ratio bias versus rare-support sampling remains unresolved.

## Prepared fixed-cap confirmation (not yet executed)

The configured filtered training population is entirely assigned to classifier
fit/early-stop splits by `join_pools`; the119002 validation conditions have
already been inspected during cap selection. No independently unused filtered
condition population has been verified in the current configuration. Therefore
the next implemented test is **Monte Carlo replication on the same conditions**,
not independent-event generalization. Do not relabel a new sample seed as a
new condition holdout.

`config/conditional_tau_cap_confirmation.yaml` freezes cap30 chosen after
`12nni684`, generator raw step1110 and classifier `pzq0nl1i`, DDIM20, all119002
conditions, K64,16 GPUs, batch1024 (feature microbatch256). It generates all
7616128 candidates afresh, with zero reused candidates, fits or policy updates.
Only three analysis arms: unweighted, raw and cap30. No cap sweep, tempering
grid, refit or selection from the new result.

The previous seed rule is930481+1000003*j+rank with j0..63. New seed64930673
uses the next64 stream indices; preflight checks every rank/candidate seed is
disjoint, all saved inputs are identical, and model/preprocessing/sampling
paths and settings are unchanged. Historical feature/score replay is performed
by the same production inference worker before new samples are accepted.

Primary criterion: capped-minus-unweighted Cij Frobenius-error difference <0
and its paired event-bootstrap95% upper bound <0. Report all nine components,
residual intervals, event/candidate ESS, maximum weight mass and the exact
leave-one influence of the leading candidate. Coarse tau histograms do not
veto this Cij-focused test. A pass supports sampling replication only; it is
not proof of independent-event generalization, an unbiased ratio, complete
conditional alignment or successful DGPO transfer. Bootstrap fixes trained
models and cannot account for unseen tails. A genuine unused-condition test
remains a later requirement before treating the selected cap as validated.

User command on the existing16-GPU Ray allocation:

```bash
shifter python3 -u scripts/run_tau_cap_confirmation.py \
  config/conditional_tau_cap_confirmation.yaml
```

New W&B run: `Does capped Cij improvement repeat? | frozen FiLM | cap30 K64 | fresh draws`.
Fresh output under `conditional_tau_cap_confirmation/confirm-<id>`; original
sampling and attribution outputs remain untouched. `--prepare-only` validates
the data/protocol without launching Ray tasks. `--analyze-only <confirm-dir>`
recovers reporting from completed shards with no resampling; it checks the
fixed cap, seed and source identity cannot change during recovery.

Validation:71 local tests pass, including disjoint seed streams (also rejection
of partially overlapping rank seeds), exact input/protocol matching, cap math
and paired intervals against the earlier estimator, full reporting using the
real W&B Summary API and unchanged previous regression tests. GPU sampling and
NERSC checkpoint/data loading have not been run by the assistant.
