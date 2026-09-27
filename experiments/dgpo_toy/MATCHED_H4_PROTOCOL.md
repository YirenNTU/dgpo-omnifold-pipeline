# Same-truth-data H4 on the normally pretrained diffusion

2026-09-21, declared before launch. User requested the EXACT same training data
as diffusion, not the old streaming millions-of-truth-events classifier.

## Round contract

- Objective: train the existing joint3 H4 on the same finite truth rows and
  determine whether residual discrimination generalizes to held-out rows.
- Success measure: verified shared rows, validation-selected raw H4, held-out
  BCE/AUC and ratio diagnostics. No forced low ESS or required positive result.
- Limiting factor: truth-data exposure confounds the previous comparison.
- First-principles claim: removing extra truth examples makes residual detection
  interpretable at the same available truth-data budget.
- Deliverable: fixed paired classifier dataset, raw classifier, report/resume state.
- Time box: validation-BCE early stop, patience20/min_delta1e-4, no step cap;
  expected minutes locally. No unseen data stream or follow-up policy training.
- Delete: new truth sampling, old H4 reuse, new architectures, diffusion updates,
  reward transformations, DGPO and EMA.
- Risks/authority: local new directory only; keep diffusion/dataset unchanged.
- Evidence debt: same data is not matched architecture, feature prior, parameter
  count, optimization objective or compute. AUC alone does not validate ratios.

## Exact sources

Truth dataset: `truth_dataset_seed17_v1/dataset.pt` used by diffusion.
Frozen generator: `truth_diffusion_earlystop_v1/best_model.pt`, raw epoch25/step1600.
Validate its report/dataset lineage. NOT old Gaussian teacher source or EMA.

Read the saved condition/target tensors directly; do not call a truth sampler.
For each row generate ONE negative with the frozen raw DDIM20 model at the
same condition. Save these fixed panels once and reuse for all training epochs:

|Split|Shared truth rows|Generated rows|Distinct conditions|
|---|---:|---:|---:|
|Train|32768|32768|32768|
|Validation|8192|8192|8192|
|Test|16384|16384|16384|

No fresh truth or negative replenishment. Separate fixed negative noise seeds
70017/71017/72017. Test metrics are computed only after classifier selection.
Dataset and diffusion byte content remain unchanged. Exact tensor equality
is checked for every split; no SHA requirement.

## Training and evaluation

Existing JointFourierClassifier: raw condition + existing per-coordinate CDF,
coordinate harmonics1..4 and all signed first-harmonic triple combinations;
width128,272769 parameters. The truth-selected four triples are not supplied,
but the explicit third-order feature prior and known marginal normalization
remain. This is NOT an architecture-matched comparison with the10444-parameter
raw-input diffusion and should not be presented as such.

AdamW3e-4,decay.001,clip1,raw/no EMA. Each shuffled epoch visits every context
once in balanced batches256 pairs (512 labelled rows),128updates/epoch.
At update50, preserve the existing function-preserving encoder-output
standardization using ONLY first2048 training pairs, and clear only head
optimizer moments as in the previous H4 implementation. No validation statistics.

Each epoch logs weighted train BCE, validation BCE/AUC/ESS/weight tails/log
mean ratio/reward means, gradient clipping, LR and epoch/update/stale clocks.
Stop after20 epochs without cumulative improvement>1e-4 from the last significant
BCE anchor. Best raw checkpoint is selected by absolute minimum validation BCE.
No early stopping by AUC, ESS, physics, joint moments or test metrics.
At stop, report selected train/validation/test metrics, preserving test isolation.
High held-out discrimination supports a residual visible to this H4 at the shared
truth-data budget. Near-chance after this protocol is absence of detected signal,
not proof of distribution equality or sufficient classifier capacity/training.

Full model/optimizer/RNG/standardization/stopping/history saves every completed
epoch; resume uses saved fixed negatives. No silent generator or reward updates.

## Launch

```bash
python -u -m experiments.dgpo_toy.matched_h4 \
  --dataset artifacts/dgpo_toy/truth_dataset_seed17_v1/dataset.pt \
  --diffusion-checkpoint artifacts/dgpo_toy/truth_diffusion_earlystop_v1/best_model.pt \
  --output artifacts/dgpo_toy/truth_diffusion_matched_h4_v1 \
  --patience 20 --min-delta 0.0001
```

The earlier streaming strong-H4 result is not the matched control: both
pretraining lineage and classifier data regime now differ. This round fits one
new classifier only. No results assumed before execution.

## Completed result

`truth_diffusion_matched_h4_v1` completed epoch24/update3072, early stop after20
epochs with no significant improvement. Selected raw epoch4/update512, NOT the
overfit last model.107 regression tests passed. Frozen diffusion unchanged;
saved truth/condition tensors equal the original diffusion dataset in all splits.
No new truth events. Runtime16.88s fit/evaluation, excluding negative preparation.

|Selected checkpoint|Train|Validation|Test|
|---|---:|---:|---:|
|BCE|.202194|.432348|.425024|
|AUC|.972327|.897126|.899245|
|ESS/N|.00216503|.00103490|.00053405|
|Top1% ratio mass|.518974|.914441|.922436|
|log mean ratio|-.827918|3.019473|2.953185|

Test has16384 generated rows; ESS fraction.00053405 corresponds to~8.75
effective samples, largest normalized weight.24848. Ratios are concentrated
and not well normalized; discrimination does not establish ratio fidelity.
There is a substantial train/held-out gap already at selection, and later
overfitting: epoch24 train online BCE.011229 versus validation1.570468.
Selection correctly kept epoch4, whose validation and test performance agree.

- Outcome: same-truth-data H4 fitted and frozen, fixed negative panels retained.
- Evidence: held-out test AUC.899245/BCE.425024 using the same finite truth splits.
- Decision: supports generalizing residual discrimination at the shared data
  budget. Extra millions of truth events are not necessary for this observed
  signal. Ratio quality remains poor, not certified for density-ratio fidelity.
- Learning index:16.88s recorded fit/evaluation for one data-exposure question;
  no new policy mechanism tested.
- Deleted: streaming truth/negative replenishment and additional DGPO runs.
- Next limiting factor: the residual/ratio-to-policy interface is not tested here.
- Better next round: preserve this generator/critic/data lineage and do not
  infer high-order-only residuals from AUC: DDIM marginal variance error also
  remains. Architecture/feature-prior/compute matching is still not established.
