# Lower caps and explicit analyzer products

Status: implemented for user-launched NERSC runs; no production job submitted.

## Why these two tests

On the fixed `fzekzrmr` K64 panel, saved context256 (`zrv2yfgt`, trained
bound30) has Cij Frobenius error 0.256479, versus unweighted 0.565440.
Saved Geometry (`n3jjqcn7`, visible p4 + six analyzer coordinates) has error
0.341437. The `gzev9j0e` factor-swap diagnostic did **not** resolve mass versus
within-event ordering as a unique cause: its two primary simultaneous bands
both included zero. These are fixed-model, inspected-panel observations.

The new tests separate two questions. They do not assume the remaining error
is necessarily a tail problem or an inability to learn products.

## A. Lower cap versus generic shrinkage at matched ESS

```bash
shifter python3 -u scripts/diagnose_tau_cap_ess_ablation.py \
  config/conditional_tau_cap_ess_ablation.yaml
```

No model training, generation, or inference. This is CPU arithmetic on the
completed16-GPU K64 panel, using16 CPU threads. No Ray connection is required.
It creates a new W&B run and a unique output under
`/pscratch/sd/y/yiren/Ztautau/conditional_tau_cap_ess_ablation/`.

- Freeze the **same context256 bound30 classifier** and its119002 x64 scores.
- Caps:30,20,10,5. The cap30 arm is the original bounded model, **not** a
  recovered unconstrained classifier.
- Weights: globally normalize `base_weight[i] * min(exp(score[i,k]), cap) / K`.
  Never normalize each event separately and never remove truth events.
- For each lower cap, compare a mixture
  `w_mix = (1-lambda)*w30 + lambda*w_unweighted` at the **same candidate ESS**.
  `w_unweighted` retains original event base weights. Lambda is determined
  from weights alone, without looking at truth or Cij. Event ESS is reported
  separately and is not constrained to match.
- Nonuniform base weights are supported, including nonmonotone mixture ESS.
  If a target ESS is unreachable, record that explicitly; do not fake a match.

Primary family: all six total-Cij-error contrasts (cap20/10/5 minus cap30,
and cap20/10/5 minus the corresponding ESS-matched mixture). Use2000 paired
whole-event bootstrap draws; all candidates and paired truth stay together.
Log pointwise intervals plus approximate simultaneous bands over all6
contrasts and all54 component contrasts.

Also report all9 components, diagonal/off-diagonal error, candidate/event ESS,
fraction of candidates clipped, original mass above the cap, unnormalized
ratio mass removed by clipping, and decay/category x fit-defined pT mass TV
and closure. Group summaries are descriptive, not subgroup significance tests.

Decision rules:

- Lower cap beats cap30 **and** its ESS-matched mixture: supports selectively
  suppressing the tail, beyond generic shrinkage at that ESS.
- Both help similarly: generic shrinkage is sufficient for this observed gain;
  does not identify a unique tail-estimation failure.
- Lower caps harm diagonals or total closure: current tail also carries useful
  correction; lower cap is not a general solution.
- Bands unresolved: retain uncertainty, not a claim of equality.

No automatic cap selection or production change. Bootstrap conditions on fixed
models, candidate panel and fitted lambda; no training uncertainty or correction
for prior test inspection. Matched ESS does not imply equivalent distributions.

Output: `cap_ess_report.json`, two W&B figures, group/component tables, local
`event_weights_and_bootstrap.npz` (event masses and numerators, not individual
candidate weights). Verify saved cap30 and unweighted endpoints before interpreting.

## B. Nine explicit analyzer products, one new classifier

```bash
shifter python3 -u scripts/run_tau_product_inputs.py \
  config/conditional_tau_product_inputs_10pct.yaml
```

Optional preflight only (no Ray, training or W&B writes):

```bash
shifter python3 -u scripts/run_tau_product_inputs.py \
  config/conditional_tau_product_inputs_10pct.yaml prepare
```

Uses the existing Ray cluster (`RAY_ADDRESS`, or automatic discovery), **16 GPUs,
1024 paired conditions/GPU**. Global nominal batch:16384 conditions, each with
one truth and one generated input, equal class weight. Final short batches and
DDP padding follow the unchanged historical trainer.

The matched control is **Geometry n3jjqcn7**, not plain context256: Geometry
already supplies the six analyzer directions and visible p4. The intervention
adds only nine outer-product coordinates:

```
[a_k*b_k, a_k*b_r, a_k*b_n,
 a_r*b_k, a_r*b_r, a_r*b_n,
 a_n*b_k, a_n*b_r, a_n*b_n]
```

These are bilinear/outer products, not the3-component vector cross product.
They have no factor9, no division by analyzing powers, no stored Cij labels,
and no empirical truth target. They are a **physics-aware representation**,
not a physics-agnostic ablation. Truth features use the truth candidate's own
tau; generated features use only that generated candidate and shared visible
information. Evaluation still uses the existing signed analyzing powers and
saved Cij convention, identically across all arms.

Architecture and training:

- Same condition256 FiLM head, candidate width64,3 blocks, bound30 BCE,
  dropout, AdamW,250-epoch cosine schedule, early stopping and data splits
  inherited from the completed Geometry experiment.
- Fresh head with the same seed and shared initial parameters; new9 input
  columns start at zero. Initial logits are checked against Geometry. New
  columns can receive gradients immediately. No trained comparator head is
  used for initialization.
- Cached raw step1110 EveNet representations; backbone and generator frozen.
  Preserve original normalization. No EMA, DGPO updates or new generation.
- Frozen training population416701:354488 fit /62213 validation; independent
  filtered test119002. Condition, weights, splits and all old candidate inputs
  must match the saved Geometry arrays exactly.
- Select best held-out **BCE**, not best test Cij. Inherited min1000 updates,
  patience25 epochs, min_delta1e-4; no new stopping or LR tuning.
- Existing Geometry and context256 checkpoints/scores are reused, not retrained.
- K1 evaluation and K64 rescoring both use the new input transform. Inference
  runs on16 GPUs. K64 raw1110 backbone feature extraction microbatch remains256,
  independent of the1024/GPU classifier batch.

Primary: paired K64 total-Cij-error change versus Geometry. Context256 is a
secondary best-existing comparator. Log all9 component changes with **pointwise**
paired intervals, K1 separately, ESS and category x pT closure. Component
intervals are not simultaneous significance claims. Better BCE/AUC alone is
not the success criterion. Do not combine this arm with a lower cap in this round.

If products beat Geometry but not context256, explicit multiplication helped
this input design but did not improve the best available estimator. If the
products receive gradients yet do not improve closure, this rules out these
explicit products alone as a sufficient fix, not every possible architecture.

New W&B display name:
`Do explicit spin products help? | FiLM context256 | bound30 | raw1110`

Output root:
`/pscratch/sd/y/yiren/Ztautau/conditional_tau_product_inputs_1110/`.
Both launchers preserve previous output directories and publish reports to the
existing `nu2flow-RL` project. No toy files are needed or synchronized.
