# Does the saved conditional tau ratio remove the residual?

Source: completed `443eg16h`, raw step-1110 samples, frozen diffusion features,
six relative-angle inputs, BCE only. Its external 119,002-event test sample
has ESS/N 0.00759. The matched Cij error is 0.5213 before weighting and 1.1106
after weighting (previous head: 2.0595). Thus less damage than the old head is
not improvement over the unweighted population. No policy change is tested.

## Two measurements

1. **Tail/error accounting**, on all source test events. Reuse saved logits;
   report top weights, top contributions to tau/angular/Cij error, signed
   contributions to the *change* in error, and sums by decay category.
   Error-change contributions sum exactly to the difference of error norms.
   These are descriptive decompositions, not causal attribution.
   Remove the highest-weight 1, 10, 0.1%, and 1% events from BOTH truth and
   generated samples, report renormalized moments and target drift. These
   selections use the observed samples and do not define an unbiased fix.
   The full, unchanged raw-weight population remains the primary result.
2. **Fresh two-sample audits**, on the source ratio's external test pool only.
   Identity-hash split 60% fit / 20% validation / 20% final test. The source
   ratio never fitted or selected a checkpoint on this population. Its
   preprocessing remains frozen; event IDs and source scores are not inputs.
   All three new heads are trained from scratch; source classifier weights
   are never loaded. Cached raw step1110 backbone features remain fixed.

Arms run sequentially, each on 16 GPUs with 1024 conditions/GPU:

| Arm | Inputs | Generated class weights |
|---|---|---|
| unweighted_joint | visible + cached tau representation + tau + relative angles | base event weights |
| weighted_joint | identical to unweighted_joint, same initialization | base weights times exp(frozen source logit) |
| weighted_condition | visible condition only | same raw ratio weights |

Truth always uses base weights. Normalize each class to mean one across the
whole training split once; never normalize weights independently per minibatch.
Validation/test normalize classes within their own fixed populations, so BCE
and AUC have balanced priors. No clipping, tempering, MMD loss, resampling, new
candidate generation, ratio refit or generator training. DDP pads the training
index set by at most workers-1 entries; paired labels and weights stay together.

Minimum 1000 optimizer updates before early stopping, patience 50, maximum
400 epochs, cosine 2e-4 to 1e-5. Select the absolute lowest validation BCE;
separately flag whether that selected checkpoint met the minimum budget.
Each arm records training curves and final weighted BCE/AUC, paired-event
bootstrap intervals, and effective support in each split. Test inference is
sharded over all 16 GPUs, with explicit complete identity checks on merge.
The two joint arms also get a paired-bootstrap difference in AUC gap and BCE.
Bootstrap intervals
condition on fitted models and samples, not classifier-training uncertainty;
low weighted ESS limits reliability.

## Interpretation, before running

- A strong unweighted audit and smaller weighted held-out AUC gap / BCE nearer
  log(2) supports removal of some residual visible to these fresh heads.
- A strong weighted joint audit rejects closure in this representation. A
  strong weighted condition-only audit additionally exposes finite-sample
  condition-marginal drift. Do not subtract their AUCs to estimate conditional
  information.
- Near-chance weighted audit is inconclusive if the unweighted control is
  weak, selected fit budget is insufficient, or weighted support is too low.
  Even a successful test does not prove full conditional closure: both joint
  heads share the frozen feature extractor and architecture.
- Tail-removal improvement supports sensitivity to those events, not that
  all remaining density ratios are correct. Persistent error after removal
  points beyond only the most extreme events; it does not identify the cause.
- The audit test split is not used for checkpoint selection. Decisions that
  adaptively reuse it in later experiments are exploratory.

## Run on the user's existing NERSC allocation

```bash
shifter python3 -u scripts/audit_conditional_tau_ratio.py \
  config/conditional_tau_weighted_audit.yaml
```

Optional `--prepare-only` does CPU preflight/tail accounting without Ray or W&B.
Every invocation creates a fresh audit subdirectory and one W&B run. The three
arms have separate metric prefixes, checkpoints, and final test-score files.
Tail JSON, moment tables, and audit endpoint results are uploaded to W&B.
No existing source files or live runs are modified. User submits the job.
