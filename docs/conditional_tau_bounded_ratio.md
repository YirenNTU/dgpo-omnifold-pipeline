# Training-time ratio bound versus post-hoc cap30

One new FiLM classifier, not a new architecture or diffusion update. Control is
completed pzq0nl1i, evaluated with cap30 on the same fzekzrmr K64 panel. The old
classifier is not fitted again. This is NOT the fresh-negative m0nfa76x protocol.

## Fixed protocol

Reuse the exact prepared inputs, identity splits, masks, normalization and raw
step1110 frozen backbone features. Both completed filtered manifests are checked
by the inherited prepare path. Original train population416701 (fit354488,
internal validation62213); external119002. Fixed K1 training negatives, same
seed42 latent parameter initialization, 3 FiLM blocks, hidden128, dropout.05.
16 GPUs x1024 paired conditions, AdamW2e-4, weight decay.001, cosine250 epochs
to1e-5. Lowest internal validation BCE selects best.pt. Patience25, min_delta1e-4,
minimum1000 updates, inherited unchanged. No Cij-based selection or new features.

## Intervention

Replace latent z output by s=log30-softplus(log29-z). Feed s, NOT z, into the
existing paired BCE, validation metrics, checkpoint scoring and reweighting.
Then r=exp(s)=30*sigmoid(z-log29), bounded by30, with r(0)=1. Only the final
parameter-free map changes. Latent parameter tensors match the original seed;
initial predictions are not exactly identical because the output map changes.
The derivative ds/dz=1-r/30 smoothly decreases near the bound. There is no
hard-clamp zero-gradient region, although floating-point saturation is possible.

With balanced classes and unrestricted functions, constrained BCE targets
min(p/q,30), approached at the upper boundary. Finite model/optimizer behavior
need not attain it. The resulting globally normalized distribution is biased:
q*min(p/q,30) is proportional to min(p,30q), not p. This is not a claim that the
true ratio is bounded. No event deletion, conditional normalization, MMD,
tempering or additional penalty is introduced. A bounded raw ratio does not
guarantee a safe normalized maximum weight or high ESS if mean ratio is tiny.

## Diagnostics and decision

Existing BCE/AUC, ratio ESS/max mass, MMD and conditional tau diagnostics remain.
Add validation truth/generated fraction with r>=27, maximum ratio, and mean
output Jacobian to expose saturation. Checkpoint metadata preserves ratio_bound
so every build_classifier inference path reapplies exactly the trained mapping.
Historic checkpoints without the key retain their original function.

After fitting, inference uses16 GPU shards, including the original K64 saved
candidates (no DDIM generation). Replay historical scores on each rank's first
batch before scoring the new head. Verify old unweighted/raw/cap30 endpoints.
Primary: bounded-training Cij Frobenius error minus old-posthoc-cap30 error,
paired2000 whole-event bootstrap. Also compare with unweighted, report all nine
component errors and pointwise intervals, fixed category x fit-only-pT Cij
groups, ESS and maximum mass. Group comparisons are descriptive, not formal
multiple-testing claims. Previously inspected test identities make this an
exploratory study, not a new independent-event confirmation.

Do not call lower BCE, higher ESS, or beating old raw alone a success. The bound
must improve on post-hoc cap30 and not conceal material component/group damage.
If it fails, training-time bounding adds no demonstrated benefit beyond existing
cap30; it does not prove all tail-robust objectives fail. No automatic deployment.

## User launch

On the existing16-GPU NERSC Ray cluster, after updating ml_pipeline:

```bash
shifter python3 -u scripts/run_tau_bounded_ratio.py \
  config/conditional_tau_bounded_ratio_10pct.yaml
```

Append `prepare` for a read-only input/protocol check without Ray or fitting.
Default `train` creates a new W&B run and unique bounded-* output. It trains only
the bounded head, then runs the paired K1 and fixed K64 endpoints. Existing
sources and checkpoints are not overwritten. CUDA/NERSC execution is user-run;
local tests exercise CPU math, configuration, worker path and checkpoint reload.

## Recover completed results after the W&B summary error

Run a7mczoed completed training and K64 scoring, then hit the W&B0.19
`SummaryDict.update` keyword-argument incompatibility. The launcher now passes
a mapping. Its saved `fixed_K64_report.json` and `fixed_K64_groups.json` can be
published without repeating training, inference, or bootstrap computation:

```bash
shifter python3 -u scripts/run_tau_bounded_ratio.py \
  config/conditional_tau_bounded_ratio_10pct.yaml report \
  --directory /pscratch/sd/y/yiren/Ztautau/conditional_tau_bounded_ratio_1110/bounded-3a83f9b24c
```

This verifies the experiment metadata and replays the saved control endpoints,
then creates a separate report-only W&B run linked to the original run ID.
It uploads both K64 reports, summaries and Cij/group tables. No Ray cluster or
GPU allocation is needed: it only republishes the existing16-GPU results.
The original failed run and its files remain unchanged; only the new
`report-recovery-*` subdirectory receives a reporting `COMPLETE` marker.
This does not resume training or declare the original training pipeline complete.
