# Explicit visible and rest-frame input ablations

Prepared, not launched. Two fresh classifier-head fits; no diffusion/DGPO updates
or new candidates. Reuse completed context256 bound30 run `zrv2yfgt` as A.
The user launches all NERSC compute.

## One question at each step

| Arm | Change relative to preceding control | Primary contrast |
| --- | --- | --- |
| A, saved zrv2yfgt | Existing inputs, condition256, candidate64, three FiLM blocks | Reused, never retrained |
| B, visible | Append eight leg-identified visible E/px/py/pz to condition | B minus A K64 Cij Frobenius error |
| C, geometry | Keep B and add six candidate-owned rest-frame direction projections | C minus B K64 Cij Frobenius error |

B receives observed visible p4 identically in both classes. Its eight features
use fitting-split-only mean/std, without clipping. Original features and their
normalization are untouched. Inventory logs whether each named p4 field appears
in the original packing specification. Missing named energy does NOT prove
energy absent from particle x: B can test both access/leg association and new
information. No certification of full redundancy is claimed.

C adds `(a_k,a_r,a_n,b_k,b_r,b_n)`, computed using exactly the existing analysis
geometry: boost to candidate tau-pair rest frame, then each candidate tau rest
frame; common A helicity basis with `n = r cross k`, fixed +z beam axis in pair
CM. These are visible-direction analyzers, not guaranteed optimal polarimeters
for every decay mode. No inverse analyzing powers, nine products, Cij values,
truth residuals, labels or IDs enter the feature vector. Truth and generated
examples each use their OWN tau reconstructed from their OWN deltas; they share
only observed visible p4. Generator reconstruction remains fixed-energy 45.6 GeV
tau directions, not newly predicted full tau kinematics. This is a physics-aware
representation ablation; neither successful closure nor this input change
independently certifies the adopted physics convention.

## Matched training and inference

- Frozen RAW step1110 backbone, fixed K1 fitting negatives and original features.
- Filtered 416701 train: 354488 fit / 62213 early-stop validation; external119002.
- Original IDs/splits/pair weights are retained; filter completion is checked.
  Packed condition is rebuilt from the saved aligned source pool and compared
  to the exact prepared inputs. Newly calculated geometry must reproduce the
  existing per-event analyzer coordinates. K64 visible p4 and conditions must
  match the external K1 population.
- 16 GPUs x1024 paired conditions per GPU; BCE class totals remain balanced.
- Fresh seed42 heads: condition256, candidate64, three residual FiLM blocks,
  bound30, AdamW2e-4, WD.001, dropout.05, cosine250 epochs to1e-5;
  early stop patience25 / min_delta1e-4 / min_steps1000, unchanged.
- Best INTERNAL validation BCE selects checkpoint, never Cij/test metrics.
- New input columns are zero initialized, preserving original shared tensors,
  random stream and initial function (floating-point roundoff may differ).
  Condition-side columns receive gradients after the zero context projection
  starts learning; the original learned paths are not zeroed.
- New preprocessing/schema travels with best.pt, manifest and materialized
  prepared.npz. K1 and K64 use the identical transformation. Six geometry
  coordinates precede relative6 and tau15, preserving legacy diagnostics.
- K64 extraction/scoring uses16 GPUs, 1024 conditions per scoring batch; existing
  frozen-backbone microbatch256 remains to bound feature-extraction memory.
- A is never retrained. C requires a completed protocol/data-matched B. `all`
  launches B then C sequentially and stops if B fails; it does not consume32 GPUs.

## Reports and decision rule

Online W&B: independent run for each fit; BCE/AUC/LR, early-stop clocks,
ratio concentration, bound saturation, condition diagnostics and explicit-input
weight/last-training-batch gradient norms. These gradient norms are utilization
diagnostics, not proofs of causal feature relevance. Final reports include K1,
K64 Cij matrices/errors and ESS, all nine component errors and pointwise paired
bootstrap intervals, plus fixed decay-category x fit-pT groups and group masses.
Saved comparator endpoints must reproduce before results are published.
The validation joint-MMD diagnostic continues using the ORIGINAL condition
coordinates and tau15; adding features does not silently change its kernel.

B compares with A. C compares with BOTH B (primary) and A (secondary).
Unweighted, old raw pzq0nl1i and old cap30 remain context, not matched input controls.
Use 2000 whole-event paired bootstrap draws, preserving all64 candidates and
truth per event. No event deletion, oracle reweighting or per-condition
normalization. No new fresh-classifier audit is included in this experiment.

A lower primary Cij error with interval below zero supports the tested input
representation. Inspect every component, especially rk/rn/nr, and groups before
claiming broad closure; inclusive improvement can conceal group tradeoffs.
C outperforming B supports usefulness of explicit rest-frame geometry; it does
not uniquely prove a boost-learning bottleneck (capacity/optimization also
change slightly). No improvement means this intervention is insufficient, not
that all conditioning hypotheses are ruled out. K64 success does not guarantee
K1 success. The repeatedly inspected test pool is exploratory; intervals exclude
model-fit uncertainty and do not certify the inherited Cij convention.

## NERSC commands

Update the existing ml_pipeline checkout (use repository upload exclusions).
On the user's existing16-GPU Ray cluster:

```bash
shifter python3 -u scripts/run_tau_explicit_inputs.py \
  config/conditional_tau_explicit_inputs_10pct.yaml all
```

Use `prepare` instead of `all` for read-only input/control checks. To run
separately, use `visible`, then `geometry`. Each training arm creates its own
new output under `/pscratch/sd/y/yiren/Ztautau/conditional_tau_explicit_inputs_1110`
and new W&B run; never overwrites A or old training runs. No local implementation
action submits a NERSC job.
