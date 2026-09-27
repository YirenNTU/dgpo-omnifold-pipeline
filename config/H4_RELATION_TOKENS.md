# EveNet whole-event relation classifier

User-authorized experiment: train a classifier initialized from the 10% pretrained
EveNet to distinguish step-1110 generated samples from the clean validation truth
pool, using 16 GPUs. No diffusion/DGPO policy updates.

## Architecture and scope
- Reuse EveNet/PET joint visible/candidate encoding and GlobalEmbedding context.
- Four learned 128-dimensional relation latents; two decoder rounds, four heads.
- Each round cross-attends to all valid PET visible tokens plus both candidate
  tokens, then self-attends among relation latents, then applies the FFN.
- Context uses existing AdaLN-Zero. Normalize each relation, mean pool, linear
  scalar classifier. No candidate or engineered-feature bypass to the output.
- Use output weight normal initialization with std .02 for this opt-in arm only:
  constant initial latents and zero AdaLN gates combined with a zero output head
  would block informative initial gradients. Legacy initialization is unchanged.
- No pair-angle Fourier features, physical projections, fixed masses/energies,
  rest-frame branch, physics losses, smoothing, decorrelation or shuffled negatives.
- The architecture encourages information aggregation but does not guarantee
  high-order structure learning or suppress all fine-angle information.

## Distinct checkpoint roles
Classifier initialization:
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`

Generated samples (weights-only, fresh experiment state):
`/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt`

Truth:
`/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`

The 10% refers to pretrained lineage, not truth-pool subsampling. Regenerate K=1
with the inherited sampler, seed and 16-rank geometry. Reuse the event-identity
split protocol; no claim of byte-identical regeneration without checking exports.

## Training and evaluation
Matched h4pair01 fit settings: one cold fit, balanced BCE, minimum 1000 / maximum
3000 updates, ten effective epochs validation patience, min_delta .001, constant
LRs (head 2e-4, decoder/adapters 5e-5, last PET block 1e-5), dropout .25, weight
decay .001, clipping 5. Preserve frozen input projectors and earlier body settings.
Checkpoint selection uses early-stop validation BCE only. Export restored-best
classifier and disjoint test scores for BCE/AUC, Brier/ECE, ESS and weight tails.
AUC reduction alone is not an improvement. Reaching the cap is not saturation.
This changes depth, readout, initialization and features; it is a new architecture
pilot, not a Fourier-only causal ablation. No across-seed stability claim.

Next structural diagnostics on held-out data: matched-condition pairing shuffle,
condition compatibility, controlled small-angle score sensitivity. These are
separate diagnostics, not modified BCE negatives or proof of physics closure.

## Launch
`python scripts/train_h4_relation_tokens.py --validate-only` checks config.
`shifter python3 scripts/train_h4_relation_tokens.py --check-only` checks remote
checkpoint/data/manifest and fresh output directory.
`sbatch NERSC/submit-h4-relation-tokens.sbatch` requests four nodes / sixteen GPUs.
W&B ID h4rel01; isolated outputs /pscratch/sd/y/yiren/Ztautau/h4_relation_tokens.

## Implementation / submission record
Local validation: 42 targeted tests plus 218 regression tests and 66 subtests
passed. The pre-existing unrelated best-decay trust test
`test_best_point_restart_opts_inherited_reference_in_at_age_zero` was excluded.
Remote checkpoint/data/manifest preflight passed. Submitted Slurm job 58728253
from isolated source `/pscratch/sd/y/yiren/Ztautau/h4rel01_code`, four GPU nodes,
16 GPUs, four-hour limit. Submission is not evidence of completed training.

User requested cancellation of job 58728253. Cancellation was accepted and the
job disappeared from squeue. All future job submissions are user-owned; the
assistant prepares code/configuration/commands only.
