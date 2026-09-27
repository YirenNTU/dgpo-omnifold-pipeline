# H4 classifier design on the fixed 10% policy lineage

## Regularized backbone fine-tuning round

Run `h4clfur1` uses
`dgpo_omnifold_ztautau_10pct_h4_classifier_unfreeze_reg.yaml`: unfreeze the
classifier backbone, retain H4 Fourier late fusion, increase head/Fourier MLP
dropout to 0.25 and AdamW weight decay to 0.001. Newly opened PET and global
embedding parameters use LR 1e-5; head, adapters and input projectors retain
2e-4. Attention dropout stays zero. This bundles unfreezing and stronger
regularization; results cannot isolate the effect of either change.

One cold fit uses 3000 updates on 16 GPUs, validation-best checkpoint selection
and a disjoint final test. The pinned policy and event/noise protocol stay fixed.
Existing training BCE includes dropout; do not interpret its gap to eval-mode
validation BCE as an exact generalization gap.

```bash
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_classifier_unfreeze_reg.yaml
```

Use `--validate-only` with the same command to check the resolved config.
Output root: `/pscratch/sd/y/yiren/Ztautau/h4_classifier_unfreeze_reg_10pct`.

## Current default: original Fourier late fusion

The launcher now defaults to `dgpo_omnifold_ztautau_10pct_h4_classifier_fourier.yaml`
and run `h4clff01`. It retains EveNet's conditional representation plus H4
Fourier late fusion, and disables the added direct logit, topology conditioning,
rest-frame context, theta-pair extras and staged warm-up. Compared with
`h4clf02`, only the direct shortcut changes active architecture; the other
extensions were already disabled. This is a simplification, not a demonstrated
fix for the early plateau or gradient spikes.

```bash
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py --validate-only
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py
```

The default output directory is
`/pscratch/sd/y/yiren/Ztautau/h4_classifier_fourier_only_10pct`.
It pins the step-1110 policy file explicitly. The audit data remain the same
separate held-out validation population used by h4clf02.

For the observed plateau, gradient clipping and initialization analysis, see
`artifacts/c4a91e07_review/H4_CLASSIFIER_PLATEAU_20260919.md`.
The remainder documents the earlier direct-logit arm, still available via
`--config config/dgpo_omnifold_ztautau_10pct_h4_classifier_direct_logit.yaml`.

## Question

Does the H4 audit learn the same held-out truth-versus-policy structure faster
when the periodic basis has a direct linear logit path, instead of reaching the
logit only through the topology encoder, late fusion, and AdaLN-Zero decoder?

This is a pure classifier-only measurement starting weights-only from the
DGPO-finetuned `c4a91e07` policy at step 1110.  The diffusion-pretrained
checkpoint remains only the classifier backbone.  No new late-fusion control
is launched; the existing fixed-policy learning curve is the historical
screening baseline.

## Probe

| Config | Audit change |
| --- | --- |
| `dgpo_omnifold_ztautau_10pct_h4_classifier_direct_logit.yaml` | One cold direct-logit classifier fit on one fixed policy pool |

The terminal execution path generates one K=1 pool from the held-out validation
shard using the fixed step-1110 policy, fits exactly one cold classifier for
3,000 classifier optimizer updates, and exits.  It performs zero reward fits,
zero OmniFold residual iterations, zero reward installations, zero DGPO policy
optimizer steps, and zero policy checkpoint writes.  Reference trust,
tempering, gradient-direction diagnostics, and staleness control are therefore
outside this experiment rather than merely disabled penalties.

`min_steps` equals the hard step cap, so validation-loss patience cannot stop
the classifier early. Validation is recorded every 40 updates; lightweight
W&B training diagnostics are recorded every 10 updates.

The run uses Ray Train DDP with 16 one-GPU workers on four Perlmutter nodes
with four GPUs per node.  The audit classifier global batch is 16,384, with a
1,024-row training microbatch per rank; validation batch 65,536 also shards
evenly across all 16 ranks. No reward classifier batch is executed.

The only point is policy step 0; the policy never updates.  The
historical late-fusion curve first exceeded AUC 0.85 at update 1,280 and
reached about 0.902.  The direct design is promising if it reaches AUC 0.85 by
update 960 and reaches at least 0.89 by update 3,000.  This is a historical
screen, not a simultaneous matched-control estimate.

## Diagnostics

W&B records train/validation loss, validation AUC, global pre-clip gradient
norm, clip scale, cumulative clip fraction, and separate pre-clip norms for:

- direct topology head;
- topology encoder and fusion context;
- context output head;
- decoder;
- PET adapters;
- trainable input projectors.

These norms are observed after distributed gradient averaging and before the
existing global clip.  W&B also records truth/generated training-logit means,
their mean separation, logit RMS, the fraction with absolute logit above 5,
and parameter RMS, gradient RMS, and gradient-to-parameter RMS ratio for the
direct head and topology context.  The logit statistics reuse the training
forward pass; there is no extra classifier evaluation.  Every classifier chart
uses the exact local optimizer update as its x-axis.  These measurements do not
alter gradients or optimizer state.

## Validation and launch

```bash
# Run both commands inside the allocated shell after the 16-GPU Ray cluster is ready.
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py --validate-only

# Launch through the NERSC Shifter image.
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_classifier_direct_logit.yaml \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_classifier_direct_logit_only_10pct/ray_results
```

The launcher rejects the run unless it resolves to the `c4a91e07` step-1110
DGPO checkpoint, `weights_only`, no diffusion-pretrain policy load, the 10%
data/backbone, exactly 16 one-GPU Ray/DDP workers, an evenly shardable audit
batch, disabled reward bootstrap and auto-resume, and an exact one-fit,
3,000-update direct-logit audit.

The valid W&B run ID is `h4clf02`, displayed as
`Does a direct H4 head learn faster? | classifier only | step-1110 fixed policy`.
Its summary must report `classifier_only/classifier_fits=1`,
`classifier_only/reward_fits=0`, and `classifier_only/policy_updates=0`.

## Last-block regularized follow-up (`h4clflb1`)

The full-backbone run `h4clfur1` developed a train/validation BCE gap:
at update 360, train BCE was about 0.634 and validation BCE about 0.753,
while validation AUC remained about 0.516. Restrict the trainable pretrained
body to the last PET transformer block; retain all internal adapters and the
classifier/Fourier heads. Freeze both input projectors, GlobalEmbedding and
all earlier pretrained PET blocks. Frozen modules remain in evaluation mode.

Keep head/topology dropout 0.25, AdamW weight decay 0.001, head/adapter LR
0.0002 and last-block LR 0.00001. Train the fixed 3000-update budget, but
restore the minimum-validation-BCE checkpoint (including early checkpoints).
Final disjoint test data are not used for checkpoint selection. This changes
capacity and selection together; compare the full validation curves as well
as the selected checkpoint. Lower capacity is a hypothesis, not a guarantee
against overfitting.

Use the same pinned step-1110 policy and 10% data on a 16-GPU allocation:

```bash
shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_classifier_lastblock_reg.yaml
```

This uses a separate `h4_classifier_lastblock_reg_10pct` output tree and W&B
ID `h4clflb1`; it does not resume or stop `h4clfur1`. Existing classifier-loss,
validation AUC, gradient/clipping and disjoint-test diagnostics remain enabled.

## Invalid first attempt

Run `h4clfdir` is not classifier-only: it entered the inherited residual-reward
bootstrap before reaching the cold audit. Treat it as a contaminated aborted
attempt and do not use its curves as evidence for the direct-logit design.

## Earlier direct-path evidence

W&B run `bg5av8mr` previously enabled the direct topology logit, but it was a
staged-training bundle: head warm-up, later body unfreezing, dropout and
learning-rate changes, reference trust, TARP, and mostly short cold audits all
changed together.  Many of its audits stopped after roughly 25--320 updates,
whereas the same fixed policy is now known to need about 2,500 updates to reach
its AUC plateau.  That run is therefore provenance, not a clean answer to the
present architecture-only question.  The new probe keeps the installed reward
and policy path identical.  It tests only the direct audit with a
3,000-step budget and uses the existing late-fusion curve as historical
context. The corrected run does not build or install any reward.

## Initialization follow-up

Do not initialize hidden layers to all ones: that makes neurons symmetric and
collapses their initial features to the same direction.  Do not initialize
AdaLN residual gates to one either; that opens the full random residual branch
and removes the pretrained near-identity starting point.

If the direct-logit treatment remains slow, the next one-variable ablation is
a small nonzero output initialization with zero bias.  Choose the weight scale
to give an initial logit RMS around 0.01--0.05, rather than choosing an
arbitrary parameter value.  A separate alternative is the already implemented
head-only topology warm-up followed by opening the context and adapters.  Do
not combine either initialization change with the present shortcut test.
