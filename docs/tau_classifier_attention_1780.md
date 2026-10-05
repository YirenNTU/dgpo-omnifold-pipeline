# Candidate-query attention at frozen DGPO step 1780

Question: does a particle-resolved, candidate-dependent condition path improve
within-condition Cij reweighting relative to the existing three-block FiLM head?

## Fixed contract

- Pin checkpoint AND runtime from the completed `5a7bf14b` experiment, at
  `tau_classifier_current_dgpo_10pct/fresh-96fminsf/`. Require step 1780.
- Raw actor frozen; independent raw1110 feature backbone frozen; zero actor updates.
- Same filtered 416701 training events and 119002 external validation events.
  Preserve saved normalization, source IDs and internal selection split.
- Generate training K=1 and validation K=8 once per experiment; share identical
  arrays across both fresh heads. Preserve prior runtime seeds/sampler. This
  regenerates samples rather than copying historical arrays; exact cross-run
  bitwise identity is not asserted. Within this run candidates are identical.
- 16 GPUs, 1024 paired events per GPU; original paired BCE, bound 30,
  AdamW/scheduler/early-stop settings inherited from the pinned runtime.
- Select each checkpoint by internal validation BCE, never external Cij.

## Arms

1. Existing three-block FiLM, condition width 256.
2. Same head plus one cross-attention residual BEFORE the three FiLM blocks.
   Query: candidate encoder output (64). Keys/values: shared Linear–SiLU–LN
   projection of normalized visible `x` rows, plus learned slot embeddings.
   Four heads, width 64. Existing feature channels and slot identity retained;
   padding mask applied. Empty events receive zero attention residual.

These are existing packed visible features, **not unpooled PET outputs**. No
feature extraction/cache schema change or truth-derived input is needed. The
existing frozen-backbone candidate features remain unchanged. All global
condition inputs remain in the original context MLP. This is an additional
particle-resolved route around global compression, not new observable information.

Only the final attention residual projection is zero initialized, so the arms
begin at the same function and shared parameters. Internal attention parameters
initialize normally; query/key/value gradients become active after the output
projection starts learning. Extra initialization preserves the shared RNG stream.

## Endpoints and interpretation

Primary: external within-condition reweighted full Cij error vs three-block arm.
Report diagonal, off-diagonal and nn errors, all nine Cij values, candidate ESS
and condition mass TV. Keep global-ratio results separate. BCE/AUC are supporting
metrics, not evidence by themselves that spin correlation improved.

W&B separates `tau/classifier_only/fresh_depth3` and
`tau/classifier_only/fresh_depth3_attention` fits/external scores; Cij probes use
`tau/reweight_fresh_depth3` and `tau/reweight_fresh_depth3_attention`.
Best heads, classifier reports, candidate shards and Cij measurement arrays are
saved in a fresh experiment directory. Existing production outputs are untouched.

A success supports this combined attention/token-path intervention, not attention
alone: parameters increase and representation routing changes. K=8 self-normalized
reweighting is finite-sample; point estimates have no significance claim. It is
not a DGPO training result. No automatic deployment of either head.

## Launch (user, existing NERSC 16-GPU allocation)

```bash
shifter python3 -u scripts/train_tau_classifier_current_dgpo.py \
  config/tau_classifier_attention_1780.yaml
```

Use `--prepare-only` to verify paths/manifests and write the resolved runtime
without starting training. Keep both source checkpoint and runtime above on NERSC.
