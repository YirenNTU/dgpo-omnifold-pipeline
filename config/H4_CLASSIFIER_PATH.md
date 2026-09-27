# Fourier-path readability experiment

Run on the allocated 16-GPU Ray cluster from the updated NERSC repo:

```bash
shifter python3 scripts/train_h4_classifier_representation.py --path-readouts --validate-only
shifter python3 scripts/train_h4_classifier_representation.py --path-readouts
```

Run ID `h4clfpath1`; W&B resume is `never` to prevent merging independent fits.
Do not rerun the same ID after interruption without deliberately assigning a
new ID and updating the launcher contract. No job is automatically submitted.

Source remains c4a91e07 step 1110, weights-only. Clean data, 16 workers, model,
initialization, constant group LRs, AdamW and stopping match h4clfrep1. No policy
updates or reward installation. Output paths are isolated under
`/pscratch/sd/y/yiren/Ztautau/h4_classifier_representation_path`.

Detached probes run at initialization, 10, 50, and every 100 optimizer updates.
All six representations use identical event-context-disjoint fit/holdout rows:
raw Fourier, Fourier input LayerNorm output, Fourier encoder, decoder, concat,
and fusion. `fourier` still means encoder output, preserving historical keys.

Raw metrics use prefix `omnifold_live/raw_staleness_audit/stability/representation_probe/`:

- `{branch}/holdout_auc`: unchanged fixed ridge lambda=1.
- `{branch}/cv/holdout_auc`, `fit_auc`, `selected_lambda`, `valid`.
- `{branch}/cv/lambda_{value}/inner_auc`: sensitivity on inner validation only.
- `path_complete`: both newly added locations and existing branches captured.

Initial measurements have the additional `initial/` prefix and explicit step 0.
W&B classifier charts expose both fixed-ridge and CV holdout curves.

Outer partition uses SHA256 byte 0 of exact packed context. Inner two-fold CV
uses byte 1, keeping identical contexts together across ranks and labels. Each
inner fold fits its own standardizer and class-balanced ridge on only its
training half. Select lambda from [0.0001, 0.01, 1, 100] by mean inner AUC,
breaking ties toward larger lambda. Refit standardization and ridge on all
outer-fit rows, then evaluate untouched outer holdout. Fewer than eight rows
of either class in an inner split emits invalid rather than a fake chance AUC.
The final independent audit test is never used by these probes.

Interpretation is descriptive: a low linear score is not proof of absent
nonlinear information; concat cannot literally delete its component features.
Do not tune the model or lambda on the outer holdout. No automated significance
claim or architecture decision is made from AUC differences. Independent
replication is required before attributing a startup delay to normalization.

Probes restore modes, buffers and RNG and never enter the production loss.
CPU solves run on rank zero, with bounded feature gathering and result broadcast
on all ranks. Additional overhead means compare optimizer updates, not wall time.
