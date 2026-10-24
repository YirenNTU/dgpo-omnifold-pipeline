# TTbar General and reconstructed-top oracle

Updated batch defaults (2026-10-23): generation1024, classifier2048 and future
DGPO512 events per GPU. Use
`config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5_batch1024.yaml`
for a new launch with these settings (run02 output). The original run01 config
and partially generated cache are preserved. K8 generation is sequential in K
(`rollout_chunk_k=1`), so each forward has1024 events rather than8192 candidates.
No process is restarted automatically and no memory-fit claim has been verified
on the four-GPU allocation for the new batch size.

Classifier fitting now loads only condition, truth/generated features and labels
into contiguous CPU arrays once per worker. It preserves cache order, selection
partition, candidate0 BCE, dtype, global shuffle, rank slicing and padded sample
weights. Validation copies only candidate0 before selecting rows; K8 physics
evaluation remains unchanged. Each training batch then does RAM indexing with no
cache file opens. The current full training features require about3.3 GB per
worker (about13.2 GB over four workers), plus selected validation and batch memory.

This loader-only change leaves saved config, normalization, model, optimizer and
RNG state compatible. After manually stopping the old fit, resume without rerunning
generation:

```bash
shifter python3 experiments/ttbar_general_classifier/run.py config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5_batch1024.yaml fit --resume
```

Resume uses the last completed epoch in `classifier-last.pt`; an incomplete epoch
is replayed. Already running Python workers need restarting to use the new loader.

## Prepared experiment

Use `config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5.yaml`.
The policy is the full-MG5 visible-FiLM diffusion, pinned to completed epoch115
(validation loss 0.5342). NERSC checkpoint inventory checked 2026-10-23;
epoch162 is the latest checkpoint, while `last.ckpt` points to an obsolete epoch11.
The policy and its normalization remain frozen. Assignment and EMA stay disabled.

The General head reproduces the Ztautau General FiLM head, with ttbar observed
conditions, frozen clean diffusion features, and ordered neutrino geometry.
Truth and source b/lepton assignments enter oracle/evaluation only.

Candidate top = reconstructed b + reconstructed lepton + predicted neutrino.
Reference top = the SAME reconstructed b + reconstructed lepton + MC neutrino.
Rank each complete two-leg candidate by mean squared normalized p4 difference.
Fit scales on training reference only: one energy RMS and one shared Cartesian
RMS across x/y/z and both legs. Freeze scales before validation. With this common
visible reference, visible p4 cancels in the difference; this is not a separate
top-mass-distance objective and does not enforce an on-shell top mass.

Within each event, General scores define normalized candidate weights q. The
evaluation uses w = 0.7 q + 0.3 one_hot(nearest candidate), with sum(w)=1.

## Manual commands

In the prepared four-GPU environment, from `~/ml_pipeline`:

On a single node the launcher starts a local Ray runtime automatically; no
separate `ray start` is needed. If `RAY_ADDRESS` is explicitly set, it connects to
that cluster instead. Cache must complete successfully before running fit.

```bash
shifter python3 experiments/ttbar_general_classifier/run.py config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5.yaml cache
shifter python3 experiments/ttbar_general_classifier/run.py config/ttbar_general_oracle/classifier_reco_truthnu_full_mg5.yaml fit
```

One seed, full MG5 training population, classifier batch1024 per GPU. Cache uses
one generated candidate per training event and eight per validation event.
The separate `evaluate` stage consumes saved caches and requires ROOT/RooUnfold.
It can run locally after downloading validation caches, classifier checkpoint,
normalization and manifest. No jobs are submitted by these preparation steps.

Compare raw, General, and General+oracle. Neutrino pT/eta/phi and top mass use
RooUnfoldSvd kreg5; native nu2flow joint bins use kreg7. See config for bin edges,
explicit absolute dilepton-phi convention, split definitions, and developmental
validation limitations. Retain closure, data and response-MC uncertainty, ESS and
conditional spread, not only total uncertainty.

For the original **nu2flow uncertainty definition**, use the separate
`evaluate_native_nu2flow.py` adapter; see [NU2FLOW_ALIGNMENT.md](NU2FLOW_ALIGNMENT.md).
It directly calls EveNet's observable, unfolding, and plotting functions and
plots predicted-neutrino count uncertainty divided by the truth-neutrino
reconstruction uncertainty. It draws one candidate per event from the K8 weights
with a shared seed, preserving the original single-entry histogram interface.
The ordinary `evaluate` stage above is an all-candidate weighted extension with
different covariance, sample splitting, and signed-phi treatment; its shape RSS
must not be presented as the original nu2flow metric. Both paths retain the
current cache's common selection, which differs from the historical notebook.

## DGPO integration

The General70 + reconstructed-top oracle30 DGPO adapter is now prepared in
[`../ttbar_dgpo_general/README.md`](../ttbar_dgpo_general/README.md).
It uses the native optimized EveNet trainer, frozen epoch115 feature model,
log probability mixture and unscaled leave-one-out advantages, with periodic
General refits and checkpointed reward/reference pairing. The dedicated config
uses16GPUs ×512events/GPU and K8. Do not launch the legacy additive-reward config.
Training remains a user-launched stage.
