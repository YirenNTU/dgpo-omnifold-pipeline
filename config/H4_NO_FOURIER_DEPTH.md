# No-Fourier attention-depth pilot

## Evidence and question

At retrieval on 2026-09-20, h4clfpath1 was running at step 970, validation
AUC 0.818955. At step 800, same-panel CV readouts were Fourier encoder
0.897984, decoder 0.500521, fusion 0.819012. Fixed-lambda scores can strongly
underestimate Fourier readability. These do not prove that decoder contains
no nonlinear information or that more attention is the solution.

The existing decoder already runs candidate self-attention, visible-memory
cross-attention, then FFN. A second block permits another candidate interaction
after the first block's visible-context update. The pilot tests whether this
extra depth helps without engineered pair features. It does not add a standalone
small model or change the backbone fine-tuning policy.

## Launch

On an allocated 16-GPU NERSC Ray cluster, run these separately, sequentially:

```bash
shifter python3 scripts/train_h4_no_fourier_depth.py --layers 1
shifter python3 scripts/train_h4_no_fourier_depth.py --layers 2
```

Append `--validate-only` for local config checks without training.
IDs are h4nfd1 and h4nfd2, resume never. No job is submitted by implementation.
Old historical classifier runs are not a matched control: budget, clean data
and fine-tuning setup may differ. The one-layer arm is a new reference, not a
claim to recreate the old classifier exactly. Two arms are needed to attribute
differences to decoder depth rather than removing Fourier.

Both arms: c4a91e07 step1110 weights-only; same clean data, seeds, 16 workers,
head LR 2e-4, adapter/decoder LR 5e-5, backbone LR 1e-5, constant AdamW,
1000 minimum updates, ten-epoch BCE patience, no maximum. One cold classifier
audit only, no policy update or reward fit. Existing classifier-only audit pool
and identity splitting are retained. Pair features, Fourier, direct topology
logits and rest-frame features are disabled in both audit and builder settings.

## Decision and diagnostics

Predeclared promising pilot: depth2 final independent test AUC exceeds depth1
by at least 0.02 and test BCE is no worse. Secondary: optimizer updates to
validation AUC 0.70, plus AUC/BCE at 100/200/400/800/1000 shared steps.
Threshold 0.02 is a practical pilot gate, not a significance claim. Replicate
seeds before claiming improvement. Never treat near-chance undertraining as closure.

Monitor decoder readout, head panel AUC, per-block gate/gradient metrics,
parameter updates, clipping and train/validation gap. Absent Fourier/fusion
curves are intentional; branches_complete now checks the modules actually
present. Representation diagnostics remain enabled, Fourier-path CV disabled.
Compare optimizer steps and GPU time: a second block costs more and changes
capacity and RNG consumption; same seed does not imply identical all-layer
initialization or dropout streams. This cannot isolate attention from its FFN.

If both arms fail, it does not prove Fourier is necessary. If depth2 succeeds,
it supports extra decoder depth under this training protocol, not a solved
DGPO gradient interface. Do not launch a depth sweep automatically.
