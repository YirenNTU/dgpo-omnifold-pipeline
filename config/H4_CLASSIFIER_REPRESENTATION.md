# Classifier startup measurement

On the allocated 16-GPU NERSC Ray cluster, from the updated repository:

```bash
shifter python3 scripts/train_h4_classifier_representation.py --validate-only
shifter python3 scripts/train_h4_classifier_representation.py
```

Run ID: `h4clfrep1`. Display name:
`Why does learning start slowly? | H4 branch probes | constant LR | clean validation`.
No job is submitted automatically. The launcher checks the checkpoint, clean
validation directory and two-event exclusion manifest before training. It does
not regenerate data or overwrite the old classifier experiment.

## Fixed protocol

- Source: c4a91e07 old-DGPO policy step 1110, weights-only with fresh experiment
  state and regenerated policy samples; not h4lbnr01 step 50.
- One cold audit classifier, no OmniFold reward install, no DGPO policy update.
- Same H4 Fourier late fusion, last PET block/adapters, AdamW, dropout/decay.
- Constant LR: head `2e-4`, decoder/adapter `5e-5`, backbone `1e-5`; no warmup.
- Same filtered 10% training pool; independent validation is now the verified
  `diffusion_val_20pct_seed42_stic_filtered_test1/val` (119002 events).
- No maximum fit steps; minimum 1000 updates; 10 classifier-epoch BCE patience,
  restore minimum-validation-BCE classifier; separate final audit test remains.
- Normalizers unchanged. Source checkpoint and validation cleaning differ from
  h4clfd1, so historical differences cannot be attributed to measurements alone.

## Measurements and W&B keys

All keys below have prefix `omnifold_live/raw_staleness_audit/` in this run.
The shared fit implementation also supports OmniFold when enabled in its fit
config, but this launcher executes only the classifier-only audit path.

Training hooks sample steps 1–5 and every 10 updates:

- `stability/layer/representation/{fourier,decoder,fusion}/activation_rms/rankmean`
- `stability/layer/representation/{fourier,decoder,fusion}/gradient_rms/rankmean`
- `stability/layer/representation/bank.decoder.blocks.0.modulation/gate_{self,cross,ffn}_rms/rankmean`
- Corresponding gate `_gradient_rms`, and rankmax variants.
- Gate absolute maxima and fraction with `abs(gate)<1e-3` (a diagnostic
  threshold, not a training gate or stopping rule).
- `stability/representation/parameter/bank.output.weight/rms` and output bias,
  decoder modulation parameter RMS, after update.
- Existing `stability/update/bank.output/rms`, `bank.topology_encoder.*`,
  `bank.fusion.*`, `bank.decoder.*` and adapter update keys are retained.

Activation/activation-gradient RMS is per-hook-call averaged over microbatches,
then over ranks, not a parameter-count-normalized comparison between branches.
Activation gradients include loss/microbatch scaling. Parameter updates are
actual post-AdamW displacements, not LR times gradient. Relative update RMS at
zero initialization is ill-conditioned: use absolute RMS there.

Fixed readouts run before the first update and after steps 10, 50, then every
100 updates. The initial baseline is emitted with step-1 logging but lives at
`stability/representation_probe/initial/*` with its explicit `step=0`.
Later results are at `stability/representation_probe/*` with explicit `step`.
If fit recovery starts later, the first probe reports that recovery boundary,
not a fictitious step-0 baseline.

- `{fourier,decoder,concat,fusion}/{fit_auc,holdout_auc,valid}`
- `current_head/{fit_auc,holdout_auc,panel_auc}` on those same panel partitions.
- Per-class row counts, feature dimensions, `branches_complete`, `error`.

The panel takes at most 128 rows per class per rank from **early-stop validation**
(up to 4096 rows at 16 GPUs), never from the final test. Rows are fixed for the
fit. SHA256 of the exact packed event context assigns fit vs holdout, shared
across classes/candidates, preventing identical contexts from crossing sides.
This is a context-identity proxy, not a separately logged source-event ID; it
assumes the deterministic packing contract. It merges identical contexts
conservatively. No RNG is consumed for partitioning.

Each readout is a new deterministic, class-balanced ridge regression on +/-1
labels with fixed lambda=1, unpenalized intercept, and standardization estimated
only on readout-fit rows. Its scores are evaluated as AUC; they are **not calibrated
probabilities or density ratios**. The readout has no tuning/early stopping.
It uses detached CPU features, never backpropagates into the classifier, and
does not affect classifier checkpoint selection. Fewer than eight rows of either
class on either side or non-finite features emits `valid=0`, not chance AUC.
Readouts are diagnostic comparisons, not independent final-test claims. A low
linear-readout AUC does not establish absence of nonlinear information.

Rank 0 solves the readouts after bounded feature gathering, then broadcasts
metrics. Every rank follows the same collective sequence. Probe forwards restore
module modes, buffers and Python/NumPy/Torch RNG, including the unregistered
shared EveNet backbone. Probe errors are logged and broadcast without changing
the training objective. Memory/compute overhead is real, so wall time is not a
fair comparison to diagnostics-off runs; compare optimizer updates.

## Decision rules

1. Gates near zero plus tiny branch updates: supports a delayed gradient-path
   explanation; does not establish that opening gates faster is safe.
2. Strong Fourier/decoder held-out readout but weak actual head on the same
   holdout: information is linearly accessible but current head/fusion has not
   extracted it. Compare concat vs fusion to localize representation loss.
3. Weak readouts and weak actual head: unresolved representation or nonlinear
   readout limitation, not proof of no physics signal.
4. High readout-fit AUC but low readout-holdout AUC: panel overfitting/limited
   sample evidence; do not call it good representation.
5. Train/validation BCE plus final independent audit remain the classifier
   performance endpoints; smoother curves alone are not success.
