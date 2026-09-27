# Complete classifier: fixed Fourier encoder-output scaling

Status: 300-update retry arms completed on NERSC; long-fit follow-up implemented,
not yet run. This is not another frozen-head replay.
Both arms train the existing full classifier for exactly 300 updates on
16 GPUs, from the same c4a91e07 step-1110 policy, clean 10%-lineage data,
initialization, sample generation and seeds. No policy updates or reward fits.

## Single intervention

Control: original encoder output -> concat with decoder -> LayerNorm -> fusion.
Standardized: (encoder output - frozen mean) / frozen std -> same concat/fusion.

At initialization, compute balanced, weighted population moments from the first
up to 8192 classifier-fit rows **per class**, never early-stop validation or
final test. Shard these replicated fit rows across all ranks and all-reduce
float64 sums. Use evaluation-mode encoder outputs, a std floor of 1e-6, restore
all module modes, buffers and RNG, then hold statistics fixed for the full fit.
The encoder remains trainable. This tests initialization-time conditioning;
statistics may drift as it learns. It is not BatchNorm, adaptive normalization,
whitening, a shortcut, or a new objective. No trainable parameters are added.
Dropout remains unchanged. Checkpoint buffers preserve the transform; older
banks without the buffers load as identity. Partial/incompatible states fail.

## Evidence and interpretation

- h4frzcap1 update50: decoder RMS ~1 with 256 dimensions; Fourier encoder
  RMS ~0.186 with 32 dimensions. Combined LayerNorm does not standardize each
  feature over events. This motivates a scaling test, not proof of suppression.
- h4readq1: standardized frozen heads improve held-out BCE over unstandardized
  heads. The ridge-AUC gate is not a density-ratio selection criterion.
- h4readb1: removing minibatch randomness did not preserve high AUC, but BCE
  improved. Full-batch is not the proposed production change.
- h4bceq1: held-out BCE improved only modestly; ESS stayed high. No evidence for
  ratio collapse or a need to add attention/capacity.

Primary endpoint: held-out balanced BCE at classifier update 200. Secondary:
50/100/300, mean of validation BCE evaluations in [200,300], train-vs-val gap,
gradient/clipping diagnostics and Fourier/decoder activation RMS. Interpret
standardized minus control: negative is improvement. Require lower BCE both
at 200 and in the late-window mean for a promising screen; no statistical
significance or convergence is claimed from one seed. AUC is auxiliary only.
Do not compare restored-best final test values as if they were update-300 BCE.
The user selected longer same-seed training next, not independent-seed
replication. No change to DGPO's objective is authorized by this experiment.

W&B includes standardizer min/max std and mean RMS, plus representation
activation/gradient RMS before and after the transform. Initial panels retain
both raw encoder and transformed features. Inspect finite statistics and
post-transform gradients before interpreting a failed fit.

## Run sequentially in the existing 16-GPU Ray allocation

```bash
shifter python3 scripts/train_h4_output_scaling.py --arm control --validate-only
shifter python3 scripts/train_h4_output_scaling.py --arm standardized --validate-only
shifter python3 scripts/train_h4_output_scaling.py --arm control && \
shifter python3 scripts/train_h4_output_scaling.py --arm standardized
```

W&B: control `h4sclc1`, standardized `h4scls1`, both resume=never. Isolated
directories `/pscratch/sd/y/yiren/Ztautau/h4_scale_{control,standardized}300/`.
Sync Python modules, launch script and YAML overlays together before running.
Existing output is not overwritten. This does not interrupt older jobs.

## Retry after runtime budget guard failure

The initial control launch failed before classifier fitting because the trainer
still enforced the inherited LR-replay 1000-update/no-cap guard. The runtime now
accepts exactly 300 updates only for the explicit quick-plateau and output-scale
protocols. Original long-replay and 3000-step architecture guards remain intact;
the LR-replay identity-split path is preserved. Regression tests execute the
actual trainer guard, not merely the launcher's YAML checks.

Sync `dgpo_trainer.py` and `train_h4_output_scaling.py` as well as the new
normalization modules. Start fresh, not with `TorchTrainer.restore`:

```bash
shifter python3 scripts/train_h4_output_scaling.py --arm control --run-suffix retry1 && \
shifter python3 scripts/train_h4_output_scaling.py --arm standardized --run-suffix retry1
```

This preserves the failed output and uses W&B h4sclc1-retry1 / h4scls1-retry1
with separate output roots ending in `-retry1`. All scientific settings and
source paths remain unchanged. Do not reuse a suffix once its fit has started.

## Completed screen and selected follow-up

Both retry arms finished 300 updates, zero policy updates. Live histories:

| Validation metric | h4sclc1-retry1 control | h4scls1-retry1 standardized |
| --- | ---: | ---: |
| BCE at 200 | 0.69327873 | 0.61065078 |
| BCE at 300 | 0.69360322 | 0.47188938 |
| Mean BCE at 200/220/240/260/280/300 | 0.69346737 | 0.53036789 |

Standardized train BCE at 300 was 0.46987325. It passes both screening
endpoints. This supports conditioning as an important early optimization
factor, not a claim that ratio estimation or DGPO is solved. Post-transform
Fourier RMS reached 15.207 at 300: fixed initial statistics do not enforce
unit variance as the encoder learns. Growth alone is not instability and
does not justify clipping away the successful signal.

Next arm `standardized-long` / W&B `h4scllong1` keeps the same seed, model,
constant per-group learning rates, AdamW, dropout, fixed training-only moments,
16 workers, clean data and source policy checkpoint 1110. Only the fit horizon
changes: maximum 3000 updates, minimum 1000, BCE patience 10 effective epochs.
Existing best-BCE restore is retained. Budget exhaustion is not saturation.
It is a **fresh cold classifier fit**, not an optimizer-state resume from 300;
compare its first 300 updates with the completed screen for consistency.

No new regularizer, activation cap, normalization refresh or scheduler is
introduced. Primary question: does validation BCE continue improving beyond
the 300-update anchor without a sustained train/validation divergence?
Inspect the full trajectory and restored-best independent test separately.
Existing W&B diagnostics include raw/transformed Fourier and decoder RMS,
gradient and parameter-update scales, clipping, and training-batch logit
fractions above absolute 5/10. Those tail fractions are **not** a full
held-out ESS or density-ratio calibration test.

```bash
shifter python3 scripts/train_h4_output_scaling.py --arm standardized-long --validate-only && \
shifter python3 scripts/train_h4_output_scaling.py --arm standardized-long
```

Sync the launcher, new YAML and trainer runtime guard together. All outputs
use `/pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long/` and do not
overwrite the quick runs. This follow-up tests stability; it does not promise
stability in advance or change production OmniFold/DGPO defaults.
