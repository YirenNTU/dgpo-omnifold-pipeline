# Direct Fourier readout for noisy invisible tokens

Prepared for user submission. No NERSC job has been submitted by the assistant.

To compare generated distributions from completed checkpoints rather than
velocity loss alone, use [COMPARE_FOURIER_CHECKPOINTS.md](COMPARE_FOURIER_CHECKPOINTS.md).
That separate evaluation freezes both generators and fits paired cold H4 audits.

## Evidence and question

The successful local cube toy has p(+|c)=0.5+0.4 sin(8 pi c); its Fourier bank
[1,2,4,8] includes the known target frequency. The adapter was zero initialized.
Three seeds reduced conditional eight-mode TV by 68--73%, while held-out
velocity MSE fell only about 2%. See
`../artifacts/dgpo_toy/cube_truth_fourier_v1/RESULTS.md` locally; do not upload
that toy artifact or toy code to NERSC. The toy supports a frequency-based
representation explanation in that constructed problem, not guaranteed gains
in EveNet. Its dimensionless c and pi scaling are not production angles.

EveNet's input k4 linear and MLP branches inject into visible-token embeddings;
the Fourier information then has to reach invisible-token predictions through
the existing attention stack. The fast linear arm increased branch magnitude
without demonstrating a validation-loss benefit. Its fixed-checkpoint scale
probe changed velocity predictions but did not show a consistent gain from
removing or weakening Fourier; doubling worsened all three noise panels.
The input MLP was reported as unhelpful by the user; no new run ID or matched
metric export was supplied for it, so its outcome remains user-reported here.

Question: does a separate query-dependent Fourier memory, read directly by
noisy invisible tokens immediately before the generation head, improve the
same supervised objective? Within this architecture, does a multiscale bank
outperform the existing contiguous bank at identical parameter count?

## Architecture

Visible raw eta/phi -> theta/phi in radians -> sin/cos at the configured four
harmonics -> Linear(16,128), GELU, LayerNorm -> memory tokens (keys/values).

Noisy invisible PET output -> LayerNorm, Linear(256,128), plus a learned
embedding of [t,sin(pi t),cos(pi t),sin(2 pi t),cos(2 pi t)] -> queries.

One four-head cross-attention layer reads the memory; a zero-init bias-free
Linear(128,256) maps its output into a residual added ONLY to valid invisible
PET output tokens before TruthGeneration. This is not the previous output
adapter, which added to visible tokens. No additional Fourier residual is
added at PET input, and visible PET outputs are unmodified by this readout.
All main parameters remain trainable; backbone gradients pass through the
readout queries and existing prediction path. No clean invisible coordinates
are used. Training and sampling use the same PET forward path.

Invalid visible memory tokens are masked, invalid invisible queries get zero
residual, and all-empty visible events get exactly zero residual without an
all-masked attention softmax. Attention dropout is zero; the baseline model's
existing dropout is unchanged. Final zero projection preserves the original
checkpoint function in both train and eval with paired RNG. Internal readout
layers begin learning after that projection moves away from zero. Toy success
with zero initialization is a reason to retain this controlled starting point,
not to initialize every projection weight to one.

Harmonics are positive integers, preserving phi periodicity. This remains a
physical angular basis, not Gaussian random Fourier features on generic input
coordinates. Bandwidth in theta also does not equal bandwidth in raw eta.

## Matched experiment

| Arm | Harmonics | Fusion | Branch peak LR |
| --- | --- | --- | --- |
| readout_k4 | 1,2,3,4 | noisy-token cross-attention | 2e-5 |
| readout_multiscale | 1,2,4,8 | identical | 2e-5 |

Both banks have 16 scalar features and identical trainable parameter shapes.
The multiscale arm replaces k3 with k8; it does not isolate adding k8 alone.
Existing source, seed42, all-backbone unfreeze, fresh optimizer/scheduler,
50 epochs, five-epoch warmup, 16 GPUs and batch2048/GPU are inherited unchanged.
Source remains the original epoch190 no-Fourier checkpoint in the common YAML,
not an already trained Fourier or DGPO checkpoint. Existing none mpvmx945 and
linear-input 98q3imcl are historical matched recipe controls.

Primary endpoint remains mean validation velocity loss over epochs45--49.
A readout-vs-input gain supports the complete new architecture, including its
extra capacity; it does not isolate cross-attention alone. A multiscale-vs-k4
gain more narrowly supports frequency selection within the new architecture.
One seed is a pilot, not a replicated result. No claim of generation improvement
should be made from velocity MSE alone; any promising checkpoint should also
receive the same held-out conditional-generation evaluation as its control.
This change does not add or run a classifier/generation-quality benchmark.

Existing train/val fourier RMS metrics now refer to the invisible-token
injection site, so do not directly compare their ratio to the old visible-token
input ratio. The fixed-checkpoint scale diagnostic still works through its
branch forward hook; point its YAML to the chosen readout runtime/checkpoint.
It measures reliance on the whole readout branch, not harmonics in isolation.

## User execution

Use the existing NERSC ml_pipeline checkout and the repository's upload
exclusions (`--exclude-from=NERSC/upload-excludes.txt` for root rsync). No toy
files or excluded classifier artifacts belong in the upload.

Inside an existing four-node / 16-GPU Ray allocation, run:

```bash
shifter python3 scripts/train_fourier_integration.py --arm readout_multiscale --preflight
shifter python3 scripts/train_fourier_integration.py --arm readout_multiscale
```

For the matched frequency control, use a separate allocation, or run after the
first arm finishes on the same allocation:

```bash
shifter python3 scripts/train_fourier_integration.py --arm readout_k4
```

Preflight for either readout checks the untouched source, linear input and
BOTH readout arms on the same two real validation events. It checks initial
prediction equality and all parameters trainable; it is not a full validation
benchmark. Config-only inspection is `--check-only` through the same Shifter
command. Alternatively the user can submit the existing Shifter batch wrapper
with arm readout_multiscale or readout_k4; it performs preflight automatically.
Each arm gets a fresh W&B run and its own seed42/<arm> output directory.

## References and limits

Tancik et al., Fourier Features Let Networks Learn High Frequency Functions in
Low Dimensional Domains: https://arxiv.org/abs/2006.10739. Motivates the
frequency/bandwidth hypothesis, not guaranteed gains for this event model.

Peebles and Xie, Scalable Diffusion Models with Transformers:
https://arxiv.org/abs/2212.09748. Studies diffusion conditioning strategies;
this specific readout is our testable design, not a reproduced DiT result.
