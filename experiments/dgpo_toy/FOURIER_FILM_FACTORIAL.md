# Fourier x FiLM: complete the missing raw-shift cell

Status: implemented; scientific training NOT launched by the assistant.
Toy-only follow-up19. The previous18-round ledger and paused automation are
untouched. User launches compute; no production changes or NERSC uploads.

Validation (2026-09-27):45 related tests passed, including the new factorial
algebra/decision rules, source-contract checks, native raw-shift resume equality,
read-only real-artifact preflight and synthetic report/plot integration.
All four initial velocities and DDIM samples match exactly on the declared
probe; each has61379 nominal parameters. Compile/JSON/whitespace checks pass.
The scientific output directory has not been created; tests are not results.

## One question

Does Fourier specifically amplify the advantage of scale+shift FiLM over
additive nonlinear conditioning, for fixed-H4 reward absorption?

| Condition basis | Shift-only | Scale+shift FiLM |
|---|---|---|
| Raw | NEW raw_shift, trained from step0 | Saved round04/raw |
| Fourier | Saved round07/shift | Saved round04/fourier |

Only ONE new policy is trained. Do not disable scale on a trained FiLM endpoint
and call that a training control. No new reward classifier, truth distribution,
training seed, pretraining, reference or objective. Fresh audit classifiers are
new at each policy checkpoint, as in the controls; the source audit is repeated
to validate exact replay before continuing training.

## Fixed setup

- Native DGPO detached gate/unscaled LOO; coefficient1 times HALF velocity-MSE.
- Original uniform-cube pretrained source, same fixed full-truth H4 (NOT the
  oracle/shape-matched H4); raw weights, never EMA; no refit or recentering.
- AdamW1e-4, weight decay.001, clip1, batch64, K8, M4, DDIM50,1000 updates.
- Same initialization, policy, monitor, panel and classifier RNG clocks as
  rounds04/07. Audit pauses preserve policy optimizer and RNG.
- Shared two-linear-layer SiLU encoder, width32, new-branch LayerNorm,
  three modulation sites. Raw=sqrt(3)*c repeated8 times; Fourier frequencies
  1/2/4/8. Gamma outputs are disabled only in shift arms. Equal nominal count
  is NOT equal active rank, effective capacity, FLOPs or velocity displacement.
- Fresh width128 Fourier audits at0/25/100/300/1000: minimum8000,
  maximum32000, validation every100, patience20 checks, delta1e-4;
  strict minimum validation BCE checkpoint. Inadequate audit stops the existing
  runner as inconclusive; it is not evidence of closure or no reward transfer.

## Endpoint and decisions

Re-score the same saved16384 IID held-out contexts with the SAME fixed critic.
There is one generated draw per context in this test panel. Training K8 and the
descriptive64-condition x512-draw monitor grid are separate. Never best-of-K.

Let G be source-relative mean fixed reward gain at1000. Primary:

`interaction = (G_fourier_film - G_fourier_shift) - (G_raw_film - G_raw_shift)`

Four simple effects accompany it: FiLM benefit under raw/Fourier, and Fourier
benefit under shift/FiLM. Simultaneous95% paired-context bootstrap family of
these five contrasts,2000 replicates, material margin .01.

- Interaction lower95>.01: supports extra FiLM benefit with Fourier.
- Interaction upper95<-.01: supports a smaller FiLM benefit with Fourier.
- Entire interval within[-.01,.01]: compatible with only a small interaction
  at this declared margin, not proof of exact equality.
- Otherwise unresolved. A CI spanning zero is not proof of no interaction.
- Inspect both FiLM simple effects to distinguish general modulation benefit
  from complementarity; do not infer that an interaction alone proves any arm
  improved over its source.

Fresh BCE/AUC, mode TV/parity error, corner shape, low-order moments,
velocity-probe distance and source-cell reward decomposition are diagnostics.
They cannot silently veto fixed-reward absorption. No demand for AUC=.5.
Separate-family sparse-curve averages use0/25/100/300/1000, not dense measured
time-to-target. No raw reward comparison between different critics.

This is adaptive exploration using three already-observed arms and reused test
data, not a preregistered independent confirmation. Intervals quantify sample
uncertainty conditional on these fitted models, not seed/selection uncertainty.
It does not isolate extreme ESS or establish causal transfer to EveNet.

## Run locally

From `/Users/yirenwu/Ztautau/ml_pipeline`:

```bash
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.fourier_film_factorial --preflight

/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.fourier_film_factorial
```

Preflight checks saved control configs, clocks, adequate audits, source panels,
baseline classifier predictions, checkpoint metadata, and actual identical
initial velocity/samples for all four cells. It trains nothing and writes no
experiment directory. The default one CPU thread matches the archived paths.

W&B defaults OFFLINE under the new output. `--wandb-mode disabled` avoids SDK
logging. Display name: "Do Fourier and FiLM cooperate? | fixed H4 | missing raw
shift control"; group "Cube conditioning closed loop". Existing runs/IDs are
not changed. Native training/fit progress is logged by the existing runner;
the combined four-cell analysis is a local report, not a second W&B run.

Outputs under `artifacts/dgpo_toy/fourier_film_factorial_v1/`:

- `raw_shift/`: new policy/optimizer/RNG checkpoints, fresh audits and fit plots.
- `SUMMARY.md`, `report.json`, `paired_reward_gains.pt`: four-cell result and
  paired interaction/simple effects, with separate structural diagnostics.
- `factorial_curves.png`: reward, fresh AUC, joint-mode error and velocity drift.
- `status.json`: running / awaiting_review / failed_or_interrupted.

Stop after this report; no automatic next experiment or training extension.
Saved controls are read-only. Existing output directories are not overwritten
by training. If only analysis/plotting was interrupted after a completed new
arm, rerun `--analysis-only` (no training). This wrapper does not offer arbitrary
interrupted-policy resume; the retained technical checkpoints preserve full
state for an explicitly reviewed continuation.
