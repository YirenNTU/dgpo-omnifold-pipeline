# Relation-access ablation: classifier first, fixed-reward diffusion second

Status: implemented, NOT scientifically run. Local toy only. The previous
18-round campaign and its paused automation are not restarted or modified.

2026-09-27 follow-up: retain the existing two-stage architecture and native
loss after the Fourier x FiLM factorial completion. Add explicit startup
`runtime.json`, all-three-arm initial-function checks, finite/paired data
checks, and direct relative-minus-particle content diagnostics. The latter
expands the simultaneous descriptive content family from six to nine
contrasts BEFORE any scientific run. A relative-over-particle BCE advantage
alone is not labeled a joint-access result: require an advantage over raw,
concordant AUC, and the relative-over-particle mode-content result as well.
This is the remaining relation-access question, not another FiLM depth test.

Default logging is now local files (`wandb_mode: disabled`) because the last
toy run's offline W&B service timed out before training. This changes logging
only. `--wandb-mode offline` explicitly opts back into offline W&B; no online
sync or remote training is enabled. Each stage prints its actual fit LR,
batch, budget and policy/audit clocks before it starts.

Validation (2026-09-27):73 related tests passed across the relational runner,
native conditioning/resume, factorial, reward-transfer and signal-attribution
tests. Classifier preflight passed on the actual archived panels; the RL
planning preflight is correctly `planned_awaiting_stage1_review`. Raw,
particle and relative policies all reproduce the visible source's initial
velocity and samples bitwise. JSON/compile checks and W&B display-name
validation passed. No scientific output directory or training job was created.

Implementation validation (2026-09-26):56 related tests passed, including
12 new relational tests; classifier and RL planning preflights passed. With
the real archived source, matched arms reproduce visible-source velocities
and DDIM samples bitwise; Cartesian round-trip versus archived scalar-source
maximum sample error was4.77e-7 on the declared probe. No scientific fit or
DGPO job was started. One inherited FiLM gradient-connectivity test now uses
a seeded continuous probe reward: its old categorical fixture could yield
zero advantage for same-mode candidates. Actual experiment rewards unchanged.

## One question per stage

1. Can a classifier recover the same conditional joint signal when `c` must
   be read from two visible directions instead of supplied directly?
2. With ONE of those classifiers frozen for BOTH policy arms, does easier
   relation access in diffusion's trainable correction improve held-out reward?

No new truth difficulty, narrower plain-only network, extra training seeds,
reward shaping, or production changes. `raw` means **no extra condition
Fourier**; output `y` Fourier remains on in every classifier.

## Encoding and the three classifier arms

Keep the exact archived `(c, truth_y, generated_y)` panels (32768/8192/16384).
Draw independent common rotation `psi` for each paired event and set
`phi1=psi+pi*c/2`, `phi2=psi-pi*c/2`. Both labels get the SAME pair of visible
unit vectors `(cos(phi1),sin(phi1),cos(phi2),sin(phi2))`. Each individual
direction is marginally uniform. The truth law and samples do not change.
Only metadata retains original `c`; learned classifier APIs accept 4-vectors,
not scalar c. Rotation draws are nuisance samples, not model-seed replications.

| Arm | Extra condition slots (all retain the four raw coordinates) |
|---|---|
| raw | 16 zeros |
| particle | sin/cos of k=1..4 for each visible direction |
| relative | sin/cos of k=1..4 of phi1-phi2, each repeated twice |

All have 47 inputs:4 visible+3 output+24 output Fourier+16 condition extras;
three GELU hidden layers, width32. B/C extras have equal RMS and count; their
active rank is NOT equal. k1 in B duplicates raw Cartesian components. C is a
deliberately supplied relation (a structural prior), not a truth/parity leak or
a claim of completely prior-free discovery. Harmonics use dot/cross products
and recurrences, not a discontinuous angle input to the learned network.

Fits share initial weights and minibatch indices. AdamW3e-4, WD.001, clip1,
batch512; validate100, minimum32k, maximum64k, patience20 checks, delta1e-4.
Select the exact minimum validation BCE even if an improvement is below the
early-stop delta. Cap without plateau is inconclusive, not model incapacity.

Primary: three paired test BCE contrasts with simultaneous bootstrap and
material margin .002. Fixed-grid A/B/C/D swaps from the saved source are ONLY
post-fit content diagnostics; mode and shape are never training features.
The nine content contrasts include relative-minus-particle, not just both
enhanced arms versus raw. Their intervals form a separate family from the
IID test BCE intervals. Failure to resolve an advantage is not equivalence.
Report common-rotation logit changes too. Learned networks retain raw absolute
directions, so even C is not architecturally forced to be rotation invariant.

## Why the second stage uses a residual correction

Simply changing the pretrained scalar-input network to four inputs changes
its initial generator, confounding learning with initial coverage. Instead:

`v_policy = v_source(x,t,c_from_visible) + F_train(x,t,visible) - F_initial(x,t,0)`

`v_source` and `F_initial` are immutable buffers, excluded from optimizer and
gradient diagnostics. `F_train` starts from the old network, but its legacy
scalar-c input is always zero. Its only trainable condition path is the visible
encoder (20->32->32, SiLU, new-branch LayerNorm) and three scale+shift FiLM
heads. Only those output heads are zero-initialized. Its backbone is trainable.

Both arms initially reproduce the visible source **bitwise**. Round-tripping
archived scalar c through Cartesian geometry has finite-precision error;
preflight separately measures velocity and DDIM differences from the original
archived source and requires <=1e-5. We do NOT claim bitwise identity to archived
samples after that coordinate transform. The frozen source still uses the
known geometric relation; this experiment isolates the **learned correction's
access**, not an entirely relation-blind end-to-end pretraining architecture.

The two policies share nominal count, initialization, draws, AdamW, reference
and frozen reward. Native loss is unchanged: detached DGPO gate, unscaled LOO,
plus coefficient1 times HALF velocity-MSE. No reward scaling/clipping, EMA or
refits. Audits cannot reset policy RNG, optimizer or original reference.

## Launch locally (user runs these commands)

From `/Users/yirenwu/Ztautau/ml_pipeline`:

```bash
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.relational_experiment \
  --plan experiments/dgpo_toy/plans/relational_classifier.json --preflight

/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.relational_experiment \
  --plan experiments/dgpo_toy/plans/relational_classifier.json
```

Then read `artifacts/dgpo_toy/relational_access_v1/classifier/SUMMARY.md`.
The runner stops at `awaiting_review`; it never starts DGPO automatically.
No source snapshot or saved results are overwritten. Use `--output` for a new
run directory; if changed, point RL plan's `classifier_result` at that result.

After review, choose `particle` or `relative`. The following is an EXAMPLE
for a result that actually supports `relative`, not a preselected conclusion:

```bash
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.relational_experiment \
  --plan experiments/dgpo_toy/plans/relational_rl.json \
  --representation relative \
  --review-note 'Reviewed stage1: relative branch has adequate BCE and mode-content benefit.'
```

The selected representation must pass stage1's fit/BCE/mode-content gates;
otherwise RL refuses to run, leaving the question unresolved. The same saved
classifier is frozen for both trajectories. Audits are independently freshly
initialized width128 relative-feature classifiers, not reward refits.

## Measurements and interpretation

- Policy update budget:1000; dense native reward/gradient monitor every25.
- Independent all-sample reward, joint/shape and fresh audit at0/25/100/300/1000.
- Primary: paired enhanced-minus-raw mean reward at1000, lower95 > .01.
  Each IID context contributes its mean across ALL8 candidates, not best-of-K.
- Same fixed reference-probe velocity distance, per-condition gain quantiles,
  positive-gain fraction, gradient conflict and branch RMS; larger motion is
  reported, not silently treated as matched-distance efficiency.
- Mode TV, order1/2/3 debiased L2 and within-mode RMS. Frozen-source reward
  decomposition describes mode versus shape/interaction; it is not unique
  causal attribution. Fixed-grid intervals are descriptive only.
- Fresh audit BCE/AUC and actual fit/selected steps. Cap-hit fits are invalid;
  keep valid reward results but never label undertrained near-chance as closure.

Output: `plan.json`, `status.json`, `progress.jsonl`, `report.json`, `SUMMARY.md`,
selected classifiers, paired score arrays and policy/optimizer/RNG checkpoints.
`fit_curves_*.png` and `policy_curves.png` are rendered automatically; invalid
fresh audits are omitted from the AUC trajectory, not drawn as closure.
W&B defaults DISABLED; local metrics, checkpoints and plots remain enabled.
Use `--wandb-mode offline` only if the local service works. Fit/policy clocks
are separate; `preflight.json` and `runtime.json` record resolved settings.
The RL summary includes fresh held-out BCE/AUC and fit/selected steps, separate
from the frozen classifier's reward gain.

Technical checkpoints preserve optimizer/RNG between built-in milestones;
this first version does not offer arbitrary interrupted-run CLI resume.

## Validation / authority

Preflight and unit tests are local diagnostic checks, not scientific training.
Only the user launches either experiment. No NERSC, allocations, rsync or
production files are touched. Test results and implementation status cannot
be cited as evidence that the conditioning hypothesis succeeded.
