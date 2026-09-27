# Toy-only conditioning closed loop

Authorized 2026-09-26 by the user: execute local toy experiments, at most 20
scientific rounds, review and summarize each before the next. No production
changes, remote jobs, allocations, uploads, or NERSC synchronization. All new
code and outputs remain in `experiments/dgpo_toy/` and `artifacts/dgpo_toy/`.

## Question and evidence order

Latest user clarification, 2026-09-26: the target is **reward-transfer
learnability**, not complete distribution closure. Determine which conditions
let diffusion absorb the same fixed classifier reward faster and more strongly
on held-out conditions/noise at matched policy-update and rollout budgets.
Fresh-classifier AUC/BCE and joint/shape attribution remain diagnostics of what
changed; neither AUC=.5 nor monotonic AUC decline is required for success.
This changes evaluation priority, NOT the native DGPO loss, frozen reward,
reference, or coefficient1. Preserve earlier plans/results with their original
endpoints; any reward-first reanalysis of observed runs is explicitly post hoc.

1. Validate the measuring instrument: Fourier on c, y, both, neither, holding
   the classifier MLP, initialization, panels and minibatch identities fixed.
2. Fresh plain-versus-Fourier audits on the saved full-truth reward trajectory
   (0, early rebound, recovery). Do not rerun a policy just to measure it.
3. Isolate one conditioning factor per paired native-DGPO round. Candidate
   factors are Fourier vs rank-one raw c, multiplicative vs additive modulation,
   encoder activations, encoder normalization, and active injection depth.
   These are a menu, not a mandatory sweep. The last result determines the next.
4. User update 2026-09-26: do not add multi-seed or seed-replication rounds.
   Use matched single-seed comparisons and independent held-out samples/fresh
   classifiers. Report conclusions as conditional on the tested training seed;
   do not claim seed robustness or change seeds to seek a preferred result.
5. Latest user fidelity challenge: EveNet plain reportedly misses discrepancies
   exposed by Fourier, whereas toy plain detects after much longer fitting.
   Distinguish finite-budget optimization, classifier capacity, relational-input
   access and condition complexity. Do not force failure by shrinking only plain,
   changing truth, supplying oracle parity, or searching many difficulties.
   Round15 tests one common width128->32 change on unchanged saved populations.
   No inference to real EveNet or to reward transfer from detection alone.

Stop early if a causal decision is sufficiently answered, no valid instrument
is available within its declared budget, the next round cannot distinguish
hypotheses, or round 20 is completed. Do not force failure by changing truth.
At most one experiment process at once; one CPU thread by default. Each policy
trajectory is capped at 3000 updates; initial screens use at most 1000.

## Fixed toy and mathematical contract

- Continuous c in [-1,1], three generated coordinates near eight cube corners.
- Truth changes triple parity probability with four localized, non-sinusoidal
  smooth bumps. Analytic modes and truth probabilities are diagnostics only;
  neither classifier nor diffusion gets parity, mode IDs, or the bump function.
- Source: ordinary velocity-pretrained uniform-cube diffusion from
  `nonperiodic_cube_classifier_extended_v1/reference.pt`. This source has
  within-mode shape errors; do not call its marginals exactly correct.
- Native detached DGPO gate and unscaled leave-one-out advantage, frozen full
  truth learned H4 reward, original frozen reference, velocity surrogate
  coefficient **1**, AdamW 1e-4, weight decay .001, clipping 1, batch64, K8,
  four noise-time samples, DDIM50, no EMA, no reward refits or transformations.
- The reference coefficient multiplies the inherited half velocity-MSE loss;
  it is a surrogate, not an exact endpoint distribution KL.
- All conditioning arms must generate bitwise-identical initial velocities
  and samples. Last modulation heads zero; encoder internals normal init.
- Optimizer, RNG, history and original reference persist across audit pauses.

## Classifier and endpoint contract

- MLP has three GELU hidden layers, width128, 36 inputs. Four raw coordinates
  plus k=1..4 sin/cos on scaled (c,y). Inactive features are zeroed. Nominal
  parameter counts match, but active input rank/effective capacity do not.
- Cold fits share initialization and batch identities. Independent fixed
  train32768/validation8192/test16384 condition/positive/negative triples.
- AdamW3e-4, weight decay .001, balanced batch512, clipping1. Validation every
  100 updates, minimum2000, patience20 checks, min_delta1e-4. Default cap32000;
  cap64000 only in a separately declared adequacy round. Each arm stops by the
  same rule, not necessarily at the same update. Select exact minimum val BCE.
- A cap hit without plateau is inconclusive, not evidence for closure or
  inability to discriminate. Do not shorten audits to make AUC look good.
- Classifier representation primary: held-out BCE differences; AUC and ratio
  ESS characterize them. New DGPO rounds use held-out fixed-reward gain
  R(t)-R(0) as primary, with identical frozen critic/logit scale across arms.
  Predeclare a matched update horizon and early measurement times to separate
  learning speed from eventual gain. Report actual rollout/update cost;
  do not confuse audit overhead with policy-learning efficiency.
- Fixed reward is mean over ALL generated candidates, not best-of-K, measured
  on conditions/noise independent of fitting and policy updates. Use paired
  context-level uncertainty for arm differences, not separate-arm intervals.
  Reward standard deviation, gradient magnitude and AUC alone are not reward
  absorption measures. Do not compare raw reward magnitudes from different
  classifiers as if calibrated to one common scale.
- Retain adequately trained fresh-classifier AUC/BCE, mode TV, within-mode RMS,
  reward mode/shape decomposition and gradient conflict as secondary
  diagnostics. They distinguish reward absorption from useful truth-directed
  change or exploitation; they are not closure gates. AUC near .5 still requires
  fit validity if interpreted. Report per-condition gain/tails and reference
  velocity distance where available, so a mean increase does not hide a few
  dominating conditions or greater policy drift. Unmeasured diagnostics must
  be labeled rather than assumed.
- Paired bootstrap resamples contexts (positive and negative together), with
  simultaneous intervals over preregistered contrasts per metric. It excludes
  classifier/policy training-seed uncertainty. Adaptive rounds are exploratory;
   seed robustness remains unmeasured; no extra seed confirmation is requested.

## Review boundary

Each round first saves `plan.json` (one question, endpoint, contrasts, decision
rule, seeds, validity budget). Training saves progress, selected models and
test scores. Completion writes `report.json` and `SUMMARY.md`, then **stops**
at `awaiting_review`. A reviewer records `DECISION.md`, including supports /
rules out as sufficient / unresolved, limitations, cost, and the next question,
before launching another round. The loop ledger records completed vs running
rounds; max20 counts scientific plans, not unit tests. W&B is local offline
telemetry; results do not silently appear in the production W&B project.

Do not treat early AUC rebound as permanent failure. Measure predeclared early,
middle and final policy checkpoints; any later horizon extension is a new round.
Historical shape-matched success is not full-truth closure, and linear versus
nonlinear comparisons that also change LayerNorm/capacity are package effects.
Full-truth closure is no longer the stopping target. Do not extend horizons
merely to chase AUC=.5; extend only when a predeclared reward-learning question
requires it. First reuse saved matched runs to quantify reward gain/dynamics
before choosing the next intervention; no additional training seed is needed.
