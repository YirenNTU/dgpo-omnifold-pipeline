# Saved plain classifier + velocity-MSE proxy, 2026-09-21

User endpoint clarification2026-09-21: the requested question is easier absorption
of each classifier's OWN reward, NOT high-order structure. The combined gate
below was an overextended interpretation; keep it as historical secondary
analysis, not a veto on the requested reward-transfer result. Completed10k
SUPPORTS more readily absorbed plain reward in this matched classifier-bundle
comparison: independent own gain+.0485130 [.0436807,.0533454] versus
strong-own-.00020228 [-.00069335,.00028880]. Initial reward-SD normalized gains
are+.0274662 [.0247303,.0302021] versus-.0002616 [-.0008968,.0003735].
This normalization only changes reporting units, not training objective or
the relative penalty. It does not isolate ESS from scale/ranking/representation.
No new experiment, fitted reward, or objective change was performed.

Status: COMPLETED10000; final results appended below. The launch/live records
are retained as historical evidence, not current status.

Declared before launch, following the user's explicit velocity-MSE choice.

- Objective: test reward transfer and high-order learning with the selected
  no-Fourier classifier under the SAME legacy velocity proxy as strong H4.
- Success measure: the common strong-H4 and known-truth moment endpoints below.
- Limiting factor: classifier-to-policy signal actionability under regularization.
- First-principles claim: a different informative reward may be easier to absorb;
  its own score improvement alone cannot establish high-order progress.
- Deliverable: one new matched10k trajectory and independent paired endpoint.
- Time box:10000 updates, approximately5-10 CPU minutes; no plateau early stop.
- Delete: no new classifier fits/features, KL sweep, second stage or endpoint
  inverse calculations; reuse the completed strong velocity control.
- Risks/authority: local toy only; preserve all source/control/endpoint files.
  No production, NERSC, W&B or external writes.
- Evidence debt: velocity MSE is a proxy, not endpoint KL. Classifier architecture,
  data regime, scale and ranking differ, so this does NOT isolate ESS causally.

## Matched conditions and formula

Restart from `low_ess_joint3_v1/reward.pt` original pretrained weights, NOT
from the interrupted weak endpoint run or a10k/20k policy. Load the exact frozen
plain model selected9900 from `conditional_fourier_fit10k_v1/seed17/models.pt`.
Same seed17, train RNG3017, DDIM20, AdamW1e-4/decay.001, batch64/K8/M4,
uniform t in[0,.7], gradient clipping1, unscaled LOO advantage and detached gate.
The reference remains frozen at original pretrained step0.

L = L_DGPO + 1 * 0.5 * mean((v_current-v_step0)^2).

Use the existing production-style reference-trust implementation on the SAME
detached current-policy candidates, t and noise as DGPO. No endpoint ratio
subtraction or density controller. Coefficient1 does NOT guarantee truth alignment.
Matched strong control: `velocity_kl1_joint3_10k_v1`, completed10000 with identical
source/config/train stream/proxy. Verify coefficient in BOTH report and checkpoint,
complete sequential history, and exact reproduction of its saved reward monitor.

## Measurements and predeclared decision

Reuse all provenance and setup gates in `WEAK_CLASSIFIER_PROTOCOL.md`:
validation replay, informative plain classifier, at least10x global ESS contrast,
nontrivial within-context signal/alignment. Global ESS is NOT within-K/gradient ESS.

- Every25: own fixed reward with paired monitoring CI, reward std, global/within-K
  ESS, head-gradient concentration, clipping/gate saturation, actual velocity MSE,
  half-MSE penalty, total/DGPO losses; full-network pre-Adam main/penalty gradient
  norms, ratio, cosine and cancellation/projection diagnostics.
- Every250: SAME strong-H4 judge, both score distributions and56 high-order moments,
  lower-order marginal/variance/covariance checks. Measurement does not advance
  training RNG; regression tests compare full model/AdamW/RNG bitwise.
- Final only at10000: fresh paired109017,4096 contexts x8 for initial,
  saved strong-velocity10k and new plain-velocity10k.500 paired bootstrap replicates
  for moment MSE differences. This panel was not consumed by the interrupted arm.

Transfer: strong-H4 gain>=.10 with lower95%CI>0, AND plain-policy minus
strong-policy common-H4 gain lower95%CI>0.
Structure:56-moment RMSE at least10% below initial, MSE-change upper95%CI<0
against BOTH initial and strong-policy control. Useful structural progress also
requires max residual mean<=.10, variance error<=.15, pair covariance<=.08.
Own plain reward progress is a diagnostic, not a substituted success criterion.
Any setup/numerical failure is unresolved. No peak selection or early stopping
on reward. Failing the declared endpoint rules out THIS intervention as sufficient
at10k/this seed, not all weaker classifiers or all future budgets.

## Historical interruption, not a failed algorithmic endpoint

`plain_classifier_endpoint_kl1_10k_v1` stopped on user direction, saved full valid
state1248. Its report contains KeyboardInterrupt; do NOT describe this as numerical
failure or a completed10k negative result. At1000 common-H4 gain+.00165139,
CI[-.00162701,.00492980]; moment RMSE .49212512 -> .49199716. Own reward improved
about.04. Frozen/direct-density results are kept separate from this new proxy arm.

## Launch

```bash
python -m experiments.dgpo_toy.weak_classifier \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --plain-checkpoint artifacts/dgpo_toy/conditional_fourier_fit10k_v1/seed17/models.pt \
  --comparator artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1 \
  --output artifacts/dgpo_toy/plain_classifier_velocity_mse1_10k_v1 \
  --regularization velocity --policy-steps 10000
```

90 tests passed before launch, including proxy formula/coefficient, no endpoint
controller, frozen sources, no classifier fit and telemetry noninterference.
Full model/AdamW/RNG saved after1/every1000/final; progress every25.
Outcome unresolved at declaration. No automated follow-up or additional arm.

## Verified live handoff

- Outcome: new velocity-proxy arm running beyond3250/10000; old endpoint
  full state preserved at1248. No change midway to either trajectory's objective.
- Evidence:90 tests; actual new checkpoint3000 has model/history/AdamW clocks3000,
  coefficient1 and no endpoint controller. First penalty and its gradient are0;
  by3250 MSE.00101125, penalty.000505625 exactly half. Own reward gain+.0414401,
  CI[.0369836,.0458966]. Common judge3000 gain+.00114088,
  CI[-.00188829,.00417005]; moment RMSE .49212512 -> .49192229.
- Decision: supports correct proxy wiring; structural progress unresolved,
  no completed10k endpoint yet. Repeated monitor intervals are descriptive,
  not final independent evidence.
- Learning index: policy hypothesis unresolved; cost per resolved policy
  decision not yet finite. One bounded10k arm, estimated5-10 CPU minutes.
- Deleted: endpoint inverse work in this arm, extra fits and extra policy arms.
- Next limiting factor: whether the plain reward produces common-judge/high-order
  improvement under the matched proxy by the predeclared endpoint.
- Better next round: distinguish own reward, common reward and true structure;
  never treat a global-ESS contrast alone as a causal diagnosis.

## Completed result

Source: `artifacts/dgpo_toy/plain_classifier_velocity_mse1_10k_v1/report.json`.
All10000 updates completed in206.93s. Independent final109017 contains4096
contexts x8; uncertainty paired by context,500 bootstrap replicates for structure.

| Independent final metric | Initial | Strong-H4 + velocity1 | Plain + velocity1 |
| --- | ---: | ---: | ---: |
| Plain reward gain |0|-.00000899|+.0485130 [.0436807,.0533454]|
| Common strong-H4 reward gain |0|-.00020228 [-.00069335,.00028880]|-.00194739 [-.00497167,.00107689]|
|56-moment RMSE (lower better)|.492759873|.492749749|.492921942|
|Plain reward std|1.766282|1.766745|1.750777|

Plain-minus-strong common-H4 gain-.00174511,95%CI[-.00479797,.00130775].
Plain-vs-initial structural MSE change+.000159748,
CI[-.000147122,.000507946]; plain-vs-strong+.000169726,
CI[-.000147879,.000505481]. Neither resolves a structure change.
Lower-order preservation passes (.051838 mean,.026392 variance error,
.047589 pair covariance); all four transfer/structure decision gates fail.

Own monitor gain at100/1000/3000/10000: .01679/.03690/.04157/.04430.
It learns early then slows; not a claim of exactly zero late learning. Independent
final own gain is.04851, and must not be mixed with the reused monitor numbers.
No clipping or gate saturation. Last1000 logged main/velocity gradient cosine
median-.31495, penalty/main norm ratio.29880,total-on-main projection.90322.
There is a partial opposing component, not complete per-batch cancellation;
these pre-Adam batch statistics alone do not identify a population equilibrium.
Full model/AdamW/history clocks all10000; final and full-state models equal;
finite weights, coefficient1, no endpoint controller and frozen sources verified.

Conclusion: supports modest generalizing OWN-reward transfer. THIS saved-plain
intervention is insufficient for the predeclared common-H4/high-order endpoint
at10k/seed17. Higher global ESS did not solve it, but ESS alone is not isolated:
features/capacity/fit regime/scale/ranking differ and within-K ESS ordering reverses.
Do not claim permanent inability, precise truth KL, or production generalization.
No extra policy run or objective change was made during this status check.
