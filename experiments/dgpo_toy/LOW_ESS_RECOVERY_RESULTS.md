# Learned low-ESS recovery: local pilot, 2026-09-21

Question: reproduce a generalizing classifier with concentrated raw ratio
weights, then test whether conditional neural diffusion fails to improve its
fixed reward. These are separate stages, not interchangeable outcomes.

Source: `artifacts/dgpo_toy/conditional_fourier_v1/seed17/models.pt`.
Frozen pretrained 12D conditional neural DDIM20; fresh paired classifier fitting
data; seed17; 20,000 updates (5,120,000 contexts); best validation BCE selection.
No logit inflation, clipping, temperature, or oracle training targets.

| Fresh-data representation | Held-out BCE | AUC | ESS/N | log mean ratio |
|---|---:|---:|---:|---:|
| Individual Fourier | .436938 | .872707 | .212573 | .033449 |
| All signed triples + individual Fourier | .206318 | .947441 | .001590 | .771115 |

Confirmation uses131,072 contexts per class. Joint3 selected step20,000;
top1% weight mass=.930791; maximum individual weight mass=.021394;
mean-ratio relative SE=.069215. Actual ESS is about208 of131,072 samples.
Joint3 is a generic order-three inductive bias with a larger input layer, not
a parameter-matched comparison or evidence of arbitrary structure discovery.

On a separate2048-context paired density panel, learned/reference BCE is
.189485/.179154 (excess .010332); truth logit RMS error is1.166700 and generated
logit RMS error .339065. Neural-vs-nominal log-density RMS is approximately.011.
The source sampler passes a sufficient global-bijection bound (minimum step
margin .278614), plus inverse residual, start-agreement and Jacobian checks.
This supports the numerical reference for this source, not arbitrary policies.

## Decision against the declared gates

**Classifier phenomenon reproduced; policy failure still unresolved.** Joint3
passes discrimination, low ESS, lower-order generator checks, within-context
signal, missing joint structure and reference BCE agreement. It fails:

- abs(log mean ratio)=.771115 versus maximum.50 (mean ratio about2.16).
- Truth logit RMS=1.166700 versus maximum1.0.

Both runs are `inconclusive_setup`; neither executed the formal300-update DGPO
or pathwise arms. Do not relax gates after seeing outcomes or select another
checkpoint using confirmation data. A constant logit offset would cancel under
LOO and cannot itself explain policy stagnation. Remaining ratio error also
includes nonconstant error, so normalization alone is not a diagnosis.

Fresh data alone rules out that intervention as sufficient at this budget.
The generic joint representation supports a much better learned concentrated
ratio, but does not establish low ESS as a cause of production stagnation.
Next decision is whether to test the approximate fixed classifier as such, with
an explicitly revised protocol, or require the stronger ratio-fidelity claim.
Do not spend another classifier search round without choosing that question.

## Artifacts and verification

- `artifacts/dgpo_toy/low_ess_recovery_v1/report.json`: coordinate,89.86s.
- `artifacts/dgpo_toy/low_ess_joint3_v1/report.json`: joint3,174.70s.
- Each folder has selected `reward.pt` and live `progress.jsonl`.
- Joint3 smoke exercised both policy arms for2 updates; not scientific evidence.
-40 tests passed; tracked diff whitespace check passed.
- No production changes, W&B writes, NERSC jobs or dataset changes.

Workflow outcome: one setup decision changed in about265 CPU wall seconds;
avoided uninterpretable policy runs. Next round should distinguish a test of
fixed learned reward transfer from a test requiring accurate density ratios.
