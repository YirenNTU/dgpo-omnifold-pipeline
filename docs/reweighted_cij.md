# Paired Cij reweighting experiment

## 2026-09-29 analysis correction: compare like with like

The first FiLM replay (`ugle45z4`) compared **stored full-truth** angles to
reconstructed angles built from generated tau directions, fixed tau energy,
and reconstructed visible momenta. Its reported worsening relative to that
physics reference remains an observation; it is **not** an isolated test of
the generated direction distribution. Do not claim the mismatch explains the
worsening before measuring the controls below.

The `matched-cij-v2` bridge now retains that exact comparison and additionally
computes, on the same selected source IDs and base weights:

| Reference | Tau four-vector | Visible four-vector | Purpose |
| --- | --- | --- | --- |
| `stored_truth` (legacy main result) | Upstream truth | Upstream truth | Original physics reference |
| `recomputed_truth` | Stored true tau p4 | Stored true visible p4 | Quantify stored/recomputed convention or precision discrepancy in C, not just angles |
| `truth_tau_reco_visible` | Stored true tau p4 | Reconstructed visible p4 | Isolate change of visible input along this comparison chain |
| `matched_target` | True direction offsets through the SAME fixed-E reconstruction as samples | Reconstructed visible p4 | Test the implemented direction-to-observable mapping |

For calibrated samples, the matched target uses the **same back-to-back
calibration**. Raw samples and raw truth targets remain uncalibrated. No true
energy or true visible momentum is inserted into generated candidates; truth
is used only in evaluation references. The model predicts tau-direction
offsets, not full neutrino four-vectors.

All references keep their original base MC weights. Candidate ratio weights
are never applied to the reference. We do not clip, temper, discard high-weight
events, refit the classifier, regenerate samples, or change `--kappa-signs`.
At K=1, within-event normalization would remove **all** reweighting, so the
experiment keeps globally self-normalized joint raw ratios.

New reports:

- `raw.json/png`, `calibrated.json/png`: original full-truth comparison,
  with additional reference comparisons and signed matrix decomposition.
- `raw_matched.json/png`, `calibrated_matched.json/png`: matched-target
  results, with paired event-bootstrap CIs for all nine elements and norm change.
- `*_angular_moments.json`: sensitivity to inverse analyzing-power factors,
  reporting **9 times the angular-product mean, NOT physical Cij**. No powers
  are changed in the main Cij reports.
- `diagnostics`: actual base/combined-weight ESS, analyzing-power quantiles,
  inverse-product tails, per-cell top-1% absolute event influence. Good ratio
  ESS does not imply a stable inverse-power moment estimator.

All reference comparisons within a report use the same event-bootstrap draws.
The signed identity `sample - stored = (sample - matched) + (matched - stored)`
is exact. Norms do not add; these reference shifts are **not** causal shares
or a proven irreducible physics-error floor.

Reprocess the already-saved FiLM scores, with a **new online W&B run**:

```bash
shifter python3 -u scripts/rescore_film_cij.py \
  --analysis-only /pscratch/sd/y/yiren/Ztautau/h4_film_step1110_cij/rescore-ca89b70725 \
  --kappa-signs 1 1 \
  --allow-truth-mismatch
```

This is CPU postprocessing only; it reuses the existing 16-GPU inference
output. W&B receives both comparison types, the angular-moment sensitivity,
weight/influence diagnostics and complete JSON reports. Existing report files
in this exact rescore directory are refreshed; scores/checkpoints are unchanged.
`cij/<raw|calibrated>/matched_target/error_change/*` is the new matched
diagnostic; `.../full_truth/error_change/*` retains the physics reference.
Negative norm-error change means closer to the explicitly named reference.

### Physics checks and remaining scope

- Local boosts match the supplied TT2L `Core`, including non-collinear,
  boosted synthetic decays. Axes are common A k/r/n, with TT2L's fixed +z
  reference in the pair CM. We have not silently changed this to a different
  beam convention or inferred charge flips from the desired result.
- In the adopted signed-kappa convention, integrating a density proportional
  to `1 + kappaA*kappaB*Cij*a_i*b_j` gives
  `Cij = 9 E[a_i*b_j]/(kappaA*kappaB)`. An exact angular-quadrature regression
  test checks the factor 9, sign, indexing and channel-dependent constant kappas.
- **`--kappa-signs 1 1` multiplies the stored powers by +1; it does NOT set
  both analyzing powers to one.** The converter passes those fields through
  from upstream and does not define their decay-channel meaning. The bridge
  cannot certify their physical validity or their independence from generated
  variables. Per-event inverse powers require an appropriate polarimeter model.
- A tau polarimeter is channel dependent and is not generally just the visible
  momentum direction. Full polarimetry and acceptance/response unfolding remain
  outside this moment comparison. See the primary research discussion of
  [tau-pair kinematic reconstruction and polarimeters](https://link.springer.com/article/10.1140/epjc/s10052-026-15804-y).
- New tests explicitly show that perfect truth direction predictions close the
  matched map while still differing from full truth when visible/energy inputs
  differ. They do not establish what fraction of the real-data gap this explains.
- This saved classifier's full-validation fit overlap is not established.
  These bootstrap intervals condition on fixed scores and candidates and are
  exploratory, not fresh-classifier or population-unfolding uncertainties.

## Executed result and diagnosis — 2026-09-29

W&B [`ehw9lwmt`](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/ehw9lwmt),
**finished**, display name `Does reweighting improve matched Cij? | saved FiLM
scores | reconstruction controls`. Complete uploaded JSON reports and history
were inspected, not only summary values. Reuses 119,002 validation events,
K=1, original step-1110 policy draws, and the kinematic FiLM pre-update step-0
reward stack (four increments). No generation, classifier fits, clipping,
tempering, policy updates or selections were added.

Question: is the previous Cij worsening explained by an inconsistent truth
reconstruction or inverse-analyzing-power amplification?

| Reconstruction / reference | Unweighted norm error | Reweighted norm error | Change, paired 95% CI |
| --- | ---: | ---: | --- |
| Raw / stored full truth | 0.591019 | 0.747331 | +0.156312 [0.017084, 0.670733] |
| Raw / matched target | 0.521308 | 0.677790 | +0.156482 [0.035813, 0.688334] |
| Calibrated / matched target | 0.399416 | 0.610073 | +0.210656 [0.081165, 0.745784] |

The full-truth to matched-raw matrix distance is 0.305502, so reconstruction
does matter to the absolute reference. Nevertheless, raw reweighting worsens
the matched comparison by almost exactly the same amount as the full-truth
comparison. The stored/recomputed full-truth matrix distance is only 1.7171e-5.
For raw C_rk, matched truth/unweighted/reweighted are
+0.116778 / +0.018782 / -0.276150; not every cell worsens (four of nine improve).

Stored |kappa| ranges from 0.33 to 1; max 1/|kappaA*kappaB| is 9.182736.
There is no near-zero-power numerical singularity. Without inverse-power
division, the raw matched **9-times-angular-moment** norm error still worsens
1.763121 -> 2.356585 (+0.593464, CI [0.544250, 0.644487]); this sensitivity
observable is not physical Cij and its magnitude is not directly comparable
to the kappa-corrected norm.

Ratio ESS/N is 0.0803866, top 1% mass 0.255968 and maximum event mass 0.000864.
The top 1% of absolute event influences account for 36.8–42.9% of each raw
reweighted matrix element's total absolute influence (different events may
enter each cell's top 1%). This is concentration, not proof of its causal role.

Hypothesis status: inconsistent reference and near-zero kappa amplification
are **ruled out as sufficient explanations** of this worsening. The fixed raw
ratio fails this matched moment endpoint. Ratio calibration, event-category
mixture shifts and within-category distortion remain **unresolved**; do not
attribute the failure to diffusion conditioning from this postprocessing run.
A useful next diagnostic is an exact category-mixture/within-category
decomposition of the same weighted moments, without fitting cuts or altering
the saved ratio. It has not been implemented or run by this result check.

Limitations: fixed-score/candidate bootstrap only; latest-classifier fit
overlap and upstream polarimeter validity remain unestablished; no acceptance
unfolding or entanglement claim. The exploratory finite truth-angle mismatch
was explicitly allowed and not silently repaired.

## Historical step-1110 / h4ratio1 replay

`scripts/diagnose_1110_cij.py` now directly consumes the historical
`h4_scale_standardized_long-ratio-health/ratio_audit/best_classifier_and_test.pt`.
The companion resolved config must name the canonical c4a91e07 step-1110
checkpoint. This is an old-DGPO endpoint, not pure diffusion pretraining.
The saved best-BCE H4 classifier and its independent test scores are reused;
no model loading for inference, EMA selection, generation, training or Ray is
needed. Split identity disjointness and exact unique parquet identity recovery
are checked. Original physical truth deltas are checked by the downstream bridge.

Run first on NERSC (after syncing the new scripts into the existing repo):

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_1110_cij.py --export-only
```

Then use the same command without `--export-only`, adding
`--kappa-signs SIGN_A SIGN_B`.
Those signs must be established from the tau analyzing-power definition,
not the top default. No TT2L checkout is required on NERSC. All other source,
event and output paths have defaults recovered from the historical configs.
`--events` can override the converted parquet if physics columns are stored
in a different sidecar; matching is by exact unique visible context and IDs,
never row position.

This artifact has K=1, so **joint** raw `exp(logit)` weighting is mandatory:
within-event normalization would erase all weighting. It changes event mixture
as well as generated-target moments and does not establish conditional closure.
Outputs under `/pscratch/sd/y/yiren/Ztautau/h4_step1110_cij/` include source
provenance, aligned candidates and separate raw/calibrated Cij JSON/PNG reports.
No clipping or fitted temperature is introduced. This reused historical test
panel is exploratory, not fresh confirmatory evidence. Source artifacts remain
unchanged; outputs in the chosen output folder may be overwritten on rerun.

## Scope

Read-only reference: `/Users/yirenwu/Ztautau/TT2L-QC-Study/analysis_core/{basis,core}.py`.
No edits to that repository. This experiment compares the full 3×3 C matrix,
not concurrence, Bell violation or an entanglement significance claim.

Estimator (unbinned, includes exact-zero observations):

`C_ij = 9 / (kappa_A*kappa_B) * weighted_mean(a_i*b_j)`.

Axes are **k,r,n** and rows are A, columns B. A is the positive parent / child
and B the negative one in the TT2L reference. The reference constructs axes
from parent A in the pair CM frame, then measures each child in its own parent
rest frame against these axes. Input cosines must use that consistently for
truth and generated samples; they are not lab-frame angles.

The reference TT2L default `(1,-1)` is NOT a Ztautau default. Tau decay
polarimeters, channel-dependent analyzing powers, charge signs and acceptance
must be established before calling these moments a physical spin C matrix.
The generic analyzer supports constant or per-event nonzero analyzing powers,
using an explicit inverse-product moment estimator. It does not implement
event-dependent vector polarimeters or detector-response unfolding.

## Input contract — one NPZ, no positional join of separate files

- `event_id`: unique `[N]` held-out source event IDs.
- `truth_a`, `truth_b`: `[N,3]` unit-direction cosines in k,r,n order.
- `sample_a`, `sample_b`: `[N,K,3]` candidate direction cosines.
- `log_ratio`: `[N,K]`, oriented log(p_truth/p_sample), already computed using
  held-out or out-of-fold scoring. Do not infer orientation from AUC.
- `event_weight`: nonnegative `[N]` base MC weights; use ones for unweighted MC.

Truth and candidates must represent the SAME selected events, and the same
candidate set is used before/after reweighting. Save provenance for selection,
kinematic reconstruction, classifier checkpoint and scoring split. Invalid
events must be investigated, not silently dropped only from one arm.

`joint` exponentiates the raw log ratio and normalizes globally in the moment
estimator; it can alter the visible event mixture. `conditional` normalizes K
candidate weights separately inside each event, preserving the event mixture
but defining a finite-K self-normalized estimator. Choose explicitly based on
the experiment; they are different targets. No implicit clipping/tempering.

## Outputs and decision

JSON + PNG: truth/unweighted/reweighted matrices, signed residual matrices,
cell-wise absolute-error changes, Frobenius-error change, paired event-bootstrap
95% intervals, candidate and event ESS, maximum weight masses.

Primary: `||C_reweighted-C_truth||F - ||C_unweighted-C_truth||F`.
Negative means closer to truth; upper CI below zero supports improvement.
Do not select cuts/weights on this test panel. Bootstrap resamples whole events,
keeping truth and all candidates paired; it conditions on the fitted classifier
and does not include classifier-fit uncertainty. Matrix CIs are pointwise.

Example ONLY after the TT2L input contract is satisfied:

```bash
python3 scripts/diagnose_reweighted_cij.py /path/to/aligned_angles_weights.npz \
  --output /path/to/cij_report.json \
  --kappas 1 -1 --weight-mode joint \
  --physics-definition 'TT2L; A=l+ B=l-; common k,r,n helicity axes; acceptance treatment: SPECIFY' \
  --bootstrap 1000
```

This is CPU postprocessing, not a training job; 16 GPUs are unnecessary.
## Ztautau bridge (implemented)

`scripts/export_ztautau_cij_candidates.py` reads the existing `fixed_k8_panel.pt`,
`fixed_k1_pool.pt`, and `counterfactual_weights.pt`. It checks the companion
weight report's source directory, recovers composite source IDs by exact unique
float32 visible-context matching against the converted parquet, and exports a
keyed NPZ. It rejects ambiguous/missing matches; it never joins by row order.
Use the ORIGINAL unmodified source panel and weight artifacts. The historical
weight file contains no candidate digest; it must not be transplanted from a
different replay with identical shapes.

```bash
python3 scripts/export_ztautau_cij_candidates.py \
  --source /path/to/reward_interface_source \
  --weights /path/to/candidate_reweighting/counterfactual_weights.pt \
  --events /path/to/converted/parquet_directory \
  --arm EXACT_SAVED_ARM_KEY --output /path/to/cij_candidates.npz
```

Append `--list-arms` to print saved arm names before choosing; no export occurs.

Then run:

```bash
python3 scripts/diagnose_ztautau_cij.py \
  --events /path/to/converted/parquet_directory \
  --candidates /path/to/cij_candidates.npz \
  --output /path/to/cij_comparison \
  --kappa-signs 1 1 --weight-mode conditional --bootstrap 1000
```

**Do not assume `--kappa-signs 1 1` is correct without checking upstream metadata.**
This example means the stored `analyzing_power_a/b` ALREADY include the correct
charge-dependent signs. The explicit factors multiply the stored values;
zero/missing powers fail rather than silently substituting top kappas. The
estimator is `9 * mean_w(cosA*cosB / (signed kappaA*kappaB))`. It is not a
histogram-fit estimator or division by the average analyzing power. For mixed
channels this requires the polarimeter assumptions for each channel; acceptance
and backgrounds are not corrected by this script. Use a predeclared boolean
`--selection COLUMN` for the intended signal/category region.

The bridge validates physical truth deltas against aligned parquet four-vectors,
optionally compares its vectorized boosts to TT2L `Core` when `--tt2l-repo`
is supplied for development, and checks truth angles reconstructed from truth
visible/tau p4 against the stored truth cosines. A mismatch stops by default;
the explicit `--allow-truth-mismatch` override retains stored truth and reports
the discrepancy. No sign flips are chosen to improve closure.
Raw candidates use the existing fixed-E tau prescription (E=45.6, m=1.777 GeV).
Calibrated candidates separately use the existing back-to-back prescription.
The two reports use the SAME saved weights, truth and event selection.

Produces `raw.json/png` and `calibrated.json/png`, each containing all nine Cij,
residuals and paired event-bootstrap comparisons. Only selected-sample moment
closure is claimed. Requires numpy, torch, pyarrow, pandas, scipy, vector and
matplotlib; no ROOT, Ray, GPU or remote training for the Cij postprocessor.
The NumPy angle/boost implementation lives inside ml_pipeline. No external
quantum module, TT2L checkout, or EveNet-private checkout is needed at runtime.
The fresh sampler additionally requires Ray, CUDA PyTorch and W&B from the
NERSC container, plus the existing data, normalization and checkpoint files.

Tests:
`python3 -m unittest scripts.test_reweighted_cij scripts.test_ztautau_cij -v`.
Only the optional live TT2L parity test skips if that checkout is unavailable.
The end-to-end fixture and fixed reference values run without it.
Production files have not been executed by these synthetic tests.
