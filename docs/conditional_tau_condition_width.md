# Condition width ablation: 64 to 256

Completed as `zrv2yfgt`. One new classifier fit, not DGPO or diffusion
training. Tests whether wider conditioning improves conditional tau/spin closure.
It changes branch capacity as well as compression; a positive result cannot by
itself establish that the original encoder discarded information.

Completed K64 endpoint: Cij Frobenius error0.256479 versus context64
0.373213, paired difference -0.116734 (95% interval[-0.144951,-0.052436]).
Unweighted error0.565440. Seven of nine component point estimates improve versus
64, but coarse within-group error is not uniformly improved and K1 remains worse
than unweighted. Supports useful condition-branch capacity, not full closure.
Saved output: `/pscratch/sd/y/yiren/Ztautau/conditional_tau_condition_width_1110/condition256-40e1f8b288`.
Next matched input arms are documented in `conditional_tau_explicit_inputs.md`.

## Intervention and controls

| Part | Saved bound30 control a7mczoed | New arm |
| --- | --- | --- |
| Condition encoder | input ->128->64, SiLU | input ->256->256, SiLU |
| Per-block context projection | 64->128 | 256->128 |
| Candidate encoder | input ->64->64 | unchanged |
| Fusion | three residual FiLM blocks, hidden64 | unchanged |
| Output | smooth bound30, paired BCE | unchanged |

The actual saved input has517 coordinates, but width is read from the verified
prepared data. No new input feature, energy, spin observable, Fourier feature,
normalization change, or candidate embedding change is introduced. Frozen raw
step1110 trunk and fixed K1 negatives remain identical. Shared candidate,
FiLM-hidden and readout initial tensors are exactly equal at seed42. Context
projections start at zero in both, yielding identical initial outputs; a forked
RNG when constructing new layers preserves shared initialization and the global
RNG stream. Condition encoders themselves differ. This is fresh classifier
training, not loading the trained narrow head into the wide one.

Reuse a7mczoed's trained best.pt, K1 and K64 scores, and completed JSON endpoints;
its reporting-only failure does not invalidate these saved results. Recovery run
u6yoq4jw uploaded those reports. Do not require a fabricated COMPLETE marker on
the original directory. Preflight validates its checkpoint, paired identities,
prepared arrays, training settings, source runs and endpoint availability.
Final analysis recomputes and verifies the saved narrow-head Cij/ESS endpoints.
The unbounded pzq0nl1i and its post-hoc cap30 remain secondary controls, not the
primary matched width comparator.

## Fixed setup

16 GPUs,1024 paired conditions/GPU; filtered416701 train events, split354488 fit
and62213 internal validation, external119002. Existing completed filter manifests
and exact sample/feature sources are checked through the original prepare chain.
Raw1110 checkpoint normalization and cached representations are unchanged.
AdamW2e-4, weight decay.001, dropout.05, cosine250 epochs to1e-5;
patience25/min_delta1e-4/min_steps1000. Minimum internal validation BCE selects
best.pt. No Cij-based selection. Frozen candidates: no new DDIM generation.
Only the classifier head is trained. All new K1/K64 scoring uses16 GPU workers.

## Diagnostics and decision

Retain every-epoch BCE/AUC/LR, ESS/tails, bound saturation and context projection
norm; inherited conditional diagnostics every5 epochs. Log both parameter counts
and input/hidden/context widths. New checkpoint dimensions persist for strict
reload in K1 and K64 inference; old checkpoints default to their original widths.

Primary: K64 Cij Frobenius error `bounded_minus_condition64`, where `bounded`
denotes the NEW context256 head, on exactly the same119002 x64 candidates.
2000 paired whole-event bootstraps. Compare against unweighted and old cap30
as well. Report all nine entries, absolute errors, pointwise component-difference
intervals, and category x fit-pT group tables including condition64. Bootstrap
excludes model-fit uncertainty and unseen tails; this repeatedly inspected test
population is exploratory, not a new independent confirmation. A scalar norm
gain cannot establish all-component closure, and multiple component intervals
are not simultaneous confidence statements.

Improved BCE alone is not a physics success. Lower Cij error against the narrow
bounded control, with uncertainty and component/group damage inspected, supports
wider condition-branch capacity as useful. A negative result weakens this
specific64-dimensional-bottleneck hypothesis, not all conditioning hypotheses.

## NERSC user launch

Update the existing ml_pipeline checkout, then on the existing16-GPU Ray cluster:

```bash
shifter python3 -u scripts/run_tau_condition_width.py \
  config/conditional_tau_condition_width_10pct.yaml
```

Append `prepare` to verify inputs/controls without training. New output root is
`/pscratch/sd/y/yiren/Ztautau/conditional_tau_condition_width_1110`.
Creates a new online W&B run; never overwrites old runs/checkpoints or refits the
control. The user submits compute jobs. No remote job is launched by implementation.
