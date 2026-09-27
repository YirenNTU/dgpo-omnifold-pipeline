# Does coarse conditional reward hide DGPO difficulty?

2026-09-24 local exploration; two1000-step paired runs. Authorized locally.

Source for BOTH: artifacts/dgpo_toy/bump_dgpo_narrow_pilot1/source.pt,
selected no-Fourier epoch4, seed17, narrow_bump width.05.
No source retraining, optimizer/state inheritance, classifier, or NERSC work.

Only intervention:32 versus128 condition bins in the fixed estimated reward.
Each run internally compares None/Early Fourier with exactly matching starting
DDIM samples and common update/evaluation RNG, following BUMP_DGPO_PROTOCOL.md.
Both resolutions calibrate against the SAME32768 contexts x64 source samples.
The previous8192-context run is not the matched resolution control.
Output bins remain64, including overflow. Total prior mass across the whole
conditional grid remains constant: pseudocount.5*32/condition_bins per cell.
Analytic truth probabilities are integrated on the SAME4096-point c grid.

Instrument validity: compare centered per-context rewards from two independent
calibration halves on the initial held-out candidate panel. Require cosine>=.95
before calling any observed failure-and-rescue. This is an estimate-stability
check, not proof of continuous density-ratio calibration or perfect resolution.

Primary: fresh final4096-context paired reward gains, same5%-of-initial-tilt
headroom margin and joint baseline-stall/Fourier-advantage requirement.
Do NOT compare absolute reward gains across resolutions as an identical scale;
the frozen rewards differ. Report normalized headroom fractions descriptively.

If both controls absorb reward,32-bin smoothing alone is insufficient to explain
the absence of a failure here. If fine reward alone exposes failure, first check
split calibration reliability before attributing it to conditioning.
This is one exploratory seed; no general EveNet inference or frequency sweep.

## Completed result

Both processes completed1000 steps successfully.38 targeted tests pass.

| c bins | None fresh reward gain [95%] | Fourier gain [95%] | Fourier minus None [95%] |
|---|---|---|---|
|32| .05555 [.05134,.05976] | .07755 [.07147,.08364] | .02200 [.01913,.02487] |
|128| .11952 [.10922,.12982] | .15488 [.14205,.16771] | .03536 [.03121,.03951] |

Calibration split cosine.999322/.998430, both pass. Initial reward ESS
fractions.936754/.903878. Both None baselines improve decisively; both Fourier
arms improve more. Failure-and-rescue=false for both. Coarse32-bin conditional
smoothing is ruled out as a sufficient explanation for the absence of failure
at this source/target/budget. Reward scales differ; raw gains across rows are
not a same-reward improvement claim.

Output directories: artifacts/dgpo_toy/bump_dgpo_resolution32_v1 and
artifacts/dgpo_toy/bump_dgpo_resolution128_v1. Shared source unchanged.
No processes from this round remain running (both exited0).

Learning: two matched local pilots resolve one discretization hypothesis;
no longer spend iterations narrowing scalar bumps solely to force a failure.
The major remaining mismatch is scalar mean correction with high reward ESS,
rather than a residual conditional joint dependency after low-order closure.
