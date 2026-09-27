# Local nonperiodic DGPO exploration — 2026-09-24

Two completed seed17 pilots,1000 policy updates each, shared initial checkpoint
within each pair, fixed binned ratio, native DGPO + velocity-MSE coefficient1.
Both use the same conditional Gaussian truth for source pretraining and DGPO.
Normal bump width.15; second pilot changes only target width to.05 (unit
variance normalization remains, so peak height also changes). Source readiness
selects epoch2 and4 respectively; cross-target comparisons are not matched
initial-checkpoint causal effects. Within each target, the Fourier comparison is.

Fresh final4096-context x32 sample reward gains:

| Target | None gain [95%] | Fourier gain [95%] | Fourier minus None [95%] |
|---|---|---|---|
| bump .15 | .12520 [.12040,.13000] | .12955 [.12444,.13467] | .00435 [.00348,.00522] |
| bump .05 | .05521 [.05100,.05942] | .07532 [.06932,.08132] | .02011 [.01735,.02287] |

Neither baseline stalls under the declared threshold. No failure-and-rescue
has been reproduced. Narrow bump shows a material Fourier advantage, not a
rescue of inability to absorb reward. Initial reward ESS fractions are high
(narrow .9417); these are not replicas of concentrated real H4 ratios.

Do not continue narrowing blindly:32 condition bins already coarsen the.05
bump. A narrower case could primarily test reward discretization. Next unresolved
link is whether reward complexity/conditioning in the real high-dimensional
problem differs from this coarse scalar ratio. Needs a separately declared
representation or reward-resolution test, not more steps in these passing cases.

The original progress JSONL completion line used `fourier_rescue` for material
advantage alone. This label was corrected in code and final report.json;
historical logs are retained. No measurements changed. True failure_and_rescue
requires baseline stall AND Fourier material advantage; false in both runs.

Outcome: target not being sinusoidal alone is insufficient to reproduce failure
in these two settings. No claim about EveNet or general DGPO convergence.
