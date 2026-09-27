# Direct endpoint-KL: completed 10k to 20k continuation

2026-09-21; local CPU, seed17. Output:
`artifacts/dgpo_toy/direct_endpoint_kl1_joint3_resume20k_v1/`.
Protocol: [DIRECT_ENDPOINT_KL_EXTENSION_PROTOCOL.md](DIRECT_ENDPOINT_KL_EXTENSION_PROTOCOL.md).

Completed exactly 10,000 new updates, ending at 20,000, in 3121.69 seconds
(52.03 minutes). Process exited successfully. Full model/AdamW/RNG/history and
density clock were resumed from 10k; the original step0 reference and frozen
H4 remained unchanged. Boundary monitor and historical-prefix checks pass.
Coefficient1, LR1e-4, weight decay.001, batch64/K8/M4 and DDIM20 are unchanged.
No new classifier fit, production run or W&B write.

## Independent endpoint

Predeclared panel97017: 4096 contexts x8 common-noise candidates. The saved
10k and final20k models are evaluated on this SAME new panel. It differs from
the original10k endpoint96017 and the reused monitor90017; do not subtract
results from different panels. Intervals describe evaluation sampling
uncertainty conditional on this one trained trajectory, not training-seed
variation.

| Measurement | Step0 | Step10k | Step20k |
|---|---:|---:|---:|
| Fixed H4 raw reward mean | -2.16774130 | -2.16804838 | -2.16679287 |
| Raw reward std | .78959954 | .79024988 | .79408866 |
| 56-moment RMSE (lower is better) | .491144513 | .491121921 | .491116214 |
| Raw ratio ESS/N | .00295696 | .00303252 | .00335081 |

- **Reward20k minus10k:** +.001255374, paired95%CI[+.000195818,+.002314929].
  A small detectable reward increase on this independent panel; not zero.
- **High-order MSE20k minus10k:** -.00000560616,
  paired-bootstrap95%CI[-.000101730,+.000104126]. Unresolved.
- Joint cosine20k minus10k: -.0000662926,
  95%CI[-.000311278,+.000178693]. Unresolved.
- Reward20k minusstep0: +.000948422,
  95%CI[-.0000768796,+.001973723]. Still far below declared meaningful gain.10.
- High-order MSE20k minusstep0: -.0000277970,
  95%CI[-.000134316,+.0000739632]. No resolved truth-alignment improvement.
- Lower-order preservation passes: max residual mean.0136728,
  max variance error.0135893, max pair covariance.0175961.
- Direct endpoint reference KL=.000677628,
  95%CI[.000327104,.001028152]. The policy remains close to its initial reference.

Extension gates: reward increase TRUE, high-order improvement FALSE,
lower-order preservation TRUE. Combined `unresolved_at_extended_budget`;
original meaningful-alignment decision `no_decisive_alignment`. The latter
labels must not erase the small positive reward contrast versus10k.

## Trajectory and numerical checks

The unchanged monitor does not show sustained reward accumulation. Its mean
step0-relative gain is .00083338 at9001-10000, .00075525 at10001-11000,
.00073278 at12001-13000, .00034498 at15001-16000, and .00001703 at19001-20000.
These reused correlated points are descriptive, not independent replicates.
The fresh endpoint is the primary comparison; its small positive contrast
and the monitor's absent sustained trend both belong in the interpretation.

Across all 10,000 NEW updates: maximum current inverse residual4.20e-14,
minimum sufficient certificate margin.288812, zero clipped gradients and
zero gate saturation. Final autodiff logdet discrepancy4.74e-15. No validity
stop. These are numerical checks, not interval-arithmetic proofs.

## Bounded conclusion

More time alone yielded a tiny detectable endpoint reward increase, but did
not establish high-order truth alignment within20k. It does not prove eternal
stagnation or identify low ESS as the cause. No claim of complete reward
gradient cancellation is justified. Approximate H4 ratio fidelity and the
DGPO surrogate remain distinct unresolved issues.

Historical no-KL and velocity controls are still10k, explicitly budget-unmatched
to this20k arm. Their fresh-panel numbers do not constitute a new matched
20k comparison. No-KL retains its successful reward-transfer interpretation;
unconstrained marginal drift does not veto that control's intended endpoint.

This closes the authorized extension. No run beyond20k has been launched.
