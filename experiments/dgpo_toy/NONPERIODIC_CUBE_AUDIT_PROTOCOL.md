# Matched fresh Fourier classifier audit

Local user-authorized run2026-09-24. No policy updates or remote uploads.
Compare source reference,raw DGPO1000,and Fourier DGPO1000 from
artifacts/dgpo_toy/nonperiodic_cube_dgpo_v1.

Three fresh Critic(True) instances, same seed41 initialization; no inherited
reward-critic weights or optimizer. Identical architecture/Fourier inputs to
the installed critic. Same32768train/8192validation/16384test context/truth
draws across arms, common DDIM noise; new seeds910041/920041/930041 disjoint
from reward fitting and policy evaluation. Every minibatch uses same indices.

AdamW3e-4,wd.001,balanced512-sample minibatch,clip1. Evaluate every100.
Minimum2000updates; joint20-stale-check patience,min_delta1e-4; max16000.
Each arm selects exact minimum validation BCE. Test only after training ends.
Max budget without plateau is explicitly inconclusive, not a chance classifier.

Primary fresh test AUC changes versus newly fitted baseline, with300 paired
context bootstrap replicates preserving truth/generated pairing and arm pairing.
Report BCE and weight ESS too. Intervals conditional on fitted classifiers;
single fit seed does not establish robustness. This tests distribution
discriminability,not whether the learned reward isolates parity alone.

Training minibatch BCE,validation BCE/AUC,selected steps,stale counters,
best and last optimizer states,panels,test scores and report saved locally.
Preflight: paired bootstrap gives exactly zero difference for identical scores.

## Completed result

Joint early stopping at9800 updates; all arms satisfy significant-improvement
patience. Exact best-validation checkpoints: baseline7400,raw9000,Fourier7800.
Held-out test:

| Policy | Cold AUC | Cold BCE | AUC change from fresh baseline [95%] |
|---|---|---|---|
| initial | .647512 | .652503 | — |
| raw DGPO1000 | .661324 | .647960 | +.013812 [.009826,.017830] |
| Fourier DGPO1000 | .662163 | .646849 | +.014651 [.010541,.018992] |

Fresh classifiers distinguish both endpoints MORE easily, despite fixed reward
gains+.15684/+.16063 and almost unchanged conditional parity error in the policy
experiment. This reproduces a phenomenological reward-versus-fresh-audit mismatch,
not its production cause. Fourier diffusion has no established rescue here.
One fit seed; intervals do not include classifier training-seed uncertainty.
Corner-shape improvement is observed but reward attribution remains unmeasured.
