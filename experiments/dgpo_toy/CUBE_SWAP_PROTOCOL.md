# Conditional mode/shape swap diagnosis

Existing DGPO endpoints and learned critics only; no diffusion updates.
Default run computes fixed-reward attribution and existing cold-judge diagnostics.
Optional --fit-audits additionally cold-fits separate swap classifiers. Never
describe default judge scores as fresh swap-specific discriminability.

Condition grid128(midpoints of[-1,1]); grid64 sensitivity repeat supported.
At each c independently generate1024 donor samples per policy/split,DDIM50.
Use all8 sign modes,not only parity. Require>=16 donors in EVERY c/mode cell;
stop instead of borrowing from another condition. Preserve full3D within-mode
shape by resampling complete donor vectors,not independent coordinates.

A=truth mode probabilities + truth shape;
B=generated modes + truth shape;
C=truth mode probabilities + generated conditional mode shape;
D=direct generator samples. A/C share mode ids; B/D share mode ids. Identical
conditions and independently sampled positive truth across all four panels.
Truth shape is width.15 Gaussian restricted to its orthant; this changes original
truth by <1e-10 crossing mass,explicitly not an exact untruncated mixture identity.
Training/validation/test donor pools have separate seeds. Test sample size256
per condition. Grid comparisons diagnose a discrete conditional distribution,
not unseen-continuous-condition generalization. Empirical donor resampling can
duplicate vectors; sparse counts are reported,not hidden.

Fixed reward decomposition: R(D)-R(A)=[R(B)-R(A)]+[R(D)-R(B)]. Also compute
R(C)-R(A) and interaction R(D)-R(B)-R(C)+R(A). Between-policy reward changes
split as delta R(B)+delta[R(D)-R(B)]. This is a distribution decomposition,
NOT causal attribution of optimizer/KL effects or proof of reward hacking.

Existing cold judge is each policy's previously fitted best classifier. Its swap
scores are OOD diagnostics; raw AUC may invert. No inferred Bayes separability.
Optional fresh fits: common cold seed73,128-wide Fourier MLP,32768train(grid128),
16384validation,32768test,AdamW3e-4,wd.001,512balancedbatch,minimum2000 updates,
joint patience20 checks every100 with min_delta1e-4,max16000. Best validation
BCE selects checkpoints. Budget-limited fits marked plateau=false. One common
A sanity fit plus3 policies x B/C/D fits. No classifier seed robustness claim.

First execute default scoring at128 and64 to check donor adequacy and resolution
stability. Fresh swap fits are implemented but require a separately launched
--fit-audits run; score-only outputs must explicitly mark these unexecuted.

## Executed score-only results

128-grid: minimum donor count per c/mode93/94/96 for baseline/raw/Fourier;
64-grid minimum101/96/97. All pass>=16; no condition mixing or missing-cell fallback.
Three unit tests pass (mode preservation, condition pairing, missing-cell stop).

| Grid | Policy | Total reward gain | Mode-at-truth-shape component | Shape-at-model-modes component |
|---|---|---|---|---|
|128|raw| .150515 | -.001273 | .151787 |
|128|Fourier| .148076 | -.001039 | .149116 |
|64|raw| .143743 | .000112 | .143632 |
|64|Fourier| .148326 | -.000105 | .148432 |

These are Monte Carlo point estimates; no training-seed robustness claim.
Mode contribution near zero and shape term dominates at both resolutions.
Shape term includes interactions under model mode weights; it is not restricted
to low-order marginals. Conditional shapes themselves may contain multivariate
structure. This does not identify KL/optimizer causality or prove reward hacking.

Existing policy-specific cold judges:128-grid A truth-vs-truth AUC approximately
.5005/.4986/.4987; B mode-only .5972/.5838/.5837; C shape-only .5564/.5805/.5807;
D original samples .6491/.6640/.6640. Since judges differ and were not fitted on
swaps, do not interpret these as comparable fresh best-response swap AUCs.
Truth-vs-truth BCE need not equal ln2 for these frozen,uncalibrated OOD judges.

Artifacts: artifacts/dgpo_toy/cube_swap_grid128_v1 and cube_swap_grid64_v1.
Both analyses completed; fresh swap-specific classifier fits remain UNEXECUTED.
