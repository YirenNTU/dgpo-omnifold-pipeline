# Fixed-transport model cheat —2026-09-22

Delete warm training. Physical model y=T_a(x,c), x=neural DDIM20(c,noise),
a=.6 fixed conditional copula transform from SAMPLE_TEACHER_PROTOCOL.md. Input
to T is base diffusion residual x-mu(c), not original Gaussian noise. This makes
the initialized actual generator partly structured without claiming a learned
mapping. T stays in the model throughout classifier fitting, DGPO and evaluation.
Fourier/true triple geometry is explicitly hardcoded into T. No teacher RL loss.

Coordinate consistency: native DGPO works on x candidates and x noisy velocity
targets. Reward is learned strong f(T(x,c),c). Fixed velocity reference is initial
base network; coefficient1 (user instruction). Do NOT feed y to the base denoising
loss, splice truth into candidates, or claim physical-space velocity-MSE KL.
Continuous T is bijective for amplitude<1; numerical tail clamps from teacher
implementation mean exact global invertibility is not asserted. Report |base
residual|>7 frequency. No pathwise gradients through T in DGPO.

Actual initialized coverage first:4096 contexts,K128,seed450017,paired identity T0
vsT.6, firstK8 nested. Gate anyK8 delta lower95CI>0, jointmean(.1,.5), marginal
maxmean<.1,varerror<.15,paircov<.08. Same region all4phaseerrors<1radian. These are
operational toy gates, not physics tolerances. Fail => no classifier/RL.

Fit SAME old streaming strong joint3 architecture/budget/config/seed17 using real
transformed negatives; truth unchanged. Fixed validation/test panels, train-only
head standardization. Freeze best validation classifier. Native DGPO10k with
same AdamW,K8,M4,train3017,monitor90017. Physical structure every250, full state
every1000. Independent endpoint460017, paired moment bootstrap460018.
Primary: own reward gain and improved56moments vs transformed initial without
low-order gate violations. InitialESS measured, never assumed high. Original
untransformed lambda1 result is a historical reference, not an ESS-only control.
No production/W&B modifications. One local fit+one10k arm, no new seed sweep.

Run: python -u -m experiments.dgpo_toy.direct_transport --output artifacts/dgpo_toy/direct_transport_joint_v1
