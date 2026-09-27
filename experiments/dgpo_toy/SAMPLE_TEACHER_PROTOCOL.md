# Sample-level oracle teacher — 2026-09-22

Replace failed batch-moment warm objective ONLY. Start same old raw Gaussian
baseline; architecture and DGPO pipeline unchanged. Given c and initial Gaussian
noise z, map u=Phi(z). For each true triple keep u1,u2, invert conditional CDF
F(v)=v+.6/(2pi)*(sin(2pi*v+offset)-sin(offset)), offset=2pi(u1+u2)-phi(c).
Density1+.6cos(triplephase) is strictly positive. Ideal marginal/pair distributions
unchanged; expected triplecos=.3. Groups independent, unlike full truth mixture.
No Bernoulli switches or target noise unavailable to model. Clamp only inverse-CDF
teacher tails at1e-12, not generator outputs. Teacher targets detached.

Warm loss is per-sample MSE between actual DDIM20 output and deterministic teacher
target for same c/noise. No oracle moment reward, marginal penalties or samplev loss.
AdamW3e-4/WD.001,batch512,clip1,max3000. Same selection/confirmation gates as previous
ORACLE_WARMSTART_PROTOCOL.md; validate teacher separately before fitting.
Primary prerequisite is learned independent coverage, not teacher availability or
lower supervised error. If it fails, no new classifier or DGPO and no relaxed gates.
If it passes, exact strong joint3 streaming fit then10k native DGPO with velocityMSE1
(explicit user correction; no lambda.1 sample-teacher run was launched).
No teacher call in policy loss or generation after initialization. Raw/noEMA.

Command: python -u -m experiments.dgpo_toy.oracle_warmstart --warm-mode sample_teacher --coefficient 1 --output artifacts/dgpo_toy/oracle_sample_teacher_v1
