# Nonperiodic high-order structure: classifier gate before DGPO

Local run authorized2026-09-24; no remote jobs/uploads. Prepared one seed pilot.

Truth: noisy3D cube corners with independent Gaussian width.15. Observed scalar
c is uniform[-1,1]. Positive parity probability=.5+.4*g(c), where g is tanh of
four signed Gaussian bumps at[-.77,-.31,.12,.63],width.06,amplitudes[1,-.85,.95,-1].
Reference: uniform eight modes (positive parity probability.5). Every conditional
single-variable and pair distribution is identical analytically. Triple parity
alone changes. No parity, bump formula, or truth mode id enters a learned model.

Stage1: ideal reference vs truth classifier gate.32768/8192/16384 paired-context
train/validation/test panels. Plain/Fourier: identical36->128->128->128->1 GELU
MLPs and initial weights, AdamW3e-4,wd.001,512 balanced samples,shared minibatches.
Plain uses raw[c,y] plus32 zero slots; Fourier activates coordinate sin/cos k1..4
(c scale pi,y scale pi/2). Nominal capacity equal; inactive input slots mean
effective first-layer input rank is intentionally different. No standardization
or additional architecture advantage. Both run same available update budget.

Minimum2000 classifier steps; evaluate every100; joint patience20 checks with
min_delta1e-4, maximum6000. Select exact best validation BCE, test only afterward.
If either lacks20 stale checks, comparison is underconverged/inconclusive.
Gate requires all: Plain BCE>=ln2-.005 and AUC<=.55; Fourier BCE<ln2-.015 and
AUC>=.60; paired test BCE Fourier-minus-Plain upper95%<- .01. This operationalizes
'Fourier needed at this capacity/budget', not a universal inability theorem.

If gate fails, STOP and record failed prerequisite. No forcing an ablation by
weakening Plain or selecting seeds. No DGPO runs on an invalid premise.

Stage2 (only on pass): ordinary velocity pretrain of no-Fourier diffusion on
uniform reference cube,NOT complete truth.32768 train8192val; earlystop20 epochs,
min_delta1e-4,raw weights,hidden128,AdamW1e-3. Test generated reference: corner
fraction>=.90 and maximum32-bin single/pair sign moment magnitude<=.06. This is
a finite-sample sign-moment gate, not proof of exact continuous marginal equality.
Repeat paired classifier fit against ACTUAL generated samples (seed23) and require
same prerequisite before DGPO; no oracle reward is substituted.

Stage3: freeze actual Fourier classifier; common source diffusion, zero-initialized
condition adapter. Inherit cube ConditionDenoiser basis k=[1,2,4,8],not a learned
target-specific basis; raw counterpart inactive. Verify exact initial velocities
and DDIM samples. Native DGPO,velocity-MSE surrogate coefficient1,1000updates,
K8,4 timesteps,DDIM50,AdamW1e-4,wd.001,identical RNG. No exact-KL claim.

Primary held-out fixed-classifier reward gain, plus conditional parity-bin MAE,
single/pair sign moments and corner fraction. No classifier AUC-closure claim
from a frozen reward increase. Data/checkpoints/progress/report persisted locally.
Any failure stops at its named stage; report distinguishes unrun stages.

## First execution and validation-budget extension

`artifacts/dgpo_toy/nonperiodic_cube_classifier_v1`:
ideal stage passes after5500 updates: Plain test BCE.693278/AUC.495484;
Fourier BCE.668101/AUC.617210. Paired BCE difference-.025177,
95%[-.027700,-.022653]. Both meet plateau criterion.

Reference earlystops at101 epochs. Corner fraction.95676; maximum conditional
single/pair sign moment.04545. Actual classifier6000-step test:
Plain BCE.691318/AUC.530751; Fourier BCE.652330/AUC.647123.
Stopped before DGPO because Fourier best was at6000 with zero stale checks,
not because discrimination failed. This is underconverged, not closure.

Authorized local continuation uses `nonperiodic_cube_classifier_extended_v1`,
same saved reference, same actual panel seeds and classifier initialization;
replays actual fits from step0 with max16000, unchanged patience/gates. This
is NOT a full-state optimizer resume; first run lacked final optimizer storage.
Earlier test has been seen: this remains exploratory, not independent confirmation.
No target changes or relaxed pass thresholds. Ideal fit/pretrain not repeated.
Actual diffusion can retain continuous marginal imperfections despite passing
sign-moment/corner checks; do not claim all learned reward is pure parity.

Extended run completed16000 classifier steps and stopped at actual-classifier
gate; NO DGPO updates ran. Plain test BCE.689236/AUC.548147 (best15700,
3 stale checks); Fourier BCE.652326/AUC.648105 (best6900,91 stale checks).
Paired BCE difference-.036910,95%[-.039820,-.034001]. Discrimination conditions
pass, but Plain has not plateaued; therefore the strict common adequacy gate
remains unresolved. Do not label the actual Plain model incapable, nor claim
DGPO failure/rescue. Both launched processes exited0, no training left running.
Eight targeted model/distribution/fork regression tests pass.
