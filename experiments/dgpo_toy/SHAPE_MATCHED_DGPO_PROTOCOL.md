# Shape-matched learned reward ablation

User authorizes local experiment. Starting checkpoint unchanged:
nonperiodic_cube_classifier_extended_v1/reference.pt. No full-truth pretraining
claim, no production changes, no NERSC submission.

Single intervention vs previous learned-reward run: classifier positive target
uses truth p(mode|c), but within each mode uses actual initial-generator shape.
Negative examples are actual initial-generator samples. No parity/mode/formula
features enter the classifier or policy. Latent labels used only for constructing
this diagnostic target; this is NOT a production physics-prior-free method.

Reuse audited swap C/D construction with1024 donors per condition; require>=16
per c/mode. Independent train/val/test donor pools. Train128grid x256,
validation128grid x64,test256(interleaved)grid x128; test conditions unseen in
training. Exact c preserved, no cross-condition shape borrowing. This is an
empirical shape match, subject to finite donor noise/duplicate resampling.

Fourier classifier unchanged,seed23,AdamW3e-4,wd.001,512balancedbatch,clip1.
Minimum2000steps,check100,min_delta1e-4,patience20checks,max16000.
Select best validation BCE; require plateau and held-out-grid AUC>.55,
BCE<ln2-.01. Test-used validity gate is exploratory,not confirmatory model selection.
Compare conditional mean learned mode scores to estimated log[p(mode|c)/q(mode|c)]
on independent initial-generation panel; global centered cosine must exceed.5.
This preference check does not prove correct within-mode gradients.

Then frozen classifier DGPO,raw/Fourier conditioning from function-matched start.
Same native loss and previous policy settings,velocity surrogate coefficient1,
1000updates,seed17,K8,4 timesteps,DDIM50,AdamW1e-4,wd.001.
Preserve nonlinear DGPO gate and unscaled leave-one-out advantages. No pathwise
reward gradients, changed reward normalization, or exact-KL claim.

Primary descriptive outcomes: independent continuous-context reward gains,
128-grid full8-mode TV change,parity MAE,low-order moments/corner diagnostics.
Material reward improvement: paired95% lower bound>.01. Mode TV reduction>=.01
is a pilot signal,not a CI-backed claim. A rescue requires raw small reward gain
and Fourier material advantage; otherwise call relative improvement only.
Fixed reference-shape score table splits total reward change into mode-probability
and residual within-mode-shape contributions. It is not KL/optimizer causality.
Saved raw checkpoints,panels,classifier progress,policy diagnostics and decomposition.
One seed; do not equate classifier fit quality with successful probability transport.

## Matched velocity-coefficient ablation

User authorizes testing coefficient0 vs1 on this experiment. New output
shape_matched_dgpo_v0 reuses shape_matched_dgpo_v1/classifier.pt(selected3100)
and the SAME original reference.pt,not DGPO endpoint weights. No classifier
refit or resumed policy optimizer. Same source/config/1000stepbudget is checked.
Initial grid reward arrays,q,p,and conditional mode scores must be bitwise
identical to saved coefficient1 baseline. Both raw/Fourier arms use common
seed17,optimizer settings,candidates and timestep draws as coefficient1.

Only policy-loss change is removal of additive velocity reference penalty.
Save coefficient in checkpoint. Re-evaluate saved coefficient1 endpoints on
the same continuous panel with same frozen critic to report paired reward
contrasts and mode-TV/parity differences. Primary interpretation still needs
mode improvement,not only exploitable within-mode reward improvement.

If removing penalty rescues mode movement,it supports reference constraint at
this budget. If both remain poor,it rules out that penalty as the sole sufficient
explanation but does NOT uniquely establish conditioning: loss credit assignment,
gradient geometry,finite budget and learned within-mode score gradients remain.
Tests before launch:11 shape-decomposition/velocity-penalty tests pass.
